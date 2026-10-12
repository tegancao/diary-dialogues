"""Judge backend on vLLM (runs inside a Modal GPU container).

Formats each judge prompt as a single user message with the judge's own chat template, then answers the scoring
module's "what are the top next tokens?" requests in batches (greedy, top-20 log-probabilities, as in the paper).

Reasoning judges (gpt-oss) first think in a hidden section, then answer. For them the judge first generates the
reasoning (greedy), cuts the output right after the marker that opens the final answer, and reads the answer's
number probabilities from there. A judge that never reaches the marker (reasoning too long) scores None.
"""
import math

from diary_dialogues.judge.scoring import number_distribution, score_from_distribution

TOP_K = 20


def chat_prompt_ids(tokenizer, text: str, **chat_kwargs) -> list[int]:
    """Token ids of `text` as one user message in the judge's chat format, ready for the answer.

    Hugging Face tokenizers return a BatchEncoding (dict-like, but not a dict); Mistral's own tokenizer returns a list.
    """
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, **chat_kwargs)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    ids = list(ids)
    if not all(isinstance(i, int) for i in ids):
        raise TypeError(f"expected token ids, got {type(ids[0]).__name__}")
    return ids


def answer_prefix(output_ids: list[int], marker_ids: list[int]) -> list[int] | None:
    """The generated ids up to and including the first `marker_ids` (the start of the final answer), else None."""
    n = len(marker_ids)
    for i in range(len(output_ids) - n + 1):
        if output_ids[i:i + n] == marker_ids:
            return output_ids[:i + n]
    return None


class VLLMJudge:
    def __init__(self, model: str, revision: str, max_model_len: int = 16384, gpu_memory_utilization: float = 0.9,
                 reasoning: dict | None = None, **vllm_kwargs):
        """reasoning (for reasoning judges): {"chat_kwargs": ..., "answer_marker": str, "max_tokens": int}."""
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, revision=revision, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, **vllm_kwargs)
        self.tok = self.llm.get_tokenizer()
        self.eos = self.tok.eos_token_id
        self.params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOP_K)
        self.reasoning = reasoning
        if reasoning:
            self.marker = self.tok.encode(reasoning["answer_marker"], add_special_tokens=False)
            self.think_params = SamplingParams(max_tokens=reasoning["max_tokens"], temperature=0.0)

    @classmethod
    def from_config(cls, cfg: dict) -> "VLLMJudge":
        """Build from an entry of modal_setup.JUDGES."""
        return cls(cfg["model"], cfg["revision"], reasoning=cfg.get("reasoning"), **cfg["vllm_kwargs"])

    def prompt_ids(self, text: str) -> list[int]:
        return chat_prompt_ids(self.tok, text, **(self.reasoning or {}).get("chat_kwargs", {}))

    def _think(self, base: list[list[int]]) -> list[list[int] | None]:
        """Prompt + reasoning, cut where the final answer starts; None if the answer was never reached."""
        from vllm.inputs import TokensPrompt
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=b) for b in base], self.think_params, use_tqdm=False)
        prefixes = [answer_prefix(list(o.outputs[0].token_ids), self.marker) for o in outs]
        return [None if p is None else b + p for b, p in zip(base, prefixes)]

    def _next_token_logprobs(self, requests):
        from vllm.inputs import TokensPrompt
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=base + ext) for base, ext in requests], self.params,
                                 use_tqdm=False)
        tops = []
        for out in outs:
            lp = out.outputs[0].logprobs[0]
            tops.append([(tid, (l.decoded_token or ""), l.logprob, tid == self.eos) for tid, l in lp.items()])
        return tops

    def yes_probability(self, prompts: list[str], min_mass: float = 0.5) -> list[float | None]:
        """P(yes) / (P(yes) + P(no)) for the judge's first answer token, per prompt (None if yes/no together get less
        than min_mass of the probability). For reasoning judges, read after the reasoning, like score()."""
        base = [self.prompt_ids(p) for p in prompts]
        if self.reasoning:
            thought = self._think(base)
            ok = [i for i, t in enumerate(thought) if t is not None]
            tops = dict(zip(ok, self._next_token_logprobs([(thought[i], []) for i in ok])))
        else:
            tops = dict(enumerate(self._next_token_logprobs([(b, []) for b in base])))
        out = []
        for i in range(len(prompts)):
            if i not in tops:
                out.append(None); continue
            p = {"yes": 0.0, "no": 0.0}
            for _, text, logprob, _ in tops[i]:
                t = text.strip().lower()
                if t in p:
                    p[t] += math.exp(logprob)
            total = p["yes"] + p["no"]
            out.append(p["yes"] / total if total >= min_mass else None)
        return out

    def complete(self, prompts: list[str], max_tokens: int = 400) -> list[str | None]:
        """The judge's written answer to each prompt (greedy). For reasoning judges: the final answer after the
        reasoning (None if the reasoning didn't finish within its limit)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        budget = max_tokens + (self.reasoning["max_tokens"] if self.reasoning else 0)
        params = SamplingParams(max_tokens=budget, temperature=0.0)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=self.prompt_ids(p)) for p in prompts], params,
                                 use_tqdm=False)
        if not self.reasoning:
            return [o.outputs[0].text for o in outs]
        answers = []
        for o in outs:
            ids = list(o.outputs[0].token_ids)
            prefix = answer_prefix(ids, self.marker)
            answers.append(None if prefix is None else self.tok.decode(ids[len(prefix):], skip_special_tokens=True))
        return answers

    def score(self, prompts: list[str]) -> list[dict]:
        """For each judge prompt: the 0-100 score (None = refusal) and the probability mass on numbers."""
        base = [self.prompt_ids(p) for p in prompts]
        if not self.reasoning:
            dists = number_distribution(self._next_token_logprobs, base)
            return [{"score": score_from_distribution(d), "number_mass": round(sum(d.values()), 4)} for d in dists]
        thought = self._think(base)
        ok = [i for i, t in enumerate(thought) if t is not None]
        dists = dict(zip(ok, number_distribution(self._next_token_logprobs, [thought[i] for i in ok])))
        results = []
        for i, t in enumerate(thought):
            d = dists.get(i, {})
            results.append({"score": score_from_distribution(d), "number_mass": round(sum(d.values()), 4),
                            **({} if t is None else {"reasoning_tokens": len(t) - len(base[i])}),
                            **({"answer_not_reached": True} if t is None else {})})
        return results
