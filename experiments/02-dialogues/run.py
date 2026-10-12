"""Generate diary conversations: steered diarists x response models, turn by turn.

    uv run --with modal modal run experiments/02-dialogues/run.py --responders phi-4,baseline --entries 8 --per-cell 1

For each profile (control + the main schemas, diary_dialogues.design) and each of `per_cell` diarists, the first
entry is written once and shared by all responders (a paired design: every responder answers the same first entry).
Then, turn by turn: every responder replies to its conversations, and the diarist writes the next entry (3 days
later) with the reply in context. Steering (layer 15 x 2.0) applies to all of the diarist's writing.

First entries are kept in <out>_first_entries.json, so responders added later answer the same first entries.
Writes <out>.json next to this file after every turn; rerunning the same command resumes from it.
"""
import json
from pathlib import Path

import modal

from diary_dialogues import design
from diary_dialogues.harness.conversation import DATES, Conversation
from diary_dialogues.modal_setup import VOLUMES, api_image, hf_token_secret, transformers_image, vllm_image, with_vectors
from diary_dialogues.paths import DIARIST_MODEL, DIARIST_REVISION
from diary_dialogues.responder.models import (API_RESPONDERS, BASELINE, BASELINE_REPLY, MAX_MODEL_LEN, RESPONDERS,
                                              SAMPLING)

HERE = Path(__file__).resolve().parent
# Qwen writes ~550 words when asked for ~400; a 700-token limit cut off ~60% of entries mid-sentence (dry run 1).
DIARIST_SAMPLING = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": int(design.ENTRY_WORDS * 2.5)}
CHUNK = 10                                       # diarist conversations per generate() call
app = modal.App("diary-dialogues-conversations")


@app.cls(image=with_vectors(transformers_image), gpu="L40S", volumes=VOLUMES, timeout=30 * 60,
         scaledown_window=20 * 60)
class Diarist:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(DIARIST_MODEL, revision=DIARIST_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(DIARIST_MODEL, revision=DIARIST_REVISION,
                                                          dtype=torch.bfloat16, device_map="cuda").eval()

    @modal.method()
    def write(self, conversations: list[list[dict]], profiles: list[str], coefs: list[float], seed: int) -> list[str]:
        """One new entry per conversation, steered toward its profile with its own strength (0 = control)."""
        import torch
        from diary_dialogues.simulator.diarist import write_batch
        from diary_dialogues.simulator.steering import load_vector, steer
        zero = torch.zeros(self.model.config.hidden_size)
        vec = {p: load_vector(p, design.LAYER, "/vectors") for p in set(profiles) if p != "control"}
        texts = []
        for start in range(0, len(conversations), CHUNK):
            rows = torch.stack([vec.get(p, zero) * c for p, c in zip(profiles[start:start + CHUNK],
                                                                      coefs[start:start + CHUNK])])
            with steer(self.model, rows, design.LAYER, 1.0):          # each row already carries its strength
                texts += write_batch(self.model, self.tok, conversations[start:start + CHUNK], seed=seed + start,
                                     **DIARIST_SAMPLING)
        return texts


@app.cls(image=vllm_image, gpu="L40S", volumes=VOLUMES, secrets=[hf_token_secret()], timeout=30 * 60,
         scaledown_window=20 * 60)
class Responder:
    name: str = modal.parameter()

    @modal.enter()
    def load(self):
        from vllm import LLM
        cfg = RESPONDERS[self.name]
        self.llm = LLM(model=cfg["model"], revision=cfg["revision"], max_model_len=MAX_MODEL_LEN,
                       gpu_memory_utilization=0.9, **cfg["vllm_kwargs"])

    @modal.method()
    def reply(self, conversations: list[list[dict]], seeds: list[int]) -> list[str]:
        from vllm import SamplingParams
        params = [SamplingParams(seed=s, **SAMPLING) for s in seeds]
        outs = self.llm.chat(conversations, params, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]


@app.function(image=api_image,
              secrets=[modal.Secret.from_name("anthropic-secret")], timeout=30 * 60)
def reply_anthropic(model: str, conversations: list[list[dict]]) -> list[str]:
    """A frontier responder through Anthropic's API: the same length cap as the open responders and the
    model's default sampling (the SDK takes no temperature for it, as in the benchmark run), no
    system prompt. An organization-level key names its workspace via ANTHROPIC_WORKSPACE_ID (optional)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    import anthropic
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(max_retries=8, default_headers={"anthropic-workspace-id": ws} if ws else None)

    def ask(msgs):
        out = client.messages.create(model=model, messages=msgs, max_tokens=SAMPLING["max_tokens"])
        return "".join(b.text for b in out.content if b.type == "text").strip()

    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(ask, conversations))


def save(path: Path, convs: list[Conversation], meta: dict):
    path.write_text(json.dumps({**meta, "conversations": [c.to_dict() for c in convs]}, indent=1, ensure_ascii=False))


@app.local_entrypoint()
def main(responders: str = "phi-4,baseline", entries: int = 8, per_cell: int = 1, out: str = "dry_run",
         first: str = "", backgrounds: bool = True, coef: float = 0.0, uptake: str = "none"):
    """first: reuse the first entries of another run (its `out` name), so separate runs stay paired.
    backgrounds: diarist k of every profile gets design.BACKGROUNDS[k] (dry runs 1-3: --no-backgrounds).
    coef: one steering strength for all steered profiles, for strength tests (0 = design.coef per profile)."""
    strength = lambda p: 0.0 if p == "control" else (coef or design.coef(p))
    names = responders.split(",")
    unknown = [n for n in names if n != BASELINE and n not in RESPONDERS and n not in API_RESPONDERS]
    if unknown or entries > len(DATES) or (backgrounds and per_cell > len(design.BACKGROUNDS)):
        raise SystemExit(f"unknown responders {unknown}" if unknown else
                         f"at most {len(DATES)} entries and {len(design.BACKGROUNDS)} diarists per profile")
    bg = lambda key: design.BACKGROUNDS[int(key.rsplit("-", 1)[1])] if backgrounds else None
    meta = {"diarist": {"model": DIARIST_MODEL, "revision": DIARIST_REVISION, "layer": design.LAYER,
                        "coef": {p: strength(p) for p in design.PROFILES}, "words": design.ENTRY_WORDS,
                        "backgrounds": backgrounds, "sampling": DIARIST_SAMPLING},
            "responders": {n: RESPONDERS.get(n) or API_RESPONDERS.get(n) or {"reply": BASELINE_REPLY} for n in names},
            "responder_sampling": SAMPLING, "entries": entries, "uptake": uptake}
    diarist = Diarist()

    # First entries: written once per (profile, k), shared by every responder.
    first_path = HERE / f"{first or out}_first_entries.json"
    first = json.loads(first_path.read_text()) if first_path.exists() else {}
    keys = [f"{p}-{k}" for p in design.PROFILES for k in range(per_cell)]
    missing = [key for key in keys if key not in first]
    if missing:
        msgs = [Conversation("", key.rsplit("-", 1)[0], "", 0, design.ENTRY_WORDS, background=bg(key)).diarist_messages()
                for key in missing]
        profiles = [key.rsplit("-", 1)[0] for key in missing]
        texts = diarist.write.remote(msgs, profiles, [strength(p) for p in profiles], seed=0)
        first.update(zip(missing, texts))
        first_path.write_text(json.dumps(first, indent=1, ensure_ascii=False))
    print(f"first entries: {len(keys)} ({len(missing)} new)")

    convs = [Conversation(f"{key}-{n}", key.rsplit("-", 1)[0], n, int(key.rsplit("-", 1)[1]), design.ENTRY_WORDS,
                          entries=[first[key]], background=bg(key), uptake=uptake) for key in keys for n in names]
    path = HERE / f"{out}.json"
    start = 0
    if path.exists():                        # resume after an interruption: continue from the saved turns
        saved = {c["id"]: c for c in json.loads(path.read_text())["conversations"]}
        if set(saved) != {c.id for c in convs}:
            raise SystemExit(f"{path.name} holds other conversations; pass a new --out")
        for c in convs:
            c.entries, c.replies = saved[c.id]["entries"], saved[c.id]["replies"]
        done = {len(c.replies) for c in convs}
        if len(done) != 1:
            raise SystemExit(f"{path.name} is inconsistent (conversations at different turns)")
        start = done.pop()
        print(f"resuming {path.name} after turn {start}")
    for turn in range(start, entries):
        calls = {}
        for n in names:
            group = [c for c in convs if c.responder == n]
            if n in API_RESPONDERS:
                calls[n] = reply_anthropic.spawn(API_RESPONDERS[n]["model"], [c.responder_messages() for c in group])
            elif n != BASELINE:
                calls[n] = Responder(name=n).reply.spawn([c.responder_messages() for c in group],
                                                         [1000 * c.seed + turn for c in group])
        for n in names:
            group = [c for c in convs if c.responder == n]
            texts = [BASELINE_REPLY] * len(group) if n == BASELINE else calls[n].get()
            for c, t in zip(group, texts):
                c.replies.append(t)
        if turn + 1 < entries:
            texts = diarist.write.remote([c.diarist_messages() for c in convs], [c.profile for c in convs],
                                         [strength(c.profile) for c in convs], seed=1000 * (turn + 1))
            for c, t in zip(convs, texts):
                c.entries.append(t)
        save(path, convs, meta)
        print(f"turn {turn + 1}/{entries}: {len(convs)} conversations, saved {path.name}")
