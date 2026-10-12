"""Turn a judge's next-token probabilities into a 0-100 score (the paper's logprob method).

The judge is asked to answer with a number from 0 to 100. Instead of reading the single sampled answer, use the
probability the judge gives each possible answer and return their probability-weighted average. If less than 0.25 of
the probability is on numbers, treat it as a refusal (None): the authors' _aggregate_0_100_score rule.

weighted_score: the paper's version, for judges whose tokenizer writes every number 0-100 as one token (Llama, OLMo):
    the first token's top probabilities already cover all answers.
number_distribution: the general version for tokenizers that split numbers into digits (Mistral: "74" -> "7", "4"):
    walk a small tree, P("74") = P("7") x P("4" | "7") x P(end | "74"), skipping branches with negligible probability.
    For single-token numbers the tree is one level deep and matches weighted_score.
"""
import math
import re
from collections.abc import Callable

DIGITS = re.compile(r"[0-9]+")   # plain ASCII digits only: str.isdigit() also accepts e.g. '₀' or '²'


def weighted_score(top_logprobs: dict[str, float], min_mass: float = 0.25) -> float | None:
    """top_logprobs: {token text: log-probability} for the judge's first output token."""
    total = weighted = 0.0
    for token, logprob in top_logprobs.items():
        if not DIGITS.fullmatch(token.strip()):
            continue
        value = int(token.strip())
        if 0 <= value <= 100:
            p = math.exp(logprob)
            total += p
            weighted += value * p
    return weighted / total if total >= min_mass else None


def score_from_distribution(dist: dict[int, float], min_mass: float = 0.25) -> float | None:
    total = sum(dist.values())
    return sum(n * p for n, p in dist.items()) / total if total >= min_mass else None


def number_distribution(next_token_logprobs: Callable[[list], list[list[tuple]]], prompts: list,
                        min_prob: float = 1e-4) -> list[dict[int, float]]:
    """Probability of each complete answer 0..100, for many judge prompts at once.

    next_token_logprobs(requests) gets a list of (prompt, extra_token_ids) and returns, for each, the judge's top
    next tokens as (token_id, token_text, logprob, is_end). An answer ends at the first token that isn't digits.
    Requests are batched level by level, so a GPU backend can process them together.
    """
    dists = [dict() for _ in prompts]
    frontier = [(i, [], "", 1.0) for i in range(len(prompts))]        # (prompt index, token ids, digits, prob)
    while frontier:
        tops = next_token_logprobs([(prompts[i], ids) for i, ids, _, _ in frontier])
        nxt = []
        for (i, ids, digits, prob), top in zip(frontier, tops):
            for token_id, text, logprob, is_end in top:
                p = prob * math.exp(logprob)
                t = text.strip()
                is_digits = DIGITS.fullmatch(t) is not None
                if digits and (is_end or not is_digits):               # the number ends here
                    dists[i][int(digits)] = dists[i].get(int(digits), 0.0) + p
                elif is_digits and not is_end:
                    value = digits + t
                    if (len(value) > 1 and value[0] == "0") or int(value) > 100 or p < min_prob:
                        continue                                       # not a valid answer, or negligible
                    nxt.append((i, ids + [token_id], value, p))
        frontier = nxt
    return dists
