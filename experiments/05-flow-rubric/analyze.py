"""Does the flow rubric transfer? Compare it with the official benchmark scores on the same replies.

    uv run python experiments/05-flow-rubric/analyze.py

For each dimension: ratings per model (share best / suboptimal / poor / not relevant), the per-reply rank
correlation between the numeric rating (best 1, suboptimal 0.5, poor 0) and the official score of the matching
MentalHealthBench axis (paper's prompt, Mistral), and whether both order the five models the same way.
"""
import json
import statistics as st
import sys
from collections import Counter

from diary_dialogues.benchmark import flow_rubric as F
from diary_dialogues.benchmark.rubric import score
from diary_dialogues.paths import DATA, ROOT

sys.path.insert(0, str(ROOT / "experiments/04-reply-grading"))
from validate import spearman  # noqa: E402

MODELS = ["claude-opus-5.5", "claude-haiku-4.5", "gemma-3-12b", "phi-4", "llama-3.1-8b"]


def ratings(case: dict, qs: list[str]) -> dict[str, str | None]:
    yes = {q: (None if p is None else p >= 0.5) for q, p in zip(qs, case["p_yes"])}
    return {d: F.rate(d, yes) for d in F.DIMENSIONS}


def official_axis(rows, official, case, axes):
    r = rows[case["id"]]
    pairs = [(it, v) for it, v in zip(r["rubric_items"], official[case["model"]][case["id"]])
             if it["behavior_axis"] in axes]
    return score([it for it, _ in pairs], [v for _, v in pairs]) if pairs else None


def main():
    d = json.loads((DATA / "flow" / f"mhb_validation100_v{F.VERSION}.json").read_text())
    qs = d["questions"]
    rows = {r["id"]: r for r in map(json.loads, (DATA / "mhb" / "mentalhealthbench_eval.jsonl").read_text().splitlines())}
    official = json.loads((DATA / "mhb" / "grades_mistral-small-3.1_sample300.json").read_text())["verdicts"]
    missing = sum(p is None for c in d["cases"] for p in c["p_yes"])
    print(f"{len(d['cases'])} replies x {len(qs)} questions; unreadable answers: {missing}\n")
    R = {c["key"]: ratings(c, qs) for c in d["cases"]}
    for dim, axes in F.MATCHING_AXES.items():
        print(f"=== {dim}  (official axis: {', '.join(a.split('_')[0] for a in axes)})")
        for m in MODELS:
            cnt = Counter(R[c["key"]][dim] for c in d["cases"] if c["model"] == m)
            n = sum(cnt.values())
            print(f"   {m:<17} " + "  ".join(f"{k[:6]} {cnt[k] / n:4.0%}" for k in (F.BEST, F.SUBOPTIMAL, F.POOR, F.NOT_RELEVANT)))
        pairs = [(official_axis(rows, official, c, axes), F.NUMERIC.get(R[c["key"]][dim]), c["model"]) for c in d["cases"]]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        if len(pairs) < 10:
            print("   too few replies with both scores\n"); continue
        means = {m: (st.mean([p[0] for p in pairs if p[2] == m] or [float("nan")]),
                     st.mean([p[1] for p in pairs if p[2] == m] or [float("nan")])) for m in MODELS}
        order = lambda i: " > ".join(m.replace("claude-", "").split("-")[0] if not m.startswith("claude")
                                     else m.replace("claude-", "") for m in sorted(MODELS, key=lambda m: -means[m][i]))
        print(f"   per-reply Spearman with the official axis score: {spearman([p[0] for p in pairs], [p[1] for p in pairs]):.2f}"
              f" ({len(pairs)} replies)")
        print(f"   model-level Spearman: {spearman([means[m][0] for m in MODELS], [means[m][1] for m in MODELS]):.2f}")
        print(f"   order official: {order(0)}\n   order rubric:   {order(1)}\n")


if __name__ == "__main__":
    main()
