"""Second judge on the diary rubric: do Mistral and gpt-oss agree on what the chatbots do?

    uv run python experiments/05-flow-rubric/compare_judges.py

Runs main, main_opus, main_haiku (rubric v2). Per question: agreement and Cohen's kappa over the same replies; for the
behaviors in the findings, the share of replies to steered diarists per chatbot under each judge.
"""
import json

from diary_dialogues.benchmark import flow_rubric as F
from diary_dialogues.paths import DATA

RUNS = ["main", "main_opus", "main_haiku"]
JUDGES = ["", "_gpt-oss-120b"]
APPS = ["gemma-3-12b", "llama-3.1-8b", "phi-4", "claude-haiku-4.5", "claude-opus-5.5", "baseline"]
KEY = {"talks about the diary as writing": "talk about the", "asks no question": "fail to ask",
       "endorses the person's view": "endorse or amp", "judge sees risk": "Has the person said anything"}


def load(suffix):
    cases, qs = {}, None
    for run in RUNS:
        d = json.loads((DATA / "flow" / f"diary_{run}_v{F.VERSION}{suffix}.json").read_text())
        qs = d["questions"]
        cases.update({c["key"]: c for c in d["cases"]})
    return qs, cases


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def main():
    (qs, m), (qs2, g) = load(JUDGES[0]), load(JUDGES[1])
    assert qs == qs2
    keys = [k for k in m if k in g]
    yes = lambda c, i: c["p_yes"][i] is not None and c["p_yes"][i] >= .5
    both = lambda c, i: c["p_yes"][i] is not None
    print(f"{len(keys)} replies graded by both judges\n")
    print(f"{'question':<70}{'agree':>7}{'kappa':>7}{'yes M':>7}{'yes G':>7}")
    for i, q in enumerate(qs):
        ks = [k for k in keys if both(m[k], i) and both(g[k], i)]
        a = [yes(m[k], i) for k in ks]; b = [yes(g[k], i) for k in ks]
        kp, po = kappa(a, b)
        print(f"{q[:68]:<70}{po:7.0%}{kp:7.2f}{sum(a) / len(a):7.0%}{sum(b) / len(b):7.0%}")
    print("\nKey behaviors, replies to steered diarists (Mistral | gpt-oss):")
    for label, s in KEY.items():
        i = next(k for k, q in enumerate(qs) if s in q)
        cells = []
        for app in APPS:
            ks = [k for k in keys if m[k]["responder"] == app and m[k]["profile"] != "control"]
            if not ks:
                continue
            cells.append(f"{app.split('-')[0][:7]} {sum(yes(m[k], i) for k in ks) / len(ks):4.0%}|"
                         f"{sum(yes(g[k], i) for k in ks) / len(ks):4.0%}")
        print(f"   {label:<34}" + "  ".join(cells))


if __name__ == "__main__":
    main()
