"""Diary replies under the flow rubric: behavior rates per app and profile.

    uv run python experiments/05-flow-rubric/diary_rates.py

Needs data/flow/diary_main.json (grade.py --source diary). For each dimension, the share of replies rated best
practice / suboptimal / poor (and not relevant), per app × profile, plus specific behaviors read from single
questions: crisis resources urged although no risk was expressed, the message treated as a text, no question asked,
the pattern endorsed or only mirrored. Writes diary_rates.csv.
"""
import csv
import json
from collections import Counter
from pathlib import Path

from diary_dialogues.benchmark import flow_rubric as F
from diary_dialogues.paths import DATA

HERE = Path(__file__).resolve().parent
APPS = ["gemma-3-12b", "llama-3.1-8b", "phi-4", "claude-haiku-4.5", "claude-opus-5.5", "baseline"]
PROFILES = ["control", "Social_Isolation", "Entitlement", "Unrelenting_Standards"]
D = F.DIMENSIONS
BEHAVIORS = {    # label -> (gate questions that must be "yes" or "no" first, question)
    "risk expressed": ([], D["calibrated_escalation"]["branch"]),
    "crisis line without risk": ([(D["calibrated_escalation"]["branch"], False)], D["calibrated_escalation"]["if_no_risk"][0][0]),
    "treated as a text": ([], D["engages_with_person"]["flow"][0][0]),
    "asks no question": ([], D["explores_before_advising"]["flow"][1][0]),
    "rigid view present": ([], D["responds_to_pattern"]["gate"]),
    "endorses the view": ([(D["responds_to_pattern"]["gate"], True)], D["responds_to_pattern"]["flow"][0][0]),
    "only mirrors the view": ([(D["responds_to_pattern"]["gate"], True)], D["responds_to_pattern"]["flow"][1][0]),
}


def main():
    d = json.loads((DATA / "flow" / f"diary_main_v{F.VERSION}.json").read_text())
    for run in ("main_opus", "main_haiku"):        # Claude responders, same first entries, graded separately
        extra = DATA / "flow" / f"diary_{run}_v{F.VERSION}.json"
        if extra.exists():
            d["cases"] += json.loads(extra.read_text())["cases"]
    qs = d["questions"]
    cases = []
    for c in d["cases"]:
        yes = {q: (None if p is None else p >= 0.5) for q, p in zip(qs, c["p_yes"])}
        cases.append({**c, "yes": yes, "rating": {dim: F.rate(dim, yes) for dim in F.DIMENSIONS}})
    print(f"{len(cases)} replies; unreadable answers: {sum(p is None for c in d['cases'] for p in c['p_yes'])}\n")

    rows = []
    for label, (conds, q) in BEHAVIORS.items():
        print(f"{label}")
        for app in APPS:
            cells = []
            for p in PROFILES:
                xs = [c for c in cases if c["responder"] == app and c["profile"] == p
                      and all(c["yes"].get(g) is v for g, v in conds)]
                rate = sum(bool(c["yes"].get(q)) for c in xs) / len(xs) if xs else None
                cells.append("   –" if rate is None else f"{rate:4.0%}")
                rows.append({"behavior": label, "app": app, "profile": p, "n": len(xs),
                             "rate": None if rate is None else round(rate, 3)})
            print(f"   {app:<13} " + "  ".join(f"{p[:12]:>12} {v}" for p, v in zip(PROFILES, cells)))
    print("\nDimension ratings, share 'best practice' among relevant replies:")
    for dim in F.DIMENSIONS:
        print(f"   {dim}")
        for app in APPS:
            cells = []
            for p in PROFILES:
                rs = Counter(c["rating"][dim] for c in cases if c["responder"] == app and c["profile"] == p)
                rel = sum(v for k, v in rs.items() if k in F.NUMERIC)
                cells.append(f"{rs[F.BEST] / rel:4.0%}" if rel else "   –")
            print(f"     {app:<13} " + "  ".join(f"{p[:12]:>12} {v}" for p, v in zip(PROFILES, cells)))
    with (HERE / "diary_rates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
