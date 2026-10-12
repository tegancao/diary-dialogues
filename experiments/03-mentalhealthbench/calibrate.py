"""The shared 300-conversation sample: five models, two judges, and placement on the published scale.

    uv run --with modal modal volume get diary-dialogues-results mhb/grades_<judge>_sample300.json data/mhb/
    uv run python experiments/03-mentalhealthbench/calibrate.py

For each judge: task-clipped scores per model with the paper's 95% intervals (mean ± 1.96 s/√N over conversations,
appendix B.3). Calibration: the two published Claude models (Haiku 4.5 41.7, Opus 5.5 52.4; paper figure 5) give a
straight line from our judge's score to the published scale; our models are placed on it. Placing a model outside
the anchors' range extends the line and is marked as such. Writes calibration.csv (aggregates only).
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

from diary_dialogues.benchmark.rubric import score
from diary_dialogues.benchmark.sample import SAMPLE_SIZE, stratified_ids
from diary_dialogues.paths import DATA

HERE = Path(__file__).resolve().parent
PUBLISHED = {"claude-haiku-4.5": 41.7, "claude-opus-5.5": 52.4}     # task-clipped %, GPT-5.6 Sol grader
OURS = ["gemma-3-12b", "llama-3.1-8b", "phi-4", "gemma-3-27b", "llama-3.3-70b", "gemma-4-31b"]
JUDGES = ["mistral-small-3.1", "gpt-oss-120b"]


def main():
    rows = [json.loads(line) for line in (DATA / "mhb" / "mentalhealthbench_eval.jsonl").read_text().splitlines()]
    keep = set(stratified_ids(rows))
    rows = [r for r in rows if r["id"] in keep]
    out = []
    for judge in JUDGES:
        path = DATA / "mhb" / f"grades_{judge}_sample{SAMPLE_SIZE}.json"
        if not path.exists():
            print(f"{judge}: no grades yet ({path.name})"); continue
        g = json.loads(path.read_text())
        total = sum(len(v) for m in g["verdicts"].values() for v in m.values())
        print(f"\n=== judge {judge}: {total} verdicts, {g['unreadable']} unreadable")
        scores = {}
        for m in g["models"]:
            xs = [100 * score(r["rubric_items"], g["verdicts"][m][r["id"]]) for r in rows]
            mean, half = st.mean(xs), 1.96 * st.stdev(xs) / math.sqrt(len(xs))
            scores[m] = (mean, half)
        (h, _), (o, _) = scores["claude-haiku-4.5"], scores["claude-opus-5.5"]
        slope = (PUBLISHED["claude-opus-5.5"] - PUBLISHED["claude-haiku-4.5"]) / (o - h)
        place = lambda x: PUBLISHED["claude-haiku-4.5"] + slope * (x - h)
        print(f"calibration line: published = {PUBLISHED['claude-haiku-4.5']} + {slope:.2f} × (ours − {h:.1f})")
        print(f"{'model':<18}{'our judge':>10}  95% CI        {'published':>10}{'placed on published scale':>28}")
        for m in g["models"]:
            mean, half = scores[m]
            pub = PUBLISHED.get(m)
            placed = None if pub else place(mean)
            note = "" if pub else ("  (extends below the anchors)" if mean < min(h, o) else "")
            print(f"{m:<18}{mean:10.1f}  [{mean - half:.1f}, {mean + half:.1f}]{(f'{pub:10.1f}' if pub else '         –')}"
                  f"{(f'{placed:28.1f}' if placed is not None else ''):>28}{note}")
            out.append({"judge": judge, "model": m, "score": round(mean, 2), "ci_half": round(half, 2),
                        "published": pub, "placed": None if placed is None else round(placed, 2)})
    if out:
        with (HERE / "calibration.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
