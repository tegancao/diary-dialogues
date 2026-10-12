"""Does the conversation history change the judge's verdicts? Full history vs. the latest exchange only.

    uv run python experiments/05-flow-rubric/history_effect.py

Same 384 app replies, same 15 questions: diary_main_v2 (the judge sees the whole conversation so far) vs.
human_view_v2 (only the latest entry and the reply, as the human raters see it). Per question: share of "yes" in
each view and how often the answers differ; split by turn (no history exists at turn 1, so turn 1 should match).
"""
import json

from diary_dialogues.benchmark import flow_rubric as F
from diary_dialogues.paths import DATA


def answers(path):
    d = json.loads(path.read_text())
    return d["questions"], {c["key"]: c for c in d["cases"]}


def main():
    qs, full = answers(DATA / "flow" / f"diary_main_v{F.VERSION}.json")
    qs2, latest = answers(DATA / "flow" / f"human_view_v{F.VERSION}.json")
    assert qs == qs2
    keys = [k for k in latest if k in full]
    yes = lambda c, j: c["p_yes"][j] is not None and c["p_yes"][j] >= .5
    print(f"{len(keys)} replies in both views\n")
    print(f"{'question (dimension)':<64}{'yes: full':>10}{'latest':>8}{'differ':>8}{'differ, turns 2-4':>19}")
    for j, q in enumerate(qs):
        dim = next(d for d, qq in F.questions() if qq == q)
        later = [k for k in keys if full[k]["turn"] > 0]
        print(f"{(q[:46] + '…  (' + dim.split('_')[0] + ')'):<64}"
              f"{sum(yes(full[k], j) for k in keys) / len(keys):10.0%}{sum(yes(latest[k], j) for k in keys) / len(keys):8.0%}"
              f"{sum(yes(full[k], j) != yes(latest[k], j) for k in keys) / len(keys):8.0%}"
              f"{sum(yes(full[k], j) != yes(latest[k], j) for k in later) / len(later):19.0%}")
    first = [k for k in keys if full[k]["turn"] == 0]
    diffs = sum(yes(full[k], j) != yes(latest[k], j) for k in first for j in range(len(qs)))
    print(f"\nsanity check, turn 1 (identical input in both views): {diffs} differing answers of {len(first) * len(qs)}")


if __name__ == "__main__":
    main()
