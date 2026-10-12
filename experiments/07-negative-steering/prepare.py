"""Proof of concept: what appears when a schema is steered AWAY from (negative strength)?

    uv run --with modal modal run experiments/02-dialogues/run.py --responders baseline --entries 1 --per-cell 8 \
        --out neg_pilot --coef -2.0                                            # first entries at −2.0
    uv run python experiments/07-negative-steering/prepare.py                 # -> items.json
    uv run --with modal modal run --detach experiments/01-steered-diarists/judge_diaries.py \
        --input $PWD/experiments/07-negative-steering/items.json --schemas all
    uv run python experiments/07-negative-steering/analyze.py

The same 8 diarists (role cards) per pattern: steered toward it (main run, design strength), away from it (−2.0),
and unsteered; all first entries, judged on all 18 schemas plus coherence.
"""
import json
from pathlib import Path

from diary_dialogues.design import MAIN_SCHEMAS
from diary_dialogues.harness.conversation import Conversation

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "02-dialogues"


def main():
    items = []
    for direction, run in (("toward", "main"), ("away", "neg_pilot")):
        first = json.loads((RUNS / f"{run}_first_entries.json").read_text())
        for key, text in first.items():
            profile, k = key.rsplit("-", 1)
            if profile == "control" and direction == "away":
                continue                                           # the unsteered entries are taken from main once
            items.append({"profile": profile, "diarist": int(k),
                          "direction": "unsteered" if profile == "control" else direction,
                          "question": Conversation("", profile, "", 0, 400).entry_prompt(0), "text": text})
    (HERE / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"{len(items)} first entries -> items.json ({', '.join(MAIN_SCHEMAS)} toward/away + unsteered)")


if __name__ == "__main__":
    main()
