"""Dyads, proof of concept: the focal pattern in each friend's letters, by round and partner type.

    uv run python experiments/06-dyads/analyze.py
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    items = json.loads((HERE / "dyads_items_scores_mistral-small-3.1.json").read_text())["items"]
    for focal in ("Entitlement", "Social_Isolation"):
        print(f"=== focal pattern: {focal} (judge score of that pattern, mean of 4 dyads; rounds 1 -> 3)")
        for partner in ("same", "unsteered", "opposite"):
            row = []
            for side in ("A", "B"):
                xs = [st.mean(it["scores"][focal]["score"] for it in items if it["focal"] == focal
                              and it["partner"] == partner and it["side"] == side and it["round"] == r) for r in range(3)]
                row.append(f"{side} {' -> '.join(f'{x:3.0f}' for x in xs)}")
            coh = st.mean(it["scores"]["coherence"]["score"] for it in items if it["focal"] == focal and it["partner"] == partner)
            print(f"   partner {partner:<10} {row[0]}   |   {row[1]}   | coherence {coh:.0f}")


if __name__ == "__main__":
    main()
