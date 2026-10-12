"""Negative steering, proof of concept: scores toward / away / unsteered, and which schemas rise when steering away.

    uv run python experiments/07-negative-steering/analyze.py
"""
import json
import statistics as st
from pathlib import Path

from diary_dialogues.design import MAIN_SCHEMAS
from diary_dialogues.paths import SCHEMAS

HERE = Path(__file__).resolve().parent


def main():
    d = json.loads((HERE / "items_scores_mistral-small-3.1.json").read_text())
    items = d["items"]
    sc = lambda it, s: it["scores"][s]["score"]
    unsteered = [it for it in items if it["direction"] == "unsteered"]
    base = {s: st.mean(sc(it, s) for it in unsteered) for s in SCHEMAS}
    print(f"unsteered (8 diarists): coherence {st.mean(sc(it, 'coherence') for it in unsteered):.0f}\n")
    for p in MAIN_SCHEMAS:
        print(f"=== {p}")
        for direction in ("toward", "away"):
            xs = [it for it in items if it["profile"] == p and it["direction"] == direction]
            print(f"   {direction:<7} own schema {st.mean(sc(it, p) for it in xs):5.1f} (unsteered {base[p]:.1f}) | "
                  f"coherence {st.mean(sc(it, 'coherence') for it in xs):4.0f}")
        away = [it for it in items if it["profile"] == p and it["direction"] == "away"]
        rises = sorted(((st.mean(sc(it, s) for it in away) - base[s], s) for s in SCHEMAS), reverse=True)[:4]
        falls = sorted(((st.mean(sc(it, s) for it in away) - base[s], s) for s in SCHEMAS))[:3]
        print("   steering away, biggest rises vs unsteered: " + ", ".join(f"{s} {v:+.0f}" for v, s in rises))
        print("   steering away, biggest falls vs unsteered: " + ", ".join(f"{s} {v:+.0f}" for v, s in falls))


if __name__ == "__main__":
    main()
