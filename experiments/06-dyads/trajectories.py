"""Dyads v2, descriptive: mean focal-pattern score per letter, by condition and seat, for both judges; and candidate
examples of one friend picking up the other's pattern.

    uv run python experiments/06-dyads/trajectories.py [--examples N]

Seat A is the focal person, B the friend. Control pairs (two unsteered friends) are scored on both patterns.
"""
import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGES = ["mistral-small-3.1", "gpt-oss-120b"]
PATTERNS = ["Entitlement", "Social_Isolation"]
FRIENDS = ["same", "unsteered", "away", "neutral", "control"]


def load(judge):
    return json.loads((HERE / f"dyads_v2_items_scores_{judge}.json").read_text())["items"]


def table(items):
    """{(pattern, friend, side, round): [scores]}"""
    out = {}
    for it in items:
        friend = it["friend"] if it["focal"] else "control"
        for p in [it["focal"]] if it["focal"] else PATTERNS:
            s = it["scores"][p].get("score")
            if s is not None:
                out.setdefault((p, friend, it["side"], int(it["round"])), []).append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()
    tabs = {j: table(load(j)) for j in JUDGES}
    for p in PATTERNS:
        print(f"\n{p}: mean score per letter 1-3, Mistral | gpt-oss")
        for friend in FRIENDS:
            for side in "AB":
                cells = []
                for j in JUDGES:
                    v = [tabs[j].get((p, friend, side, r)) for r in range(3)]
                    cells.append(" ".join(f"{st.mean(x):4.0f}" if x else "   -" for x in v))
                who = "focal" if side == "A" else "friend"
                print(f"  {friend:<10}{who:<8}{cells[0]}  |  {cells[1]}")
    if not args.examples:
        return
    # Friends who start low and end high, under both judges: candidates for quoting.
    items = {j: {(i["dyad"], i["side"], int(i["round"])): i for i in load(j)} for j in JUDGES}
    m = items[JUDGES[0]]
    rows = []
    for (dyad, side, r), it in m.items():
        if side != "B" or r != 2 or it["focal"] is None or it["friend"] != "unsteered":
            continue
        p = it["focal"]
        try:
            rise = [items[j][(dyad, "B", 2)]["scores"][p]["score"] - items[j][(dyad, "B", 0)]["scores"][p]["score"]
                    for j in JUDGES]
        except (KeyError, TypeError):
            continue
        rows.append((min(rise), dyad, p))
    for rise, dyad, p in sorted(rows, reverse=True)[:args.examples]:
        print(f"\n=== {dyad}  friend rise (min over judges) {rise:+.0f}")
        for side, r in [("B", 0), ("A", 1), ("B", 2)]:
            it = m[(dyad, side, r)]
            sc = " / ".join(f"{items[j][(dyad, side, r)]['scores'][p]['score']:.0f}" for j in JUDGES)
            print(f"--- {it['name']} ({'focal' if side == 'A' else 'friend'}), letter {r + 1}, {p} {sc}")
            print(it["text"][:900])


if __name__ == "__main__":
    main()
