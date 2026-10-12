"""How many entries does a conversation need? Measures by entry number for a scored run.

    uv run python experiments/02-dialogues/analyze_length.py dry_run3

Needs <run>.json (run.py) and <run>_scores.json (judge_run.py). By entry number:
  diarist  own-schema score per profile (controls: mean of the 4 schemas), coherence, diary realism, writes as an AI
  replies  addresses the person (judge), length in words, overlap with the same conversation's earlier replies
Prints the table and writes <run>_by_entry.png.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

from diary_dialogues.design import MAIN_SCHEMAS, PROFILES

HERE = Path(__file__).resolve().parent
AI = re.compile(r"\b(as an AI|AI language model|I am an AI|I'm an AI|my programming)\b", re.I)


def trigrams(text: str) -> set:
    w = re.findall(r"\w+", text.lower())
    return set(zip(w, w[1:], w[2:]))


def repetition(replies: list[str]) -> list[float | None]:
    """For each reply: the largest share of its word trigrams found in one earlier reply (None for the first)."""
    sets = [trigrams(r) for r in replies]
    return [None] + [max(len(s & p) / max(len(s), 1) for p in sets[:t]) for t, s in enumerate(sets) if t > 0]


def own(scores: dict, profile: str) -> float:
    if profile == "control":
        return st.mean(scores[s]["score"] for s in MAIN_SCHEMAS)
    return scores[profile]["score"]


def table(run: str) -> list[dict]:
    convs = json.loads((HERE / f"{run}.json").read_text())["conversations"]
    d = json.loads((HERE / f"{run}_scores.json").read_text())
    rep = {c["id"]: repetition(c["replies"]) for c in convs if c["responder"] != "baseline"}
    responders = sorted({r["responder"] for r in d["replies"]})
    rows = []
    for t in range(max(e["turn"] for e in d["entries"]) + 1):
        es = [e for e in d["entries"] if e["turn"] == t]
        rs = [r for r in d["replies"] if r["turn"] == t]
        row = {"entry": t + 1}
        for p in PROFILES:
            row[p] = st.mean(own(e["scores"], p) for e in es if e["profile"] == p)
        row["coherence"] = st.mean(e["scores"]["coherence"]["score"] for e in es)
        row["realism"] = st.mean(e["scores"]["realism"]["score"] for e in es)
        row["as AI"] = sum(bool(AI.search(e["text"])) for e in es)
        for x in responders:
            row[f"addr {x.split('-')[0]}"] = st.mean(r["scores"]["addresses_person"]["score"] for r in rs
                                                     if r["responder"] == x)
        row["reply words"] = st.mean(len(r["reply"].split()) for r in rs)
        reps = [rep[r["id"]][t] for r in rs if rep[r["id"]][t] is not None]
        row["repetition"] = st.mean(reps) if reps else None
        rows.append(row)
    return rows


def plot(run: str, rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["entry"] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for p in PROFILES:
        axes[0].plot(x, [r[p] for r in rows], marker="o",
                     label=p.replace("_", " ") + (" (mean of 4 schemas)" if p == "control" else ""))
    axes[0].set_title("Diarist: own-schema score", weight="bold"); axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=7.5, loc="lower left")
    for key, c in (("coherence", "black"), ("realism", "#0f766e")):
        axes[1].plot(x, [r[key] for r in rows], marker="o", color=c, label=key)
    axes[1].set_title("Diarist: coherence and realism", weight="bold"); axes[1].set_ylim(0, 100); axes[1].legend()
    for key in [k for k in rows[0] if k.startswith("addr ")]:
        axes[2].plot(x, [r[key] for r in rows], marker="o", label=key[5:])
    axes[2].set_title("Replies: address the person", weight="bold"); axes[2].set_ylim(0, 100); axes[2].legend()
    axes[3].plot(x, [r["repetition"] for r in rows], marker="o", color="#c2410c")
    axes[3].set_ylim(0, 1); axes[3].set_title("Replies: overlap with earlier replies", weight="bold")
    for ax in axes:
        ax.set_xlabel("entry number")
    fig.suptitle(f"{run}: measures by entry number (judge Mistral-Small-3.1)", weight="bold")
    fig.savefig(HERE / f"{run}_by_entry.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "dry_run3"
    rows = table(run)
    cols = list(rows[0])
    print(" ".join(f"{c[:11]:>11}" for c in cols))
    for r in rows:
        print(" ".join(f"{r[c]:11.1f}" if isinstance(r[c], float) else f"{str(r[c]):>11}" for c in cols))
    plot(run, rows)
