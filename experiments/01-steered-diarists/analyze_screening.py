"""Summarize the screening: per schema and entry length, how strongly and how specifically does it come through?

    uv run experiments/01-steered-diarists/analyze_screening.py [--judge mistral-small-3.1]

target       mean judge score (0-100) on the schema the diary was steered toward
specificity  target minus the mean score on the other 17 schemas (high = this schema, not generic negative mood)
rank         rank of the target among the 18 judged schemas (1 = judged most present)
Writes screening_summary_<judge>.csv and screening_heatmap_<judge>.png next to this file.
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diary_dialogues.paths import SCHEMAS

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="mistral-small-3.1")
    judge = ap.parse_args().judge
    d = json.loads((HERE / f"screening_diaries_scores_{judge}.json").read_text())
    items, lengths = d["items"], sorted({it["words_target"] for it in d["items"]})
    sc = lambda it, s: it["scores"][s]["score"]
    rows, mats = [], {}
    for L in lengths:
        M = np.zeros((len(SCHEMAS) + 1, len(SCHEMAS)))
        for r, steered in enumerate(["control"] + SCHEMAS):
            its = [it for it in items if it["words_target"] == L and it["schema"] == steered]
            M[r] = [st.mean(sc(it, s) for it in its if sc(it, s) is not None) for s in SCHEMAS]
            if steered == "control":
                continue
            t = SCHEMAS.index(steered)
            ranks = [1 + sum(M[r] > M[r, t])]
            rows.append({"schema": steered, "words": L, "target": round(M[r, t], 1),
                         "off_target": round(float(np.delete(M[r], t).mean()), 1),
                         "specificity": round(M[r, t] - float(np.delete(M[r], t).mean()), 1), "rank": ranks[0],
                         "control_target": round(M[0, t], 1),
                         "coherence": round(st.mean(sc(it, "coherence") for it in its), 1), "n": len(its)})
        mats[L] = M
    with (HERE / f"screening_summary_{judge}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fig, axes = plt.subplots(1, len(lengths), figsize=(8.6 * len(lengths), 8.5), gridspec_kw={"wspace": 0.42})
    labels = [s.replace("_", " ") for s in SCHEMAS]
    for ax, L in zip(axes, lengths):
        im = ax.imshow(mats[L], cmap="magma", vmin=0, vmax=100)
        ax.set_xticks(range(18), labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(19), ["control"] + labels, fontsize=8)
        for r in range(1, 19):
            ax.add_patch(plt.Rectangle((r - 1.5, r - 0.5), 1, 1, fill=False, ec="cyan", lw=1.2))
        ax.set_title(f"~{L}-word diaries", fontsize=13, weight="bold")
        ax.set_xlabel("judged as"); ax.set_ylabel("steered toward")
    fig.colorbar(im, ax=axes, fraction=0.015, label="judge score (0-100)")
    fig.suptitle(f"Schema screening: steered vs. judged schema (layer 15 x 2.0; judge {judge}); cyan = own schema",
                 fontsize=14, weight="bold")
    fig.savefig(HERE / f"screening_heatmap_{judge}.png", dpi=110, bbox_inches="tight")

    print(f"{'schema':<26}" + "".join(f"  ~{L}w: tgt spec rk coh" for L in lengths))
    for s in SCHEMAS:
        r = {x["words"]: x for x in rows if x["schema"] == s}
        print(f"{s:<26}" + "".join(f"  {r[L]['target']:5.0f} {r[L]['specificity']:+4.0f} {r[L]['rank']:2d} {r[L]['coherence']:3.0f}" for L in lengths))


if __name__ == "__main__":
    main()
