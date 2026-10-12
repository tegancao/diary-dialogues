"""Figures for docs/steering-explained.md.

    uv run python docs/make_steering_figures.py

Needs data/ems (scripts/download_data.py) and experiments/01-steered-diarists/steering_demo.json (steering_demo.py).
Writes PNGs to docs/figures/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from diary_dialogues import figstyle as fs
from diary_dialogues.paths import ROOT, SCHEMAS, VECTOR_DIR

OUT = ROOT / "docs" / "figures"
DEMO = ROOT / "experiments" / "01-steered-diarists" / "steering_demo.json"
POS, NEG, STEER = fs.APP["claude-opus-5.5"], fs.ACCENT, fs.APP["llama-3.1-8b"]


def vec(schema):
    return torch.load(VECTOR_DIR / f"{schema}_response_avg_diff.pt").float()


def toy_difference_of_means():
    """A 2-D cartoon of step 2-3: two clouds of hidden states, the arrow between their means."""
    rng = np.random.default_rng(0)
    neg = rng.normal([0, 0], 0.55, (40, 2)); pos = rng.normal([2.2, 1.3], 0.55, (40, 2))
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(*neg.T, s=18, c=NEG, alpha=.55, label="replies told to avoid the schema")
    ax.scatter(*pos.T, s=18, c=POS, alpha=.55, label="replies told to show the schema")
    mn, mp = neg.mean(0), pos.mean(0)
    for m, c in ((mn, NEG), (mp, POS)):
        ax.scatter(*m, s=160, c=c, edgecolor=fs.PAPER, zorder=3, marker="X")
    ax.annotate("", xy=mp, xytext=mn, arrowprops=dict(arrowstyle="-|>", lw=2.5, color=fs.INK))
    ax.text(*(mn + mp) / 2 + [-.35, .55], "schema vector\n= mean(show) − mean(avoid)", ha="right", fontsize=10,
            color=fs.INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel("(real hidden states have 3,584 dimensions; this is a 2-D cartoon)", fontsize=8, color=fs.MUTED)
    ax.legend(loc="lower right", fontsize=9)
    fs.heading(fig, "Extracting a schema vector: difference of means")
    fs.save(fig, OUT / "1-difference-of-means.png")


def norms_by_layer(hidden_norm):
    fig, ax = plt.subplots(figsize=(6.8, 4))
    for s, c in (("Unrelenting_Standards", STEER), ("Abandonment", POS), ("Entitlement", NEG)):
        ax.plot(range(29), vec(s).norm(dim=1), marker="o", ms=3.5, color=c, label=s.replace("_", " "))
    ax.axvline(15, color=fs.LINE, ls="--", lw=1)
    ax.text(15.4, ax.get_ylim()[1] * .9, "layer 15\n(used for steering)", fontsize=8, color=fs.MUTED)
    ax.set_xlabel("layer (0 = embeddings, 28 = last block)"); ax.set_ylabel("length of the schema vector")
    fs.heading(fig, "Schema vectors are stored for every layer")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fs.save(fig, OUT / "2-vector-length-by-layer.png")


def size_of_the_nudge(demo):
    h = np.median([n for x in demo["plain"] for n in x["norm"]])
    nudge = demo["coef"] * demo["vector_norm"]
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.barh([1, 0], [h, nudge], color=[fs.LINE, STEER], height=.55)
    ax.set_yticks([1, 0], ["hidden state at layer 15\n(median over diary tokens)", f"nudge = {demo['coef']:g} × vector"])
    for y, v in ((1, h), (0, nudge)):
        ax.text(v + 1, y, f"{v:.1f}", va="center", fontsize=10)
    ax.set_xlim(0, h * 1.15); ax.set_xlabel("vector length")
    fs.heading(fig, f"The nudge is about {nudge / h:.0%} of the state it is added to")
    fs.save(fig, OUT / "4-size-of-the-nudge.png")


def cosine_matrix():
    V = torch.stack([vec(s)[15] for s in SCHEMAS]); U = V / V.norm(dim=1, keepdim=True); C = (U @ U.T).numpy()
    fig, ax = plt.subplots(figsize=(8.4, 7.4))
    im = ax.imshow(C, cmap=fs.DIVERGING, vmin=-1, vmax=1); ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    labels = [s.replace("_", " ") for s in SCHEMAS]
    ax.set_xticks(range(18), labels, rotation=60, ha="right", fontsize=8.5); ax.set_yticks(range(18), labels, fontsize=8.5)
    for i in range(18):
        for j in range(18):
            ax.text(j, i, f"{C[i, j]:.1f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=6.5,
                    color="white" if abs(C[i, j]) > .6 else fs.INK)
    cb = fig.colorbar(im, fraction=.04, label="cosine similarity"); cb.outline.set_visible(False)
    fs.heading(fig, "How similar the 18 schema vectors are (layer 15)")
    fs.save(fig, OUT / "6-vector-similarity.png")


def projection_traces(demo, window=15):
    fig, ax = plt.subplots(figsize=(7.4, 4))
    for key, c, lab in (("plain", NEG, "unsteered diaries"), ("steered", STEER, "steered diaries (layer 15 × 2.0)")):
        for k, x in enumerate(demo[key]):
            p = np.array(x["proj"]); sm = np.convolve(p, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(len(sm)) + window // 2, sm, color=c, alpha=.8, lw=1.4, label=lab if k == 0 else None)
        m = np.mean([np.mean(x["proj"]) for x in demo[key]])
        ax.axhline(m, color=c, ls="--", lw=1)
        ax.text(1.01, m, f"mean\n{m:.1f}", color=c, fontsize=9, va="center", transform=ax.get_yaxis_transform())
    ax.set_xlabel("token position in the diary"); ax.set_ylabel("projection onto the schema direction")
    fs.heading(fig, "Re-read without steering, the text itself carries the schema")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fs.save(fig, OUT / "5-projection-per-token.png")


if __name__ == "__main__":
    fs.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    demo = json.loads(DEMO.read_text())
    toy_difference_of_means(); norms_by_layer(None); size_of_the_nudge(demo); cosine_matrix(); projection_traces(demo)
    print(sorted(p.name for p in OUT.glob("*.png")))
