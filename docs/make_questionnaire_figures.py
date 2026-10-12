"""Figures for docs/self-report-vs-text.md.

    uv run python docs/make_questionnaire_figures.py

Needs experiments/01-steered-diarists/screening_questionnaire.json (questionnaire.py) and
screening_diaries_scores_mistral-small-3.1.json (judge_diaries.py). Writes PNGs to docs/figures/.
"""
import json
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diary_dialogues import figstyle as fs
from diary_dialogues.paths import ROOT
from diary_dialogues.simulator.questionnaire import SCALES

EXP = ROOT / "experiments" / "01-steered-diarists"
OUT = ROOT / "docs" / "figures"
Q = json.loads((EXP / "screening_questionnaire.json").read_text())
J = json.loads((EXP / "screening_diaries_scores_mistral-small-3.1.json").read_text())["items"]
assert all(q["schema"] == j["schema"] and q["words_target"] == j["words_target"] for q, j in zip(Q, J))
PROFILES = ["control", "Unrelenting_Standards", "Entitlement", "Social_Isolation", "Enmeshment"]
COLORS = fs.PROFILE
label = lambda s: s.replace("_", " ")


def two_views(words=400):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(SCALES)); w = 0.16
    for k, p in enumerate(PROFILES):
        qs = [q for q in Q if q["schema"] == p and q["words_target"] == words]
        js = [j for j in J if j["schema"] == p and j["words_target"] == words]
        self_rep = [st.mean(q[s]["score"] for q in qs) for s in SCALES]
        judged = [st.mean(j["scores"][schema]["score"] for j in js) for schema, _ in SCALES.values()]
        axes[0].bar(x + (k - 2) * w, self_rep, w, color=COLORS[p], label=label(p))
        axes[1].bar(x + (k - 2) * w, judged, w, color=COLORS[p])
    axes[0].set_xticks(x, [f"{label(s)}\n(↔ {label(sc)})" for s, (sc, _) in SCALES.items()], fontsize=9.5)
    axes[1].set_xticks(x, [label(sc) for sc, _ in SCALES.values()], fontsize=9.5)
    axes[0].set_ylim(1, 5); axes[0].set_ylabel("self-rating, mean of items (1–5)")
    axes[1].set_ylim(0, 100); axes[1].set_ylabel("judge score of the diary text (0–100)")
    axes[0].set_title("What the diarists say about themselves", fontsize=9.5)
    axes[1].set_title("What the judge reads in their diaries", fontsize=9.5)
    axes[0].legend(title="diarist steered toward", fontsize=8.5, title_fontsize=9, ncol=2, loc="upper left")
    fs.heading(fig, "Self-report and text tell different stories", f"~{words}-word screening diaries, 4 per profile.",
               note_y=-0.08)
    fs.save(fig, OUT / "q1-self-report-vs-judge.png")


def withdrawal_items():
    _, items = SCALES["Social_Withdrawal"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    y = np.arange(len(items))
    for p, off in (("control", -0.18), ("Social_Isolation", 0.18)):
        qs = [q for q in Q if q["schema"] == p]
        ax.barh(y + off, [st.mean(q["Social_Withdrawal"]["items"][i] for q in qs) for i in range(len(items))], 0.34,
                color=COLORS[p], label=label(p) + (" (unsteered)" if p == "control" else " diarists"))
    ax.set_yticks(y, [("(reverse) " if k < 0 else "") + t for t, k in items], fontsize=9.5)
    ax.invert_yaxis(); ax.set_xlim(1, 5); ax.axvline(3, color=fs.LINE, lw=.8, ls=":")
    ax.set_xlabel("rating: 1 = very inaccurate … 5 = very accurate (all three lengths)")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.spines["left"].set_visible(False)
    fs.heading(fig, "Unsteered diarists agree with opposite statements; steered ones don't")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fs.save(fig, OUT / "q2-withdrawal-items.png")


def scatter():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (scale, (schema, _)) in zip(axes, SCALES.items()):
        for p in ["other"] + PROFILES:
            sel = [(j["scores"][schema]["score"], q[scale]["score"]) for q, j in zip(Q, J)
                   if (j["schema"] == p if p != "other" else j["schema"] not in PROFILES)]
            if sel:
                a, b = zip(*sel)
                ax.scatter(a, b, s=16 if p == "other" else 26, alpha=.35 if p == "other" else .85,
                           color=fs.LINE if p == "other" else COLORS[p], label="other 14 schemas" if p == "other" else label(p))
        a = [j["scores"][schema]["score"] for j in J]; b = [q[scale]["score"] for q in Q]
        r = np.corrcoef(a, b)[0, 1]
        ax.set_title(f"{label(scale)} vs. judged {label(schema).lower()} (r = {r:.2f})", fontsize=9.5)
        ax.set_xlabel("judge score (0–100)"); ax.set_xlim(-3, 100)
    axes[0].set_ylabel("self-rating (1–5)"); axes[0].set_ylim(1, 5)
    axes[2].legend(fontsize=8, loc="lower right")
    fs.heading(fig, "Per diary, self-ratings barely follow the text", "All 228 screening diaries; r = Pearson correlation.",
               note_y=-0.05)
    fs.save(fig, OUT / "q3-per-diary.png")


if __name__ == "__main__":
    fs.apply()
    two_views(); withdrawal_items(); scatter()
    print(sorted(p.name for p in OUT.glob("q*.png")))
