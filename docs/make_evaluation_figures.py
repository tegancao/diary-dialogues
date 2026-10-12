"""Picture for docs/evaluation.md: three ways to grade a reply, side by side.

    uv run python docs/make_evaluation_figures.py
"""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from diary_dialogues import figstyle as fs
from diary_dialogues.paths import ROOT

OUT = ROOT / "docs" / "figures"
WAYS = [   # title, color, steps (reads, criteria, judge, result), verdict, verdict is good
    ("Benchmark criteria", fs.APP["claude-opus-5.5"],
     ["A benchmark conversation + the chatbot's reply",
      "2–15 criteria written by clinicians for this exact case",
      "Is each criterion met?",
      "Score = points met ÷ points possible"],
     "The reference, but only exists for benchmark cases", True),
    ("Borrowed criteria", fs.MUTED,
     ["Any conversation + the chatbot's reply",
      "150 general clinician criteria; the judge picks the relevant ones",
      "Is each criterion met?",
      "Score per behavior axis"],
     "Ranks models correctly, but per reply ≈ random (ρ = 0.04). Not used", False),
    ("Behavior rubric", fs.ACCENT,
     ["A diary conversation + the chatbot's reply",
      "15 fixed yes/no questions on 4 behaviors (after VERA-MH)",
      "Answers each question",
      "% of replies showing each behavior"],
     "Used for the diaries, graded by two judges", True),
]
LABELS = ["reads", "criteria", "judge", "result"]


def three_ways():
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02)
    for ax, (title, color, steps, verdict, good) in zip(axes, WAYS):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.plot([0, 1], [1, 1], color=color, lw=2.2, solid_capstyle="butt")
        ax.text(0, 0.95, title, color=color, fontsize=10.5, weight="semibold", va="top")
        for k, (lab, step) in enumerate(zip(LABELS, steps)):
            y = 0.86 - k * 0.18
            ax.add_patch(FancyBboxPatch((0, y - 0.13), 1, 0.13, boxstyle="round,pad=0,rounding_size=0.02",
                                        fc=fs.SOFT, ec="none"))
            ax.text(0.03, y - 0.065, lab, color=fs.MUTED, fontsize=7.5, va="center")
            ax.text(0.22, y - 0.065, "\n".join(textwrap.wrap(step, 40)), color=fs.INK, fontsize=8.5, va="center",
                    linespacing=1.35)
        ax.text(0, 0.06, ("✓  " if good else "✗  ") + "\n    ".join(textwrap.wrap(verdict, 52)),
                color=fs.INK if good else fs.MUTED, fontsize=8.5, va="center", linespacing=1.35)
    fs.heading(fig, "Three ways to grade a reply")
    fs.save(fig, OUT / "e0-three-ways.png")


if __name__ == "__main__":
    fs.apply()
    three_ways()
    print("e0-three-ways.png")
