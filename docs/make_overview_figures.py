"""Illustrations for docs/project-overview.md, drawn from real run data.

    uv run python docs/make_overview_figures.py

Needs experiments/02-dialogues/bg_test.json (Rosa's entries) and dry_run3.json (replies). Style: diary_dialogues.figstyle.
"""
import json
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

from diary_dialogues import figstyle as fs
from diary_dialogues.design import BACKGROUNDS
from diary_dialogues.paths import ROOT

OUT = ROOT / "docs" / "figures"
RUNS = ROOT / "experiments" / "02-dialogues"


def text_column(ax, label, color, body, width=40, size=9, label_size=9):
    """A plain column: colored rule, small label, body text (spacing in points, so it holds at any axes size)."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot([0, 1], [1, 1], color=color, lw=2.2, solid_capstyle="butt")
    below = lambda pts: offset_copy(ax.transAxes, fig=ax.figure, y=-pts, units="points")
    ax.text(0, 1, label, color=color, fontsize=label_size, weight="semibold", va="top", transform=below(8))
    ax.text(0, 1, "\n".join(textwrap.wrap(body, width)), color=fs.INK, fontsize=size, va="top", linespacing=1.55,
            transform=below(12 + 1.6 * label_size))


def rosa_three_ways():
    convs = {c["id"]: c for c in json.loads((RUNS / "bg_test.json").read_text())["conversations"]}
    k = next(i for i, b in enumerate(BACKGROUNDS) if b["name"] == "Rosa")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.3), gridspec_kw={"wspace": 0.14})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.05)
    for ax, profile in zip(axes, ["control", "Social_Isolation", "Entitlement"]):
        text = " ".join(convs[f"{profile}-{k}-baseline"]["entries"][0].split()[:42]) + " …"
        label = "No nudge" if profile == "control" else f"Nudged: {fs.PROFILE_LABEL[profile].lower()}"
        text_column(ax, label, fs.PROFILE[profile], text, width=26, size=10.5, label_size=10)
    fs.heading(fig, "Rosa, 47, nurse: the same Monday, three ways",
               "First entries from a test run (Qwen2.5-7B, layer 15, strength 2.0). Identical instructions; only the "
               "nudge differs.")
    fs.save(fig, OUT / "o1-rosa-three-ways.png")


ABOUT = {"Daniel": "first-year college student, shares a dorm room in Columbus",
         "Aisha": "barista and part-time art student, two roommates, Philadelphia",
         "Sam": "software tester working from home, lives with their partner, Minneapolis",
         "Marcus": "warehouse shift supervisor, married, a two-year-old daughter, Atlanta",
         "Mei": "accountant, single, lives alone in Seattle",
         "Rosa": "hospital nurse, divorced, lives with her teenage son, San Antonio",
         "Tom": "electrician, married, grown children moved out, small-town Iowa",
         "Walter": "retired postal worker, widowed, lives alone, daughter nearby, Pittsburgh"}


def diarist_cards():
    fig, axes = plt.subplots(2, 4, figsize=(11, 3.6), gridspec_kw={"wspace": 0.15, "hspace": 0.3})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02)
    for ax, b in zip(axes.flat, BACKGROUNDS):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.plot([0, 1], [1, 1], color=fs.LINE, lw=1)
        age = b["card"].split(", ")[1]
        ax.text(0, 0.9, b["name"], fontsize=11, weight="semibold", color=fs.INK, va="top")
        ax.text(0.98, 0.9, age, fontsize=11, color=fs.MUTED, va="top", ha="right")
        ax.text(0, 0.68, f"{b['work']} · {b['household'].replace(', children moved out', ', empty nest')}",
                fontsize=7.5, color=fs.MUTED, va="top")
        ax.text(0, 0.5, "\n".join(textwrap.wrap(ABOUT[b["name"]], 32)), fontsize=8.5, color=fs.INK, va="top",
                linespacing=1.45)
    fs.heading(fig, "The eight diarists: each writes under every nudge", y=1.02)
    fs.save(fig, OUT / "o2-diarists.png")


def three_apps():
    convs = {c["id"]: c for c in json.loads((RUNS / "dry_run3.json").read_text())["conversations"]}
    entry = " ".join(convs["Social_Isolation-0-phi-4"]["entries"][0].split()[:24]) + " …"
    fig = plt.figure(figsize=(7.5, 5.0))
    top = fig.add_axes([0.01, 0.72, 0.98, 0.25])
    text_column(top, "The diarist shares an entry", fs.PROFILE["Social_Isolation"],
                "“I've been keeping a diary. Here's what I wrote today, I just wanted to share it with someone: "
                + entry + "”", width=88, size=10.5, label_size=10)
    for i, key in enumerate(["gemma-3-12b", "llama-3.1-8b", "phi-4"]):
        ax = fig.add_axes([0.01 + i * 0.335, 0, 0.31, 0.62])
        reply = convs[f"Social_Isolation-0-{key}"]["replies"][0].replace("**", "").replace("*", "")
        text_column(ax, f"{fs.APP_LABEL[key]} replies", fs.APP[key], " ".join(reply.split()[:32]) + " …",
                    width=26, size=10.5, label_size=10)
    fs.heading(fig, "The same entry, three chatbots", "First replies from dry run 3 (diarist nudged toward social "
               "isolation; excerpts).", y=1.0)
    fs.save(fig, OUT / "o3-three-chatbots.png")


CHECKS = {   # what a diary mentions -> pattern (case-insensitive)
    "works in an office": r"\b(office|boss|meeting)\b",
    "a colleague named Sarah": r"\bSarah\b",
    "a boss named Mr. Thompson": r"\bThompson\b",
    "a partner": r"\b(husband|wife|boyfriend|girlfriend|partner)\b",
    "children": r"\b(my (son|daughter|kids|children|toddler)|our (son|daughter|kids))\b",
    'signs off "[Your Name]"': r"Your Name",
}


def why_role_cards():
    exp = ROOT / "experiments"
    profiles = ["control", "Social_Isolation", "Enmeshment", "Entitlement", "Unrelenting_Standards"]
    before = list(json.loads((RUNS / "style_test_first_entries.json").read_text()).values())
    before += [d["text"] for d in json.loads((exp / "01-steered-diarists" / "screening_diaries.json").read_text())
               if d["words_target"] == 400 and d["schema"] in profiles]
    after = [c["entries"][0] for c in json.loads((RUNS / "bg_test.json").read_text())["conversations"]]
    share = lambda texts, pat: 100 * sum(bool(re.search(pat, t, re.I)) for t in texts) / len(texts)
    labels = list(CHECKS)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    y = range(len(labels))
    for off, texts, color, name in ((-0.18, before, fs.LINE, f"without role cards ({len(before)} diaries)"),
                                    (0.18, after, fs.ACCENT, f"with role cards ({len(after)} diaries)")):
        vals = [share(texts, CHECKS[k]) for k in labels]
        ax.barh([i + off for i in y], vals, 0.34, color=color, label=name)
        for i, v in zip(y, vals):
            ax.text(v + 1, i + off, f"{v:.0f}%", va="center", fontsize=7.5, color=fs.MUTED)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("share of diaries that mention it")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right")
    fs.heading(fig, "Without role cards, most diarists are the same office worker")
    fs.save(fig, OUT / "o4-why-role-cards.png")


if __name__ == "__main__":
    fs.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    rosa_three_ways(); diarist_cards(); three_apps(); why_role_cards()
    print(sorted(p.name for p in OUT.glob("o*.png")))
