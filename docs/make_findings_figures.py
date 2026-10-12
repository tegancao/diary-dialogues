"""Figures for docs/findings.md.

    uv run python docs/make_findings_figures.py

Needs data/flow/diary_main_v2.json (experiments/05-flow-rubric/grade.py --source diary) and
experiments/03-mentalhealthbench/calibration.csv. Writes PNGs to docs/figures/.
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diary_dialogues import figstyle as fs
from diary_dialogues.benchmark import flow_rubric as F
from diary_dialogues.paths import DATA, ROOT

OUT = ROOT / "docs" / "figures"
APPS = ["gemma-3-12b", "llama-3.1-8b", "phi-4", "claude-haiku-4.5", "claude-opus-5.5"]
PROFILES = ["control", "Social_Isolation", "Entitlement", "Unrelenting_Standards"]
D = F.DIMENSIONS
SHORT = {"control": "none", "Social_Isolation": "isolation", "Entitlement": "entitlement",
         "Unrelenting_Standards": "standards"}
BEHAVIORS = [   # label, conditions (question, required answer), question
    ("talks about the diary as writing", [], D["engages_with_person"]["flow"][0][0]),
    ("asks no question at all", [], D["explores_before_advising"]["flow"][1][0]),
    ("endorses the person's rigid view", [(D["responds_to_pattern"]["gate"], True)], D["responds_to_pattern"]["flow"][0][0]),
    ("judge sees risk in the conversation", [], D["calibrated_escalation"]["branch"]),
]


def behavior_grid():
    d = json.loads((DATA / "flow" / f"diary_main_v{F.VERSION}.json").read_text())
    for run in ("main_opus", "main_haiku"):        # Claude responders, same first entries, graded separately
        extra = DATA / "flow" / f"diary_{run}_v{F.VERSION}.json"
        if extra.exists():
            d["cases"] += json.loads(extra.read_text())["cases"]
    qs = d["questions"]
    cases = [{**c, "yes": {q: (p is not None and p >= .5) for q, p in zip(qs, c["p_yes"])}} for c in d["cases"]]
    apps = [a for a in APPS if any(c["responder"] == a for c in cases)]
    fig, axes = plt.subplots(1, len(BEHAVIORS), figsize=(12, 2.4), sharey=True, gridspec_kw={"wspace": 0.06})
    fig.subplots_adjust(top=0.8, bottom=0.12)
    for ax, (label, conds, q) in zip(axes, BEHAVIORS):
        M = np.full((len(apps), len(PROFILES)), np.nan)
        for r, a in enumerate(apps):
            for col, p in enumerate(PROFILES):
                xs = [c for c in cases if c["responder"] == a and c["profile"] == p and all(c["yes"][g] == v for g, v in conds)]
                if xs:
                    M[r, col] = 100 * sum(c["yes"][q] for c in xs) / len(xs)
        ax.imshow(M, cmap=fs.CMAP, vmin=0, vmax=130, aspect="auto")
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        for r in range(len(apps)):
            for col in range(len(PROFILES)):
                v = M[r, col]
                ax.text(col, r, "–" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 70 else fs.INK)
        ax.set_xticks(range(len(PROFILES)), [SHORT[p] for p in PROFILES], fontsize=7.5)
        ax.set_yticks(range(len(apps)), [fs.APP_LABEL[a] for a in apps])
        ax.set_title(label, fontsize=9.5)
    fs.heading(fig, "What the chatbots do, by who is writing (% of replies)",
               "Columns: the diarist's pattern. Judge: Mistral-Small-3.1, behavior rubric v2; 32 replies per cell; "
               "'endorses' only where the person states a rigid view.", note_y=0.0)
    fs.save(fig, OUT / "f1-behaviors.png")


def calibration():
    rows = list(csv.DictReader((ROOT / "experiments/03-mentalhealthbench/calibration.csv").open()))
    published = {"GPT-6 Astra": 57.3, "GPT-6 Sol": 53.9, "Claude Opus 5.5": 52.4, "GPT-6 Luna": 50.2,
                 "Muse Spark 1.3": 48.6, "GPT-5.6 Sol": 47.0, "Claude Fable 5.1": 46.4, "GPT-5.6 Luna": 44.9,
                 "Claude Sonnet 5": 44.5, "GPT-5 Thinking": 42.9, "Claude Haiku 4.5": 41.7, "Grok 4.7": 41.3,
                 "Gemini 3.8 Flash": 35.5, "Gemini 2.5 Flash": 33.5, "GPT-4o (Mar 2025)": 32.1,
                 "Gemini 3.1 Pro": 32.1, "Gemini 2.5 Pro": 29.5}
    ours = {r["model"]: (float(r["score"]), float(r["ci_half"])) for r in rows if r["judge"] == "mistral-small-3.1"}
    names = {"gemma-3-12b": "Gemma-3-12B (ours)", "phi-4": "Phi-4 (ours)", "llama-3.1-8b": "Llama-3.1-8B (ours)",
             "gemma-3-27b": "Gemma-3-27B (ours)", "llama-3.3-70b": "Llama-3.3-70B (ours)",
             "gemma-4-31b": "Gemma-4-31B (ours)",
             "claude-haiku-4.5": "Claude Haiku 4.5 (our grader)", "claude-opus-5.5": "Claude Opus 5.5 (our grader)"}
    items = [(n, v, None, "published") for n, v in published.items()] + \
            [(names[m], s, h, "ours" if "claude" not in m else "anchor") for m, (s, h) in ours.items()]
    items.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, 5.6))
    fig.subplots_adjust(top=0.93)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)
    colors = {"published": fs.LINE, "ours": fs.ACCENT, "anchor": fs.APP["claude-opus-5.5"]}
    for y, (n, v, h, kind) in enumerate(items):
        ax.barh(y, v, color=colors[kind], height=.62)
        if h:
            ax.errorbar(v, y, xerr=h, color=fs.MUTED, capsize=0, lw=1)
        ax.text(v + (h or 0) + .8, y, f"{v:.1f}", va="center", fontsize=7.5, color=fs.MUTED)
    ax.set_yticks(range(len(items)), [i[0] for i in items])
    ax.set_xlabel("MentalHealthBench task-clipped score (%)")
    ax.set_xlim(0, 66)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, ["published (GPT-5.6 Sol grader, all 1,215 cases)", "our models (Mistral grader, 300 cases)",
                        "published models through our grader"], loc="upper left", bbox_to_anchor=(-0.4, -0.09))
    fs.heading(fig, "Where our models sit on the published MentalHealthBench scale")
    fs.save(fig, OUT / "f2-benchmark-placement.png")


def steering_away():
    """s1: own-pattern score on the first entry, steered toward, unsteered, steered away (8 diarists each)."""
    import statistics as st
    items = json.loads((ROOT / "experiments/07-negative-steering/items_scores_mistral-small-3.1.json").read_text())["items"]
    pats = ["Social_Isolation", "Entitlement", "Unrelenting_Standards"]
    rows = [("toward", "steered toward"), ("unsteered", "unsteered"), ("away", "steered away")]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True); ax.spines["left"].set_visible(False)
    for i, p in enumerate(pats):
        for j, (key, label) in enumerate(rows):
            xs = [it["scores"][p]["score"] for it in items
                  if it["direction"] == key and (key == "unsteered" or it["profile"] == p)]
            v = st.mean(xs); y = i * 4 + j
            color = fs.PROFILE[p] if key == "toward" else (fs.LINE if key == "unsteered" else fs.MUTED)
            ax.barh(y, v, 0.8, color=color)
            ax.text(v + 1, y, f"{label}  {v:.0f}", va="center", fontsize=7.5, color=fs.MUTED)
    ax.set_yticks([i * 4 + 1 for i in range(3)], [fs.PROFILE_LABEL[p] for p in pats])
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("pattern score of the first entry (0–100)")
    fs.heading(fig, "Steering away lowers a pattern below an ordinary day",
               "8 diarists per bar; strength +2.0 (standards +2.5) toward, −2.0 away; judge: Mistral-Small-3.1.",
               note_y=-0.06)
    fs.save(fig, OUT / "s1-steering-away.png")


def dyads():
    """s2: dyads v2: mean focal-pattern score per letter for the steered writer and the friend, by kind of friend
    (32 pairs per line; primary judge)."""
    import statistics as st
    import sys
    sys.path.insert(0, str(ROOT / "experiments/06-dyads"))
    from trajectories import load, table
    tab = table(load("mistral-small-3.1"))
    panels = [("A", "the steered writer", ["same", "neutral", "unsteered", "away"]),
              ("B", "their friend", ["same", "unsteered", "away", "control"])]
    label = {"same": "friend steered the same way", "unsteered": "friend unsteered", "away": "friend steered away",
             "neutral": "fixed neutral letter", "control": "two unsteered friends"}
    style = lambda focal, cond: {"same": (fs.PROFILE[focal], "-"), "unsteered": (fs.INK, "-"),
                                 "away": (fs.MUTED, "-"), "neutral": (fs.LINE, "--"), "control": (fs.LINE, "--")}[cond]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharey=True, gridspec_kw={"hspace": 0.5, "wspace": 0.75})
    for r, focal in enumerate(["Entitlement", "Social_Isolation"]):
        for c, (side, who, conds) in enumerate(panels):
            ax = axes[r, c]
            ends = []
            for cond in conds:
                v = [st.mean(tab[(focal, cond, "B" if cond == "control" else side, k)]) for k in range(3)]
                color, ls = style(focal, cond)
                ax.plot(range(3), v, ls, color=color, lw=1.8, marker="o", ms=3.5)
                ends.append([v[2], cond, color])
            ends.sort()                     # spread end labels so they don't overlap
            for i in range(1, len(ends)):
                ends[i][0] = max(ends[i][0], ends[i - 1][0] + 7)
            for y, cond, color in ends:
                text = label[cond] if side == "A" else label[cond].removeprefix("friend ")
                ax.text(2.12, y, text, va="center", fontsize=7.5, color=color if color != fs.LINE else fs.MUTED)
            ax.set_xticks(range(3), ["letter 1", "letter 2", "letter 3"])
            ax.set_xlim(-0.15, 2.1); ax.set_ylim(0, 100)
            ax.set_title(f"{fs.PROFILE_LABEL[focal]} · {who}", fontsize=9, loc="left")
            if c == 0:
                ax.set_ylabel(f"{fs.PROFILE_LABEL[focal].lower()} score")
    fs.heading(fig, "Two steered friends, letter by letter",
               "Mean score of the steered writer's pattern in each letter; 32 pairs per line; judge: Mistral-Small-3.1 "
               "(gpt-oss-120b gives the same picture).", note_y=0.0)
    fs.save(fig, OUT / "s2-dyads.png")


if __name__ == "__main__":
    fs.apply()
    behavior_grid(); calibration(); steering_away(); dyads()
    print("f1-behaviors.png, f2-benchmark-placement.png")
