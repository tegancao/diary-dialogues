"""Validate borrowed criteria on MentalHealthBench: does it agree with the official criterion-by-criterion scores?

    uv run python experiments/04-reply-grading/validate.py

Same replies (100 sample conversations × our 3 models), two methods:
  official    the conversation's own clinician criteria, paper's grader prompt (Mistral; grades_..._sample300.json)
  borrowed criteria: reused bank criteria with a relevance check (6 axes) and axis-level scores (4 axes) (mhb_validation100)
Per axis: Spearman correlation across replies, and whether both methods rank the three models the same way.
"""
import json
import math
import statistics as st
from pathlib import Path

from diary_dialogues.benchmark.bank import CRITERIA_AXES
from diary_dialogues.benchmark.reply_grading import AXIS_DEFINITIONS, RELEVANT_MIN
from diary_dialogues.benchmark.rubric import score
from diary_dialogues.paths import DATA



def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):                      # average ranks for ties
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else float("nan")




def plot(overall, models):
    """docs/figures/e1-borrowed-criteria.png: per reply, and per model (mean ± 95% CI), official vs. borrowed criteria."""
    import random as rnd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diary_dialogues import figstyle as fs
    from diary_dialogues.paths import ROOT
    fs.apply()
    rng = rnd.Random(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.25})
    for m in models:
        pts = [(o, x) for o, x, mm in overall if mm == m]
        a.scatter([o + rng.uniform(-.015, .015) for o, _ in pts], [x + rng.uniform(-.015, .015) for _, x in pts], s=13,
                  alpha=.45, color=fs.APP[m], label=fs.APP_LABEL[m], lw=0)
        ci = lambda v: 1.96 * st.stdev(v) / math.sqrt(len(v))
        o, x = [p[0] for p in pts], [p[1] for p in pts]
        b.errorbar(st.mean(o), st.mean(x), xerr=ci(o), yerr=ci(x), fmt="o", ms=7, capsize=0, color=fs.APP[m],
                   label=fs.APP_LABEL[m])
    rho = spearman([p[0] for p in overall], [p[1] for p in overall])
    for ax, title in ((a, f"Each reply (rank correlation {rho:.2f})"), (b, "Each model: mean with 95% interval")):
        ax.plot([0, 1], [0, 1], ls=":", color=fs.MUTED, lw=1)
        ax.grid(axis="x", visible=True)
        ax.set_xlim(-.05, 1.05) if ax is a else ax.set_xlim(0, .7)
        ax.set_ylim(-.05, 1.05) if ax is a else ax.set_ylim(0, .8)
        ax.set_xlabel("official score (the case's own criteria)")
        ax.set_ylabel("borrowed-criteria score")
        ax.set_title(title, fontsize=9.5)
    b.legend(loc="lower right")
    fs.heading(fig, "Borrowed criteria rank models, not replies",
               "Same benchmark replies graded two ways (judge: Mistral-Small-3.1). Dotted line: perfect agreement.",
               note_y=-0.03)
    out = ROOT / "docs" / "figures" / "e1-borrowed-criteria.png"
    fs.save(fig, out)
    print(f"figure: {out.relative_to(ROOT)}")


def main():
    rows = {r["id"]: r for r in map(json.loads, (DATA / "mhb" / "mentalhealthbench_eval.jsonl").read_text().splitlines())}
    official = json.loads((DATA / "mhb" / "grades_mistral-small-3.1_sample300.json").read_text())["verdicts"]
    b = json.loads((DATA / "mhb_validation100.json").read_text())
    extra = DATA / "mhb_validation100_claude.json"            # the Claude anchors' replies, for a wide quality range
    if extra.exists():
        b["cases"] += json.loads(extra.read_text())["cases"]
    criteria = b["criteria"]
    models = list(dict.fromkeys(c["model"] for c in b["cases"]))
    print(f"{len(b['cases'])} replies; bank of {len(criteria)} criteria; relevant at >= {RELEVANT_MIN}")
    n_rel = [sum(1 for s in c["relevance"] if s is not None and s >= RELEVANT_MIN) for c in b["cases"]]
    print(f"relevant bank criteria per reply: median {st.median(n_rel)}, range {min(n_rel)}-{max(n_rel)}\n")

    def method_b(case, axis):
        if axis in AXIS_DEFINITIONS:
            return case["axis_scores"][axis]
        items = [{"points": criteria[int(k)]["points"], "behavior_axis": criteria[int(k)]["axis"]}
                 for k in case["verdicts"]]
        return score(items, list(case["verdicts"].values()), axis)

    def official_score(case, axis):
        r = rows[case["id"]]
        return score(r["rubric_items"], official[case["model"]][case["id"]], axis)

    def pooled(case):
        items = [{"points": criteria[int(k)]["points"], "behavior_axis": criteria[int(k)]["axis"]} for k in case["verdicts"]]
        return score(items, list(case["verdicts"].values()))

    overall = [(official_score(c, None), pooled(c), c["model"]) for c in b["cases"]]
    overall = [p for p in overall if p[1] is not None]
    print(f"Overall (all criteria; {len(overall)} replies): per-reply Spearman "
          f"{spearman([p[0] for p in overall], [p[1] for p in overall]):.2f}")
    print(f"{'model':<18}{'official':>10}{'borrowed':>10}")
    for m in sorted(models, key=lambda m: -st.mean(p[0] for p in overall if p[2] == m)):
        print(f"{m:<18}{st.mean(p[0] for p in overall if p[2] == m):10.3f}{st.mean(p[1] for p in overall if p[2] == m):10.3f}")
    plot(overall, models)
    means = {m: (st.mean(p[0] for p in overall if p[2] == m), st.mean(p[1] for p in overall if p[2] == m)) for m in models}
    print(f"model-level Spearman (official vs borrowed-criteria means): "
          f"{spearman([means[m][0] for m in models], [means[m][1] for m in models]):.2f}\n")

    print(f"{'axis':<52}{'pairs':>6}{'Spearman':>10}   model order: official | borrowed")
    for axis in [*CRITERIA_AXES, *AXIS_DEFINITIONS]:
        pairs = [(official_score(c, axis), method_b(c, axis), c["model"]) for c in b["cases"]]
        pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
        if len(pairs) < 10:
            print(f"{axis:<52}{len(pairs):6d}   (too few)"); continue
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        short = lambda m: m.replace("claude-", "").split("-")[0] if not m.startswith("claude") else m.replace("claude-", "")
        present = [m for m in models if any(p[2] == m for p in pairs)]
        order = lambda i: " > ".join(short(m) for m in sorted(
            present, key=lambda m: -st.mean(p[i] for p in pairs if p[2] == m)))
        print(f"{axis:<52}{len(pairs):6d}{rho:10.2f}   {order(0)} | {order(1)}")


if __name__ == "__main__":
    main()
