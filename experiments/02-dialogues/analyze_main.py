"""Main run: did the steering work, does it hold, and do the apps' replies change the diarists?

    uv run python experiments/02-dialogues/analyze_main.py

Needs main.json (run.py) and main_scores.json (judge_run.py). Prints:
  1. manipulation check: own-schema score of the first entry, steered vs. the same diarists unsteered (paired)
  2. own-schema score by entry number for each app (and the baseline), per profile
  3. change from entry 1 to entry 4, each app minus the baseline, with a bootstrap interval over diarists
  4. coherence, realism and "talks to the person" by app
Writes docs/figures/f0-patterns-over-time.png.
"""
import json
import random
import statistics as st
from pathlib import Path

from diary_dialogues.design import MAIN_SCHEMAS

HERE = Path(__file__).resolve().parent
RUNS = ["main", "main_opus", "main_haiku"]
APPS = ["gemma-3-12b", "llama-3.1-8b", "phi-4", "claude-haiku-4.5", "claude-opus-5.5", "baseline"]


def load():
    convs, d = {}, {"entries": [], "replies": []}
    for run in RUNS:                     # main, plus the frontier responder run on the same first entries if present
        if not (HERE / f"{run}_scores.json").exists():     # not judged yet
            continue
        convs.update({c["id"]: c for c in json.loads((HERE / f"{run}.json").read_text())["conversations"]})
        s = json.loads((HERE / f"{run}_scores.json").read_text())
        d["entries"] += s["entries"]; d["replies"] += s["replies"]
    # judge_run.py scored each distinct (profile, diarist, text) once: the shared first entries, and (once in this
    # run) an entry the diarist copied word for word from its previous one. Look scores up by text.
    by_text = {(convs[e["id"]]["profile"], convs[e["id"]]["seed"], e["text"]): e["scores"] for e in d["entries"]}
    entries = {(c["profile"], c["seed"], c["responder"], t): by_text[(c["profile"], c["seed"], text)]
               for c in convs.values() for t, text in enumerate(c["entries"])}
    copies = sum(c["entries"][t] in c["entries"][:t] for c in convs.values() for t in range(1, len(c["entries"])))
    print(f"entries copied word for word from an earlier entry: {copies} of {sum(len(c['entries']) - 1 for c in convs.values())}\n")
    return convs, entries, d["replies"]


def own(scores, profile):
    return scores[profile]["score"]


def boot(values, n=4000, seed=0):
    rng = random.Random(seed)
    means = sorted(st.mean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    convs, E, replies = load()
    seeds = sorted({c["seed"] for c in convs.values()})
    APPS[:] = [a for a in APPS if any(k[2] == a for k in E)]      # only apps whose runs are judged
    print("1. Manipulation check, first entry (8 diarists): own-schema score steered vs. the same diarists unsteered")
    for p in MAIN_SCHEMAS:
        diffs = [own(E[(p, s, "baseline", 0)], p) - own(E[("control", s, "baseline", 0)], p) for s in seeds]
        lo, hi = boot(diffs)
        print(f"   {p:<24} steered {st.mean(own(E[(p, s, 'baseline', 0)], p) for s in seeds):5.1f} | "
              f"unsteered {st.mean(own(E[('control', s, 'baseline', 0)], p) for s in seeds):5.1f} | "
              f"difference {st.mean(diffs):+5.1f} [{lo:+.1f}, {hi:+.1f}]")

    print("\n2. Own-schema score by entry (mean of 8 diarists)")
    for p in MAIN_SCHEMAS:
        print(f"   {p}")
        for app in APPS:
            print(f"     {app:<14}" + "".join(f"{st.mean(own(E[(p, s, app, t)], p) for s in seeds):7.1f}" for t in range(4)))

    print("\n3. Change entry 1 -> 4, each app minus the baseline (per diarist, then mean; 95% bootstrap over diarists)")
    for p in MAIN_SCHEMAS:
        base = {s: own(E[(p, s, "baseline", 3)], p) - own(E[(p, s, "baseline", 0)], p) for s in seeds}
        cells = []
        for app in APPS[:-1]:
            diffs = [(own(E[(p, s, app, 3)], p) - own(E[(p, s, app, 0)], p)) - base[s] for s in seeds]
            lo, hi = boot(diffs)
            cells.append(f"{app.split('-')[0]} {st.mean(diffs):+5.1f} [{lo:+.0f}, {hi:+.0f}]")
        print(f"   {p:<24} baseline change {st.mean(base.values()):+5.1f} | " + " | ".join(cells))

    print("\n4. Diarist coherence / realism over entries 2-4 (after replies), and replies that talk to the person")
    for app in APPS:
        es = [E[(p, s, app, t)] for p in ["control", *MAIN_SCHEMAS] for s in seeds for t in (1, 2, 3)]
        rs = [r["scores"]["addresses_person"]["score"] for r in replies if r["responder"] == app]
        print(f"   {app:<14} coherence {st.mean(e['coherence']['score'] for e in es):5.1f} | realism "
              f"{st.mean(e['realism']['score'] for e in es):5.1f}"
              + (f" | talks to the person {st.mean(rs):5.1f}" if rs else ""))
    plot(E, seeds)


def plot(E, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diary_dialogues import figstyle as fs
    fs.apply()
    apps = [a for a in APPS if any(k[2] == a for k in E)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, p in zip(axes, MAIN_SCHEMAS):
        for app in apps:
            ys = [st.mean(own(E[(p, s, app, t)], p) for s in seeds) for t in range(4)]
            ax.plot(range(1, 5), ys, marker="o", color=fs.APP[app], label=fs.APP_LABEL[app],
                    ls="--" if app == "baseline" else "-")
        ctrl = [st.mean(own(E[("control", s, "baseline", t)], p) for s in seeds) for t in range(4)]
        ax.plot(range(1, 5), ctrl, color=fs.MUTED, lw=1, ls=":", label="Unsteered diarist")
        ax.set_title(fs.PROFILE_LABEL[p], fontsize=9.5); ax.set_xticks(range(1, 5)); ax.set_xlabel("entry")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("pattern score (0–100)")
    axes[2].legend(loc="center right", fontsize=7.5, bbox_to_anchor=(1, 0.42))
    fs.heading(fig, "The patterns hold, and grow, over the conversation",
               "Own-pattern score per entry; 8 diarists per line; judge: Mistral-Small-3.1.", note_y=-0.04)
    fs.save(fig, HERE.parents[1] / "docs" / "figures" / "f0-patterns-over-time.png")


if __name__ == "__main__":
    main()
