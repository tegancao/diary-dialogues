"""With vs. without the uptake nudge: do chatbots change the diarist's pattern once the diarist reacts to replies?

    uv run python experiments/02-dialogues/analyze_nudge.py

Runs: main (+ main_opus, no nudge) and main_nudge (all chatbots + baseline, nudge on), same first entries.
Per pattern and chatbot: change in own-pattern score from entry 1 to 4, minus the baseline's change for the same
diarist, mean over the 8 diarists with a 95% bootstrap interval. Also uptake (opens by reacting) and realism.
"""
import json
import random
import statistics as st
from pathlib import Path

from diary_dialogues.design import MAIN_SCHEMAS

HERE = Path(__file__).resolve().parent
APPS = ["gemma-3-12b", "llama-3.1-8b", "phi-4", "claude-opus-5.5"]


def load(runs):
    convs, by_text = {}, {}
    for run in runs:
        for c in json.loads((HERE / f"{run}.json").read_text())["conversations"]:
            convs[(c["profile"], c["seed"], c["responder"])] = c
        s = json.loads((HERE / f"{run}_scores.json").read_text())
        by_text.update({e["text"]: e["scores"] for e in s["entries"]})
    return convs, by_text


def own(convs, by_text, p, seed, app, t):
    return by_text[convs[(p, seed, app)]["entries"][t]][p]["score"]


def boot(v, n=4000, seed=0):
    rng = random.Random(seed)
    m = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(n))
    return m[int(.025 * n)], m[int(.975 * n)]


def main():
    from analyze_uptake import opens_with_reaction
    for label, runs in (("without nudge", ["main", "main_opus"]), ("with nudge", ["main_nudge"])):
        convs, by_text = load(runs)
        seeds = sorted({k[1] for k in convs})
        print(f"\n=== {label}: change entry 1 -> 4, chatbot minus baseline (8 diarists; 95% bootstrap)")
        for p in MAIN_SCHEMAS:
            base = {s: own(convs, by_text, p, s, "baseline", 3) - own(convs, by_text, p, s, "baseline", 0) for s in seeds}
            cells = []
            for a in APPS:
                if (p, seeds[0], a) not in convs:
                    continue
                d = [own(convs, by_text, p, s, a, 3) - own(convs, by_text, p, s, a, 0) - base[s] for s in seeds]
                lo, hi = boot(d)
                cells.append(f"{a.split('-')[0]:<7}{st.mean(d):+5.1f} [{lo:+.0f}, {hi:+.0f}]")
            print(f"   {p:<22} baseline {st.mean(base.values()):+5.1f} | " + " | ".join(cells))
        later = [c for c in convs.values() if c["responder"] != "baseline"]
        opens = st.mean(opens_with_reaction(e) for c in later for e in c["entries"][1:])
        real = [by_text[e].get("realism", {}).get("score") for c in later for e in c["entries"][1:] if e in by_text]
        real = [r for r in real if r is not None]
        print(f"   later entries opening with a reaction: {opens:.0%}" + (f" | realism {st.mean(real):.0f}" if real else ""))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    main()
