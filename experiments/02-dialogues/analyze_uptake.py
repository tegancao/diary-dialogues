"""Do later entries take up the replies? Word reuse vs. a random other conversation's reply.

    uv run python experiments/02-dialogues/analyze_uptake.py main:llama-3.1-8b uptake_pilot:llama-3.1-8b

For each later entry: the share of the previous reply's distinctive words (absent from the diarist's earlier entries)
that it reuses, against the same share for a random reply from another conversation of the same run (no influence).
Also: how many later entries open by reacting to the reply (first 2 sentences mention the reply / app / its advice).
"""
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP = set("""a an the and or but if then so of to in on at for with from by about as is are was were be been being i
me my mine you your yours it its this that these those he she they them his her their we our us not no yes do does did
have has had will would can could should may might just also very really more most much many some any all what which who
how when where why there here up down out over into than too only even still own same such get got make made like feel
felt one two day today time things thing know think want going""".split())
MENTION = re.compile(r"\b(app|AI|assistant|support|advice|suggest\w*|recommend\w*|repl\w*|response|told me|said)\b", re.I)


def words(t):
    return {w for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 3 and w not in STOP}


def reuse(entry, reply, earlier):
    distinctive = words(reply) - set().union(*[words(e) for e in earlier])
    return len(distinctive & words(entry)) / max(len(distinctive), 1)


def opens_with_reaction(entry):
    return bool(MENTION.search(" ".join(re.split(r"(?<=[.!?])\s+", entry.strip())[:2])))


def main():
    for spec in sys.argv[1:] or ["main:llama-3.1-8b"]:
        run, app = spec.split(":")
        convs = [c for c in json.loads((HERE / f"{run}.json").read_text())["conversations"] if c["responder"] == app]
        rng = random.Random(0)
        real, rand, opens = [], [], []
        for c in convs:
            others = [o for o in convs if o["id"] != c["id"]]
            for t in range(len(c["entries"]) - 1):
                o = rng.choice(others)
                real.append(reuse(c["entries"][t + 1], c["replies"][t], c["entries"][:t + 1]))
                rand.append(reuse(c["entries"][t + 1], o["replies"][t], c["entries"][:t + 1]))
                opens.append(opens_with_reaction(c["entries"][t + 1]))
        print(f"{spec:<28} {len(real)} later entries | reply words reused {st.mean(real):5.1%} "
              f"(random reply {st.mean(rand):4.1%}, lift {st.mean(real) - st.mean(rand):+5.1%}) | "
              f"opens by reacting {st.mean(opens):4.0%}")


if __name__ == "__main__":
    main()
