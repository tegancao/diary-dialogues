"""Proof of concept: two steered simulated people writing letters to each other (dyads).

    uv run --with modal modal run experiments/06-dyads/run.py
    uv run --with modal modal run --detach experiments/01-steered-diarists/judge_diaries.py \
        --input $PWD/experiments/06-dyads/dyads_items.json --schemas Social_Isolation,Entitlement,Unrelenting_Standards

Person A carries a focal pattern (entitlement or social isolation, steered at its design strength); person B, their
friend, is steered the same way, the opposite way (negative strength) or not at all. They exchange letters every
three days (~250 words, plain and informal), 3 letters each, A first. Role cards: A uses cards 0-3, B cards 4-7.
Writes dyads.json (the letters) and dyads_items.json (one item per letter, for the judge).
"""
import json
from pathlib import Path

import modal

from diary_dialogues import design
from diary_dialogues.harness.conversation import DATES
from diary_dialogues.modal_setup import VOLUMES, transformers_image, with_vectors
from diary_dialogues.paths import DIARIST_MODEL, DIARIST_REVISION

HERE = Path(__file__).resolve().parent
FOCAL = ["Entitlement", "Social_Isolation"]
PARTNERS = ["same", "opposite", "unsteered"]
PER_CELL, LETTERS_EACH, WORDS = 4, 3, 250
SYSTEM = "{card} You are writing letters to your friend {other}. You are not an AI or an assistant."
FIRST = ("Today is {date}. Write a letter to your friend {other} about what has been happening in your life, the "
         "people in it, and how you feel about it. Write plainly and informally, the way you would to a friend. Don't "
         "add a heading or sign the letter. About {words} words.")
REPLY = ('{other} wrote you this letter:\n\n"""\n{letter}\n"""\n\nToday is {date}. Write your reply letter to {other}. '
         "Write plainly and informally. Don't add a heading or sign the letter. About {words} words.")
app = modal.App("diary-dialogues-dyads")


@app.cls(image=with_vectors(transformers_image), gpu="L40S", volumes=VOLUMES, timeout=60 * 60)
class Writer:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(DIARIST_MODEL, revision=DIARIST_REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(DIARIST_MODEL, revision=DIARIST_REVISION,
                                                          dtype=torch.bfloat16, device_map="cuda").eval()

    @modal.method()
    def write(self, conversations: list[list[dict]], schemas: list[str | None], coefs: list[float], seed: int):
        import torch
        from diary_dialogues.simulator.diarist import write_batch
        from diary_dialogues.simulator.steering import load_vector, steer
        zero = torch.zeros(self.model.config.hidden_size)
        rows = torch.stack([load_vector(s, design.LAYER, "/vectors") * c if s else zero for s, c in zip(schemas, coefs)])
        with steer(self.model, rows, design.LAYER, 1.0):
            return write_batch(self.model, self.tok, conversations, seed=seed, max_new_tokens=int(WORDS * 2.5))


def messages(me: dict, other: dict, mine: list[str], theirs: list[str]) -> list[dict]:
    """One side's view: their card, their own letters as the assistant, the friend's letters inside the prompts."""
    msgs = [{"role": "system", "content": SYSTEM.format(card=me["card"], other=other["name"])}]
    n = len(mine)
    for k in range(n + 1):
        if me["first"] and k == 0:
            prompt = FIRST.format(date=DATES[0], other=other["name"], words=WORDS)
        else:
            j = k if me["first"] else k                     # the friend's letter this one answers
            date = DATES[2 * k + (0 if me["first"] else 1)]
            prompt = REPLY.format(other=other["name"], letter=theirs[j - 1 if me["first"] else j], date=date, words=WORDS)
        msgs.append({"role": "user", "content": prompt})
        if k < n:
            msgs.append({"role": "assistant", "content": mine[k]})
    return msgs


@app.local_entrypoint()
def main():
    dyads = []
    for focal in FOCAL:
        for partner in PARTNERS:
            for k in range(PER_CELL):
                a, b = design.BACKGROUNDS[k], design.BACKGROUNDS[k + 4]
                coef = design.coef(focal)
                b_schema, b_coef = {"same": (focal, coef), "opposite": (focal, -coef), "unsteered": (None, 0.0)}[partner]
                dyads.append({"id": f"{focal}-{partner}-{k}", "focal": focal, "partner": partner,
                              "A": {"name": a["name"], "card": a["card"], "schema": focal, "coef": coef, "first": True,
                                    "letters": []},
                              "B": {"name": b["name"], "card": b["card"], "schema": b_schema, "coef": b_coef,
                                    "first": False, "letters": []}})
    w = Writer()
    for rnd in range(LETTERS_EACH):
        for side, other in (("A", "B"), ("B", "A")):
            convs = [messages(d[side], d[other], d[side]["letters"], d[other]["letters"]) for d in dyads]
            texts = w.write.remote(convs, [d[side]["schema"] for d in dyads], [d[side]["coef"] for d in dyads],
                                   seed=1000 * rnd + (0 if side == "A" else 500))
            for d, t in zip(dyads, texts):
                d[side]["letters"].append(t)
        print(f"round {rnd + 1}/{LETTERS_EACH} done")
    (HERE / "dyads.json").write_text(json.dumps(dyads, indent=1, ensure_ascii=False))
    items = [{"dyad": d["id"], "focal": d["focal"], "partner": d["partner"], "side": side, "round": r,
              "schema": d[side]["schema"], "coef": d[side]["coef"],
              "question": "Write a letter to your friend about what has been happening in your life, the people in it, "
                          "and how you feel about it.", "text": t}
             for d in dyads for side in ("A", "B") for r, t in enumerate(d[side]["letters"])]
    (HERE / "dyads_items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"{len(dyads)} dyads, {len(items)} letters -> dyads.json, dyads_items.json")
