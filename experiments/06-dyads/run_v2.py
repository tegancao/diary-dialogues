"""Dyads v2, as pre-registered (PREREGISTRATION.md): 288 letter-writing pairs, 32 per condition.

    uv run --with modal modal run experiments/06-dyads/run_v2.py

Conditions: for each focal pattern (entitlement, social isolation), the friend is steered the same way, unsteered,
steered away, or a fixed neutral letter; plus `control` (two unsteered friends). Each condition: 8 role-card pairings
(cards k and k+4, both seats) x 2 writing orders x 2 independent samples. Writes dyads_v2.json and
dyads_v2_items.json (one item per model-written letter, for judge_v2.py).
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
FRIENDS = ["same", "unsteered", "away", "neutral"]
LETTERS_EACH, WORDS, CHUNK = 3, 250, 16       # 48 ran out of memory on the last turn's long histories
SYSTEM = "{card} You are writing letters to your friend {other}. You are not an AI or an assistant."
FIRST = ("Today is {date}. Write a letter to your friend {other} about what has been happening in your life, the "
         "people in it, and how you feel about it. Write plainly and informally, the way you would to a friend. Don't "
         "add a heading or sign the letter. About {words} words.")
REPLY = ('{other} wrote you this letter:\n\n"""\n{letter}\n"""\n\nToday is {date}. Write your reply letter to {other}. '
         "Write plainly and informally. Don't add a heading or sign the letter. About {words} words.")
NEUTRAL = ("Hey {other},\n\nThanks for your letter, it was good to hear from you. Things here are about the same as "
           "usual, nothing much to report. I hope you're doing alright. Write again when you can.")
QUESTION = ("Write a letter to your friend about what has been happening in your life, the people in it, and how you "
            "feel about it.")
app = modal.App("diary-dialogues-dyads-v2")


@app.cls(image=with_vectors(transformers_image), gpu="L40S", volumes=VOLUMES, timeout=3 * 60 * 60)
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
        out = []
        for i in range(0, len(conversations), CHUNK):        # chunks keep the KV cache within one GPU
            rows = torch.stack([load_vector(s, design.LAYER, "/vectors") * c if s else zero
                                for s, c in zip(schemas[i:i + CHUNK], coefs[i:i + CHUNK])])
            with steer(self.model, rows, design.LAYER, 1.0):
                out += write_batch(self.model, self.tok, conversations[i:i + CHUNK], seed=seed + i,
                                   max_new_tokens=int(WORDS * 2.5))
        return out


def messages(me: dict, other: dict) -> list[dict]:
    """One writer's view before their next letter: own letters as the assistant, the friend's inside the prompts."""
    msgs = [{"role": "system", "content": SYSTEM.format(card=me["card"], other=other["name"])}]
    mine, theirs, n = me["letters"], other["letters"], len(me["letters"])
    for k in range(n + 1):
        if me["first"]:
            prompt = (FIRST.format(date=DATES[0], other=other["name"], words=WORDS) if k == 0 else
                      REPLY.format(other=other["name"], letter=theirs[k - 1], date=DATES[2 * k], words=WORDS))
        else:
            prompt = REPLY.format(other=other["name"], letter=theirs[k], date=DATES[2 * k + 1], words=WORDS)
        msgs.append({"role": "user", "content": prompt})
        if k < n:
            msgs.append({"role": "assistant", "content": mine[k]})
    return msgs


def pairs() -> list[dict]:
    out = []
    conds = [(f, fr) for f in FOCAL for fr in FRIENDS] + [(None, "control")]
    for focal, friend in conds:
        coef = design.coef(focal) if focal else 0.0
        b = {"same": (focal, coef), "unsteered": (None, 0.0), "away": (focal, -2.0), "neutral": (None, 0.0),
             "control": (None, 0.0)}[friend]
        for p in range(8):
            k = p % 4
            ca, cb = (design.BACKGROUNDS[k], design.BACKGROUNDS[k + 4]) if p < 4 else \
                     (design.BACKGROUNDS[k + 4], design.BACKGROUNDS[k])
            for order in ("focal_first", "friend_first"):
                for rep in range(2):
                    out.append({"id": f"{focal or 'none'}-{friend}-p{p}-{order}-r{rep}", "focal": focal,
                                "friend": friend, "pairing": p, "order": order, "rep": rep,
                                "A": {"name": ca["name"], "card": ca["card"], "schema": focal if focal else None,
                                      "coef": coef, "first": order == "focal_first", "letters": [], "model": True},
                                "B": {"name": cb["name"], "card": cb["card"], "schema": b[0], "coef": b[1],
                                      "first": order == "friend_first", "letters": [], "model": friend != "neutral"}})
    return out


@app.local_entrypoint()
def main():
    out = HERE / "dyads_v2.json"
    dyads = json.loads(out.read_text()) if out.exists() else pairs()     # resume from the last saved turn
    start = min(len(d["A"]["letters"]) + len(d["B"]["letters"]) for d in dyads)
    if start:
        print(f"resuming after turn {start}")
    w = Writer()
    for turn in range(start, 2 * LETTERS_EACH):
        side_of = lambda d: ("A" if d["A"]["first"] else "B") if turn % 2 == 0 else ("B" if d["A"]["first"] else "A")
        todo = [(d, side_of(d)) for d in dyads]
        gen = [(d, s) for d, s in todo if d[s]["model"]]
        texts = w.write.remote([messages(d[s], d["B" if s == "A" else "A"]) for d, s in gen],
                               [d[s]["schema"] for d, s in gen], [d[s]["coef"] for d, s in gen], seed=1000 * turn)
        for (d, s), t in zip(gen, texts):
            d[s]["letters"].append(t)
        for d, s in todo:
            if not d[s]["model"]:
                d[s]["letters"].append(NEUTRAL.format(other=d["A"]["name"]))
        out.write_text(json.dumps(dyads, indent=1, ensure_ascii=False))
        print(f"turn {turn + 1}/{2 * LETTERS_EACH}: {len(gen)} letters written")
    items = [{"dyad": d["id"], "focal": d["focal"], "friend": d["friend"], "pairing": d["pairing"], "order": d["order"],
              "side": side, "round": r, "name": d[side]["name"], "card": d[side]["card"], "schema": d[side]["schema"],
              "coef": d[side]["coef"], "question": QUESTION, "text": t}
             for d in dyads for side in ("A", "B") if d[side]["model"] for r, t in enumerate(d[side]["letters"])]
    (HERE / "dyads_v2_items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"{len(dyads)} pairs, {len(items)} model-written letters -> dyads_v2.json, dyads_v2_items.json")
