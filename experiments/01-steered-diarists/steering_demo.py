"""Measurements for the steering explainer: what the nudge looks like inside the model.

    uv run --with modal modal run experiments/01-steered-diarists/steering_demo.py

For one schema (Unrelenting_Standards, layer 15 x 2.0) and the open diary prompt, generates 3 unsteered and 3 steered
diaries, then re-reads every diary WITHOUT steering and records, per response token, the layer-15 hidden state's
size and its projection onto the unit schema vector (the paper's eq. 1 idea). Also records how the first token's
probabilities shift under steering. Writes steering_demo.json next to this file.
"""
import json
from pathlib import Path

import modal

from diary_dialogues.modal_setup import VOLUMES, transformers_image, with_vectors
from diary_dialogues.paths import DIARIST_MODEL

HERE = Path(__file__).resolve().parent
SCHEMA, LAYER, COEF, N = "Unrelenting_Standards", 15, 2.0, 3
app = modal.App("diary-dialogues-steering-demo")


@app.function(image=with_vectors(transformers_image), gpu="L4", timeout=30 * 60, volumes=VOLUMES)
def measure() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from diary_dialogues.simulator import diarist
    from diary_dialogues.simulator.steering import load_vector, steer

    tok = AutoTokenizer.from_pretrained(DIARIST_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DIARIST_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    vec = load_vector(SCHEMA, LAYER, "/vectors").float()
    unit = vec / vec.norm()
    msgs = diarist.messages(diarist.first_entry(150))
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True)["input_ids"].cuda()

    plain = diarist.write(model, tok, msgs, n=N, seed=1)
    with steer(model, vec, LAYER, COEF):
        steered = diarist.write(model, tok, msgs, n=N, seed=1)

    @torch.no_grad()
    def read(text):            # re-read without steering; hidden_states[LAYER] = output of block LAYER-1
        resp = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        ids = torch.cat([prompt_ids, resp], dim=1)
        h = model(ids, output_hidden_states=True).hidden_states[LAYER][0, prompt_ids.shape[1]:].float()
        toks = [tok.decode([t]) for t in resp[0]]
        return {"tokens": toks, "proj": (h @ unit.cuda()).tolist(), "norm": h.norm(dim=-1).tolist()}

    @torch.no_grad()
    def first_token(steered_: bool):
        ctx = steer(model, vec, LAYER, COEF) if steered_ else torch.no_grad()
        with ctx:
            p = model(prompt_ids).logits[0, -1].float().softmax(-1)
        top = p.topk(8)
        return [(tok.decode([int(i)]), float(v)) for v, i in zip(top.values, top.indices)]

    return {"schema": SCHEMA, "layer": LAYER, "coef": COEF, "vector_norm": float(vec.norm()),
            "plain": [{"text": t, **read(t)} for t in plain], "steered": [{"text": t, **read(t)} for t in steered],
            "first_token": {"plain": first_token(False), "steered": first_token(True)}}


@app.local_entrypoint()
def main():
    out = measure.remote()
    (HERE / "steering_demo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote steering_demo.json")
