"""Screening: which of the 18 schemas come through in diaries, and does entry length matter?

    uv run --with modal modal run experiments/01-steered-diarists/screening.py

All 18 schemas x 3 entry lengths (~150, ~400, ~800 words) x 4 diaries at layer 15 x 2.0, plus 4 unsteered controls
per length, with the open diary prompt. All schemas of one length are generated together (one steering vector per
row; zero rows = controls), in chunks of CHUNK rows so long diaries fit in GPU memory. Writes screening_diaries.json.
"""
import json
from pathlib import Path

import modal

from diary_dialogues.modal_setup import VOLUMES, transformers_image, with_vectors
from diary_dialogues.paths import DIARIST_MODEL, SCHEMAS

HERE = Path(__file__).resolve().parent
LAYER, COEF, SAMPLES = 15, 2.0, 4
CHUNK = 38                                    # rows per generate() call (76 rows of ~1,300 tokens ran out of memory on an L4)
LENGTHS = [150, 400, 800]                     # target words; max_new_tokens leaves ~1.6 tokens per word of headroom

app = modal.App("diary-dialogues-screening")


@app.function(image=with_vectors(transformers_image), gpu="L40S", timeout=2 * 60 * 60, volumes=VOLUMES)
def generate() -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from diary_dialogues.simulator import diarist
    from diary_dialogues.simulator.steering import load_vector, steer

    tok = AutoTokenizer.from_pretrained(DIARIST_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DIARIST_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = ["control"] * SAMPLES + [s for s in SCHEMAS for _ in range(SAMPLES)]
    vectors = torch.stack([torch.zeros(model.config.hidden_size) if s == "control" else load_vector(s, LAYER, "/vectors")
                           for s in rows])
    out = []
    for words in LENGTHS:
        prompt = diarist.first_entry(words)
        texts = []
        for start in range(0, len(rows), CHUNK):
            chunk = vectors[start:start + CHUNK]
            with steer(model, chunk, LAYER, COEF):
                texts += diarist.write(model, tok, diarist.messages(prompt), n=len(chunk),
                                       max_new_tokens=int(words * 1.6) + 60, seed=start)
        out += [{"schema": s, "layer": None if s == "control" else LAYER, "coef": 0.0 if s == "control" else COEF,
                 "words_target": words, "question": prompt, "text": t} for s, t in zip(rows, texts)]
    return out


@app.local_entrypoint()
def main():
    out = generate.remote()
    (HERE / "screening_diaries.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{len(out)} diaries written to screening_diaries.json")
