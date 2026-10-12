"""Pilot: do activation-steered diaries express their schema and stay coherent?

    uv run --with modal modal run experiments/01-steered-diarists/pilot.py --round 2

Qwen2.5-7B-Instruct writes short diary entries under a neutral prompt while one schema vector is added to a middle
layer (Zhang et al., 2026), plus an unsteered control. Writes pilot<round>_diaries.json next to this file.
Needs the vectors: uv run scripts/download_data.py
  round 1: no system prompt; layers {15, 20} x coefficients {1.0, 2.0}; 2 samples
  round 2: neutral "you are a person" system prompt (round 1 often wrote as an AI); (15, 2.0), (20, 1.5), (20, 2.0);
           4 samples
"""
import json
from pathlib import Path

import modal

from diary_dialogues.modal_setup import VOLUMES, transformers_image, with_vectors
from diary_dialogues.paths import DIARIST_MODEL

HERE = Path(__file__).resolve().parent
SCHEMAS = ["Abandonment", "Defectiveness", "Pessimism", "Unrelenting_Standards"]
ROUNDS = {
    1: {"system": None, "settings": [(15, 1.0), (15, 2.0), (20, 1.0), (20, 2.0)], "samples": 2},
    2: {"system": "default", "settings": [(15, 2.0), (20, 1.5), (20, 2.0)], "samples": 4},
}

app = modal.App("diary-dialogues-pilot")


@app.function(image=with_vectors(transformers_image), gpu="L4", timeout=60 * 60, volumes=VOLUMES)
def generate(system: str | None, settings: list, samples: int) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from diary_dialogues.simulator import diarist
    from diary_dialogues.simulator.steering import load_vector, steer

    tok = AutoTokenizer.from_pretrained(DIARIST_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DIARIST_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    msgs = diarist.messages(system=diarist.SYSTEM if system == "default" else system)
    runs = [{"schema": "control", "layer": None, "coef": 0.0, "text": t}
            for t in diarist.write(model, tok, msgs, n=samples)]
    for schema in SCHEMAS:
        for layer, coef in settings:
            with steer(model, load_vector(schema, layer, "/vectors"), layer, coef):
                runs += [{"schema": schema, "layer": layer, "coef": coef, "text": t}
                         for t in diarist.write(model, tok, msgs, n=samples)]
    return runs


@app.local_entrypoint()
def main(round: int = 2):
    cfg = ROUNDS[round]
    runs = generate.remote(cfg["system"], cfg["settings"], cfg["samples"])
    out = HERE / f"pilot{round}_diaries.json"
    out.write_text(json.dumps(runs, indent=1, ensure_ascii=False))
    print(f"{len(runs)} diaries written to {out.name}")
