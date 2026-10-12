"""Self-report check: each screening diarist rates itself on the IPIP scales right after writing its diary.

    uv run --with modal modal run experiments/01-steered-diarists/questionnaire.py

For every diary in screening_diaries.json: the same steering as when it was written (zero vector for controls),
the diary in context, then each of the 19 items (diary_dialogues.simulator.questionnaire) asked separately.
Writes screening_questionnaire.json (per diary: item ratings and scale scores).
"""
import json
from pathlib import Path

import modal

from diary_dialogues.modal_setup import VOLUMES, transformers_image, with_vectors
from diary_dialogues.paths import DIARIST_MODEL

HERE = Path(__file__).resolve().parent
BATCH = 32
app = modal.App("diary-dialogues-questionnaire")


@app.function(image=with_vectors(transformers_image), gpu="L40S", timeout=60 * 60, volumes=VOLUMES)
def rate(diaries: list[dict]) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from diary_dialogues.simulator import diarist
    from diary_dialogues.simulator.questionnaire import OPTIONS, SCALES, expected_rating, item_messages, scale_score
    from diary_dialogues.simulator.steering import load_vector, steer

    tok = AutoTokenizer.from_pretrained(DIARIST_MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(DIARIST_MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    option_ids = [tok.encode(o, add_special_tokens=False) for o in OPTIONS]
    assert all(len(i) == 1 for i in option_ids), option_ids
    option_ids = [i[0] for i in option_ids]
    layer = next(d["layer"] for d in diaries if d["layer"])
    zero = torch.zeros(model.config.hidden_size)
    vecs = {s: load_vector(s, layer, "/vectors") for s in {d["schema"] for d in diaries} if s != "control"}

    jobs = []                                        # (diary index, scale, item index, text, coef * vector)
    for k, d in enumerate(diaries):
        msgs = diarist.messages(d["question"])
        for scale, (_, items) in SCALES.items():
            for j, (item, _) in enumerate(items):
                text = tok.apply_chat_template(item_messages(msgs, d["text"], item), add_generation_prompt=True,
                                               tokenize=False)
                jobs.append((k, scale, j, text, vecs[d["schema"]] * d["coef"] if d["coef"] else zero))
    ratings = {}
    for start in range(0, len(jobs), BATCH):
        chunk = jobs[start:start + BATCH]
        enc = tok([c[3] for c in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(), steer(model, torch.stack([c[4] for c in chunk]), layer, 1.0):   # rows carry coef
            logits = model(**enc, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)   # last position only
        for c, row in zip(chunk, logits):
            ratings[c[:3]] = expected_rating({o: float(row[i]) for o, i in zip(OPTIONS, option_ids)})
    out = []
    for k, d in enumerate(diaries):
        rec = {key: d[key] for key in ("schema", "layer", "coef", "words_target")}
        for scale, (_, items) in SCALES.items():
            r = [ratings[(k, scale, j)] for j in range(len(items))]
            rec[scale] = {"items": r, "score": scale_score(r, [key for _, key in items])}
        out.append(rec)
    return out


@app.local_entrypoint()
def main(input: str = "screening_diaries.json"):
    diaries = json.loads((HERE / input).read_text())
    out = rate.remote(diaries)
    (HERE / "screening_questionnaire.json").write_text(json.dumps(out, indent=1))
    print(f"rated {len(out)} diaries on {sum(1 for k in out[0] if isinstance(out[0][k], dict))} scales")
