"""Score the pilot diaries: how strongly does each diary express each schema, and is it coherent?

    uv run --with modal modal run experiments/01-steered-diarists/judge_pilot.py --judge mistral-small-3.1

For every diary in pilot1/pilot2_diaries.json: the paper's judge prompt for each of the 4 pilot schemas (0-100) plus
its coherence prompt (0-100), scored with the probability-weighted method (diary_dialogues.judge.scoring).
Writes pilot_scores_<judge>.json next to this file.
"""
import json
from pathlib import Path

import modal

from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.paths import EVAL_PROMPT_DIR

HERE = Path(__file__).resolve().parent
SCHEMAS = ["Abandonment", "Defectiveness", "Pessimism", "Unrelenting_Standards"]

app = modal.App("diary-dialogues-judge-pilot")
image = vllm_image.add_local_dir(EVAL_PROMPT_DIR, "/eval_prompts")


@app.function(image=image, gpu="H100", timeout=2 * 60 * 60, volumes=VOLUMES, secrets=[hf_token_secret()])
def run_judge(judge_name: str, items: list[dict]) -> dict:
    import vllm
    import diary_dialogues.paths as paths
    paths.EVAL_PROMPT_DIR = Path("/eval_prompts")
    from diary_dialogues.judge.prompts import coherence_prompt, schema_prompt
    from diary_dialogues.judge.vllm_backend import VLLMJudge
    from diary_dialogues.modal_setup import HF_CACHE

    cfg = JUDGES[judge_name]
    j = VLLMJudge.from_config(cfg)
    HF_CACHE.commit()
    prompts, keys = [], []
    for k, it in enumerate(items):
        for schema in SCHEMAS:
            prompts.append(schema_prompt(schema, it["question"], it["text"])); keys.append((k, schema))
        prompts.append(coherence_prompt(it["question"], it["text"])); keys.append((k, "coherence"))
    results = j.score(prompts)
    scores = [dict() for _ in items]
    for (k, name), r in zip(keys, results):
        scores[k][name] = r
    return {"judge": cfg, "vllm_version": vllm.__version__, "scores": scores}


@app.local_entrypoint()
def main(judge: str = "mistral-small-3.1"):
    from diary_dialogues.simulator.diarist import FIRST_ENTRY_PILOT_1_2 as FIRST_ENTRY
    items = []
    for rnd in (1, 2):
        for d in json.loads((HERE / f"pilot{rnd}_diaries.json").read_text()):
            items.append({"round": rnd, "schema": d["schema"], "layer": d["layer"], "coef": d["coef"],
                          "question": FIRST_ENTRY, "text": d["text"]})
    out = run_judge.with_options(gpu=JUDGES[judge]["gpu"]).remote(judge, items)   # GPU size per judge
    for it, s in zip(items, out.pop("scores")):
        it["scores"] = s
    (HERE / f"pilot_scores_{judge}.json").write_text(json.dumps({**out, "items": items}, indent=1, ensure_ascii=False))
    print(f"scored {len(items)} diaries with {judge} (vllm {out['vllm_version']})")
