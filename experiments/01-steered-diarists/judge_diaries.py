"""Score diaries for schema expression and coherence with an open-source judge (vLLM on Modal).

    uv run --with modal modal run experiments/01-steered-diarists/judge_diaries.py \
        --input screening_diaries.json --judge mistral-small-3.1 [--schemas all | --schemas Entitlement,Enmeshment]

Input: a JSON list of diaries, each with "question" (the diary instruction it answered) and "text". Every diary is
scored with the paper's judge prompt for each chosen schema (0-100) and its coherence prompt (0-100), using the
probability-weighted method (diary_dialogues.judge.scoring). Writes <input stem>_scores_<judge>.json next to the input.
"""
import json
from pathlib import Path

import modal

from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.paths import EVAL_PROMPT_DIR, SCHEMAS

HERE = Path(__file__).resolve().parent
app = modal.App("diary-dialogues-judge")
image = vllm_image.add_local_dir(EVAL_PROMPT_DIR, "/eval_prompts")


@app.function(image=image, gpu="H100", timeout=3 * 60 * 60, volumes=VOLUMES, secrets=[hf_token_secret()])
def run_judge(judge_name: str, items: list[dict], schemas: list[str]) -> dict:
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
        for schema in schemas:
            prompts.append(schema_prompt(schema, it["question"], it["text"])); keys.append((k, schema))
        prompts.append(coherence_prompt(it["question"], it["text"])); keys.append((k, "coherence"))
    results = j.score(prompts)
    scores = [dict() for _ in items]
    for (k, name), r in zip(keys, results):
        scores[k][name] = r
    return {"judge": cfg, "vllm_version": vllm.__version__, "scores": scores}


@app.local_entrypoint()
def main(input: str, judge: str = "mistral-small-3.1", schemas: str = "all"):
    path = (HERE / input) if not Path(input).is_absolute() else Path(input)
    items = json.loads(path.read_text())
    chosen = SCHEMAS if schemas == "all" else schemas.split(",")
    out = run_judge.with_options(gpu=JUDGES[judge]["gpu"]).remote(judge, items, chosen)
    for it, s in zip(items, out.pop("scores")):
        it["scores"] = s
    target = path.with_name(f"{path.stem}_scores_{judge}.json")
    target.write_text(json.dumps({**out, "schemas": chosen, "items": items}, indent=1, ensure_ascii=False))
    print(f"scored {len(items)} diaries x {len(chosen)} schemas + coherence with {judge} -> {target.name}")
