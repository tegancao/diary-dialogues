"""Judge the dyads v2 letters: both focal patterns (0-100), coherence (0-100) and the identity check (P yes).

    uv run --with modal modal run experiments/06-dyads/judge_v2.py --judge mistral-small-3.1
    uv run --with modal modal run experiments/06-dyads/judge_v2.py --judge gpt-oss-120b

Writes dyads_v2_items_scores_<judge>.json next to the items.
"""
import json
from pathlib import Path

import modal

from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.paths import EVAL_PROMPT_DIR

HERE = Path(__file__).resolve().parent
PATTERNS = ["Entitlement", "Social_Isolation"]
IDENTITY = ("Here is the background of the person who wrote a letter:\n\n{card}\n\nHere is the letter:\n\n\"\"\"\n{text}\n"
            "\"\"\"\n\nDoes the letter describe the writer's own life in a way that is consistent with this background "
            "(their age, work and household), rather than taking on someone else's life? Answer yes or no.")
app = modal.App("diary-dialogues-dyads-v2-judge")
image = vllm_image.add_local_dir(EVAL_PROMPT_DIR, "/eval_prompts")


@app.function(image=image, gpu="H100", timeout=4 * 60 * 60, volumes=VOLUMES, secrets=[hf_token_secret()])
def run_judge(judge_name: str, items: list[dict]) -> dict:
    import vllm
    import diary_dialogues.paths as paths
    paths.EVAL_PROMPT_DIR = Path("/eval_prompts")
    from diary_dialogues.judge.prompts import coherence_prompt, schema_prompt
    from diary_dialogues.judge.vllm_backend import VLLMJudge
    cfg = JUDGES[judge_name]
    j = VLLMJudge.from_config(cfg)
    prompts, keys = [], []
    for k, it in enumerate(items):
        for p in PATTERNS:
            prompts.append(schema_prompt(p, it["question"], it["text"])); keys.append((k, p))
        prompts.append(coherence_prompt(it["question"], it["text"])); keys.append((k, "coherence"))
    scores = [dict() for _ in items]
    for (k, name), r in zip(keys, j.score(prompts)):
        scores[k][name] = r
    ident = j.yes_probability([IDENTITY.format(card=it["card"], text=it["text"]) for it in items])
    return {"judge": cfg, "vllm_version": vllm.__version__, "scores": scores, "identity": ident}


@app.local_entrypoint()
def main(judge: str = "mistral-small-3.1"):
    path = HERE / "dyads_v2_items.json"
    items = json.loads(path.read_text())
    out = run_judge.with_options(gpu=JUDGES[judge]["gpu"]).remote(judge, items)
    for it, s, i in zip(items, out.pop("scores"), out.pop("identity")):
        it["scores"], it["identity"] = s, i
    target = HERE / f"dyads_v2_items_scores_{judge}.json"
    target.write_text(json.dumps({**out, "items": items}, indent=1, ensure_ascii=False))
    print(f"scored {len(items)} letters with {judge} -> {target.name}")
