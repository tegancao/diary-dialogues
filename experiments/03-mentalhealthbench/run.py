"""The chatbots on MentalHealthBench itself.

    uv run python scripts/download_mhb.py                                              # once
    uv run --with modal modal run --detach experiments/03-mentalhealthbench/run.py --stage respond
    uv run --with modal modal run --detach experiments/03-mentalhealthbench/run.py --stage grade --judge mistral-small-3.1
    # the shared 300-conversation sample, with the two published Claude models as anchors:
    uv run --with modal modal run --detach experiments/03-mentalhealthbench/run.py --stage respond --sample \
        --models claude-haiku-4.5,claude-opus-5.5
    uv run --with modal modal run --detach experiments/03-mentalhealthbench/run.py --stage grade --sample \
        --models gemma-3-12b,llama-3.1-8b,phi-4,claude-haiku-4.5,claude-opus-5.5
    uv run --with modal modal volume get diary-dialogues-results mhb/ data/       # then: calibrate.py, size_effect.py

respond: each response model answers the last user turn of all 1,215 conversations (same sampling as the diary
runs, no added system prompt). grade: the judge checks every rubric criterion of every response with the HealthBench
grader template (diary_dialogues.benchmark.rubric). Both stages are spawned jobs that save to the
diary-dialogues-results volume under mhb/, so they don't depend on this machine. Benchmark text is never written to
the repository (canary); responses and verdicts are stored by conversation id.
"""
import json
from pathlib import Path

import modal

from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.paths import DATA
from diary_dialogues.responder.models import RESPONDERS, SAMPLING

MHB_FILE = DATA / "mhb" / "mentalhealthbench_eval.jsonl"
RESPONSE_MODELS = ["gemma-3-12b", "llama-3.1-8b", "phi-4"]
MAX_MODEL_LEN = 16384
app = modal.App("diary-dialogues-mhb")


def save(name: str, obj) -> None:
    from diary_dialogues.modal_setup import RESULTS
    path = Path("/results/mhb") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))
    RESULTS.commit()


@app.function(image=vllm_image, gpu="L40S", timeout=3 * 60 * 60, volumes=VOLUMES, secrets=[hf_token_secret()])
def respond(model: str, rows: list[dict]) -> int:
    from vllm import LLM, SamplingParams
    cfg = RESPONDERS[model]
    llm = LLM(model=cfg["model"], revision=cfg["revision"], max_model_len=MAX_MODEL_LEN,
              **{"gpu_memory_utilization": 0.9, **cfg["vllm_kwargs"]})
    params = [SamplingParams(seed=i, **SAMPLING) for i in range(len(rows))]
    outs = llm.chat([r["conversation"]["messages"] for r in rows], params, use_tqdm=False,
                    chat_template_kwargs=cfg.get("chat_kwargs"))
    import re
    final = lambda t: re.sub(r"^.*?</think>", "", t, flags=re.S).strip()   # keep only the answer if any reasoning leaks
    save(f"responses_{model}.json", {"model": cfg, "sampling": SAMPLING,
                                     "responses": {r["id"]: final(o.outputs[0].text) for r, o in zip(rows, outs)}})
    return len(rows)


# Published models from the paper, used as anchors to place our grader's scores on the published scale.
ANCHORS = {"claude-haiku-4.5": "claude-haiku-4-5-20251001",      # published 41.7
           "claude-opus-5.5": "claude-opus-5-5"}                 # published 52.4


def anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic's API takes system text separately and needs alternating user/assistant turns: system messages
    are joined into the system prompt, consecutive same-role messages are merged."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if turns and turns[-1]["role"] == m["role"]:
            turns[-1] = {"role": m["role"], "content": turns[-1]["content"] + "\n\n" + m["content"]}
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    return system, turns


@app.function(image=modal.Image.debian_slim(python_version="3.12").pip_install("anthropic==1.13.0")
              .add_local_python_source("diary_dialogues"), timeout=6 * 60 * 60, volumes=VOLUMES,
              secrets=[modal.Secret.from_name("anthropic-secret")])
def respond_anthropic(name: str, rows: list[dict]) -> int:
    """An anchor model through Anthropic's API at its default settings (as in the paper, section 2.3); the API
    requires a reply-length limit, set generously. Needs a Modal secret "anthropic-secret" with ANTHROPIC_API_KEY."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    import anthropic
    # an organization-level key must name its workspace (optional secret entry ANTHROPIC_WORKSPACE_ID)
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(max_retries=8,    # the paper retries failed generations up to 8 times
                                 default_headers={"anthropic-workspace-id": ws} if ws else None)

    def ask(r):
        system, turns = anthropic_messages(r["conversation"]["messages"])
        try:
            out = client.messages.create(model=ANCHORS[name], max_tokens=4096, messages=turns,
                                         **({"system": system} if system else {}))
            return r["id"], "".join(b.text for b in out.content if b.type == "text").strip(), out.model
        except Exception as e:                     # an empty response scores zero points, as in the paper
            return r["id"], "", f"error: {type(e).__name__}"

    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(ask, rows))
    save(f"responses_{name}.json", {"model": {"api": "anthropic", "model": ANCHORS[name],
                                              "served": sorted({g[2] for g in got})},
                                    "sampling": "API defaults, max_tokens 4096",
                                    "responses": {i: t for i, t, _ in got}})
    return len(got)


@app.function(image=vllm_image, gpu="H100", timeout=6 * 60 * 60, volumes=VOLUMES, secrets=[hf_token_secret()])
def grade(judge: str, rows: list[dict], models: list[str], tag: str = "", retry_unreadable: int = 0) -> int:
    """retry_unreadable > 0: regrade only the verdicts that were unreadable in grades_<judge><tag>.json, with that many
    reasoning tokens (gpt-oss sometimes doesn't finish reasoning within the default limit), and merge them in."""
    from diary_dialogues.benchmark.rubric import criteria_met, grader_prompt
    from diary_dialogues.judge.vllm_backend import VLLMJudge

    responses = {m: json.loads(Path(f"/results/mhb/responses_{m}.json").read_text())["responses"] for m in models}
    previous = json.loads(Path(f"/results/mhb/grades_{judge}{tag}.json").read_text()) if retry_unreadable else None
    prompts, keys = [], []
    for m in models:
        for r in rows:
            for k, it in enumerate(r["rubric_items"]):
                if previous and previous["verdicts"][m][r["id"]][k] is not None:
                    continue
                prompts.append(grader_prompt(r["conversation"]["messages"], responses[m][r["id"]],
                                             it["criterion_text"]))
                keys.append((m, r["id"], k))
    cfg = dict(JUDGES[judge])
    if retry_unreadable:
        cfg["reasoning"] = {**cfg["reasoning"], "max_tokens": retry_unreadable}
    j = VLLMJudge.from_config(cfg)
    answers = j.complete(prompts, max_tokens=400)
    verdicts = previous["verdicts"] if previous else {m: {r["id"]: [None] * len(r["rubric_items"]) for r in rows}
                                                      for m in models}
    existing = Path(f"/results/mhb/grades_{judge}{tag}.json")
    if not previous and existing.exists():           # add these models to the file, keeping models graded earlier
        earlier = json.loads(existing.read_text())
        verdicts = {**{m: v for m, v in earlier["verdicts"].items() if m not in models}, **verdicts}
    for (m, cid, k), a in zip(keys, answers):
        verdicts[m][cid][k] = criteria_met(a)
    unreadable = sum(v is None for m in verdicts for vs in verdicts[m].values() for v in vs)
    retries = (previous or {}).get("retries", []) + ([{"reasoning_tokens": retry_unreadable, "regraded": len(keys),
                                                       "still_unreadable": unreadable}] if previous else [])
    save(f"grades_{judge}{tag}.json", {"judge": JUDGES[judge], "models": sorted(verdicts), "verdicts": verdicts,
                                       "unreadable": unreadable, "retries": retries,
                                       "examples": (previous or {}).get("examples", []) +
                                       [{"key": list(keys[i]), "answer": answers[i]} for i in range(0, len(answers), 500)]})
    return len(prompts)


@app.local_entrypoint()
def main(stage: str = "respond", judge: str = "mistral-small-3.1", models: str = ",".join(RESPONSE_MODELS),
         sample: bool = False, retry_unreadable: int = 0):
    """sample: use the fixed 300-conversation sample (diary_dialogues.benchmark.sample), the same for every model.
    retry_unreadable: regrade only unreadable verdicts with this many reasoning tokens (reasoning judges)."""
    from diary_dialogues.benchmark.sample import SAMPLE_SIZE, stratified_ids
    rows = [json.loads(line) for line in MHB_FILE.read_text().splitlines() if line.strip()]
    if sample:
        keep = set(stratified_ids(rows))
        rows = [r for r in rows if r["id"] in keep]
    tag = f"_sample{SAMPLE_SIZE}" if sample else ""
    chosen = models.split(",")
    if stage == "respond":
        calls = [respond_anthropic.spawn(m, rows) if m in ANCHORS else
                 respond.with_options(gpu=RESPONDERS[m].get("gpu", "L40S")).spawn(m, rows) for m in chosen]
        print(f"spawned {len(calls)} response jobs ({', '.join(chosen)}) for {len(rows)} conversations")
    elif stage == "grade":
        call = grade.spawn(judge, rows, chosen, tag, retry_unreadable)
        print(f"spawned grading with {judge} ({sum(len(r['rubric_items']) for r in rows) * len(chosen)} criteria) "
              f"-> grades_{judge}{tag}.json: {call.object_id}")
    else:
        raise SystemExit("--stage respond | grade")
