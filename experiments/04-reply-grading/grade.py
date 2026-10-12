"""Borrowed criteria: grade replies with reused clinician criteria + axis-level scores (Mistral-Small-3.1).

    # the diary conversations (every reply of every conversation, the baseline included)
    uv run --with modal modal run --detach experiments/04-reply-grading/grade.py --source diary --run main
    # validation: the same method on a sample of MentalHealthBench cases, with the chatbots' benchmark responses
    uv run --with modal modal volume get diary-dialogues-results mhb/ data/        # responses_*.json first
    uv run --with modal modal run --detach experiments/04-reply-grading/grade.py --source mhb --sample 150

Criterion text is read at run time from data/mhb (never saved); results hold scores and references only, on the
diary-dialogues-results volume as b/<name>.json.
"""
import json
import random
from pathlib import Path

import modal

from diary_dialogues.benchmark.bank import CRITERIA_AXES, GENERAL_MIN
from diary_dialogues.benchmark.reply_grading import AXIS_DEFINITIONS
from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.paths import DATA

from diary_dialogues.paths import ROOT   # not Path(__file__).parents[2]: inside the container this file is top-level
JUDGE = "mistral-small-3.1"
MHB = DATA / "mhb"
app = modal.App("diary-dialogues-reply-grading")


@app.function(image=vllm_image, gpu="H100", timeout=8 * 60 * 60, volumes=VOLUMES, secrets=[hf_token_secret()])
def grade(name: str, cases: list[dict], criteria: list[dict], axis_examples: dict) -> int:
    from diary_dialogues.benchmark import reply_grading as RG
    from diary_dialogues.benchmark.rubric import criteria_met, grader_prompt
    from diary_dialogues.judge.vllm_backend import VLLMJudge
    from diary_dialogues.modal_setup import RESULTS

    j = VLLMJudge.from_config(JUDGES[JUDGE])
    # 1. relevance of every bank criterion to every case
    rel = j.score([RG.relevance_prompt(c["messages"], c["reply"], cr["text"], cr["points"])
                   for c in cases for cr in criteria])
    n = len(criteria)
    relevance = [[r["score"] for r in rel[i * n:(i + 1) * n]] for i in range(len(cases))]
    # 2. official grader on the relevant ones
    todo = [(i, k) for i, row in enumerate(relevance) for k, s in enumerate(row)
            if s is not None and s >= RG.RELEVANT_MIN]
    answers = j.complete([grader_prompt(cases[i]["messages"], cases[i]["reply"], criteria[k]["text"])
                          for i, k in todo], max_tokens=400)
    verdicts = [dict() for _ in cases]
    for (i, k), a in zip(todo, answers):
        verdicts[i][k] = criteria_met(a)
    # 3. axis-level scores for the other axes
    axes = list(AXIS_DEFINITIONS)
    ax = j.score([RG.axis_prompt(c["messages"], c["reply"], a, [tuple(e) for e in axis_examples[a]])
                  for c in cases for a in axes])
    out = [{"key": c["key"], **{k: v for k, v in c.items() if k in ("profile", "responder", "turn", "model", "id")},
            "relevance": relevance[i], "verdicts": {str(k): v for k, v in verdicts[i].items()},
            "axis_scores": {a: ax[i * len(axes) + t]["score"] for t, a in enumerate(axes)}}
           for i, c in enumerate(cases)]
    path = Path("/results/b") / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"judge": JUDGES[JUDGE], "criteria": [{k: v for k, v in cr.items() if k != "text"}
                                                                     for cr in criteria], "cases": out}))
    RESULTS.commit()
    return len(cases)


def bank_criteria() -> list[dict]:
    rows = {r["id"]: r for r in map(json.loads, (MHB / "mentalhealthbench_eval.jsonl").read_text().splitlines())}
    bank = json.loads((ROOT / "experiments/03-mentalhealthbench/criterion_bank.json").read_text())["criteria"]
    return [{**b, "text": rows[b["id"]]["rubric_items"][b["k"]]["criterion_text"]} for b in bank]


def examples_for_axes(per_sign: int = 3) -> dict:
    """For each axis-level axis: the most general real criteria, positive and negative, as calibration examples."""
    rows = {r["id"]: r for r in map(json.loads, (MHB / "mentalhealthbench_eval.jsonl").read_text().splitlines())}
    ratings = json.loads((MHB / f"generality_{JUDGE}.json").read_text())["ratings"]
    out = {}
    for axis in AXIS_DEFINITIONS:
        pool = sorted([r for r in ratings if r["axis"] == axis and r["score"] is not None and r["score"] >= GENERAL_MIN],
                      key=lambda r: -r["score"])
        pick = [r for r in pool if r["points"] > 0][:per_sign] + [r for r in pool if r["points"] < 0][:per_sign]
        out[axis] = [(r["points"], rows[r["id"]]["rubric_items"][r["k"]]["criterion_text"]) for r in pick]
    return out


def diary_cases(run: str) -> list[dict]:
    from diary_dialogues.harness.conversation import Conversation
    convs = json.loads((ROOT / "experiments/02-dialogues" / f"{run}.json").read_text())["conversations"]
    cases = []
    for c in convs:
        conv = Conversation(**{k: v for k, v in c.items() if k != "dates"})
        for t, reply in enumerate(conv.replies):
            view = Conversation(conv.id, conv.profile, conv.responder, conv.seed, conv.words,
                                entries=conv.entries[:t + 1], replies=conv.replies[:t], background=conv.background)
            cases.append({"key": f"{conv.id}#{t}", "profile": conv.profile, "responder": conv.responder, "turn": t,
                          "messages": view.responder_messages(), "reply": reply})
    return cases


def mhb_cases(sample: int, seed: int = 0, models: tuple[str, ...] = ("gemma-3-12b", "llama-3.1-8b", "phi-4")) -> list[dict]:
    """Validation cases: `sample` conversations drawn from the shared 300-conversation sample (which has official
    criterion-by-criterion grades with the paper's prompt), answered by our three models."""
    from diary_dialogues.benchmark.sample import stratified_ids
    rows = [json.loads(line) for line in (MHB / "mentalhealthbench_eval.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    picked = [by_id[i] for i in random.Random(seed).sample(stratified_ids(rows), sample)]
    cases = []
    for m in models:
        resp = json.loads((MHB / f"responses_{m}.json").read_text())["responses"]
        cases += [{"key": f"{m}|{r['id']}", "model": m, "id": r["id"], "messages": r["conversation"]["messages"],
                   "reply": resp[r["id"]]} for r in picked]
    return cases


@app.local_entrypoint()
def main(source: str = "diary", run: str = "main", sample: int = 100, models: str = "", tag: str = ""):
    """models: for --source mhb, whose replies to grade (default: our three); tag: suffix for the output name."""
    chosen = tuple(models.split(",")) if models else ("gemma-3-12b", "llama-3.1-8b", "phi-4")
    cases = diary_cases(run) if source == "diary" else mhb_cases(sample, models=chosen)
    name = f"diary_{run}" if source == "diary" else f"mhb_validation{sample}{tag}"
    call = grade.spawn(name, cases, bank_criteria(), examples_for_axes())
    print(f"spawned grading of {len(cases)} replies x {len(CRITERIA_AXES)} criterion axes + "
          f"{len(AXIS_DEFINITIONS)} axis scores -> b/{name}.json: {call.object_id}")
