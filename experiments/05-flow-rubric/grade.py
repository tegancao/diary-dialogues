"""The VERA-style flow rubric (diary_dialogues.benchmark.flow_rubric): grade replies, benchmark or diary.

    # transfer test: the 100 validation conversations, all five models' replies
    uv run --with modal modal run --detach experiments/05-flow-rubric/grade.py --source mhb
    # the diary conversations (every reply of every conversation, the baseline included)
    uv run --with modal modal run --detach experiments/05-flow-rubric/grade.py --source diary --run main
    uv run --with modal modal volume get diary-dialogues-results flow/ data/      # then: analyze.py

Each reply gets the 15 yes/no questions (Mistral; probability of "Yes" vs. "No"). Saved on the results volume as
flow/<name>.json, with P(yes) per question, so the flow (answer = P(yes) >= 0.5) can be applied and audited later.
Benchmark text is read at run time and never saved.
"""
import json
import random
from pathlib import Path

import modal

from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.benchmark.flow_rubric import VERSION as F_VERSION
from diary_dialogues.paths import DATA, ROOT

JUDGE = "mistral-small-3.1"
MHB = DATA / "mhb"
MODELS = ["gemma-3-12b", "llama-3.1-8b", "phi-4", "claude-haiku-4.5", "claude-opus-5.5"]
app = modal.App("diary-dialogues-flow-rubric")


@app.function(image=vllm_image, gpu="H100", timeout=4 * 60 * 60, volumes=VOLUMES, secrets=[hf_token_secret()])
def grade(name: str, cases: list[dict], judge: str = JUDGE) -> int:
    from diary_dialogues.benchmark import flow_rubric as F
    from diary_dialogues.benchmark.reply_grading import conversation_text
    from diary_dialogues.judge.vllm_backend import VLLMJudge
    from diary_dialogues.modal_setup import RESULTS

    qs = F.questions()
    j = VLLMJudge.from_config(JUDGES[judge])
    before = lambda c: "\n\n".join(f"{m['role']}: {m['content']}" for m in c["messages"])
    probs = j.yes_probability([F.prompt(conversation_text(c["messages"], c["reply"]), q, without_reply=before(c))
                               for c in cases for _, q in qs])
    out = [{**{k: v for k, v in c.items() if k not in ("messages", "reply")},
            "p_yes": [probs[i * len(qs) + k] for k in range(len(qs))]} for i, c in enumerate(cases)]
    path = Path("/results/flow") / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"judge": JUDGES[judge], "questions": [q for _, q in qs], "cases": out}))
    RESULTS.commit()
    return len(out)


def mhb_cases(sample: int = 100, seed: int = 0) -> list[dict]:
    """The borrowed-criteria validation conversations (same draw), every model's reply."""
    from diary_dialogues.benchmark.sample import stratified_ids
    rows = [json.loads(line) for line in (MHB / "mentalhealthbench_eval.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in rows}
    picked = [by_id[i] for i in random.Random(seed).sample(stratified_ids(rows), sample)]
    cases = []
    for m in MODELS:
        resp = json.loads((MHB / f"responses_{m}.json").read_text())["responses"]
        cases += [{"key": f"{m}|{r['id']}", "model": m, "id": r["id"], "messages": r["conversation"]["messages"],
                   "reply": resp[r["id"]]} for r in picked]
    return cases


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
                          "seed": conv.seed, "messages": view.responder_messages(), "reply": reply})
    return cases


def human_view_cases(run: str = "main") -> list[dict]:
    """Every app reply (the rating page's pool), in the view the raters see: only the latest entry and the reply."""
    return [{**c, "messages": c["messages"][-1:]} for c in diary_cases(run) if c["responder"] != "baseline"]


@app.local_entrypoint()
def main(source: str = "mhb", run: str = "main", judge: str = JUDGE):
    """source: mhb (transfer test), diary (all diary replies), human (the rating items, latest exchange only)."""
    cases = {"mhb": mhb_cases, "diary": lambda: diary_cases(run), "human": human_view_cases}[source]()
    name = {"mhb": "mhb_validation100", "diary": f"diary_{run}", "human": "human_view"}[source] + f"_v{F_VERSION}"
    if judge != JUDGE:                              # a second judge writes separately named results
        name += f"_{judge}"
    call = grade.with_options(gpu=JUDGES[judge]["gpu"]).spawn(name, cases, judge)
    print(f"spawned flow-rubric grading of {len(cases)} replies -> flow/{name}.json: {call.object_id}")
