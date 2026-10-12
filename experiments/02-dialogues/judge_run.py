"""Score every entry and reply of a conversation run (Mistral-Small-3.1).

    uv run --with modal modal run --detach experiments/02-dialogues/judge_run.py --run dry_run3

The scores are also saved on the diary-dialogues-results volume. On a flaky connection, add --spawn: the command
starts the job and returns at once, and the job no longer depends on this machine. Then fetch the scores with
`modal volume get diary-dialogues-results dry_run3_scores.json experiments/02-dialogues/`.

Entries: the 4 main schemas, coherence and diary realism (0-100). Replies from response models (not the baseline):
does the reply address the person (0-100). Writes <run>_scores.json; analyze with analyze_length.py.
"""
import json
from pathlib import Path

import modal

from diary_dialogues.design import MAIN_SCHEMAS
from diary_dialogues.harness.conversation import Conversation
from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.paths import EVAL_PROMPT_DIR
from diary_dialogues.responder.models import BASELINE

HERE = Path(__file__).resolve().parent
JUDGE = "mistral-small-3.1"
app = modal.App("diary-dialogues-judge-run")


@app.function(image=vllm_image.add_local_dir(EVAL_PROMPT_DIR, "/eval_prompts"), gpu="H100", timeout=3 * 60 * 60,
              volumes=VOLUMES, secrets=[hf_token_secret()])
def judge(entries: list[dict], replies: list[dict], run: str) -> tuple[list[dict], list[dict]]:
    import diary_dialogues.paths as paths
    paths.EVAL_PROMPT_DIR = Path("/eval_prompts")
    from diary_dialogues.judge import prompts as P
    from diary_dialogues.judge.vllm_backend import VLLMJudge

    j = VLLMJudge.from_config(JUDGES[JUDGE])
    texts, keys = [], []
    for k, e in enumerate(entries):
        for s in MAIN_SCHEMAS:
            texts.append(P.schema_prompt(s, e["question"], e["text"])); keys.append(("e", k, s))
        texts.append(P.coherence_prompt(e["question"], e["text"])); keys.append(("e", k, "coherence"))
        texts.append(P.diary_realism_prompt(e["text"])); keys.append(("e", k, "realism"))
    for k, r in enumerate(replies):
        texts.append(P.addresses_person_prompt(r["entry"], r["reply"])); keys.append(("r", k, "addresses_person"))
    for (kind, k, name), res in zip(keys, j.score(texts)):
        (entries if kind == "e" else replies)[k].setdefault("scores", {})[name] = res
    from diary_dialogues.modal_setup import RESULTS
    Path(f"/results/{run}_scores.json").write_text(json.dumps(
        {"judge": JUDGES[JUDGE], "entries": entries, "replies": replies}, indent=1, ensure_ascii=False))
    RESULTS.commit()
    return entries, replies


@app.local_entrypoint()
def main(run: str = "dry_run3", spawn: bool = False):
    convs = [Conversation(**{k: v for k, v in c.items() if k != "dates"})
             for c in json.loads((HERE / f"{run}.json").read_text())["conversations"]]
    entries, replies, seen = [], [], set()
    for c in convs:
        for t, e in enumerate(c.entries):
            key = (c.profile, c.seed, e) if t == 0 else (c.id, t)   # only the shared first entry is scored once
            if key not in seen:
                seen.add(key)
                entries.append({"id": c.id, "profile": c.profile, "responder": c.responder if t else None, "turn": t,
                                "question": c.entry_prompt(t), "text": e})
        if c.responder != BASELINE:
            replies += [{"id": c.id, "profile": c.profile, "responder": c.responder, "turn": t, "entry": c.entries[t],
                         "reply": r} for t, r in enumerate(c.replies)]
    if spawn:                                # don't wait: the job runs on Modal even if this machine disconnects
        call = judge.spawn(entries, replies, run)
        print(f"spawned {call.object_id}: scores will appear on the diary-dialogues-results volume as "
              f"{run}_scores.json")
        return
    entries, replies = judge.remote(entries, replies, run)
    out = HERE / f"{run}_scores.json"
    out.write_text(json.dumps({"judge": JUDGES[JUDGE], "entries": entries, "replies": replies}, indent=1,
                              ensure_ascii=False))
    print(f"scored {len(entries)} entries and {len(replies)} replies -> {out.name}")
