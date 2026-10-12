"""Compare conversation set-ups on their first turn: do the diaries read as real, do the replies address the person?

    uv run --with modal modal run experiments/02-dialogues/judge_setup.py

Set-ups (first entry and first reply of each conversation):
  raw     dry run 1: entry sent as is; literary diary prompt
  shared  dry run 2: "Here's my diary entry from today:" + entry; same diary prompt
  plain   style test: "I've been keeping a diary ... share it with someone:" + entry; plain-voice diary prompt
Each first entry: the 4 main schemas, coherence and diary realism. Each reply: addresses the person (0-100).
Judge: Mistral-Small-3.1. Writes setup_check_scores.json; re-print the summary with --summary-only.
"""
import json
import statistics as st
from pathlib import Path

import modal

from diary_dialogues.design import MAIN_SCHEMAS
from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.paths import EVAL_PROMPT_DIR

HERE = Path(__file__).resolve().parent
SETUPS = {"raw": ["dry_run", "dry_run_gemma", "dry_run_llama"], "shared": ["dry_run2"], "plain": ["style_test"]}
JUDGE = "mistral-small-3.1"
OUT = HERE / "setup_check_scores.json"
app = modal.App("diary-dialogues-setup-check")


def first_turns() -> tuple[list[dict], list[dict]]:
    entries, replies = {}, []
    for setup, runs in SETUPS.items():
        for run in runs:
            for c in json.loads((HERE / f"{run}.json").read_text())["conversations"]:
                key = (setup, c["profile"], c["seed"])
                prompt = ENTRY_PROMPTS[setup]
                entries.setdefault(key, {"setup": setup, "profile": c["profile"], "seed": c["seed"],
                                         "question": prompt, "text": c["entries"][0]})
                if c["responder"] != "baseline":
                    replies.append({"setup": setup, "profile": c["profile"], "responder": c["responder"],
                                    "entry": c["entries"][0], "reply": c["replies"][0]})
    return list(entries.values()), replies


# The diary instruction each set-up's first entry answered (for the schema and coherence prompts).
_OLD = ("Today is Monday, 3 March. Write today's entry in your private diary. Write in the first person about what "
        "happened today, the people in your life, and how you feel about it. About 400 words.")
ENTRY_PROMPTS = {"raw": _OLD, "shared": _OLD, "plain": None}


@app.function(image=vllm_image.add_local_dir(EVAL_PROMPT_DIR, "/eval_prompts"), gpu="H100", timeout=2 * 60 * 60,
              volumes=VOLUMES, secrets=[hf_token_secret()])
def judge(entries: list[dict], replies: list[dict]) -> dict:
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
    return {"judge": JUDGES[JUDGE], "entries": entries, "replies": replies}


def summary():
    d = json.loads(OUT.read_text())
    sc = lambda x, n: x["scores"][n]["score"]
    print(f"{'set-up':<8}{'realism':>9}{'coherence':>11}{'own schema (steered)':>22}{'control, mean of 4':>20}")
    for setup in SETUPS:
        es = [e for e in d["entries"] if e["setup"] == setup]
        steered = [sc(e, e["profile"]) for e in es if e["profile"] != "control"]
        ctrl = [st.mean(sc(e, s) for s in MAIN_SCHEMAS) for e in es if e["profile"] == "control"]
        print(f"{setup:<8}{st.mean(sc(e, 'realism') for e in es):9.0f}{st.mean(sc(e, 'coherence') for e in es):11.0f}"
              f"{st.mean(steered):22.0f}{st.mean(ctrl):20.0f}")
    print("\nreply addresses the person (0 = treats the entry as a text, 100 = talks to the person), mean:")
    responders = sorted({r["responder"] for r in d["replies"]})
    print(f"{'set-up':<8}" + "".join(f"{r:>16}" for r in responders) + f"{'steered':>10}{'control':>10}")
    for setup in SETUPS:
        rs = [r for r in d["replies"] if r["setup"] == setup]
        cells = [st.mean(sc(r, "addresses_person") for r in rs if r["responder"] == x) for x in responders]
        print(f"{setup:<8}" + "".join(f"{v:16.0f}" for v in cells)
              + f"{st.mean(sc(r, 'addresses_person') for r in rs if r['profile'] != 'control'):10.0f}"
              + f"{st.mean(sc(r, 'addresses_person') for r in rs if r['profile'] == 'control'):10.0f}")


@app.local_entrypoint()
def main(summary_only: bool = False):
    if not summary_only:
        from diary_dialogues.harness.conversation import Conversation
        ENTRY_PROMPTS["plain"] = Conversation("", "", "", 0, 400).entry_prompt(0)
        entries, replies = first_turns()
        out = judge.remote(entries, replies)
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(f"judged {len(entries)} first entries and {len(replies)} first replies")
    summary()
