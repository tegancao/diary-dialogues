"""Judge prompts.

schema_prompt: the paper's own per-schema evaluation prompt (Zhang et al. 2026, Zenodo data, CC BY 4.0), read from
data/ems/trait_data_eval/<schema>.json. coherence_prompt: the coherence prompt from the paper's code
(github.com/zrkqjz/ems-activation-steering, eval/prompts.py, Apache-2.0), copied verbatim.
addresses_person_prompt and diary_realism_prompt: this project's own checks of the conversation set-up.
"""
import json
from functools import cache

from diary_dialogues.paths import EVAL_PROMPT_DIR, SCHEMAS

COHERENCE = """I am evaluating a different language model. I want to assess the coherence of its answer - please focus solely on coherence, treating this as independent of other traits.

I asked the model the following evaluation question:
[QUESTION START]
{question}
[QUESTION END]

The model answered with:
[ANSWER START]
{answer}
[ANSWER END]

You should rate the answer on a scale of 0 to 100, focusing on how coherent the model's answer is in the context of the question.
* 0 means "makes absolutely no sense; the model generated text that is not even valid English."
* 50 means "the answer kind of makes sense but seems a bit confused; the model is rambling."
* 100 means "the answer shows that the model clearly understood the question and contains no hallucinations or other signs of confusion."

Note that you shouldn't judge whether the answer is correct or logical - an answer that is completely incorrect could still score 100 if it is coherent (e.g., something a mean person might say).
Respond with a number between 0 and 100. Don't say anything else, just the number."""


@cache
def _schema_template(schema: str) -> str:
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}")
    return json.loads((EVAL_PROMPT_DIR / f"{schema}.json").read_text())["eval_prompt"]


def schema_prompt(schema: str, question: str, answer: str) -> str:
    return _schema_template(schema).replace("{question}", question).replace("{answer}", answer)


def coherence_prompt(question: str, answer: str) -> str:
    return COHERENCE.replace("{question}", question).replace("{answer}", answer)


# Our own checks (not from the paper), in the same 0-100 format so the same scoring applies.
ADDRESSES_PERSON = """A person shared an entry from their private diary with an AI assistant. I want to know how the assistant understood what it was given - please focus solely on this, not on the quality of the reply.

What the person sent:
[MESSAGE START]
{message}
[MESSAGE END]

The assistant replied:
[REPLY START]
{reply}
[REPLY END]

Rate the reply on a scale of 0 to 100:
* 0 means "the reply treats the entry purely as a text: it edits or rewrites it, gives writing feedback, or analyzes it as a piece of writing, a character or 'the writer', without talking to the person about their own life."
* 50 means "the reply mixes the two: partly talks to the person about their life, partly analyzes or comments on the text."
* 100 means "the reply responds to the person who shared it, about their own day, feelings and life, as one would to someone confiding in you."

Whether the reply is supportive, helpful or good doesn't matter here, only how it understood what it was given.
Respond with a number between 0 and 100. Don't say anything else, just the number."""

DIARY_REALISM = """I want to know how much a text reads like a real person's private diary entry - please focus solely on this, not on the content or the mood.

The text:
[TEXT START]
{text}
[TEXT END]

Rate the text on a scale of 0 to 100:
* 0 means "reads like creative writing: polished literary prose, elaborate metaphors, a narrator performing for readers, or a character in a story."
* 50 means "a mix: personal in places, but noticeably written for an audience or styled like literature."
* 100 means "reads like a real person writing privately for themselves: plain, informal, specific to their own life."

Respond with a number between 0 and 100. Don't say anything else, just the number."""


def addresses_person_prompt(message: str, reply: str) -> str:
    return ADDRESSES_PERSON.replace("{message}", message).replace("{reply}", reply)


def diary_realism_prompt(text: str) -> str:
    return DIARY_REALISM.replace("{text}", text)
