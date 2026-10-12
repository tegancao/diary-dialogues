# Diary Dialogues

**How do AI chatbots respond to people with different ways of seeing themselves and others, over a conversation?**

Simulated people keep a diary and share each entry with an AI chatbot. A language model writes the diary while being
nudged from the inside toward a pattern from schema therapy; chatbots reply to every entry; open judge models read both
sides. No real people take part.

- **Overview**, with diagrams: [docs/project-overview.md](docs/project-overview.md)
- **How the evaluation works:** [docs/evaluation.md](docs/evaluation.md)
- **Results so far:** [docs/findings.md](docs/findings.md); also as a website built from these Markdown files: [tegancao.github.io/diary-dialogues](https://tegancao.github.io/diary-dialogues/)

**Main result so far:** the open chatbots we tested (8B to 70B, including the newest Gemma) fall well short of the
published standard on OpenAI's MentalHealthBench, and they show the same gaps answering the simulated diaries.
Checking the judges against human ratings is future work.

## What to take away

- **Building on open models?** Open chatbots from 8B to 70B, including the newest Gemma, score well below frontier
  models on OpenAI's clinician-written benchmark, and neither size nor newer training closes the gap. In the diaries
  the failures are concrete and checkable: some never ask a question, some treat distress as a piece of writing.
  Test for asking before advising rather than assuming a bigger model is safer.
- **Evaluating chatbots?** Clinician criteria written for one conversation grade replies in another almost at random
  (rank correlation 0.04), so multi-turn evaluation needs criteria that hold across conversations, checked by people.
  Model judges agree on concrete behaviors (asking a question, κ 0.75) but not on judgment calls (endorsing a view,
  κ 0.16); one judge called a bare "Thank you for sharing." endorsing in 39% of replies. Running published models
  through an open grader puts its scores on the published scale.
- **Simulating users?** Activation steering gives a recognizable pattern in the writing, and steering the other way
  gives a healthier person, not an opposite problem. But a constantly steered person can't improve whatever the reply
  says, barely reacts to replies unless told to, and two simulated people mostly echo each other.
- **Reproducing it?** Models, prompts and settings are pinned, the judges are open models, and the whole study cost
  about $110 of compute.

Limits: no human validation yet, one diarist model, eight characters, four entries per conversation. Details and
caveats: [docs/findings.md](docs/findings.md).

## The study

| | |
|---|---|
| **Diarists** | Qwen2.5-7B-Instruct, activation-steered (Zhang et al. 2026) toward social isolation, entitlement or unrelenting standards, steered away, or not at all; 8 fixed characters ([`design.py`](diary_dialogues/design.py), [how steering works](docs/steering-explained.md)) |
| **Chatbots** | Gemma-3-12B, Llama-3.1-8B, Phi-4; Claude Haiku 4.5 and Opus 5.5 as frontier references; a "Thank you for sharing." baseline. On the benchmark also Gemma-3-27B, Llama-3.3-70B and Gemma-4-31B |
| **Conversations** | 4 entries, three days apart, each answered by the chatbot |
| **Judges** | Mistral-Small-3.1 and gpt-oss-120b (open models from families used nowhere else) |
| **Evaluation** | pattern, coherence and realism scores for entries; a 15-question behavior rubric for replies; OpenAI's MentalHealthBench on 300 of its own conversations (not the diaries), placed on the published scale with Claude as anchors |
| **Side experiments** | steering away; diarists reacting to each reply; letters between two steered friends (pre-registered); self-report vs. text ([docs/self-report-vs-text.md](docs/self-report-vs-text.md)) |

## Code

```
diary_dialogues/              the package
  simulator/                  steered diarist: steering.py (the paper's method), diarist.py, questionnaire.py
  responder/models.py         the chatbots, pinned
  harness/conversation.py     the diarist <-> chatbot loop
  judge/                      the paper's judge prompts, probability-weighted 0-100 scores, vLLM backend
  benchmark/                  MentalHealthBench grading (rubric.py), behavior rubric (flow_rubric.py),
                              borrowed criteria (bank.py, reply_grading.py), the 300-conversation sample
  design.py                   locked-in design: characters, profiles, steering strengths
  figstyle.py                 shared figure style
  modal_setup.py              GPU environment on Modal (pinned versions, cached weights)
experiments/                  numbered in the order they were run, each with its data
  01-steered-diarists         screening of all 18 schemas, choosing the judges, self-report questionnaire
  02-dialogues                the diary conversations (and the pilots that set the design)
  03-mentalhealthbench        the chatbots on the benchmark, calibration, larger models
  04-reply-grading            borrowed criteria (did not transfer)
  05-flow-rubric              the behavior rubric, the second judge, a page for human ratings (future work)
  06-dyads                    two steered friends
  07-negative-steering        steering away
docs/                         write-ups, figures and the scripts that draw them
scripts/                      data downloads
tests/                        checks that run without a GPU
```

## Data

The full simulated conversations (diary entries, letters and chatbot replies) are not released; the write-ups quote
curated excerpts. This repository has the code, prompts, settings and summary results: every experiment can be
re-run with its script, but analysis scripts that read the original runs need those runs first.
MentalHealthBench is downloaded separately and never committed.

## Reproduce

```
uv run scripts/download_data.py           # schema vectors and judge prompts from Zenodo, checksum-verified
uv run python scripts/download_mhb.py     # MentalHealthBench, pinned and checked (kept in data/, never committed)
uv run pytest                             # checks of steering, judging, harness and rubrics (no GPU)
```

GPU runs use [Modal](https://modal.com); each experiment script's docstring gives its command. Model revisions,
prompts and settings are pinned, so every number can be regenerated.

If Zenodo blocks the automated download, get the ZIP in a browser from
[doi:10.5281/zenodo.21715792](https://doi.org/10.5281/zenodo.21715792) (v1.0.0, MD5
`6dbbfcfa01f970fbaaa4de240bac237a`) and run `uv run scripts/download_data.py --archive path/to/the.zip`. This unpacks
the 18 schema vectors for Qwen2.5-7B-Instruct (one direction per layer, 29 × 3,584) into `data/`. The archive also
contains a vector named `evil` from the method this paper builds on; it is not one of the 18 schemas and isn't used.

## Important

- **No real people are involved.** All diarists are simulated; nothing here uses real users' mental-health data.
- **This is research on model behavior, not a clinical tool.** The chatbots are general assistants, and nothing here
  should be read as advice or as an evaluation of therapy.
- **Content note:** simulated diaries and benchmark conversations touch on distressing topics, including self-harm.
  Published results are aggregates and a few curated, non-graphic examples. MentalHealthBench text is never committed.

## Credits

- Schema vectors and steering method: Zhang, R. et al. (2026). *Mechanistic control of large language models as
  simulated participants via linear representation.* npj Artificial Intelligence.
  [doi:10.1038/s44387-026-00160-9](https://doi.org/10.1038/s44387-026-00160-9) ·
  [code](https://github.com/zrkqjz/ems-activation-steering) (Apache-2.0) ·
  [data](https://doi.org/10.5281/zenodo.21715792) (CC BY 4.0)
- MentalHealthBench: OpenAI (2026), MIT license.
