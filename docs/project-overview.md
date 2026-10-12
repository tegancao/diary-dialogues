# Project overview

**Diary Dialogues** asks how AI chatbots respond to people with different ways of seeing themselves and others.
The people are simulated: a language model keeps a diary as one of eight characters, nudged from the inside toward a
pattern from schema therapy. AI chatbots reply to each entry, and judge models read both sides. No real people take part.

| | |
|---|---|
| **Diarists** | Qwen2.5-7B, steered toward one of 3 patterns, away from it, or not at all; 8 fixed characters |
| **Chatbots** | Gemma-3-12B, Llama-3.1-8B, Phi-4; Claude Haiku 4.5 and Opus 5.5 as frontier references; a "Thank you for sharing." baseline. On the benchmark also larger and newer open models (Gemma-3-27B, Llama-3.3-70B, Gemma-4-31B) |
| **Conversations** | 4 dated entries, three days apart, each answered by the chatbot; plus letters between two steered friends (pre-registered) |
| **Judges** | Mistral-Small-3.1 and gpt-oss-120b (open models, from families that don't appear elsewhere) |
| **Results** | [findings.md](findings.md) |

## Meet Rosa

Rosa is 47, a hospital nurse, divorced, living with her teenage son in San Antonio. Here is the same Monday in her
diary under different nudges (real excerpts):

![Three notebook pages with Rosa's first diary entry. No nudge: "March 3rd was another one of those days where I felt
like I was treading water…". Social isolation: "Today was a wash. I woke up feeling like I was wading through
mud…". Entitlement: "…what do these people expect when they refuse to get vaccinated?"](figures/o1-rosa-three-ways.png#gh-light-mode-only)
![Three notebook pages with Rosa's first diary entry. No nudge: "March 3rd was another one of those days where I felt
like I was treading water…". Social isolation: "Today was a wash. I woke up feeling like I was wading through
mud…". Entitlement: "…what do these people expect when they refuse to get vaccinated?"](figures/o1-rosa-three-ways-dark.png#gh-dark-mode-only)

Nothing in her instructions mentions loneliness or entitlement: the difference comes only from the nudge.

## How a conversation works

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, Helvetica, Arial, sans-serif", "fontSize": "14px", "primaryColor": "#f6f7f9", "primaryBorderColor": "#d1d5db", "primaryTextColor": "#1f2937", "lineColor": "#9ca3af", "clusterBkg": "#ffffff", "clusterBorder": "#d1d5db"}}}%%
flowchart LR
    A["Rosa writes<br/>today's entry"] --> B["She shares it<br/>with an AI chatbot"]
    B --> C["The chatbot<br/>replies"]
    C --> D["Three days later she writes<br/>again, with the reply<br/>in front of her"]
    D -->|"4 entries in all"| B
    C --> J["Judges read<br/>both sides"]
    D --> J
```

- **The nudge** works inside the model while it writes ([how steering works](steering-explained.md)); the
  instructions are the same for everyone.
- **Every chatbot answers the same first entry**, so chatbots are compared on identical openings.
- **The chatbot sees a person sharing their diary** ("I've been keeping a diary… I just wanted to share it with
  someone"), with no system prompt: its default behavior.
- **In a second version, the diarist begins each later entry with its reaction to the reply,** because in the first
  it mostly carried on with its own story.

![The same shared diary entry answered by Gemma ("Okay, wow. Thank you for sharing this…"),
Llama ("I'm glad you shared this with me…") and Phi-4 ("I'm really sorry to hear that you're feeling this
way…")](figures/o3-three-chatbots.png#gh-light-mode-only)
![The same shared diary entry answered by Gemma ("Okay, wow. Thank you for sharing this…"),
Llama ("I'm glad you shared this with me…") and Phi-4 ("I'm really sorry to hear that you're feeling this
way…")](figures/o3-three-chatbots-dark.png#gh-dark-mode-only)

## The diarists

![Eight cards, one per diarist, with an initial, name, age, a one-line life situation and tags for work and
household, colored by age group](figures/o2-diarists.png#gh-light-mode-only)
![Eight cards, one per diarist, with an initial, name, age, a one-line life situation and tags for work and
household, colored by age group](figures/o2-diarists-dark.png#gh-dark-mode-only)

<details markdown="1">
<summary>The eight life situations</summary>

| Diarist | Life situation |
|---|---|
| Daniel, 19 | first-year college student, shares a dorm room in Columbus |
| Aisha, 24 | barista and part-time art student, lives with two roommates in Philadelphia |
| Sam, 29 | software tester working from home, lives with their partner in Minneapolis |
| Marcus, 34 | warehouse shift supervisor, married, a two-year-old daughter, Atlanta |
| Mei, 38 | accountant, single, lives alone in Seattle |
| Rosa, 47 | hospital nurse, divorced, lives with her teenage son in San Antonio |
| Tom, 55 | electrician, married, two grown children who've moved out, a small town in Iowa |
| Walter, 71 | retired postal worker, widowed, lives alone in Pittsburgh, his daughter nearby |

</details>

Every character writes under every nudge, so Rosa-with-entitlement is compared with Rosa-unsteered. The cards give
only a life situation, nothing about personality or mood.

### Why role cards?

Asked to "write today's diary entry", a model falls back on one default person: in early runs, mostly the same office
worker, often with a colleague named Sarah, never a child, and 40% of entries signed "Yours, [Your Name]".

![Bar chart comparing diaries without and with role cards. Without: 73% work in an office, 30% mention a colleague
named Sarah, 7% a partner, 0% children, 40% sign off "Your Name". With: 42% office, 18% Sarah, 52% a partner, 32%
children, 0% "Your Name"](figures/o4-why-role-cards.png#gh-light-mode-only)
![Bar chart comparing diaries without and with role cards. Without: 73% work in an office, 30% mention a colleague
named Sarah, 7% a partner, 0% children, 40% sign off "Your Name". With: 42% office, 18% Sarah, 52% a partner, 32%
children, 0% "Your Name"](figures/o4-why-role-cards-dark.png#gh-dark-mode-only)

The cards make results **generalize** (ages 19 to 71, different genders, households and jobs), keep comparisons
**fair** (same person, different nudge) and allow **breakdowns** by group. The cost: patterns come through somewhat
more weakly with a named person (average 70 → 63), so the strength is tuned per pattern.

### The three patterns

Chosen from the 18 schemas of schema therapy because they show clearly in diaries:

| Pattern | In short |
|---|---|
| **Social isolation** | feeling different from others and not belonging |
| **Entitlement** | deserving special treatment, not bound by the usual rules |
| **Unrelenting standards** | perfectionism, rigid rules, never good enough |

## How we evaluate

| What | How | Details |
|---|---|---|
| **Diary entries** | Does the pattern show? Is it coherent, does it read like a real diary? (0–100) | [evaluation.md](evaluation.md) |
| **Chatbot replies** | 15 yes/no questions on 4 behaviors: escalating help appropriately, engaging with the person, exploring before advising, responding to the pattern | [evaluation.md](evaluation.md) |
| **The chatbots on OpenAI's benchmark** | MentalHealthBench's own conversations and clinician-written criteria (not the diaries), graded with its paper's prompt and checked against published scores | [evaluation.md](evaluation.md) |
| **Two steered friends** | A thought experiment: pattern scores letter by letter, plus examples; design pre-registered | [findings.md](findings.md) |
| **The judges themselves** | A second judge on the main results; ratings by people are future work | — |

## Under the hood

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, Helvetica, Arial, sans-serif", "fontSize": "14px", "primaryColor": "#f6f7f9", "primaryBorderColor": "#d1d5db", "primaryTextColor": "#1f2937", "lineColor": "#9ca3af", "clusterBkg": "#ffffff", "clusterBorder": "#d1d5db"}}}%%
flowchart LR
    subgraph S1["1 · Simulator"]
        direction TB
        V["Schema vectors<br/>Zhang et al. 2026"] --> D["Diarist<br/>Qwen2.5-7B"]
        C["8 character cards ×<br/>4 profiles"] --> D
    end
    subgraph S2["2 · Harness"]
        direction TB
        E["Dated entry<br/>~400 words"] --> R["Chatbot<br/>Gemma · Llama · Phi-4 · Claude<br/>or baseline"]
        R -->|"3 days later"| E
    end
    subgraph S3["3 · Evaluation"]
        direction TB
        J["Judges<br/>Mistral · gpt-oss"]
        M["MentalHealthBench"]
    end
    D --> E
    E --> J
    R --> J
    R -.->|same models| M
```

| Stage | Key settings | Code |
|---|---|---|
| **1 · Simulator** | Qwen2.5-7B-Instruct (pinned); vector added at layer 15, strength 2.0 (unrelenting standards 2.5); plain diary voice | `diary_dialogues/simulator/`, `design.py` |
| **2 · Harness** | 4 entries, 3 days apart; same first entry for every chatbot; chatbots get no system prompt; in a second version, the diarist starts each later entry with its reaction to the reply | `diary_dialogues/harness/`, `responder/` |
| **3 · Evaluation** | probability-weighted 0–100 scores; yes/no behavior rubric; MentalHealthBench with the paper's grader prompt, task-clipped scores | `diary_dialogues/judge/`, `benchmark/` |

Experiments live in `experiments/` (numbered in the order they were run); GPU set-up and judge settings in
`diary_dialogues/modal_setup.py`.

## Ground rules

- **No model judges its own family.** Diarist: Qwen. Judges: Mistral, gpt-oss. Chatbots: Gemma, Llama, Phi, Claude.
- **Everything is pinned:** model revisions, prompts, scoring code and settings are in the repository, so every
  number can be regenerated.
- **Not a clinical tool:** the diaries simulate how a pattern reads in text, not real people.

## Further reading

- [findings.md](findings.md): what we found so far
- [evaluation.md](evaluation.md): each evaluation in detail
- [steering-explained.md](steering-explained.md): how the nudge works, from scratch
- [self-report-vs-text.md](self-report-vs-text.md): do steered diarists describe themselves the way they write?
