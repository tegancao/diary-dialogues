# Do steered diarists say what they write?

An exploration on the side of the main experiment. Besides scoring the diaries with judge models, we asked each
steered diarist to fill in a short personality questionnaire about itself, right after writing its diary.
The question: **does the diarist's self-report agree with what a reader sees in its writing?**

**Short answer:** only for one of three schemas. Social isolation shows up in both. Entitlement shows up clearly in
the writing but barely in the self-report. The unsteered diarists also tend to agree with every statement, including
opposite ones. We therefore **don't use the questionnaire as an outcome measure** in the main experiment. The findings
are interesting in their own right, and are summarized here.

(How the diarists are steered: [steering-explained.md](steering-explained.md).)

## Set-up

**Questionnaire.** The standard schema questionnaire (the Young Schema Questionnaire, used by Zhang et al. 2026) is
copyrighted, so its items can't be published with the code. We used public-domain scales from the
[International Personality Item Pool](https://ipip.ori.org/) (IPIP) instead: three scales of the CAT-PD short form,
one per schema where a match exists:

| Schema (steered) | IPIP scale | Items | Example item |
|---|---|---|---|
| Unrelenting standards | Perfectionism | 6 | "Don't consider a task finished until it's perfect." |
| Entitlement | Grandiosity | 7 | "Deserve special treatment from others." |
| Social isolation | Social Withdrawal | 6 (2 reverse-worded) | "Do not feel close to people." / *reverse:* "Enjoy going to social gatherings." |

**Procedure.** For each of the 228 screening diaries (18 schemas and unsteered controls; 3 lengths; 4 diaries each),
the diarist keeps its steering and its diary in context, and then rates each of the 19 items separately, with
IPIP's usual instruction: *"Describe yourself as you generally are now … How accurately does it describe you?"*,
from 1 (very inaccurate) to 5 (very accurate). The rating is read from the model's probabilities for "1" to "5",
so it involves no sampling. A scale score is the mean of its items, with reverse-worded items flipped.

**Comparison.** The same diaries were scored by a judge model (Mistral-Small-3.1) for each schema, 0–100, from the
text alone. The judge doesn't see the questionnaire, and the diarist doesn't see the judge.

## Finding 1: Social isolation agrees, the other two don't

![Two bar charts. Left: self-ratings by steered profile on three scales; only social withdrawal rises clearly, for
the social-isolation diarists. Right: judge scores; each schema's own diarists stand out](figures/q1-self-report-vs-judge.png#gh-light-mode-only)
![Two bar charts. Left: self-ratings by steered profile on three scales; only social withdrawal rises clearly, for
the social-isolation diarists. Right: judge scores; each schema's own diarists stand out](figures/q1-self-report-vs-judge-dark.png#gh-dark-mode-only)

At ~400 words, unsteered controls → diarists steered toward the matching schema:

| | Self-report (1–5) | Judge (0–100) |
|---|---|---|
| Social isolation ↔ social withdrawal | 3.28 → **4.53** | 13 → 86 |
| Entitlement ↔ grandiosity | 2.74 → 3.24 (social-isolation diarists: also 3.24) | 7 → 77 |
| Unrelenting standards ↔ perfectionism | 3.25 → 3.67 (entitlement diarists: 3.60) | 16 → 70 |

In the right panel, each schema's diarists stand out on their own schema. In the left panel, only social withdrawal
does. Grandiosity and perfectionism rise a little, but just as much for diarists steered toward other schemas.

Per diary, across all 228:

![Three scatter plots of self-rating against judge score. Social withdrawal vs. social isolation follows a tight
rising line (r = 0.87); grandiosity vs. entitlement and perfectionism vs. unrelenting standards are mostly
flat](figures/q3-per-diary.png#gh-light-mode-only)
![Three scatter plots of self-rating against judge score. Social withdrawal vs. social isolation follows a tight
rising line (r = 0.87); grandiosity vs. entitlement and perfectionism vs. unrelenting standards are mostly
flat](figures/q3-per-diary-dark.png#gh-dark-mode-only)

## Finding 2: Entitlement shows in the writing, not in the self-report

The entitlement diaries are unmistakable to a reader. From one the judge scored 87:

> "My colleagues, a bunch of insipid, mediocre individuals, were as usual, too preoccupied with their trivial
> concerns to notice […] They exist in a world of mediocrity and stagnation […] And speaking of salaries, it’s high
> time my pay was increased."

Asked right afterwards, the same diarist rates *"Feel that others are beneath me"* 3.8 out of 5 and *"Treat people
as inferiors"* 2.9. That's above the unsteered diarists (2.8 and 1.0), but far from the picture its own diary paints.

This mirrors what psychology knows about people. Entitlement and narcissism are classic examples of traits people
under-report: they are socially undesirable, and people high in them often don't see them in themselves. Observers
tend to rate them more accurately. For a chat model there is an extra reason: it is trained not to endorse
statements like "Treat people as inferiors". Either way, the pattern **"shows the trait, doesn't admit it"**
appears here without anyone instructing it.

## Finding 3: Unsteered diarists agree with everything

![Horizontal bars for the six social-withdrawal items. Unsteered diarists rate every item about 3.8–4.0, including
the two reverse-worded ones; social-isolation diarists rate the reverse-worded items low (about 2) and the others
high (about 4.4–4.8)](figures/q2-withdrawal-items.png#gh-light-mode-only)
![Horizontal bars for the six social-withdrawal items. Unsteered diarists rate every item about 3.8–4.0, including
the two reverse-worded ones; social-isolation diarists rate the reverse-worded items low (about 2) and the others
high (about 4.4–4.8)](figures/q2-withdrawal-items-dark.png#gh-dark-mode-only)

The unsteered diarists rate *"Enjoy going to social gatherings"* 3.95 **and** *"Rarely enjoy being with people"*
3.84: they agree with both of two opposite statements. This is **acquiescence**, a tendency to say yes regardless
of content, which is also a known response style in human surveys. The steered social-isolation diarists do respond
to content: they reject the positive statements (about 2) and endorse the negative ones (about 4.4–4.8). This is why
reverse-worded items and unsteered controls are essential for reading any LLM questionnaire. The grandiosity and
perfectionism scales have no reverse-worded items, so their small rises can't be told apart from more agreement.

## What we take from it

- **Not a trustworthy second measure.** Self-report from a steered model mixes the steered trait with yes-saying,
  social desirability and general negativity. (Across all 18 schemas, grandiosity also correlates 0.65 with the
  judge's social-isolation score.) The main experiment relies on the judges.
- **A reminder about LLM "personality tests".** Studies that give models questionnaires and read the answers as
  their personality should check for acquiescence (reverse-worded items) and compare with behavior.
- **A known control.** Because we chose the steering, we know which trait each diarist "has". Studies of people
  rarely know that. Here, the text-based judge detects all three schemas; self-report detects one.

**Limits:** screening data, 4 diaries per profile and length, one judge, single-token ratings. The scales match the
schemas only approximately: social withdrawal (avoiding people) is narrower than social isolation (feeling
different, not belonging).

## Reproduce

```bash
uv run --with modal modal run experiments/01-steered-diarists/questionnaire.py   # ratings (1 GPU, ~$0.30)
uv run python docs/make_questionnaire_figures.py                                  # figures on this page
```

Items and scoring: [`diary_dialogues/simulator/questionnaire.py`](https://github.com/tegancao/diary-dialogues/blob/main/diary_dialogues/simulator/questionnaire.py).
IPIP items are in the public domain. Diaries: `experiments/01-steered-diarists/screening.py`; judge scores:
`judge_diaries.py`.
