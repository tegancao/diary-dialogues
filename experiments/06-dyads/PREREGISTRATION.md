# Dyads v2: pre-registration

Written 2026-10-11, before any v2 letter was generated, and committed to git before the run (the commit time is the
record). The pilot (`run.py`, 4 pairs per condition) motivated these hypotheses; its data are not reused.

## Question

When two simulated people with steered patterns write letters to each other, does a pattern pass from one to the
other, and does the friend change the focal person's pattern?

## Design

- **Writers:** Qwen2.5-7B-Instruct with the project's role cards, writing ~250-word letters every three days, three
  letters each (six per pair). Each writer sees their own earlier letters and every letter from the friend.
- **Focal patterns:** entitlement and social isolation, steered at the design strength (layer 15, +2.0).
- **Friend conditions, per focal pattern:**
  - `same`: friend steered toward the same pattern (+2.0);
  - `unsteered`: friend not steered;
  - `away`: friend steered away from the pattern (−2.0);
  - `neutral`: the friend's letters are a fixed, content-free text (no model), so any change in the focal person
    comes from their own steered letters accumulating.
- **Shared control `control`:** two unsteered friends. Scores of both patterns are measured in it.
- **Counterbalancing:** 8 pairings of role cards (cards k and k+4, k = 0..3, each in both seats), each in both writing
  orders (focal first / friend first), each twice (independent samples): 32 pairs per condition, 9 conditions,
  288 pairs.
- **Judges:** Mistral-Small-3.1 (primary) and gpt-oss-120b (replication), with the schema and coherence prompts of
  Zhang et al. (2026), probability-weighted 0–100 scores; plus an identity check (below).

## Measures

- **Outcome:** a writer's score for the focal pattern, averaged over that writer's 2nd and 3rd letters (the 1st letter
  is the noisiest and, for whoever writes first, is written before reading the friend).
- **Identity check:** for each letter, P(yes) that it "describes the writer's own life consistently with their
  background" (role card shown). A letter with P(yes) < 0.5 is an identity slip.

## Hypotheses

For each focal pattern separately (entitlement, social isolation):

- **H1, spread:** the unsteered friend scores higher on the focal pattern than an unsteered friend in `control`
  (friend outcome, `unsteered` − `control`).
- **H2, escalation:** the focal person scores higher with a same-steered friend than with a neutral-letter friend
  (focal outcome, `same` − `neutral`).
- **H3, a healthy friend helps:** the focal person scores lower with an away-steered friend than with a neutral-letter
  friend (focal outcome, `away` − `neutral`, expected negative). The pilot pointed against this; we test it anyway.

Exploratory (no claims): spread with an `away` friend; whether entitlement spreads more than isolation; the focal
person with an `unsteered` friend vs `neutral`; identity slips by condition; trends over letters.

## Analysis

- **Unit:** the pair. Differences between conditions are differences of condition means, with 95% bootstrap intervals
  (10,000 resamples of pairs within each condition).
- **Support:** a hypothesis counts as supported when the 95% interval excludes zero in the predicted direction for
  **both** judges. With 6 confirmatory tests, we also report Holm-adjusted bootstrap p-values (primary judge).
- **Identity slips:** the primary analysis uses all letters. A robustness analysis drops identity-slip letters (primary
  judge's check); if a conclusion changes, we report both.
- **Exclusions:** letters the judge cannot score (number probability < 0.25) are dropped and counted. No other
  exclusions.
- **Power:** single letters vary by about ±15 points (pilot); averaging two letters and 32 pairs per condition gives
  about 80% power for differences of roughly 8–10 points.

## Deviations

Any change after this file is committed is listed here, with the reason.
- 2026-10-11, during the run: the 6th (last) turn ran out of GPU memory in batches of 48 and was generated in batches
  of 16 after resuming from the saved first five turns. This changes only the random draws, not the design.
- After the run (observation, not a change): the identity check flags many letters that are not slips but say little
  about the writer (abstract despair, or only about the friend). The pre-registered robustness analysis was run as
  planned; its interpretation is in docs/findings.md.
