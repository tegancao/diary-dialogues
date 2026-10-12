# Diary Dialogues

**How do AI chatbots answer people who see themselves and others differently?**

Eight simulated people keep a diary. Inside the model that writes it, each is nudged toward a pattern from schema
therapy: feeling alone, feeling entitled, never good enough. They share each entry with an AI chatbot, and two open
judge models read what the chatbots say back, entry after entry. No real people take part.

**Main result so far:** the open chatbots we tested (8B to 70B, including the newest Gemma) fall well short of the
published standard on OpenAI's MentalHealthBench, and they show the same gaps answering the simulated diaries.

![Rosa's first diary entry three ways: with no nudge, nudged toward social isolation, and nudged toward
entitlement](figures/o1-rosa-three-ways.png#gh-light-mode-only)
![Rosa's first diary entry three ways: with no nudge, nudged toward social isolation, and nudged toward
entitlement](figures/o1-rosa-three-ways-dark.png#gh-dark-mode-only)

## Read more

- [Findings](findings.md)
    - [In brief](findings.md#in-brief)
    - [The diarists](findings.md#the-diarists)
    - [The chatbots on OpenAI's benchmark](findings.md#the-chatbots-on-openais-benchmark)
    - [The chatbots in the diary conversations](findings.md#the-chatbots-in-the-diary-conversations)
    - [Clinician criteria don't transfer](findings.md#clinician-criteria-dont-transfer)
    - [Two steered friends (a thought experiment)](findings.md#two-steered-friends-a-thought-experiment)
    - [Limitations](findings.md#limitations) · [Compute](findings.md#compute)
- [Future work](findings.md#future-work): human ratings, post-training, interpretability, measurement
- [How we evaluate](evaluation.md): how we test, the judges, the benchmark, the behavior rubric
- [Project overview](project-overview.md): the diarists, how a conversation works, the set-up
- [How steering works](steering-explained.md): the nudge, from scratch
- [Self-report vs. text](self-report-vs-text.md): do steered diarists describe themselves the way they write?
- [Code on GitHub](https://github.com/tegancao/diary-dialogues)

## What to take away

- **Building on open models?** Open chatbots from 8B to 70B, including the newest Gemma, score well below frontier
  models on OpenAI's clinician-written benchmark, and neither size nor newer training closes the gap. In the diaries
  the failures are concrete and checkable: some never ask a question, some treat distress as a piece of writing.
  Test for asking before advising rather than assuming a bigger model is safer.
- **Evaluating chatbots?** Clinician criteria written for one conversation grade replies in another almost at random
  (rank correlation 0.04), so multi-turn evaluation needs criteria that hold across conversations, checked by people.
  Model judges agree on concrete behaviors but not on judgment calls: their agreement beyond chance (Cohen's κ, where
  1 is perfect and 0 is chance) is 0.75 on asking a question and 0.16 on endorsing a view. One judge called a bare "Thank you for sharing." endorsing in 39% of replies. Running published models
  through an open grader puts its scores on the published scale.
- **Simulating users?** Activation steering gives a recognizable pattern in the writing, and steering the other way
  gives a healthier person, not an opposite problem. But a constantly steered person can't improve whatever the reply
  says, barely reacts to replies unless told to, and two simulated people mostly echo each other.
- **Reproducing it?** Models, prompts and settings are pinned, the judges are open models, and the whole study cost
  about $110 of compute.

Limits: no human validation yet, one diarist model, eight characters, four entries per conversation.

## References

- Zhang, R. et al. (2026). Mechanistic control of large language models as simulated participants via linear
  representation. *npj Artificial Intelligence.*
  [doi:10.1038/s44387-026-00160-9](https://doi.org/10.1038/s44387-026-00160-9) · schema vectors and steering method
  ([code](https://github.com/zrkqjz/ems-activation-steering), [data](https://doi.org/10.5281/zenodo.21715792))
- OpenAI (2026). [MentalHealthBench](https://openai.com/index/introducing-mentalhealthbench/): the benchmark
  conversations, clinician criteria and grader prompt
- Belli et al. (2026). VERA-MH: the model for the behavior rubric's yes/no question flows
- Young, J. E., Klosko, J. S., & Weishaar, M. E. (2003). *Schema Therapy: A Practitioner's Guide.* Guilford Press:
  the patterns (early maladaptive schemas)
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*,
  20(1), 37–46: κ, the judges' agreement beyond chance
- [International Personality Item Pool](https://ipip.ori.org/): the public-domain self-report scales

*The diaries are simulations of how a pattern reads in text, not real people; some contain simulated hopelessness.
Nothing here is a clinical assessment of any model or anyone.*
