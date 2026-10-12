"""Criterion bank: MentalHealthBench criteria general enough to grade replies in other conversations.

Most criteria refer to details of their own case; some describe general practice (such as acknowledging that the
person's situation sounds hard). A judge rates each
criterion's generality 0-100 (our own prompt below); criteria at or above GENERAL_MIN form the bank. The bank is
stored as (conversation id, item index) references, never as benchmark text.
"""

GENERAL_MIN = 70

# Axes with enough general criteria to grade diary replies criterion by criterion (others get axis-level scores).
CRITERIA_AXES = ["empathy_emotional_attunement_and_support", "context_seeking_and_assessment",
                 "actionable_guidance_and_coping_strategies", "interpretation_and_cognitive_reframing",
                 "collaboration_and_user_agency", "over_alarmism_and_under_response"]
# Near-duplicates are grouped (sentence embeddings, per axis and sign); one real criterion per group is kept.
GROUPS_PER_SIGN = {"positive": 15, "negative": 10}

GENERALITY = """Here is a rubric criterion that was written to grade an AI assistant's reply in one specific conversation. I want to know whether the same criterion could be reused, unchanged, to grade replies in many other conversations where a person shares their day, feelings or struggles (for example, entries from their personal diary) - please focus solely on this, not on whether the criterion is good.

The criterion (points in brackets; negative points mark undesirable behavior):
[CRITERION START]
[{points}] {criterion}
[CRITERION END]

Rate it on a scale of 0 to 100:
* 0 means "only makes sense in its original conversation: it refers to specific people, events, objects, places, symptoms or requests from that case."
* 50 means "partly general: the behavior is general, but it assumes a particular kind of situation (for example a medical symptom, a specific risk, or a specific request)."
* 100 means "fully general: it describes a behavior that can be checked in any reply to a person sharing their feelings or difficulties."

Respond with a number between 0 and 100. Don't say anything else, just the number."""


def generality_prompt(criterion: str, points: int) -> str:
    return GENERALITY.replace("{points}", str(points)).replace("{criterion}", criterion)
