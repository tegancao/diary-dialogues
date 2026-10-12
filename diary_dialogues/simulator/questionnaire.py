"""Self-report questionnaire for the diarist: public-domain IPIP scales matching three of the main schemas.

Items are from the CAT-PD short form (v1.1) keys on the International Personality Item Pool (https://ipip.ori.org/
newCAT-PD-SFv1.1Keys.htm), which are in the public domain. Enmeshment has no IPIP counterpart (judge only).
Each item is asked on its own, after the diary entry, with IPIP's usual 1-5 accuracy scale. The rating is the
probability-weighted mean of the answer tokens "1".."5" (no sampling), like the judge's 0-100 score.
"""
import math

SCALES = {   # scale -> (matching schema, [(item, keyed +1 or reverse -1)])
    "Perfectionism": ("Unrelenting_Standards", [
        ("Expect nothing less than perfection.", 1),
        ("Don't consider a task finished until it's perfect.", 1),
        ("Am not happy until all the details are taken care of.", 1),
        ("Set high standards for myself and others.", 1),
        ("Demand perfection in others.", 1),
        ("Strive in every way possible to be flawless.", 1)]),
    "Grandiosity": ("Entitlement", [
        ("Deserve special treatment from others.", 1),
        ("Should get special privileges.", 1),
        ("Believe that I am better than others.", 1),
        ("Don't think I should have to wait in lines like others.", 1),
        ("Feel that others are beneath me.", 1),
        ("Believe that I am always right.", 1),
        ("Treat people as inferiors.", 1)]),
    "Social_Withdrawal": ("Social_Isolation", [
        ("Enjoy going to social gatherings.", -1),
        ("Feel comfortable around people.", -1),
        ("Keep to myself even when I'm around other people.", 1),
        ("Rarely enjoy being with people.", 1),
        ("Do not feel close to people.", 1),
        ("Find it difficult to approach others.", 1)]),
}

OPTIONS = ["1", "2", "3", "4", "5"]
QUESTION = (
    "Now a question about yourself. Below is a phrase describing people's behaviors. Describe yourself as you "
    "generally are now, not as you wish to be in the future, as honestly as you can.\n\n"
    'Phrase: "{item}"\n\n'
    "How accurately does it describe you? Answer with a single number: 1 = very inaccurate, 2 = moderately "
    "inaccurate, 3 = neither accurate nor inaccurate, 4 = moderately accurate, 5 = very accurate.")


def item_messages(diary_msgs: list[dict], entry: str, item: str) -> list[dict]:
    """The diary conversation so far, the entry just written, then one questionnaire item."""
    return [*diary_msgs, {"role": "assistant", "content": entry}, {"role": "user", "content": QUESTION.format(item=item)}]


def expected_rating(option_logprobs: dict[str, float], min_mass: float = 0.25) -> float | None:
    """Probability-weighted 1-5 rating from the log-probabilities of the tokens "1".."5"; None if they're unlikely."""
    p = {o: math.exp(option_logprobs[o]) for o in OPTIONS if o in option_logprobs}
    total = sum(p.values())
    return sum(int(o) * v for o, v in p.items()) / total if total >= min_mass else None


def scale_score(ratings: list[float | None], keys: list[int]) -> float | None:
    """Mean item rating after reversing reverse-keyed items (6 - rating); None if any item is missing."""
    if any(r is None for r in ratings):
        return None
    return sum(r if k > 0 else 6 - r for r, k in zip(ratings, keys)) / len(ratings)
