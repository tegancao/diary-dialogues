"""The VERA-style flow: gates, branches and severity order."""
from diary_dialogues.benchmark.flow_rubric import (BEST, DIMENSIONS, NOT_RELEVANT, POOR, SUBOPTIMAL, prompt,
                                                   questions, rate)


def answers(**overrides):
    """All questions answered "no", except overrides given as {question: answer}."""
    base = {q: False for _, q in questions()}
    base.update(overrides)
    return base


def test_all_no_is_best_practice_except_gated():
    a = answers()
    assert rate("engages_with_person", a) == BEST
    assert rate("calibrated_escalation", a) == BEST                 # no risk, no alarm
    assert rate("explores_before_advising", a) == NOT_RELEVANT       # gate answered no
    assert rate("responds_to_pattern", a) == NOT_RELEVANT


def test_first_failure_in_severity_order_decides():
    d = DIMENSIONS["engages_with_person"]["flow"]
    a = answers(**{d[0][0]: True, d[1][0]: True})
    assert rate("engages_with_person", a) == POOR                   # the more severe one wins
    assert rate("engages_with_person", answers(**{d[2][0]: True})) == SUBOPTIMAL


def test_escalation_branches_on_risk():
    spec = DIMENSIONS["calibrated_escalation"]
    alarm_without_risk = answers(**{spec["if_no_risk"][0][0]: True})
    assert rate("calibrated_escalation", alarm_without_risk) == SUBOPTIMAL
    no_help_with_risk = answers(**{spec["branch"]: True, spec["if_risk"][1][0]: True})
    assert rate("calibrated_escalation", no_help_with_risk) == POOR
    assert rate("calibrated_escalation", answers(**{spec["branch"]: True})) == BEST


def test_pattern_endorsed_is_poor():
    spec = DIMENSIONS["responds_to_pattern"]
    assert rate("responds_to_pattern", answers(**{spec["gate"]: True, spec["flow"][0][0]: True})) == POOR
    assert rate("responds_to_pattern", answers(**{spec["gate"]: True, spec["flow"][1][0]: True})) == SUBOPTIMAL


def test_questions_about_the_person_never_see_the_reply():
    from diary_dialogues.benchmark.flow_rubric import about_person
    gate = DIMENSIONS["responds_to_pattern"]["gate"]
    branch = DIMENSIONS["calibrated_escalation"]["branch"]
    reply_q = DIMENSIONS["engages_with_person"]["flow"][0][0]
    assert about_person(gate) and about_person(branch) and not about_person(reply_q)
    full, before = "user: hi\n\nassistant: CALL A HOTLINE", "user: hi"
    assert "HOTLINE" not in prompt(full, branch, without_reply=before)
    assert "HOTLINE" in prompt(full, reply_q, without_reply=before)


def test_missing_answer_gives_none_and_prompt_is_filled():
    a = answers()
    a.pop(DIMENSIONS["engages_with_person"]["flow"][0][0])
    assert rate("engages_with_person", a) is None
    p = prompt("user: hi\n\nassistant: hello", "Is it kind?")
    assert "user: hi\n\nassistant: hello" in p and "# Question\nIs it kind?" in p and "<<" not in p
    assert len(questions()) == len({q for _, q in questions()}) == 15
