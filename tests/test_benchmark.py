"""MentalHealthBench grading: prompt, verdict parsing, task-clipped scores."""
import pytest

from diary_dialogues.benchmark.rubric import criteria_met, grader_prompt, score


def test_mentalhealthbench_grader_prompt_has_its_own_sections():
    p = grader_prompt([{"role": "user", "content": "I can't sleep."}], "Have you tried a routine?", "Asks about sleep")
    assert "<conversation_prefix>\nuser: I can't sleep.\n</conversation_prefix>" in p
    assert "<completion_to_grade>\nHave you tried a routine?\n</completion_to_grade>" in p
    assert "<rubric_criterion>\nAsks about sleep\n</rubric_criterion>" in p and "{TASK_DEFINITION}" not in p


@pytest.mark.parametrize("text,expected", [
    ('```json\n{"explanation": "It asks.", "criteria_met": true}\n```', True),
    ('{"explanation": "no", "criteria_met": false}', False),
    ('Sure! {"explanation": "a "quoted" word", "criteria_met": true} done', True),   # invalid JSON: regex fallback
    ('{"explanation": "It asks.", "satisfaction": 1}', True),
    ('```json\n{"explanation": "No.", "satisfaction": 0}\n```', False),
    ('{"explanation": "a "quoted" x", "satisfaction": 1}', True),
    ("I think it is met.", None),
    (None, None),
])
def test_criteria_met(text, expected):
    assert criteria_met(text) is expected


def test_task_clipped_score_and_axes():
    items = [{"points": 6, "behavior_axis": "empathy"}, {"points": 4, "behavior_axis": "context"},
             {"points": -8, "behavior_axis": "empathy"}]
    assert score(items, [True, True, False]) == pytest.approx(1.0)
    assert score(items, [True, False, True]) == 0.0                  # 6 - 8 < 0: clipped
    assert score(items, [False, True, None]) == pytest.approx(0.4)   # unreadable verdict counts as not met
    assert score(items, [True, False, True], axis="context") == 0.0
    assert score([{"points": -5, "behavior_axis": "x"}], [False]) is None   # no positive points: undefined


def test_stratified_sample_is_proportional_and_reproducible():
    from collections import Counter
    from diary_dialogues.benchmark.sample import stratified_ids
    acuity = ["non_acute"] * 650 + ["emergent"] * 344 + ["high_acuity"] * 221
    rows = [{"id": f"c{i}", "acuity": a} for i, a in enumerate(acuity)]
    ids = stratified_ids(rows)
    assert len(ids) == len(set(ids)) == 300 and ids == stratified_ids(rows)
    of = {r["id"]: r["acuity"] for r in rows}
    assert Counter(of[i] for i in ids) == {"non_acute": 160, "emergent": 85, "high_acuity": 55}


def test_anthropic_messages_moves_system_text_and_merges_turns():
    import importlib.util
    from diary_dialogues.paths import ROOT
    spec = importlib.util.spec_from_file_location("mhb_run", ROOT / "experiments/03-mentalhealthbench/run.py")
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    system, turns = run.anthropic_messages([{"role": "system", "content": "Context."}, {"role": "user", "content": "A"},
                                            {"role": "user", "content": "B"}, {"role": "assistant", "content": "C"},
                                            {"role": "user", "content": "D"}])
    assert system == "Context."
    assert turns == [{"role": "user", "content": "A\n\nB"}, {"role": "assistant", "content": "C"},
                     {"role": "user", "content": "D"}]


def test_reply_grading_prompts():
    from diary_dialogues.benchmark.reply_grading import AXIS_DEFINITIONS, axis_prompt, relevance_prompt
    msgs = [{"role": "user", "content": "Here's my diary entry."}]
    r = relevance_prompt(msgs, "Thanks.", "Advises a crisis line", 10)
    assert "user: Here's my diary entry.\n\nassistant: Thanks." in r and "[10] Advises a crisis line" in r
    axis = next(iter(AXIS_DEFINITIONS))
    a = axis_prompt(msgs, "Thanks.", axis, [(5, "Good thing"), (-4, "Bad thing")])
    assert "- [+5] Good thing\n- [-4] Bad thing" in a and AXIS_DEFINITIONS[axis] in a and "<<" not in a
