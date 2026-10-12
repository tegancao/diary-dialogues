import math

import pytest

from diary_dialogues.simulator.questionnaire import SCALES, expected_rating, item_messages, scale_score


def test_scales_map_to_main_schemas():
    from diary_dialogues.design import MAIN_SCHEMAS
    assert {schema for schema, _ in SCALES.values()} <= set(MAIN_SCHEMAS)
    assert sum(len(items) for _, items in SCALES.values()) == 19


def test_expected_rating():
    assert expected_rating({"2": math.log(.5), "4": math.log(.5), "x": 0.0}) == pytest.approx(3.0)
    assert expected_rating({"5": math.log(.1)}) is None            # too little probability on the options


def test_scale_score_reverses_reverse_keyed_items():
    assert scale_score([5, 1], [1, -1]) == pytest.approx(5.0)
    assert scale_score([5, None], [1, 1]) is None


def test_item_messages_append_entry_and_question():
    msgs = item_messages([{"role": "user", "content": "write"}], "Dear diary", "Am happy.")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"] and '"Am happy."' in msgs[-1]["content"]
