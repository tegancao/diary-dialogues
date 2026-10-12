import math

import pytest

from diary_dialogues.judge.prompts import coherence_prompt, schema_prompt
from diary_dialogues.judge.scoring import weighted_score

lp = math.log


def test_weighted_average_of_number_tokens():
    assert weighted_score({"70": lp(0.6), "80": lp(0.4)}) == pytest.approx(74.0)


def test_ignores_non_numbers_and_out_of_range():
    assert weighted_score({"50": lp(0.5), "REFUSAL": lp(0.3), "150": lp(0.2)}) == pytest.approx(50.0)


def test_refusal_when_little_mass_on_numbers():
    assert weighted_score({"REFUSAL": lp(0.8), "10": lp(0.2)}) is None


def test_prompts_contain_the_diary():
    p = schema_prompt("Abandonment", "Write a diary entry.", "Nobody ever stays.")
    assert "**Abandonment**" in p and "Nobody ever stays." in p and "{answer}" not in p
    assert "Nobody ever stays." in coherence_prompt("Write a diary entry.", "Nobody ever stays.")
    with pytest.raises(ValueError):
        schema_prompt("evil", "q", "a")


from diary_dialogues.judge.scoring import number_distribution, score_from_distribution


def fake_judge(table):
    """A fake judge: table maps the digits written so far to {next token text: probability}; "<end>" ends."""
    def next_token_logprobs(requests):
        out = []
        for prompt, ids in requests:
            written = "".join(ids)
            out.append([(tok, tok, math.log(p), tok == "<end>") for tok, p in table[written].items()])
        return out
    return next_token_logprobs


def test_digit_tree_for_split_numbers():
    # P(70) = .6*.5*1, P(74) = .6*.5*1, P(8)= .4*1  -> mean = (.3*70 + .3*74 + .4*8) / 1.0
    table = {"": {"7": 0.6, "8": 0.4}, "7": {"0": 0.5, "4": 0.5}, "70": {"<end>": 1.0}, "74": {"<end>": 1.0},
             "8": {"<end>": 1.0}}
    [dist] = number_distribution(fake_judge(table), ["prompt"])
    assert dist == pytest.approx({70: 0.3, 74: 0.3, 8: 0.4})
    assert score_from_distribution(dist) == pytest.approx(0.3 * 70 + 0.3 * 74 + 0.4 * 8)


def test_digit_tree_matches_first_token_method_for_single_token_numbers():
    table = {"": {"70": 0.6, "80": 0.4}, "70": {"<end>": 1.0}, "80": {"<end>": 1.0}}
    [dist] = number_distribution(fake_judge(table), ["prompt"])
    assert score_from_distribution(dist) == pytest.approx(weighted_score({"70": lp(0.6), "80": lp(0.4)}))


def test_digit_tree_rejects_over_100_and_leading_zeros_and_detects_refusal():
    table = {"": {"1": 0.1, "0": 0.1, "REFUSAL": 0.8}, "1": {"0": 1.0}, "10": {"0": 0.5, "5": 0.5},
             "100": {"<end>": 1.0}, "0": {"<end>": 0.5, "5": 0.5}}
    [dist] = number_distribution(fake_judge(table), ["prompt"])
    assert dist == pytest.approx({100: 0.05, 0: 0.05})                 # 105 and "05" rejected
    assert score_from_distribution(dist) is None                       # most probability on REFUSAL


def test_chat_prompt_ids_with_a_hugging_face_tokenizer():
    from transformers import AutoTokenizer
    from diary_dialogues.judge.vllm_backend import chat_prompt_ids
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")   # small download; returns a BatchEncoding
    ids = chat_prompt_ids(tok, "Rate this diary from 0 to 100.")
    assert all(isinstance(i, int) for i in ids)
    assert "Rate this diary" in tok.decode(ids) and tok.decode(ids).rstrip().endswith("assistant")


def test_unicode_digit_lookalikes_are_not_numbers():
    # Llama proposed '₀' (subscript zero): str.isdigit() accepts it, int() does not
    table = {"": {"7": 0.9, "₀": 0.05, "²": 0.05}, "7": {"<end>": 1.0}}
    [dist] = number_distribution(fake_judge(table), ["prompt"])
    assert dist == pytest.approx({7: 0.9})
    assert weighted_score({"7": lp(0.9), "₀": lp(0.05), "²": lp(0.05)}) == pytest.approx(7.0)


def test_answer_prefix_cuts_where_the_final_answer_starts():
    from diary_dialogues.judge.vllm_backend import answer_prefix
    marker = [5, 9, 8]                                  # e.g. <|channel|> final <|message|>
    out = [5, 1, 8, 3, 3, 7, 5, 9, 8, 42, 2]            # reasoning (incl. a partial match), marker, answer "42", end
    assert answer_prefix(out, marker) == out[:9]
    assert answer_prefix([5, 1, 8, 3], marker) is None  # reasoning cut off before any answer


def test_chat_prompt_ids_passes_template_options():
    calls = {}

    class FakeTok:
        def apply_chat_template(self, messages, add_generation_prompt, **kw):
            calls.update(kw); return [1, 2, 3]

    from diary_dialogues.judge.vllm_backend import chat_prompt_ids
    assert chat_prompt_ids(FakeTok(), "x", reasoning_effort="medium") == [1, 2, 3]
    assert calls == {"reasoning_effort": "medium"}
