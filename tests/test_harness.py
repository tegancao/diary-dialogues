"""Conversation bookkeeping, and batched diarist writing with left padding."""
import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from diary_dialogues.harness.conversation import DATES, Conversation
from diary_dialogues.simulator.diarist import SYSTEM, write_batch


def conv(**kw):
    return Conversation(id="c0", profile="Entitlement", responder="phi-4", seed=1, words=400, **kw)


def test_first_entry_prompt_is_dated_and_has_no_reply():
    msgs = conv().diarist_messages()
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    assert msgs[1]["content"].startswith(f"Today is {DATES[0]}.") and "replied" not in msgs[1]["content"]
    assert "About 400 words." in msgs[1]["content"] and "plainly and informally" in msgs[1]["content"]


def test_background_card_opens_the_system_prompt():
    from diary_dialogues.design import BACKGROUNDS
    c = conv(background=BACKGROUNDS[0])
    system = c.diarist_messages()[0]["content"]
    assert system.startswith(BACKGROUNDS[0]["card"])
    assert system.endswith("You are writing in your private diary. You are not an AI or an assistant.")
    assert "don't sign the entry" in c.entry_prompt(0)


def test_backgrounds_are_schema_neutral():
    from diary_dialogues.design import BACKGROUNDS
    words = r"lonel|isolat|perfect|deserv|entitled|special|close to|depend|anxious|sad|happy|lost|struggl"
    import re
    assert not [b["name"] for b in BACKGROUNDS if re.search(words, b["card"], re.I)]
    assert len({b["name"] for b in BACKGROUNDS}) == len(BACKGROUNDS)


def test_diarist_sees_its_entries_and_the_replies():
    c = conv(entries=["E1", "E2"], replies=["R1", "R2"])
    msgs = c.diarist_messages()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "E1" and msgs[4]["content"] == "E2"
    assert "R1" in msgs[3]["content"] and DATES[1] in msgs[3]["content"]
    assert "R2" in msgs[5]["content"] and msgs[5]["content"].endswith("About 400 words.") and DATES[2] in msgs[5]["content"]


def test_strong_uptake_asks_for_a_reaction_to_the_reply_only_after_a_reply():
    c = conv(entries=["E1"], replies=["R1"], uptake="strong")
    assert "Begin today's entry with how the reply made you feel" not in c.entry_prompt(0)
    p = c.entry_prompt(1)
    assert p.index("R1") < p.index("Begin today's entry with how the reply made you feel") < p.index("Today is")
    assert "Begin today's entry" not in conv(entries=["E1"], replies=["R1"]).entry_prompt(1)    # default unchanged


def test_responder_sees_entries_shared_by_the_user_without_system_prompt():
    c = conv(entries=["E1", "E2"], replies=["R1"])
    share = "I've been keeping a diary. Here's what I wrote today, I just wanted to share it with someone:\n\n"
    assert c.responder_messages() == [{"role": "user", "content": share + "E1"}, {"role": "assistant", "content": "R1"},
                                      {"role": "user", "content": share + "E2"}]


def test_turn_order_is_enforced():
    with pytest.raises(ValueError):
        conv(entries=["E1"]).diarist_messages()             # reply missing
    with pytest.raises(ValueError):
        conv(entries=["E1"], replies=["R1"]).responder_messages()   # no new entry


def test_to_dict_records_dates():
    assert conv(entries=["E1", "E2"], replies=["R1"]).to_dict()["dates"] == DATES[:2]


@pytest.fixture(scope="module")
def tiny_qwen():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).eval(), tok


def test_batched_writing_matches_writing_alone(tiny_qwen):
    # near-greedy sampling: a conversation's text must not depend on the (padded) batch it is written in
    model, tok = tiny_qwen
    short = conv().diarist_messages()
    long = conv(entries=["A much longer first entry " * 5], replies=["A reply."]).diarist_messages()
    kw = dict(max_new_tokens=6, temperature=1e-4, top_p=1.0)
    together = write_batch(model, tok, [short, long], **kw)
    assert together == [write_batch(model, tok, [short], **kw)[0], write_batch(model, tok, [long], **kw)[0]]
