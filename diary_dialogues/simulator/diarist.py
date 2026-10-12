"""Diary prompts and generation for the steered diarist.

The prompts are deliberately neutral: they say nothing about the schema, which comes only from steering. The system
prompt stops Qwen from writing as an AI assistant (in pilot 1, 38% of diaries did; with it, 2%).
"""
import torch

SYSTEM = "You are a person writing in your private diary. You are not an AI or an assistant."
# Open but schema-neutral: mentioning "the people in your life" lets relational schemas (e.g. abandonment) show,
# without suggesting how the diarist feels about them (pilots 1-2 used the version without it).
def first_entry(words: int = 150) -> str:
    return ("Write today's entry in your private diary. Write in the first person about what happened today, "
            f"the people in your life, and how you feel about it. About {words} words.")


FIRST_ENTRY = first_entry(150)
FIRST_ENTRY_PILOT_1_2 = ("Write today's entry in your private diary. Write in the first person about what happened "
                         "today and how you feel about it. About 150 words.")


def messages(prompt: str = FIRST_ENTRY, system: str | None = SYSTEM) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


@torch.no_grad()
def write(model, tokenizer, msgs: list[dict], n: int = 1, max_new_tokens: int = 320, temperature: float = 0.8,
          top_p: float = 0.95, seed: int = 0) -> list[str]:
    """Sample n diary texts for the same conversation."""
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                        return_dict=True)["input_ids"].to(model.device)
    batch = ids.repeat(n, 1)                 # n samples of the same conversation (rows may be steered differently)
    torch.manual_seed(seed)
    out = model.generate(batch, attention_mask=torch.ones_like(batch), max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True).strip() for o in out]


@torch.no_grad()
def write_batch(model, tokenizer, conversations: list[list[dict]], max_new_tokens: int = 700,
                temperature: float = 0.8, top_p: float = 0.95, seed: int = 0) -> list[str]:
    """One new text for each conversation, all in one batch (conversations may differ in length).

    Padding goes on the left, so every row's last position is its real last token: that is where steering adds the
    vector (see steering.py), with one vector per row.
    """
    texts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in conversations]
    tokenizer.padding_side = "left"
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    torch.manual_seed(seed)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return [tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
