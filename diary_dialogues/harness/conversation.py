"""One diary conversation: the diarist writes dated entries; after each one, a response model replies.

The diarist sees its own entries and each reply, then writes the next entry three days later. The response model
sees each entry as a user sharing their diary ("Here's my diary entry from today: ...") and its own replies, with no
system prompt (its default behavior).
"""
from dataclasses import asdict, dataclass, field

from diary_dialogues.simulator.diarist import SYSTEM

# Fixed 3-day gaps, the same for every conversation (weekdays match 2025; the year is never shown).
DATES = ["Monday, 3 March", "Thursday, 6 March", "Sunday, 9 March", "Wednesday, 12 March", "Saturday, 15 March",
         "Tuesday, 18 March", "Friday, 21 March", "Monday, 24 March", "Thursday, 27 March", "Sunday, 30 March"]

# The plain-voice sentence keeps entries from reading like creative writing (dry runs 1-2: ornate prose and date
# headings led response models to edit, analyze or critique the entry as a text). It says nothing about the schema.
ENTRY = ("Today is {date}. Write today's entry in your private diary. Write in the first person about what happened "
         "today, the people in your life, and how you feel about it. Write the way people really write in a private "
         "diary: plainly and informally, for no one but yourself. Don't add a title or date heading, and don't sign "
         "the entry. About {words} words.")
# How each entry reaches the response model: a person sharing their diary, with a reason to share but no request for
# a particular kind of reply. Dry run 1 (raw entry): Gemma-3 critiqued 32/32 steered entries as writing. Dry run 2
# ("Here's my diary entry from today:"): Gemma critiqued every entry, Phi-4 rewrote or analyzed several.
SHARE = "I've been keeping a diary. Here's what I wrote today, I just wanted to share it with someone:\n\n{entry}"
AFTER_REPLY = ("You shared your last diary entry with an AI support app, and it replied:\n\n"
               '"""\n{reply}\n"""\n\n'
               "Three days have passed. ")
# Uptake nudge (2026-10-11, user decision): with the plain prompt, later entries reused only 6.2% of a reply's
# distinctive words (4.0% for a random other reply), so the replies barely reached the diarist. "strong" asks the
# diarist to begin with its reaction to the reply. Runs before this keep uptake="none".
UPTAKE = {"none": "",
          "strong": "Begin today's entry with how the reply made you feel and what you thought about it, then go on "
                    "with your day. "}


@dataclass
class Conversation:
    id: str
    profile: str               # "control" or a schema
    responder: str
    seed: int
    words: int
    entries: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    background: dict | None = None     # a design.BACKGROUNDS card; None = no background (dry runs 1-3)
    uptake: str = "none"               # UPTAKE key: how strongly the diarist is asked to react to each reply

    def system_prompt(self) -> str:
        """The diarist's system prompt: its background (if any), then the diary framing."""
        return SYSTEM if not self.background else f"{self.background['card']} {SYSTEM.replace('a person ', '')}"

    def entry_prompt(self, turn: int) -> str:
        """The instruction for entry number `turn` (0-based), including the reply to the previous entry."""
        prompt = ENTRY.format(date=DATES[turn], words=self.words)
        if turn == 0:
            return prompt
        return AFTER_REPLY.format(reply=self.replies[turn - 1]) + UPTAKE[self.uptake] + prompt

    def diarist_messages(self) -> list[dict]:
        """What the diarist sees before writing its next entry."""
        if len(self.replies) != len(self.entries):
            raise ValueError("the last entry has no reply yet")
        msgs = [{"role": "system", "content": self.system_prompt()}]
        for t, entry in enumerate(self.entries):
            msgs += [{"role": "user", "content": self.entry_prompt(t)}, {"role": "assistant", "content": entry}]
        return msgs + [{"role": "user", "content": self.entry_prompt(len(self.entries))}]

    def responder_messages(self) -> list[dict]:
        """What the response model sees before replying to the latest entry."""
        if len(self.entries) != len(self.replies) + 1:
            raise ValueError("no new entry to reply to")
        msgs = []
        for t, entry in enumerate(self.entries):
            msgs.append({"role": "user", "content": SHARE.format(entry=entry)})
            if t < len(self.replies):
                msgs.append({"role": "assistant", "content": self.replies[t]})
        return msgs

    def to_dict(self) -> dict:
        return {**asdict(self), "dates": DATES[:len(self.entries)]}
