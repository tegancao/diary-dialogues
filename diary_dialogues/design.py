"""Locked-in design of the main experiment (decided 2026-10-10 from the screening)."""

# Steered diarist profiles: strong, specific and coherent in ~400-word diaries, one per schema domain where possible
# (other-directedness has no steerable schema). "control" = unsteered diarist.
# Enmeshment was dropped (2026-10-10): weak and unspecific in dry run 3 (~50) and with background cards (42).
MAIN_SCHEMAS = ["Social_Isolation", "Entitlement", "Unrelenting_Standards"]
PROFILES = ["control", *MAIN_SCHEMAS]

LAYER = 15             # steer the output of block 15 of 28
COEF = 2.0             # default strength; layer 15 x 2.0 kept coherence at 81-94 in the screening
# Per-schema strengths, where they differ from COEF. Strength test with background cards (bg_test vs bg_c25):
# unrelenting standards 71 -> 81 at the same coherence (87); social isolation gained nothing at 2.5 (79 -> 78) and
# lost realism; entitlement at 2.5 broke coherence in 3/8 diaries (56-58) and realism fell to 46.
COEFS: dict[str, float] = {"Unrelenting_Standards": 2.5}


def coef(profile: str) -> float:
    """Steering strength for a profile (0 for the unsteered control)."""
    return 0.0 if profile == "control" else COEFS.get(profile, COEF)
ENTRY_WORDS = 400      # target length of each diary entry
ENTRIES = 4            # entries per conversation: dry run 3 showed the diarist copying itself from entry ~5 on

# Fixed, schema-neutral life situations (a life, no traits or problems). Each is crossed with every profile, so
# diarist k has the same background in every condition. Without them, diarists defaulted to one person (26/30 office
# workers; a colleague "Sarah" in 9/30). The tags allow results to be broken down by group.
BACKGROUNDS = [
    {"name": "Daniel", "card": "You are Daniel, 19, a first-year college student. You live in a dorm with a roommate "
     "in Columbus, Ohio.", "age_group": "young adult", "gender": "man", "household": "shared", "work": "student"},
    {"name": "Aisha", "card": "You are Aisha, 24, a barista and part-time art student. You live with two roommates "
     "in Philadelphia.", "age_group": "young adult", "gender": "woman", "household": "shared", "work": "service"},
    {"name": "Sam", "card": "You are Sam, 29, a software tester who works from home. You use they/them pronouns and "
     "live with your partner in Minneapolis.", "age_group": "young adult", "gender": "nonbinary",
     "household": "partner", "work": "office"},
    {"name": "Marcus", "card": "You are Marcus, 34, a warehouse shift supervisor. You are married and have a "
     "two-year-old daughter. You live in Atlanta.", "age_group": "adult", "gender": "man",
     "household": "partner and child", "work": "manual"},
    {"name": "Mei", "card": "You are Mei, 38, an accountant. You are single and live alone in Seattle.",
     "age_group": "adult", "gender": "woman", "household": "alone", "work": "office"},
    {"name": "Rosa", "card": "You are Rosa, 47, a hospital nurse. You are divorced and live with your teenage son "
     "in San Antonio.", "age_group": "midlife", "gender": "woman", "household": "single parent", "work": "care"},
    {"name": "Tom", "card": "You are Tom, 55, an electrician. You are married; your two grown children have moved "
     "out. You live in a small town in Iowa.", "age_group": "midlife", "gender": "man",
     "household": "partner, children moved out", "work": "trade"},
    {"name": "Walter", "card": "You are Walter, 71, a retired postal worker. You are widowed and live alone in "
     "Pittsburgh; your daughter lives nearby.", "age_group": "older", "gender": "man", "household": "alone",
     "work": "retired"},
]

# Third-person summaries of the cards, for readers (docs, the human-rating page); never shown to the models.
BACKGROUND_SUMMARIES = {
    "Daniel": "Daniel, 19: first-year college student, shares a dorm room in Columbus, Ohio",
    "Aisha": "Aisha, 24: barista and part-time art student, lives with two roommates in Philadelphia",
    "Sam": "Sam, 29 (they/them): software tester working from home, lives with their partner in Minneapolis",
    "Marcus": "Marcus, 34: warehouse shift supervisor, married, a two-year-old daughter, Atlanta",
    "Mei": "Mei, 38: accountant, single, lives alone in Seattle",
    "Rosa": "Rosa, 47: hospital nurse, divorced, lives with her teenage son in San Antonio",
    "Tom": "Tom, 55: electrician, married, two grown children who have moved out, small town in Iowa",
    "Walter": "Walter, 71: retired postal worker, widowed, lives alone in Pittsburgh, his daughter nearby",
}
