"""Where the data lives, and the 18 schemas."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EMS_DIR = DATA / "ems"                                     # filled by scripts/download_data.py
VECTOR_DIR = EMS_DIR / "vectors" / "persona_vectors" / "Qwen2.5-7B-Instruct"
EVAL_PROMPT_DIR = EMS_DIR / "trait_data_eval"              # the paper's per-schema judge prompts

DIARIST_MODEL = "Qwen/Qwen2.5-7B-Instruct"                 # the vectors only fit this model
DIARIST_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"   # pinned for the dialogues (from 2026-10-10)

# The 18 early maladaptive schemas (file-name spelling). The archive also has a vector named "evil", which comes from
# the persona-vectors method the paper builds on and is not a schema.
SCHEMAS = [
    "Abandonment", "Approval_Seeking", "Defectiveness", "Dependence", "Emotional_Deprivation",
    "Emotional_Inhibition", "Enmeshment", "Entitlement", "Failure_to_Achieve", "Insufficient_Self-Control",
    "Mistrust_or_Abuse", "Pessimism", "Self-Punitiveness", "Self-Sacrifice", "Social_Isolation", "Subjugation",
    "Unrelenting_Standards", "Vulnerability_to_Harm",
]
