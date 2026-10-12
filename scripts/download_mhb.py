"""Download MentalHealthBench (MIT) at a pinned commit and check it against the paper's statistics.

    uv run python scripts/download_mhb.py

Source: github.com/MercuriusDream/MentalHealthBench (a mirror of OpenAI's release). The data carries a canary
string, so it stays in data/ (not committed) and is never published. Writes data/mhb/mentalhealthbench_eval.jsonl.
"""
import hashlib
import json
import urllib.request

from diary_dialogues.paths import DATA

COMMIT = "77a658c730f88ef622d2aa617cfcd08eb11705b4"
URL = f"https://raw.githubusercontent.com/MercuriusDream/MentalHealthBench/{COMMIT}/mentalhealthbench_eval.jsonl"
MHB_DIR = DATA / "mhb"
MHB_FILE = MHB_DIR / "mentalhealthbench_eval.jsonl"


def main():
    MHB_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as r:
        raw = r.read()
    MHB_FILE.write_bytes(raw)
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    criteria = sum(len(r["rubric_items"]) for r in rows)
    print(f"{len(rows)} conversations, {criteria} rubric criteria; sha256 {hashlib.sha256(raw).hexdigest()}")
    if (len(rows), criteria) != (1215, 5262):
        raise SystemExit("does not match the paper's statistics (1,215 conversations, 5,262 criteria)")


if __name__ == "__main__":
    main()
