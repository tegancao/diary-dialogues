"""Does size, or newer training, help? Paired by conversation on the 300-case sample, both judges.

    uv run python experiments/03-mentalhealthbench/size_effect.py
"""
import json
import random
import statistics as st

from diary_dialogues.benchmark.rubric import score
from diary_dialogues.benchmark.sample import SAMPLE_SIZE, stratified_ids
from diary_dialogues.paths import DATA

PAIRS = [("gemma-3-27b", "gemma-3-12b"), ("llama-3.3-70b", "llama-3.1-8b"), ("claude-haiku-4.5", "llama-3.3-70b"),
         ("claude-haiku-4.5", "gemma-3-27b"),
         ("gemma-4-31b", "gemma-3-27b"), ("claude-haiku-4.5", "gemma-4-31b")]


def main():
    rows = [json.loads(line) for line in (DATA / "mhb" / "mentalhealthbench_eval.jsonl").read_text().splitlines()]
    keep = set(stratified_ids(rows))
    rows = [r for r in rows if r["id"] in keep]
    rng = random.Random(0)
    for judge in ("mistral-small-3.1", "gpt-oss-120b"):
        v = json.loads((DATA / "mhb" / f"grades_{judge}_sample{SAMPLE_SIZE}.json").read_text())["verdicts"]
        conv = lambda m: {r["id"]: score(r["rubric_items"], v[m][r["id"]]) for r in rows}
        print(f"=== {judge} (paired by conversation, 95% bootstrap)")
        for a, b in PAIRS:
            sa, sb = conv(a), conv(b)
            d = [100 * (sa[i] - sb[i]) for i in sa if sa[i] is not None and sb[i] is not None]
            boots = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(5000))
            print(f"   {a:<17} − {b:<15} {st.mean(d):+5.1f} [{boots[125]:+.1f}, {boots[4875]:+.1f}]  (n = {len(d)})")


if __name__ == "__main__":
    main()
