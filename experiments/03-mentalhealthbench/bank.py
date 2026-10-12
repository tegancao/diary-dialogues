"""Build the criterion bank: rate every MentalHealthBench criterion's generality (Mistral), report coverage per axis.

    uv run --with modal modal run --detach experiments/03-mentalhealthbench/bank.py            # spawn the rating job
    uv run --with modal modal volume get diary-dialogues-results mhb/generality_mistral-small-3.1.json data/mhb/
    uv run python experiments/03-mentalhealthbench/bank.py --report                            # coverage per axis

Ratings are stored by (conversation id, item index) only. The report prints counts, never criterion text.
"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import modal

from diary_dialogues.modal_setup import JUDGES, VOLUMES, hf_token_secret, vllm_image
from diary_dialogues.paths import DATA

JUDGE = "mistral-small-3.1"
MHB_FILE = DATA / "mhb" / "mentalhealthbench_eval.jsonl"
OUT = f"generality_{JUDGE}.json"
app = modal.App("diary-dialogues-mhb-bank")


@app.function(image=vllm_image, gpu="H100", timeout=2 * 60 * 60, volumes=VOLUMES, secrets=[hf_token_secret()])
def rate(items: list[dict]) -> int:
    from diary_dialogues.benchmark.bank import generality_prompt
    from diary_dialogues.judge.vllm_backend import VLLMJudge
    from diary_dialogues.modal_setup import RESULTS

    j = VLLMJudge.from_config(JUDGES[JUDGE])
    res = j.score([generality_prompt(it["criterion_text"], it["points"]) for it in items])
    out = [{"id": it["id"], "k": it["k"], "axis": it["axis"], "points": it["points"], **r} for it, r in zip(items, res)]
    path = Path("/results/mhb") / OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"judge": JUDGES[JUDGE], "ratings": out}))
    RESULTS.commit()
    return len(out)


def report():
    from diary_dialogues.benchmark.bank import GENERAL_MIN
    ratings = json.loads((DATA / "mhb" / OUT).read_text())["ratings"]
    ok = [r for r in ratings if r["score"] is not None]
    bank = [r for r in ok if r["score"] >= GENERAL_MIN]
    print(f"{len(ratings)} criteria rated ({len(ratings) - len(ok)} unreadable); bank (generality >= {GENERAL_MIN}): "
          f"{len(bank)} ({len(bank) / len(ratings):.0%})\n")
    total = Counter(r["axis"] for r in ratings)
    pos = Counter(r["axis"] for r in bank if r["points"] > 0)
    neg = Counter(r["axis"] for r in bank if r["points"] < 0)
    print(f"{'axis':<52}{'all':>6}{'bank +':>8}{'bank -':>8}{'share':>7}")
    for axis, n in total.most_common():
        print(f"{axis:<52}{n:6d}{pos[axis]:8d}{neg[axis]:8d}{(pos[axis] + neg[axis]) / n:7.0%}")
    print(f"\nmedian generality: {st.median(r['score'] for r in ok):.0f}")


@app.local_entrypoint()
def main():
    rows = [json.loads(line) for line in MHB_FILE.read_text().splitlines() if line.strip()]
    items = [{"id": r["id"], "k": k, "axis": it["behavior_axis"], "points": it["points"],
              "criterion_text": it["criterion_text"]} for r in rows for k, it in enumerate(r["rubric_items"])]
    call = rate.spawn(items)
    print(f"spawned generality rating of {len(items)} criteria: {call.object_id}")


def build():
    """Condense the bank: per covered axis and sign, group near-duplicate criteria and keep each group's most typical
    real criterion (the one closest to the group's centre). Writes criterion_bank.json (references only)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering

    from diary_dialogues.benchmark.bank import CRITERIA_AXES, GENERAL_MIN, GROUPS_PER_SIGN

    rows = {r["id"]: r for r in map(json.loads, MHB_FILE.read_text().splitlines())}
    ratings = json.loads((DATA / "mhb" / OUT).read_text())["ratings"]
    general = [r for r in ratings if r["score"] is not None and r["score"] >= GENERAL_MIN and r["axis"] in CRITERIA_AXES]
    model = SentenceTransformer(EMBEDDER, revision=EMBEDDER_REVISION)
    bank = []
    for axis in CRITERIA_AXES:
        for sign, n_groups in GROUPS_PER_SIGN.items():
            pool = [r for r in general if r["axis"] == axis and (r["points"] > 0) == (sign == "positive")]
            if not pool:
                continue
            texts = [rows[r["id"]]["rubric_items"][r["k"]]["criterion_text"] for r in pool]
            emb = model.encode(texts, normalize_embeddings=True)
            k = min(n_groups, len(pool))
            labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(emb) \
                if k > 1 else np.zeros(len(pool), dtype=int)
            for g in range(k):
                idx = np.where(labels == g)[0]
                centre = emb[idx].mean(0)
                rep = idx[np.argmax(emb[idx] @ centre)]
                points = [pool[i]["points"] for i in idx]
                bank.append({"id": pool[rep]["id"], "k": pool[rep]["k"], "axis": axis, "sign": sign,
                             "group_size": len(idx), "points": pool[rep]["points"],
                             "group_median_points": float(np.median(points))})
    out = HERE / "criterion_bank.json"
    out.write_text(json.dumps({"embedder": EMBEDDER, "revision": EMBEDDER_REVISION, "general_min": GENERAL_MIN,
                               "criteria": bank}, indent=1))
    print(f"criterion bank: {len(bank)} representative criteria from {len(general)} general ones -> {out.name}")
    for axis in CRITERIA_AXES:
        xs = [b for b in bank if b["axis"] == axis]
        print(f"  {axis:<46} {sum(b['sign'] == 'positive' for b in xs):3d} positive, "
              f"{sum(b['sign'] == 'negative' for b in xs):3d} negative (covering {sum(b['group_size'] for b in xs)})")


HERE = Path(__file__).resolve().parent
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"          # Apache-2.0
EMBEDDER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

if __name__ == "__main__" and "--report" in sys.argv:
    report()
if __name__ == "__main__" and "--build" in sys.argv:
    build()
