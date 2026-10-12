"""A fixed MentalHealthBench sample, the same for every model: proportional to the benchmark's urgency levels."""
import random

SAMPLE_SIZE, SAMPLE_SEED = 300, 0


def stratified_ids(rows: list[dict], n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[str]:
    """n conversation ids, drawn per acuity level in proportion to the benchmark (largest remainders), sorted."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for r in rows:
        groups.setdefault(r["acuity"], []).append(r["id"])
    exact = {a: n * len(ids) / len(rows) for a, ids in groups.items()}
    take = {a: int(x) for a, x in exact.items()}
    for a in sorted(exact, key=lambda a: exact[a] - take[a], reverse=True)[:n - sum(take.values())]:
        take[a] += 1
    picked = [i for a in sorted(groups) for i in rng.sample(sorted(groups[a]), take[a])]
    return sorted(picked)
