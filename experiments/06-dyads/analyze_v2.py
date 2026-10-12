"""Dyads v2 analysis, as pre-registered (PREREGISTRATION.md).

    uv run python experiments/06-dyads/analyze_v2.py

Outcome: a writer's focal-pattern score averaged over their 2nd and 3rd letters. Unit: the pair. Contrasts: differences
of condition means with 95% bootstrap intervals (10,000 resamples of pairs within each condition), for both judges.
Confirmatory (per focal pattern): H1 friend unsteered - control; H2 focal same - neutral; H3 focal away - neutral.
Holm-adjusted bootstrap p-values for the 6 confirmatory tests (primary judge). Robustness: identity slips dropped.
"""
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGES = ["mistral-small-3.1", "gpt-oss-120b"]
PATTERNS = ["Entitlement", "Social_Isolation"]
N_BOOT = 10_000


def outcomes(items, drop_slips=None):
    """{(focal, friend): {"A": [per-pair outcome], "B": [...]}} for each pattern scored (control: both patterns).
    drop_slips: {(dyad, side, round)} letters to drop. Returns also the count of unscorable letters."""
    letters, missing = {}, 0
    for it in items:
        if drop_slips and (it["dyad"], it["side"], it["round"]) in drop_slips:
            continue
        for p in PATTERNS if it["focal"] is None else [it["focal"]]:
            s = it["scores"][p].get("score")
            if s is None:
                missing += 1; continue
            letters.setdefault((it["dyad"], it["focal"] or p, it["friend"], it["side"]), {})[it["round"]] = s
    out = {}
    for (dyad, p, friend, side), byr in letters.items():
        later = [byr[r] for r in (1, 2) if r in byr]
        if later:
            out.setdefault((p, friend), {}).setdefault(side, []).append(st.mean(later))
    return out, missing


def contrast(a, b, seed=0):
    rng = random.Random(seed)
    diffs = sorted(st.mean(rng.choices(a, k=len(a))) - st.mean(rng.choices(b, k=len(b))) for _ in range(N_BOOT))
    est = st.mean(a) - st.mean(b)
    p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / N_BOOT
    return est, diffs[int(.025 * N_BOOT)], diffs[int(.975 * N_BOOT)], min(p, 1.0)


def tests(out):
    rows = []
    for p in PATTERNS:
        rows.append((f"H1 spread    {p}", "friend: unsteered - control", contrast(out[(p, "unsteered")]["B"], out[(p, "control")]["B"]), +1))
        rows.append((f"H2 escalate  {p}", "focal: same - neutral", contrast(out[(p, "same")]["A"], out[(p, "neutral")]["A"]), +1))
        rows.append((f"H3 helps     {p}", "focal: away - neutral", contrast(out[(p, "away")]["A"], out[(p, "neutral")]["A"]), -1))
    return rows


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    data = {j: json.loads((HERE / f"dyads_v2_items_scores_{j}.json").read_text())["items"]
            for j in JUDGES if (HERE / f"dyads_v2_items_scores_{j}.json").exists()}
    results = {}
    for j, items in data.items():
        out, missing = outcomes(items)
        results[j] = tests(out)
        print(f"\n=== {j}: {len(items)} letters, {missing} unscorable")
        for p in PATTERNS:
            print(f"   {p}: means of letters 2-3, focal (A) | friend (B)")
            for fr in ("same", "unsteered", "away", "neutral", "control"):
                g = out.get((p, fr), {})
                a = f"{st.mean(g['A']):5.1f}" if g.get("A") else "    -"
                b = f"{st.mean(g['B']):5.1f}" if g.get("B") else "    -"
                print(f"      {fr:<10} {a} | {b}   (n = {len(g.get('A', []))})")
    primary = JUDGES[0]
    if primary in results:
        adj = holm([r[2][3] for r in results[primary]])
        print("\n=== confirmatory tests (supported = 95% interval excludes 0 in the predicted direction for both judges)")
        for k, (name, what, (est, lo, hi, p), sign) in enumerate(results[primary]):
            line = f"   {name}  {what:<28} {primary.split('-')[0]} {est:+5.1f} [{lo:+.1f}, {hi:+.1f}] Holm p {adj[k]:.3f}"
            ok = [(sign > 0 and lo > 0) or (sign < 0 and hi < 0)]
            if len(results) > 1:
                e2, l2, h2, _ = results[JUDGES[1]][k][2]
                line += f" | gpt-oss {e2:+5.1f} [{l2:+.1f}, {h2:+.1f}]"
                ok.append((sign > 0 and l2 > 0) or (sign < 0 and h2 < 0))
            print(line + ("   SUPPORTED" if all(ok) and len(ok) == 2 else "   not supported" if len(ok) == 2 else ""))
        items = data[primary]
        slips = {(it["dyad"], it["side"], it["round"]) for it in items if it["identity"] is not None and it["identity"] < .5}
        print(f"\n=== identity slips (primary judge): {len(slips)} of {len(items)} letters")
        for fr in ("same", "unsteered", "away", "neutral", "control"):
            xs = [it for it in items if it["friend"] == fr]
            print(f"   {fr:<10} {sum((it['dyad'], it['side'], it['round']) in slips for it in xs) / len(xs):5.1%}")
        out2, _ = outcomes(items, drop_slips=slips)
        print("\n=== robustness: identity slips dropped (primary judge)")
        for name, what, (est, lo, hi, p), sign in tests(out2):
            print(f"   {name}  {what:<28} {est:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
        out, _ = outcomes(items)
        print("\n=== exploratory (primary judge)")
        for p in PATTERNS:
            for name, a, b in (("friend: away - control", out[(p, "away")]["B"], out[(p, "control")]["B"]),
                               ("focal: unsteered - neutral", out[(p, "unsteered")]["A"], out[(p, "neutral")]["A"])):
                est, lo, hi, _ = contrast(a, b)
                print(f"   {p:<17} {name:<28} {est:+5.1f} [{lo:+.1f}, {hi:+.1f}]")
        e1 = contrast(out[("Entitlement", "unsteered")]["B"], out[("Entitlement", "control")]["B"])[0]
        e2 = contrast(out[("Social_Isolation", "unsteered")]["B"], out[("Social_Isolation", "control")]["B"])[0]
        print(f"   spread, entitlement minus isolation: {e1 - e2:+.1f} (point estimate)")


if __name__ == "__main__":
    main()
