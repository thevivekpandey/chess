"""
D1: Policy target entropy histogram + summary stats.
D2: Value target distribution.

Reads replay_buffer.csv from a run work-dir and reports.
"""
import csv
import math
import sys
from collections import Counter

import numpy as np


def policy_entropy(visits_str: str) -> float:
    """Entropy in nats of the visit-count distribution."""
    counts = np.fromstring(visits_str, sep=",", dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def policy_top_share(visits_str: str) -> float:
    """Fraction of visits that went to the most-visited move."""
    counts = np.fromstring(visits_str, sep=",", dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    return float(counts.max() / total)


def policy_n_moves(visits_str: str) -> int:
    return len(visits_str.split(","))


def policy_effective_n(visits_str: str) -> float:
    """Effective number of moves (exp of entropy)."""
    return math.exp(policy_entropy(visits_str))


def main(buffer_path: str, sample_n: int = 50000):
    print(f"Reading {buffer_path}, sampling {sample_n} rows for stats...\n")

    entropies = []
    top_shares = []
    eff_ns = []
    n_moves = []
    value_targets = []

    rng = np.random.default_rng(0)
    with open(buffer_path) as f:
        n_total = sum(1 for _ in f) - 1  # minus header

    skip_prob = max(0.0, 1.0 - sample_n / n_total)

    with open(buffer_path) as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if rng.random() < skip_prob:
                continue
            try:
                ent = policy_entropy(row["policy_visits"])
                top = policy_top_share(row["policy_visits"])
                nm = policy_n_moves(row["policy_visits"])
                if not math.isnan(ent):
                    entropies.append(ent)
                    top_shares.append(top)
                    eff_ns.append(math.exp(ent))
                    n_moves.append(nm)
                value_targets.append(float(row["value_target"]))
            except (KeyError, ValueError):
                continue
            if len(entropies) >= sample_n:
                break

    entropies = np.array(entropies)
    top_shares = np.array(top_shares)
    eff_ns = np.array(eff_ns)
    n_moves = np.array(n_moves)
    value_targets = np.array(value_targets)

    print(f"Total rows in buffer: {n_total:,}")
    print(f"Sampled: {len(entropies):,}\n")

    # ---- D1: Policy entropy ----
    print("=" * 60)
    print("D1: POLICY TARGET ENTROPY (nats)")
    print("=" * 60)
    print(f"  mean:   {entropies.mean():.3f}")
    print(f"  median: {np.median(entropies):.3f}")
    print(f"  std:    {entropies.std():.3f}")
    print(f"  min:    {entropies.min():.3f}")
    print(f"  max:    {entropies.max():.3f}")
    print()
    print("  Percentiles:")
    for p in [5, 25, 50, 75, 95]:
        print(f"    p{p:02d}: {np.percentile(entropies, p):.3f}")
    print()
    print("  Histogram (entropy bins, count):")
    bins = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]
    hist, edges = np.histogram(entropies, bins=bins)
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        bar = "#" * int(40 * c / max(hist))
        pct = 100 * c / hist.sum()
        print(f"    [{lo:4.2f}, {hi:4.2f}): {c:6,d} ({pct:5.1f}%) {bar}")
    print()
    print(f"  Effective # of moves (exp(entropy)): mean {eff_ns.mean():.2f}, median {np.median(eff_ns):.2f}")
    print(f"  Top-1 share (visits to most-visited / total): mean {top_shares.mean():.3f}, median {np.median(top_shares):.3f}")
    print(f"  # legal moves: mean {n_moves.mean():.1f}, median {np.median(n_moves):.1f}")
    print()

    # Compare to "model loss = 1.93" — what would target entropy of 1.93 imply?
    print(f"  Reference: log(7) = {math.log(7):.3f}, log(10) = {math.log(10):.3f}")
    print(f"  Model policy loss has been ~1.93. If target H << 1.93, model can't fit targets — has too much entropy.")
    print(f"  If target H ~~ 1.93, model already at the entropy ceiling — targets ARE the wall.")
    print()

    # ---- D2: Value target distribution ----
    print("=" * 60)
    print("D2: VALUE TARGET DISTRIBUTION")
    print("=" * 60)
    print(f"  mean:   {value_targets.mean():+.4f}")
    print(f"  std:    {value_targets.std():.4f}")
    print(f"  min:    {value_targets.min():+.3f}")
    print(f"  max:    {value_targets.max():+.3f}")
    print()
    print("  Distribution:")
    bins = [-1.001, -0.999, -0.5, -0.001, 0.001, 0.5, 0.999, 1.001]
    labels = [
        "= -1.0 (loss)",
        "(-1, -0.5]",
        "(-0.5, 0)",
        "= 0.0 (draw)",
        "(0, 0.5]",
        "(0.5, 1)",
        "= +1.0 (win)",
    ]
    hist, _ = np.histogram(value_targets, bins=bins)
    for label, c in zip(labels, hist):
        bar = "#" * int(40 * c / max(hist))
        pct = 100 * c / hist.sum()
        print(f"    {label:<18s}: {c:6,d} ({pct:5.1f}%) {bar}")
    print()
    n_extreme = ((value_targets == 1.0) | (value_targets == -1.0)).sum()
    n_zero = (value_targets == 0.0).sum()
    print(f"  Fraction at exactly 0.0:  {n_zero / len(value_targets):.3f} (these provide weakest gradient)")
    print(f"  Fraction at +/-1.0:       {n_extreme / len(value_targets):.3f} (strongest gradient)")
    print()
    print(f"  Variance of value targets: {value_targets.var():.4f}")
    print(f"  Reference: a constant predictor (output 0) has MSE = var(target) = {value_targets.var():.4f}")
    print(f"  Model value MSE (latest): ~0.13. Compare to the constant-baseline above.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/chess/attempt_17/selfplay_run08/replay_buffer.csv"
    main(path)
