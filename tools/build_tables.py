"""Build the precomputed preflop tables shipped with the bot.

Output: bot/data/holdem_tables.npz with
  E169        169x169 float32: equity of class i vs class j (all-in, preflop)
  order       169 int16: classes from strongest to weakest
  pct_lo/hi   169 float32: where each class sits in the ranking, as a fraction
              of all 1326 combos (class covers [pct_lo, pct_hi))
  eq_random   169 float32: equity vs a random hand
  eq_rank     169 float32: equity vs the reference range used for ranking

Ranking rule (from the team design): sort hands by equity against a
mid-strength range (top ~25%), not against a random hand. Doing that needs a
ranking to define "top 25%", so we iterate a few times until it is stable.

Run:  python3 tools/build_tables.py [--samples 4000]
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

from pb import kernels as K  # noqa: E402


def rank_classes(E, ref_width=0.25, iters=6):
    n = K.CLASS_COUNTS.astype(np.float64)
    w = np.ones(169)  # start: random hand
    order = None
    for _ in range(iters):
        eq = (E * (n * w)[None, :]).sum(1) / (n * w).sum()
        order = np.argsort(-eq)
        cum = np.cumsum(n[order]) / n.sum()
        start = cum - n[order] / n.sum()
        w = np.zeros(169)
        # fraction of each class inside the top ref_width of combos
        frac = np.clip((ref_width - start) / (cum - start), 0.0, 1.0)
        w[order] = frac
    eq = (E * (n * w)[None, :]).sum(1) / (n * w).sum()
    return order, eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    t = time.time()
    E = K.class_matrix(args.samples, args.seed, K.COMBOS, K.COMBO_CLASS, K.POP, K.HIGH, K.STRAIGHT, K.CATR)
    print(f"E169 done in {time.time() - t:.1f}s")
    n = K.CLASS_COUNTS.astype(np.float64)
    eq_random = (E * n[None, :]).sum(1) / n.sum()
    order, eq_rank = rank_classes(E)
    cum = np.cumsum(n[order]) / n.sum()
    pct_hi = np.zeros(169)
    pct_lo = np.zeros(169)
    pct_hi[order] = cum
    pct_lo[order] = cum - n[order] / n.sum()
    out = os.path.join(ROOT, "bot", "data", "holdem_tables.npz")
    np.savez_compressed(out, E169=E.astype(np.float32), order=order.astype(np.int16),
                        pct_lo=pct_lo.astype(np.float32), pct_hi=pct_hi.astype(np.float32),
                        eq_random=eq_random.astype(np.float32), eq_rank=eq_rank.astype(np.float32))
    names = K.CLASS_NAMES
    print("top 30:", " ".join(names[i] for i in order[:30]))
    print("ranks 31-60:", " ".join(names[i] for i in order[30:60]))
    for nm in ("AA", "KK", "AKs", "AKo", "QQ", "JTs", "A2o", "72o", "22", "K2o", "65s"):
        i = names.index(nm)
        print(f"{nm:4s} pct {pct_lo[i]:.3f}-{pct_hi[i]:.3f}  eq_random {eq_random[i]:.3f}  eq_vs_top25 {eq_rank[i]:.3f}")
    def e(a, b):
        return E[names.index(a), names.index(b)]
    print(f"checks: AA-KK {e('AA','KK'):.3f} (~0.82)  AKs-QQ {e('AKs','QQ'):.3f} (~0.46)  "
          f"KK-AKo {e('KK','AKo'):.3f} (~0.70)  AKo-22 {e('AKo','22'):.3f} (~0.47)")
    print("wrote", out)


if __name__ == "__main__":
    main()
