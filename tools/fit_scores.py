"""Fit the flop/turn hand-strength score used to order opponents' ranges.

Plan: sort an opponent's hands by equity (not by hand rank) before the river,
and get cheap equity estimates from "a scoring function ... fit its weights
against a small sample of computed equities".

For random flops and turns we compute, for every hand in the universe:
  * the features the bot can compute in microseconds (kernels.hand_features):
    made-hand percentile p, flush draw, open-ender, gutshot, backdoor flush draw
  * its true equity against a random hand (average, over the runouts, of the
    share of live hands it beats at showdown)
and fit equity ~ coef . features by least squares, separately for the flop
and the turn. Output: bot/data/scorefit_<variant>.npz (coef[2, NF]).

Run:  python3 tools/fit_scores.py            (configured variant; ~10-60 s)
      PB_VARIANT=omaha4 python3 tools/fit_scores.py
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))


def pct_of(vals):
    """Percentile (ties averaged) among live hands; nan for dead hands."""
    out = np.full(vals.shape[0], np.nan)
    live = vals >= 0
    lv = vals[live]
    if lv.size < 2:
        return out
    _, inv, counts = np.unique(lv, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    out[live] = (start + (counts - 1) / 2.0)[inv] / (lv.size - 1)
    return out


def ehs(g, board, rng, max_runouts):
    """Equity vs a random hand for every universe hand on this board."""
    left = [int(c) for c in g.deck if c not in board]
    need = g.board_total - len(board)
    runouts = list(itertools.combinations(left, need))
    if len(runouts) > max_runouts:
        idx = rng.choice(len(runouts), max_runouts, replace=False)
        runouts = [runouts[i] for i in idx]
    tot = np.zeros(g.M)
    cnt = np.zeros(g.M)
    for ro in runouts:
        fb = list(board) + list(ro)
        p = pct_of(g.strengths(fb, g.dead_mask(fb)))
        ok = ~np.isnan(p)
        tot[ok] += p[ok]
        cnt[ok] += 1
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def old_bonus_score(X):
    """The hand-picked draw bonuses this fit replaces (for comparison only)."""
    p, fd, oe, gs, bd = X[:, 1], X[:, 3], X[:, 5], X[:, 7], X[:, 9]
    bonus = np.where(fd > 0, 0.35, np.where(bd > 0, 0.04, 0.0))
    bonus = np.where(oe > 0, np.maximum(bonus, 0.25) + np.where(bonus >= 0.35, 0.1, 0.0), bonus)
    bonus = np.where((gs > 0) & (oe == 0), np.maximum(bonus, 0.10) + np.where(bonus >= 0.35, 0.05, 0.0), bonus)
    return np.minimum(p + bonus * (1 - p), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=0, help="boards per street (default: by variant)")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    from pb import game as G
    from pb import kernels as K
    from pb import settings
    g = G.get_game()
    name = settings.VARIANT["name"]
    rng = np.random.default_rng(args.seed)
    n_boards = args.boards or (70 if g.exact else 60)
    max_ro = 1200 if g.exact else 160
    coef = np.zeros((2, K.NF))
    t0 = time.time()
    for row, nb in ((0, 3), (1, 4)):
        Xs, ys = [], []
        for _ in range(n_boards):
            board = [int(c) for c in rng.choice(g.deck, nb, replace=False)]
            X, _ = g.board_features(board, g.dead_mask(board))
            y = ehs(g, board, rng, max_ro)
            ok = (X[:, 1] >= 0) & ~np.isnan(y)
            Xs.append(X[ok])
            ys.append(y[ok])
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        cols = [f for f in range(K.NF) if np.abs(X[:, f]).sum() > 0]   # e.g. no backdoor draws on the turn
        beta, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        coef[row, cols] = beta
        pred = np.clip(X @ coef[row], 0, 1)

        def rmse(z):
            return float(np.sqrt(np.mean((z - y) ** 2)))

        def rank_corr(z):
            a = np.argsort(np.argsort(z))
            b = np.argsort(np.argsort(y))
            return float(np.corrcoef(a, b)[0, 1])

        print(f"{'flop' if nb == 3 else 'turn'}: {len(y)} hands on {n_boards} boards | "
              f"fitted: rmse {rmse(pred):.4f}, rank corr {rank_corr(pred):.4f} | "
              f"made-hand percentile only: rmse {rmse(X[:, 1]):.4f}, rank corr {rank_corr(X[:, 1]):.4f} | "
              f"old hand-picked bonuses: rmse {rmse(old_bonus_score(X)):.4f}, rank corr {rank_corr(old_bonus_score(X)):.4f}")
        print("   coef [1, p, p^2, fd, fd*p, oe, oe*p, gs, gs*p, bd] =", np.round(coef[row], 3).tolist())
    out = os.path.join(ROOT, "bot", "data", f"scorefit_{name}.npz")
    np.savez(out, coef=coef, variant=np.array(name))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
