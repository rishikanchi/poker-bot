"""Solve near-equilibrium push/fold charts by iterated best response.

The plan: "start with guessed jam and call ranges for each stack depth;
repeatedly replace each range with its best response; a few dozen rounds
with damping."

Model (chip EV, all stacks equal to S big blinds, blinds 0.5/1, optional ante):
  * The shover is first in with k players behind (k=1: small blind vs big
    blind, k=2: button, ... k=7: under the gun at 8-handed).
  * The shover jams the top J of hands (by the plug-in's preflop percentile).
  * Each player behind calls with the top C of hands. Two caller types, as the
    plan simplified: the big blind (it already has 1 bb in) and everyone else
    (one shared width, priced as a player with nothing in).
  * P(first caller is j) = prod(earlier players fold) * C_j. Overcalls ignored.
  * Hand-vs-range equity comes from the variant plug-in: the class matrix
    E169 for hold'em, the pre_T table (percentile bins x top-x) otherwise.

Best responses are computed per hand group (EV of jamming/calling > 0), the
new width is the share of hands with positive EV, and each width moves 30% of
the way toward its best response per round until nothing moves more than
0.002 (or 300 rounds).

Output: bot/data/pushfold_<variant>.npz with S (bb grid), K (1..7),
jam[k, s], call_bb[k, s], call_other[k, s]  (fractions of all hands).

Run:  python3 tools/build_pushfold.py [--ante 0.0]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

S_GRID = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20, 25], float)
K_GRID = np.arange(1, 8)
X_GRID = np.linspace(0.0025, 1.0, 400)  # top-x widths the equity table is built on


class EquityModel:
    """Hand groups (weight, equity vs top-x) for the configured variant."""

    def __init__(self, wt, eqx, lo, hi):
        self.wt = wt            # [G] share of all hands in each group (sums to 1)
        self.eqx = eqx          # [G, len(X_GRID)] equity vs the top-x band
        self.lo = lo            # [G] percentile band each group covers
        self.hi = hi

    def top_frac(self, x):
        """Fraction of each group that lies inside the top-x band."""
        return np.clip((x - self.lo) / np.maximum(self.hi - self.lo, 1e-9), 0.0, 1.0)

    def eq(self, x):
        """Equity of every group vs the top-x band (linear in x between grid points)."""
        x = float(min(max(x, X_GRID[0]), 1.0))
        i = int(np.searchsorted(X_GRID, x))
        if i <= 0:
            return self.eqx[:, 0]
        if i >= len(X_GRID):
            return self.eqx[:, -1]
        t = (x - X_GRID[i - 1]) / (X_GRID[i] - X_GRID[i - 1])
        return (1 - t) * self.eqx[:, i - 1] + t * self.eqx[:, i]


def holdem_model():
    from pb import kernels as K
    t = np.load(os.path.join(ROOT, "bot", "data", "holdem_tables.npz"))
    E = t["E169"].astype(np.float64)
    lo = t["pct_lo"].astype(np.float64)
    hi = t["pct_hi"].astype(np.float64)
    n = K.CLASS_COUNTS.astype(np.float64)
    # W[j, xi] = combos of class j inside the top-x band
    W = np.clip((np.minimum(X_GRID[None, :], hi[:, None]) - lo[:, None])
                / (hi - lo)[:, None], 0.0, 1.0) * n[:, None]
    eqx = (E @ W) / W.sum(0)[None, :]
    return EquityModel(n / n.sum(), eqx, lo, hi)


def generic_model(game):
    if getattr(game, "pre_T", None) is None:
        game._build_pre_table()
    T = game.pre_T                      # [bins, len(PRE_XS)]
    xs = game.PRE_XS
    nb = T.shape[0]
    eqx = np.vstack([np.interp(X_GRID, xs, T[b], left=T[b, 0]) for b in range(nb)])
    return EquityModel(np.full(nb, 1.0 / nb), eqx, np.arange(nb) / nb, (np.arange(nb) + 1.0) / nb)


def build_model():
    from pb import game as G
    from pb import settings
    g = G.get_game()
    if g.E169 is not None:
        return holdem_model(), settings.VARIANT["name"]
    return generic_model(g), settings.VARIANT["name"]


def _width(model, ev):
    return float(model.wt[ev > 0].sum())


def solve_spot(model, S, k, ante=0.0, n_table=8, damping=0.3, tol=0.002, max_rounds=3000):
    """Jam width for the shover with k behind, and the two call widths."""
    J, Cb, Co = 0.5, 0.3, 0.2  # guesses
    shover_posted = 0.5 if k == 1 else 0.0
    pot_now = 1.5 + ante * n_table
    # who acts after the shover, in order: (k-2) others, the SB, then the BB
    callers = []
    if k >= 3:
        callers += [("other", 0.0)] * (k - 2)
    if k >= 2:
        callers.append(("other", 0.5))   # the small blind, priced with its 0.5 in
    callers.append(("bb", 1.0))
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        # shover best response to the current call widths
        ev = np.zeros(model.wt.shape[0])
        p_none = 1.0
        for kind, posted in callers:
            c = Cb if kind == "bb" else Co
            final = pot_now + (S - shover_posted - ante) + (S - posted - ante)
            ev += p_none * c * (model.eq(c) * final - (S - shover_posted - ante))
            p_none *= (1.0 - c)
        ev += p_none * pot_now
        J_br = _width(model, ev)
        # callers' best responses to the current jam width (no one called yet)
        eq_vs_j = model.eq(J)
        final_bb = pot_now + (S - shover_posted - ante) + (S - 1.0 - ante)
        Cb_br = _width(model, eq_vs_j * final_bb - (S - 1.0 - ante))
        final_o = pot_now + (S - shover_posted - ante) + (S - 0.0 - ante)
        Co_br = _width(model, eq_vs_j * final_o - (S - 0.0 - ante))
        dJ, dB, dO = J_br - J, Cb_br - Cb, Co_br - Co
        # damped steps first, then a shrinking (fictitious-play) step: best
        # responses jump between discrete widths, so a fixed step oscillates
        step = damping if rounds <= 30 else max(damping * 30.0 / rounds, 0.002)
        J += step * dJ
        Cb += step * dB
        Co += step * dO
        if rounds > 30 and max(abs(dJ), abs(dB), abs(dO)) * step < tol * 0.05:
            break
    return J, Cb, Co, rounds


def residual(model, S, k, J, Cb, Co, ante=0.0, n_table=8):
    """Largest gap between a width and its exact best response (0 = equilibrium)."""
    shover_posted = 0.5 if k == 1 else 0.0
    pot_now = 1.5 + ante * n_table
    callers = ([("other", 0.0)] * max(k - 2, 0) + ([("other", 0.5)] if k >= 2 else []) + [("bb", 1.0)])
    ev = np.zeros(model.wt.shape[0])
    p_none = 1.0
    for kind, posted in callers:
        c = Cb if kind == "bb" else Co
        final = pot_now + (S - shover_posted - ante) + (S - posted - ante)
        ev += p_none * c * (model.eq(c) * final - (S - shover_posted - ante))
        p_none *= (1.0 - c)
    ev += p_none * pot_now
    eq_vs_j = model.eq(J)
    fb = pot_now + (S - shover_posted - ante) + (S - 1.0 - ante)
    fo = pot_now + (S - shover_posted - ante) + S - ante
    return max(abs(_width(model, ev) - J), abs(_width(model, eq_vs_j * fb - (S - 1.0 - ante)) - Cb),
               abs(_width(model, eq_vs_j * fo - (S - ante)) - Co))


W_GRID = np.array([0.10, 0.15, 0.20, 0.30, 0.45, 0.80])   # opener's opening width
OPEN_BB = 2.0                                               # plan: opens ~2-2.2bb at these depths


def solve_rejam(model, S, W, ante=0.0, n_table=8, damping=0.3, max_rounds=3000):
    """Re-shove over an open (plan: "a re-shove range over a small open").

    The opener raised to OPEN_BB with the top W of hands. A player behind (not
    in the blinds) with S bb jams the top R; the opener calls with the top C
    (C <= W); everyone else folds. Chip EV, iterated best responses:
      re-jam: P(opener folds) * (blinds + open) + P(call) * (eq vs top-C * final - S)
      call:   eq vs top-R * final - (S - OPEN_BB)   for hands inside the opening range
    Returns (R, C) as fractions of all hands.
    """
    R, C = min(0.15, W), min(0.08, W)
    pot_open = 1.5 + ante * n_table + OPEN_BB
    final = 1.5 + ante * n_table + 2.0 * S
    in_open = model.top_frac(W)
    for rounds in range(1, max_rounds + 1):
        pc = min(1.0, C / max(W, 1e-9))
        ev = (1.0 - pc) * pot_open + pc * (model.eq(max(C, 1e-4)) * final - S)
        R_br = _width(model, ev)
        evc = model.eq(max(R, 1e-4)) * final - (S - OPEN_BB)
        C_br = float((model.wt * in_open * (evc > 0)).sum())
        dR, dC = R_br - R, C_br - C
        step = damping if rounds <= 30 else max(damping * 30.0 / rounds, 0.002)
        R += step * dR
        C += step * dC
        if rounds > 30 and max(abs(dR), abs(dC)) * step < 1e-4:
            break
    return R, C


def solve_all_rejam(model, ante=0.0):
    rj = np.zeros((len(W_GRID), len(S_GRID)))
    rc = np.zeros_like(rj)
    for wi, W in enumerate(W_GRID):
        for si, S in enumerate(S_GRID):
            if S <= OPEN_BB:
                rj[wi, si], rc[wi, si] = 1.0, W
                continue
            rj[wi, si], rc[wi, si] = solve_rejam(model, S, float(W), ante)
    return rj, rc


def solve_all(model, ante=0.0):
    jam = np.zeros((len(K_GRID), len(S_GRID)))
    cbb = np.zeros_like(jam)
    coth = np.zeros_like(jam)
    rounds = np.zeros_like(jam, dtype=int)
    res = np.zeros_like(jam)
    for ki, k in enumerate(K_GRID):
        for si, S in enumerate(S_GRID):
            j, b, o, r = solve_spot(model, S, int(k), ante)
            jam[ki, si], cbb[ki, si], coth[ki, si], rounds[ki, si] = j, b, o, r
            res[ki, si] = residual(model, S, int(k), j, b, o, ante)
    return jam, cbb, coth, rounds, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ante", type=float, default=0.0, help="ante per player in big blinds")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    model, name = build_model()
    t1 = time.time()
    jam, cbb, coth, rounds, res = solve_all(model, args.ante)
    rejam, rejam_call = solve_all_rejam(model, args.ante)
    t2 = time.time()
    out = args.out or os.path.join(ROOT, "bot", "data", f"pushfold_{name}.npz")
    np.savez(out, S=S_GRID, K=K_GRID, jam=jam, call_bb=cbb, call_other=coth,
             W=W_GRID, rejam=rejam, rejam_call=rejam_call,
             ante=np.array(args.ante), variant=np.array(name))
    print(f"variant {name}: model {t1 - t0:.1f}s, solve {t2 - t1:.2f}s, max rounds {rounds.max()}, "
          f"largest best-response gap {res.max():.3f} (median {np.median(res):.3f})")
    hdr = "  S(bb) " + " ".join(f"{int(s):>5d}" for s in S_GRID)
    for label, arr in (("jam", jam), ("call BB", cbb), ("call other", coth)):
        print(f"\n{label} width by players behind the shover (rows k) and stack (cols):")
        print(hdr)
        for ki, k in enumerate(K_GRID):
            print(f"  k={k}    " + " ".join(f"{100 * arr[ki, si]:5.1f}" for si in range(len(S_GRID))))
    hdr = "  S(bb) " + " ".join(f"{int(s):>5d}" for s in S_GRID)
    for label, arr in (("re-jam over an open", rejam), ("opener calls the re-jam", rejam_call)):
        print(f"\n{label}, by the opener's width (rows) and stack (cols):")
        print(hdr)
        for wi, W in enumerate(W_GRID):
            print(f"  W={W:.2f} " + " ".join(f"{100 * arr[wi, si]:5.1f}" for si in range(len(S_GRID))))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
