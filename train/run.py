"""Train the lookup tables by iterated self-play.

    python3 train/run.py --gens 5 --post 300000 --pre 24000000 --procs 4

Generation 0 is the stated starting point: linear opening ranges (15% first to
act, +5% per seat), equity realization 1.0 with -0.15 out of position, +0.15
for nut draws and -0.15 for non-nut draws and made hands under top pair with a
jack kicker. Each later generation:

  A. turn + river: play hands with the previous tables, measure the EV of every
     turn and river action, rebuild those tables
  B. flop: play again through the new turn / river tables, measure the flop
     EVs, the value of each flop bucket (used by the auction), the equity
     realization of each starting hand, and what payments say about a hand
  C. preflop: measure the EV of every preflop action, where a hand that sees a
     flop is worth equity x measured equity realization x pot

Continue later with  --resume train/out/gen3.npz --gens 2  (runs generations 4 and 5).

Output: train/out/gen<N>.npz per generation, train/out/report.json, and the
final tables in bot/data/trained_holdem.npz.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
sys.path.insert(0, ROOT)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

from pb import absx as X  # noqa: E402
from pb import kernels as K  # noqa: E402
from train import sim as S  # noqa: E402

CFG = {
    # ---- stated in the spec ----
    "open_first": 0.15, "open_step": 0.05,       # generation-0 opening ranges
    "eqr_oop": 0.15, "eqr_nut_draw": 0.15, "eqr_weak": 0.15,
    "kick_min": 9,                               # jack: top pair needs this kicker to count as a made hand
    # ---- choices made here (see train/README.md) ----
    "tau_post": 0.05,                            # pots: EV gap over which two actions still mix
    "tau_pre": 0.10,                             # big blinds
    "trim": 0.03,                                # actions under this probability are dropped
    "eps_post": 0.35, "eps_pre": 0.5,            # chance a deviating player deviates at a decision
    "onpolicy": 0.35,                            # share of hands with no deviation
    "k_post": 40.0, "k_fine": 60.0, "k_pre": 30.0, "k_vf": 30.0,   # prior weights (in samples)
    "k_eqr_group": 200.0, "k_eqr_class": 250.0,
    "eqr_carry": 0.5,                            # share of earlier generations' equity-realization samples kept
    "damp": 0.5,                                 # from generation 2 on, move this share of the way to the new EVs
    "nut_top": 0.12,                             # "the nuts" = strongest 12% of both ranges
    "lambda0": 0.004,                            # big blinds one flop-chip is worth, at the reference balance
    "kappa0": 0.5,                               # share of a flop's worth the opponent model bids
    "bal_lo": 150.0, "bal_hi": 1000.0, "bal_ref": 600.0,
    "pay_target": 0.08,                          # share of the balance paid per auction the price is tuned to
    "opp_rows": 24,                              # opponent hands sampled per auction
    "band_weights": [0.15, 0.25, 0.3, 0.3],
}


def params(cfg, lam, kappa, pay_update):
    P = np.zeros(S.P_N)
    P[S.P_TAU_POST] = cfg["tau_post"]
    P[S.P_TAU_PRE] = cfg["tau_pre"]
    P[S.P_TRIM] = cfg["trim"]
    P[S.P_EPS_POST] = cfg["eps_post"]
    P[S.P_EPS_PRE] = cfg["eps_pre"]
    P[S.P_ONPOLICY] = cfg["onpolicy"]
    P[S.P_LAMBDA] = lam
    P[S.P_KAPPA] = kappa
    P[S.P_KICK] = cfg["kick_min"]
    P[S.P_NUT_TOP] = cfg["nut_top"]
    P[S.P_BAL_LO] = cfg["bal_lo"]
    P[S.P_BAL_HI] = cfg["bal_hi"]
    P[S.P_BAL_REF] = cfg["bal_ref"]
    P[S.P_OPP_ROWS] = cfg["opp_rows"]
    for i in range(4):
        P[S.P_BAND_W0 + i] = cfg["band_weights"][i]
    P[S.P_PAY_UPDATE] = 1.0 if pay_update else 0.0
    return P


# ------------------------------------------------------------------------------
# Generation 0
# ------------------------------------------------------------------------------
@njit(cache=True)
def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@njit(cache=True)
def build_q0(Q, oop, nut, weak):
    """Postflop EVs (in pots) from equity x equity realization, before any
    self-play: a bet is called by the part of a range the pot odds require."""
    for st in range(X.NST):
        for pos in range(X.NPOS):
            for spr in range(X.NSPR):
                sprc = X.SPR_MID[spr]
                for fac in range(X.NFAC):
                    for ctx in range(X.NCTX):
                        for rv in range(X.NRV):
                            for nu in range(X.NNU):
                                base = X.cell_index(st, pos, spr, fac, ctx, rv, nu) * X.NB
                                for b in range(X.NB):
                                    s = X.S_MID[b // X.NE]
                                    ec = b % X.NE
                                    eqr = 1.0 - (oop if pos == 0 else 0.0)
                                    if ec == 2:
                                        eqr += nut
                                    elif ec == 0:
                                        eqr -= weak
                                    q = Q[base + b]
                                    if fac == 0:
                                        q[0] = -1.0
                                        q[1] = _clip01(s * eqr)
                                        for i in range(5):
                                            f = X.FRACS[i] if i < 4 else sprc
                                            phi = f / (1.0 + f)
                                            e = _clip01(_clip01((s - phi) / (1.0 - phi)) * eqr)
                                            q[2 + i] = phi + (1.0 - phi) * (e * (1.0 + 2.0 * f) - f)
                                    else:
                                        sz = (fac - 1) % 5
                                        raised = fac > 5
                                        x = 1.5 if sz == 4 else X.FRACS[sz]
                                        c = x / (1.0 + x)
                                        theta = 0.25 + 0.2 * min(x, 1.5) / 1.5 + (0.15 if raised else 0.0)
                                        bl = x / (1.0 + 2.0 * x) * (0.5 if raised else 1.0)
                                        eb = (1.0 - bl) * _clip01((s - theta) / (1.0 - theta)) + bl * min(1.0, s / 0.2)
                                        q[0] = 0.0
                                        q[1] = _clip01(eb * eqr) * (1.0 + c) - c
                                        for i in range(5):
                                            f = X.FRACS[i] if i < 4 else 0.0
                                            r = f * (1.0 + c) if i < 4 else max(sprc - c, 0.1)
                                            phi = r / (1.0 + c + r)
                                            e = _clip01(_clip01((eb - phi) / (1.0 - phi)) * eqr)
                                            q[2 + i] = phi + (1.0 - phi) * (e * (1.0 + c + 2.0 * r) - c - r)


def build_vf0(cfg):
    vf = np.zeros((X.NPOS, X.NCTX, X.NSPR, X.NB))
    for pos in range(2):
        for b in range(X.NB):
            eqr = 1.0 - (cfg["eqr_oop"] if pos == 0 else 0.0)
            ec = b % X.NE
            eqr += cfg["eqr_nut_draw"] if ec == 2 else (-cfg["eqr_weak"] if ec == 0 else 0.0)
            vf[pos, :, :, b] = min(max(X.S_MID[b // X.NE] * eqr, 0.0), 1.0)
    return vf.reshape(-1)


def build_qp0(cfg, pct):
    """Generation-0 preflop policy, stored as EVs whose softmax reproduces it.

    Unopened: the top (15% + 5% per seat) of hands raise (jam under 10 bb; at
    10-20 bb the better half min-raises and the rest jams). Facing action:
    continue with a share of the opener's width, re-raising the top third."""
    tau = cfg["tau_pre"]
    opens = np.array([cfg["open_first"] + cfg["open_step"] * s for s in range(8)])
    ag_open = np.array([0.0, opens[1], opens[4], opens[6]])
    jam_call = np.array([0.28, 0.16, 0.09, 0.05])
    ag_fac = np.array([1.0, 0.8, 1.0, 1.2])
    pi = np.full((X.NBAND, X.NSIT, X.NSEAT, X.NAG, X.NCLS, X.NPA), 1e-3)
    F, C, R, J = 0, 1, 2, 3

    def put(view, mask, main):
        view[mask, main] = 1.0
        if main == R:
            view[mask, J] = 0.02
        elif main == J:
            view[mask, C] = 0.02
        elif main == F:
            view[mask, C] = 0.02

    for band in range(X.NBAND):
        for seat in range(X.NSEAT):
            for ag in range(X.NAG):
                o = opens[seat]
                done = np.zeros(X.NCLS, bool)

                def tier(sit, top, main):
                    m = (pct < top) & ~done
                    put(pi[band, sit, seat, ag], m, main)
                    done[:] |= m

                def rest(sit, main):
                    put(pi[band, sit, seat, ag], ~done, main)
                    done[:] = False

                small = R if band >= 2 else J
                # unopened
                if band == 1:
                    tier(0, 0.5 * o, R)
                tier(0, o, small if band != 1 else J)
                rest(0, F)
                # facing limpers
                if band == 1:
                    tier(1, 0.4 * o, R)
                tier(1, 0.8 * o, small if band != 1 else J)
                if seat >= 4:
                    tier(1, 1.3 * o, C)
                rest(1, F)
                # facing an open
                oo = ag_open[ag] if ag else 0.3
                cont = min(1.3 * oo, 0.6) if seat == 7 else 0.5 * oo
                tier(2, cont / 3.0, small)
                tier(2, cont, C)
                rest(2, F)
                # limped, now facing a raise
                tier(3, 0.06, R if band == 3 else J)
                tier(3, 0.30, C)
                rest(3, F)
                # raised, now facing a re-raise
                tier(4, 0.12 * o + 0.01, R if band == 3 else J)
                tier(4, 0.45 * o, C)
                rest(4, F)
                # facing a raise and a re-raise cold
                tier(5, 0.03, J)
                rest(5, F)
                # facing a jam
                tier(6, jam_call[band] * ag_fac[ag] * (1.3 if seat == 7 else 1.0), C)
                rest(6, F)
    q = tau * (np.log(pi) - np.log(pi[..., :1]))     # fold is worth exactly 0
    return q.reshape(X.NPROW, X.NPA)


def build_eqp0(cfg):
    e = np.ones((X.NBAND, X.NPT, 2, X.NCLS))
    e[:, :, 0, :] = 1.0 - cfg["eqr_oop"]
    return e


# ------------------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------------------
G = {}


def _post_worker(args):
    n, seed, lo, hi = args
    QS = np.zeros((X.NROW, X.NA))
    QN = np.zeros((X.NROW, X.NA))
    nvf = X.NPOS * X.NCTX * X.NSPR * X.NB
    neq = X.NBAND * X.NPT * 2 * X.NCLS
    VFS, VFN = np.zeros(nvf), np.zeros(nvf)
    EQS, EQE, EQN = np.zeros(neq), np.zeros(neq), np.zeros(neq)
    PAYC = np.zeros(X.NBAND * 2 * X.NPAY * X.NB)
    AST = np.zeros(32)
    S.run_post(n, seed, G["QP"], G["EQP"], G["E169"], G["QH"], G["VFH"], G["PAYL"], G["coef"],
               G["P"], lo, hi, QS, QN, VFS, VFN, EQS, EQE, EQN, PAYC, AST)
    used = np.nonzero(QN.sum(axis=1))[0]
    return used, QS[used], QN[used], VFS, VFN, EQS, EQE, EQN, PAYC, AST


def _pre_worker(args):
    n, seed = args
    QS = np.zeros((X.NPROW, X.NPA))
    QN = np.zeros((X.NPROW, X.NPA))
    kinds = S.run_preflop(n, seed, G["QP"], G["EQP"], G["E169"], G["P"], QS, QN)
    return QS, QN, kinds


def run_post(pool, procs, n, seed, lo, hi):
    jobs = [(n // procs, seed + 1000 * i, lo, hi) for i in range(procs)]
    parts = pool.map(_post_worker, jobs) if pool else [_post_worker(j) for j in jobs]
    QS = np.zeros((X.NROW, X.NA))
    QN = np.zeros((X.NROW, X.NA))
    rest = [np.zeros_like(x) for x in parts[0][3:]]
    for used, qs, qn, *others in parts:
        QS[used] += qs
        QN[used] += qn
        for acc, x in zip(rest, others):
            acc += x
    return (QS, QN, *rest)


def run_pre(pool, procs, n, seed):
    jobs = [(n // procs, seed + 1000 * i) for i in range(procs)]
    parts = pool.map(_pre_worker, jobs) if pool else [_pre_worker(j) for j in jobs]
    return sum(p[0] for p in parts), sum(p[1] for p in parts), sum(p[2] for p in parts)


# ------------------------------------------------------------------------------
# Table updates
# ------------------------------------------------------------------------------
POST_SHAPE = (X.NST, X.NPOS, X.NSPR, X.NFAC, X.NCTX, X.NRV, X.NNU, X.NB, X.NA)


def update_post(QH, QS, QN, streets, cfg, damp=1.0):
    """New EV tables for ``streets``: measured mean shrunk to the previous
    generation; flop cells split by range-vs-range and nut advantage shrink to
    the unsplit cell."""
    q = QH.reshape(POST_SHAPE).copy()
    s = QS.reshape(POST_SHAPE)
    n = QN.reshape(POST_SHAPE)
    for st in streets:
        k0 = cfg["k_post"]
        coarse = (s[st, ..., X.ANY, X.ANY, :, :] + k0 * q[st, ..., X.ANY, X.ANY, :, :]) / (
            n[st, ..., X.ANY, X.ANY, :, :] + k0)
        if st == 0:
            k1 = cfg["k_fine"]
            fine = (s[st] + k1 * coarse[..., None, None, :, :]) / (n[st] + k1)
            q[st] = fine
        q[st, ..., X.ANY, X.ANY, :, :] = coarse
    q = q.reshape(X.NROW, X.NA)
    q = QH + damp * (q - QH)                             # damped: plain best responses swing back and forth
    q[:, 0] = np.where(QH[:, 0] < 0, QH[:, 0], 0.0)     # folding is worth exactly 0
    return q.astype(np.float32)


def update_pre(QP, QS, QN, cfg, damp=1.0):
    k = cfg["k_pre"]
    q = (QS + k * QP) / (QN + k)
    q = QP + damp * (q - QP)
    q[:, 0] = 0.0
    return q.astype(np.float32)


def update_vf(VFH, VFS, VFN, cfg):
    return ((VFS + cfg["k_vf"] * VFH) / (VFN + cfg["k_vf"])).astype(np.float32)


def update_eqp(EQS, EQE, EQN, cfg):
    """Measured equity realization: pots won from the flop on / preflop equity,
    per starting hand, shrunk to its (stack band, pot type, position) group."""
    shp = (X.NBAND, X.NPT, 2, X.NCLS)
    s, e = EQS.reshape(shp), EQE.reshape(shp)
    prior = np.array([1.0 - cfg["eqr_oop"], 1.0])[None, None, :]
    kg, kc = cfg["k_eqr_group"], cfg["k_eqr_class"]
    grp = (s.sum(-1) + kg * prior * 0.5) / (e.sum(-1) + kg * 0.5)
    out = (s + kc * grp[..., None] * 0.5) / (e + kc * 0.5)
    return np.clip(out, 0.4, 1.8), grp


def update_payl(PAYC):
    """P(payment bucket | flop bucket) relative to the average hand: the factor a
    seat's range is multiplied by after it pays."""
    c = PAYC.reshape(X.NBAND, 2, X.NPAY, X.NB)
    tot_b = c.sum(axis=2, keepdims=True)                   # per flop bucket
    marg = c.sum(axis=3, keepdims=True) / np.maximum(c.sum(axis=(2, 3), keepdims=True), 1.0)
    alpha = 20.0
    like = (c + alpha * marg) / (tot_b + alpha)
    ratio = like / np.maximum(marg, 1e-9)
    ratio[:, :, 0, :] = 1.0                                # could not bid: no information
    return np.clip(ratio, 0.05, 20.0).reshape(-1).astype(np.float32)


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def preflop_summary(QP, cfg):
    """Share of hands taking each action when unopened, per seat and stack band."""
    q = QP.reshape(X.NBAND, X.NSIT, X.NSEAT, X.NAG, X.NCLS, X.NPA).astype(np.float64)
    wts = K.CLASS_COUNTS / K.CLASS_COUNTS.sum()
    out = {}
    names = ["<10bb", "10-20bb", "20-40bb", "40+bb"]
    for band in range(X.NBAND):
        rows = []
        for seat in range(7):
            leg = np.array([seat != 7, True, band >= 1, band != 3])
            z = q[band, 0, seat, 0]
            z = np.where(leg[None, :], z, -1e9)
            p = np.exp((z - z.max(1, keepdims=True)) / cfg["tau_pre"])
            p /= p.sum(1, keepdims=True)
            p[p < cfg["trim"]] = 0
            p /= p.sum(1, keepdims=True)
            fr = (p * wts[:, None]).sum(0)
            rows.append({"fold": round(float(fr[0]), 3), "limp": round(float(fr[1]), 3), "raise": round(float(fr[2]), 3),
                         "jam": round(float(fr[3]), 3)})
        out[names[band]] = rows
    return out


def auction_summary(AST):
    n = max(AST[0], 1.0)
    h = max(AST[13:17].sum(), 1.0)
    return {
        "mean_chips_paid": round(AST[1] / n, 2), "mean_chips_bid": round(AST[2] / n, 2),
        "mean_share_of_balance_paid": round(AST[5] / n, 4), "paid_nothing": round(AST[6] / n, 3),
        "bb_gained_over_default_flop": round(AST[4] / n, 4),
        "pay_buckets": [round(x / n, 3) for x in AST[7:13]],
        "flops_by_stack_band": [round(x / h, 3) for x in AST[13:17]],
        "pot_type": [round(x / h, 3) for x in AST[17:20]],
        "mean_pot_bb": round(AST[20] / h, 2), "mean_spr": round(AST[21] / h, 2),
        "last_street_with_betting": [round(x / h, 3) for x in AST[22:26]],
        "heads_up_flop_rate": round(AST[27] / max(AST[26], 1.0), 4),
    }


# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--post", type=int, default=300000, help="heads-up flops per postflop pass")
    ap.add_argument("--pre", type=int, default=24000000, help="hands per preflop pass")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--resume", default="", help="gen<N>.npz to continue from (numbering continues at N+1)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "out"))
    ap.add_argument("--final", default=os.path.join(ROOT, "bot", "data", "trained_holdem.npz"))
    a = ap.parse_args()
    cfg = dict(CFG)
    os.makedirs(a.out, exist_ok=True)
    tabs = np.load(os.path.join(ROOT, "bot", "data", "holdem_tables.npz"))
    pct = ((tabs["pct_lo"] + tabs["pct_hi"]) / 2).astype(np.float64)

    QH = np.zeros((X.NROW, X.NA), np.float32)
    q0 = np.zeros((X.NROW, X.NA))
    build_q0(q0, cfg["eqr_oop"], cfg["eqr_nut_draw"], cfg["eqr_weak"])
    QH[:] = q0
    G["E169"] = tabs["E169"].astype(np.float64)
    G["coef"] = np.load(os.path.join(ROOT, "bot", "data", "scorefit_holdem.npz"))["coef"]
    G["QH"] = QH
    G["QP"] = build_qp0(cfg, pct).astype(np.float32)
    G["EQP"] = build_eqp0(cfg)
    G["VFH"] = build_vf0(cfg).astype(np.float32)
    G["PAYL"] = np.ones(X.NBAND * 2 * X.NPAY * X.NB, np.float32)
    lam, kappa = cfg["lambda0"], cfg["kappa0"]
    report = {"config": cfg, "args": vars(a), "generations": []}
    report["gen0_preflop"] = preflop_summary(G["QP"], cfg)
    eq_acc = None
    first = 1
    if a.resume:
        d = np.load(a.resume)
        G["QH"] = d["QH"].astype(np.float32)
        G["QP"] = d["QP"].astype(np.float32)
        G["EQP"] = d["EQP"].astype(np.float64)
        G["VFH"], G["PAYL"] = d["VFH"], d["PAYL"]
        lam, kappa = float(d["lam"]), float(d["kappa"])
        first = int(d["generation"]) + 1 if "generation" in d.files else int(
            "".join(ch for ch in os.path.basename(a.resume) if ch.isdigit()) or 0) + 1
        old = os.path.join(a.out, "report.json")
        if os.path.exists(old):
            report["generations"] = json.load(open(old)).get("generations", [])[: first - 1]

    # compile before forking so the workers inherit the machine code
    t0 = time.time()
    G["P"] = params(cfg, lam, kappa, False)
    _post_worker((20, 1, 0, 2))
    _pre_worker((200, 1))
    print(f"compiled in {time.time() - t0:.0f}s", flush=True)
    ctx = mp.get_context("fork")

    def with_pool(fn, *args):
        # a fresh pool per pass: forked workers see the tables as they are now
        if a.procs <= 1:
            return fn(None, 1, *args)
        with ctx.Pool(a.procs) as pool:
            return fn(pool, a.procs, *args)

    last = first + a.gens - 1
    for g in range(first, last + 1):
        t0 = time.time()
        G["P"] = params(cfg, lam, kappa, g > 1)
        seed = a.seed + 100000 * g
        damp = 1.0 if g == 1 else cfg["damp"]
        # A: turn + river
        QS, QN, *_ = with_pool(run_post, a.post, seed + 1, 1, 2)
        G["QH"] = update_post(G["QH"], QS, QN, (1, 2), cfg, damp)
        ta = time.time() - t0
        # B: flop, played through the new turn / river tables
        QS, QN, VFS, VFN, EQS, EQE, EQN, PAYC, AST = with_pool(run_post, a.post, seed + 2, 0, 0)
        G["QH"] = update_post(G["QH"], QS, QN, (0,), cfg, damp)
        G["VFH"] = update_vf(G["VFH"], VFS, VFN, cfg)
        new = np.stack([EQS, EQE, EQN])
        eq_acc = new if eq_acc is None else cfg["eqr_carry"] * eq_acc + new
        G["EQP"], grp = update_eqp(eq_acc[0], eq_acc[1], eq_acc[2], cfg)
        G["PAYL"] = update_payl(PAYC)
        tb = time.time() - t0 - ta
        # C: preflop from equity x measured equity realization
        QS, QN, kinds = with_pool(run_pre, a.pre, seed + 3)
        G["QP"] = update_pre(G["QP"], QS, QN, cfg, damp)
        # auction: opponent-model scale and the price of a flop-chip
        asum = auction_summary(AST)
        n = max(AST[0], 1.0)
        kappa_new = float(np.clip(AST[2] / max(AST[3], 1e-9), 0.05, 1.5))
        frac = AST[5] / n
        lam_new = float(lam * np.clip((max(frac, 1e-4) / cfg["pay_target"]) ** 0.7, 0.5, 2.0))
        info = {
            "generation": g, "seconds": round(time.time() - t0, 1),
            "lambda_used": lam, "kappa_used": kappa, "auction": asum,
            "eqr_group_oop_ip_by_band_pot": np.round(grp, 3).tolist(),
            "preflop_unopened": preflop_summary(G["QP"], cfg),
            "preflop_end": {k: round(float(v / kinds.sum()), 4) for k, v in
                            zip(["everyone_folds", "all_in", "heads_up_flop", "multiway_flop"], kinds)},
            "post_rows_with_data": int((QN.sum(1) > 0).sum()),
            "pre_rows_with_data": int((QN.sum(1) > 0).sum()),
        }
        lam, kappa = lam_new, 0.5 * kappa + 0.5 * kappa_new
        report["generations"].append(info)
        np.savez_compressed(os.path.join(a.out, f"gen{g}.npz"), QH=G["QH"].astype(np.float16),
                            QP=G["QP"].astype(np.float16), EQP=G["EQP"].astype(np.float32),
                            VFH=G["VFH"], PAYL=G["PAYL"], lam=lam, kappa=kappa, generation=g)
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(f"gen {g}: A {ta:.0f}s B {tb:.0f}s total {time.time() - t0:.0f}s | "
              f"paid {asum['mean_chips_paid']} chips ({asum['mean_share_of_balance_paid']:.1%} of balance) "
              f"lambda {lam:.5f} kappa {kappa:.2f} | HU flop rate {asum['heads_up_flop_rate']}", flush=True)
    np.savez_compressed(a.final, QH=G["QH"].astype(np.float16), QP=G["QP"].astype(np.float16),
                        EQP=G["EQP"].astype(np.float32), VFH=G["VFH"], PAYL=G["PAYL"],
                        lam=lam, kappa=kappa, kick_min=cfg["kick_min"], nut_top=cfg["nut_top"],
                        tau_post=cfg["tau_post"], tau_pre=cfg["tau_pre"], trim=cfg["trim"],
                        bal_ref=cfg["bal_ref"], generations=last)
    print("wrote", a.final)


if __name__ == "__main__":
    main()
