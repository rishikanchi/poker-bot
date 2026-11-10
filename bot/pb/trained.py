"""Decisions from the self-play tables (train/run.py -> data/trained_holdem.npz).

Three independent hooks, each switched on in settings.py:
  preflop()   fold / call / small raise / jam by seat, stack band and situation
  auction()   five bids: best of a fixed candidate set against modelled opponent bids
  postflop()  heads-up pots: check / call / fold / bet 10-33-66-125% / all-in

Each returns None when the spot is outside what was trained (the caller then
uses the hand-built logic), so a missing table can never break a decision.
The tables are chip-EV and were trained on equal stacks: ICM is not in them.
"""
from __future__ import annotations

import os

import numpy as np

from . import absx as X
from . import kernels as K
from . import ranges as Rg
from . import view as V

T = None
OPEN_TO = (0.0, 2.0, 2.2, 2.5)
RERAISE_X = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 3.3, 2.3))
MAX_SMALL = (0, 1, 2, 3)
POOL = 16
RUNOUTS = 6


class Tables:
    pass


def load(path=None):
    """Load the tables once (at import). Returns True when they are usable."""
    global T
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "trained_holdem.npz")
    try:
        d = np.load(path)
        t = Tables()
        t.QH = np.ascontiguousarray(d["QH"].astype(np.float32))
        t.QP = np.ascontiguousarray(d["QP"].astype(np.float32))
        t.VFH = d["VFH"].astype(np.float64)
        t.PAYL = d["PAYL"].astype(np.float64)
        t.EQP = d["EQP"].astype(np.float64)
        t.lam, t.kappa = float(d["lam"]), float(d["kappa"])
        t.kick = int(d["kick_min"])
        t.tau_post, t.tau_pre, t.trim = float(d["tau_post"]), float(d["tau_pre"]), float(d["trim"])
        t.bal_ref = float(d["bal_ref"])
        if t.QH.shape != (X.NROW, X.NA) or t.QP.shape != (X.NPROW, X.NPA):
            return False
        T = t
        return True
    except Exception:  # noqa: BLE001 - no tables: every hook returns None
        T = None
        return False


def _pick(probs, na, u):
    c = 0.0
    last = 1
    for a in range(na):
        if probs[a] > 0.0:
            c += probs[a]
            last = a
            if u < c:
                return a
    return last


def _last_raiser(acts):
    for a in reversed(acts):
        if a.kind == "raise":
            return a.seat
    return None


# ------------------------------------------------------------------------------
def preflop(c):
    v = c.v
    if T is None or len(v.hole) != 2 or v.hero is None or v.seat not in v.pre_order:
        return None
    order = v.pre_order
    n = len(order)
    if n < 2 or n > 8:
        return None
    bb = float(v.bb)
    label = {s: 8 - n + i for i, s in enumerate(order)}
    L = nlimp = vol = 0
    aggr = None
    jam = False
    for a in v.acts[0]:
        if a.kind == "raise":
            L += 1
            aggr = a.seat
            jam = bool(a.allin)
            if a.seat == v.seat:
                vol = 2
        elif a.kind == "call":
            if L == 0:
                nlimp += 1
            if a.seat == v.seat:
                vol = max(vol, 1)
    others = [s for s in v.alive if s != v.seat and v.players[s].in_hand]
    if not others:
        return None
    eff = min(v.hero.start, max(v.players[s].start for s in others)) / bb
    band = int(X.band_of(eff))
    if jam:
        sit = 6
    elif L == 0:
        sit = 0 if nlimp == 0 else 1
    elif L == 1:
        sit = 3 if vol == 1 else 2
    else:
        sit = 4 if vol == 2 else 5
    lab = label.get(aggr, -1) if aggr is not None else -1
    ag = 0 if lab < 0 else (1 if lab <= 2 else (2 if lab <= 5 else 3))
    cur = max(p.bet for p in v.players.values()) / bb
    raise_to = OPEN_TO[band] + nlimp if L == 0 else RERAISE_X[band][min(L, 2)] * cur
    can_raise = "raise" in v.legal
    leg = np.zeros(X.NPA, np.bool_)
    leg[0] = "fold" in v.legal and v.to_call > 0
    leg[1] = True
    leg[2] = can_raise and not jam and L < MAX_SMALL[band] and cur < raise_to < 0.5 * eff
    leg[3] = can_raise and not jam and eff > cur and not (band == 3 and L == 0)
    cls = int(K.COMBO_CLASS[X.CIDX[v.hole[0], v.hole[1]]])
    probs = np.zeros((1, X.NPA))
    X.policy_rows(T.QP, X.pre_row(band, sit, label[v.seat], ag, cls), 1, leg, X.NPA, T.tau_pre, T.trim, probs)
    a = _pick(probs[0], X.NPA, c.rng.random())
    if a == 0:
        return ("check", None) if "check" in v.legal else ("fold", None)
    if a == 1:
        return ("check", None) if "check" in v.legal else ("call", None)
    if a == 2:
        return "raise", int(raise_to * bb)
    return "raise", int(v.legal["raise"][1])


# ------------------------------------------------------------------------------
def postflop(c, ranges, sc):
    v, g = c.v, c.game
    if T is None or g.M != K.COMBOS.shape[0] or v.street < 1 or v.hero is None:
        return None
    opps = [s for s in v.alive if s != v.seat and v.players[s].in_hand]
    if len(opps) != 1 or opps[0] not in ranges:
        return None
    opp = v.players[opps[0]]
    hero = v.hero
    bb = float(v.bb)
    w = np.ascontiguousarray(ranges[opps[0]] * g.live(g.dead_mask(v.hole)), dtype=np.float64)
    sc = np.ascontiguousarray(sc, dtype=np.float64)
    hk = int(X.CIDX[v.hole[0], v.hole[1]])
    if sc[hk] < 0 or w.sum() <= 0:
        return None
    pct = np.zeros(g.M)
    X.pct_vs(np.argsort(sc), sc, w, pct)
    board = np.zeros(5, np.int64)
    nb = len(v.board)
    board[:nb] = v.board
    ec = int(X.eqr_classes(K.COMBOS[hk:hk + 1], board, nb, np.int64(0), K.POP, K.STRAIGHT, T.kick)[0])
    bucket = int(X.s_bucket(pct[hk])) * X.NE + max(ec, 0)
    pos = 1 if V.hero_in_position(v) else 0
    pot = v.pot / bb
    to_call = v.to_call / bb
    eff = min(hero.stack, opp.stack + v.to_call) / bb
    here = v.acts[v.street]
    raises = [a for a in here if a.kind == "raise"]
    nr = len(raises)
    last_inc = (raises[-1].by / bb) if raises else 0.0
    lr = _last_raiser(v.acts[v.street - 1])
    ctx = 2 if lr is None else (0 if lr == v.seat else 1)
    fac = int(X.fac_code(to_call, pot, nr, opp.stack <= 0))
    cell = int(X.cell_index(v.street - 1, pos, int(X.spr_bucket(eff / max(pot, 1e-9))), fac, ctx, X.ANY, X.ANY))
    amt = np.zeros(X.NA)
    leg = np.zeros(X.NA, np.bool_)
    X.legal_post(pot, to_call, eff, nr, last_inc, 1.0, amt, leg)
    if "raise" not in v.legal:
        leg[2:] = False
    leg[0] = "fold" in v.legal and v.to_call > 0
    probs = np.zeros((1, X.NA))
    X.policy_rows(T.QH, cell * X.NB + bucket, 1, leg, X.NA, T.tau_post, T.trim, probs)
    a = _pick(probs[0], X.NA, c.rng.random())
    if a == 0:
        return ("check", None) if "check" in v.legal else ("fold", None)
    if a == 1:
        return ("check", None) if "check" in v.legal else ("call", None)
    if a == 6:
        return "raise", int(v.legal["raise"][1])
    return "raise", int(hero.bet + amt[a] * bb)


# ------------------------------------------------------------------------------
def auction(c, P):
    v, g = c.v, c.game
    if T is None or g.M != K.COMBOS.shape[0] or len(v.hole) != 2 or len(v.auction_cards) != 5:
        return None
    balance = int((v.legal.get("bid", (0, 0))[1]) or 0)
    if balance <= 0:
        return (0, 0, 0, 0, 0)
    auc = np.asarray(v.auction_cards, np.int64)
    ranges, _ = Rg.build_ranges(v, g, c.opps, P, {}, {}, g.dead_mask(list(auc)))
    opps = [s for s in ranges if s != v.seat and v.players[s].in_hand]
    if not opps:
        return (0, 0, 0, 0, 0)
    opps = opps[:3]
    live = g.live(g.dead_mask(list(v.hole) + list(auc))).astype(np.float64)
    ws = []
    for s in opps:
        w = ranges[s] * live
        ws.append(w if w.sum() > 0 else live)
    cdfs = np.cumsum(np.vstack(ws), axis=1)
    hero = np.asarray(v.hole, np.int64)
    pool = np.zeros((POOL, 2), np.int64)
    eq = np.zeros((POOL + 1, 10))
    seed = V.stable_seed(v.table_id, v.hand_number, "auction")
    X.auction_pool(hero, auc, g.deck, K.COMBOS, cdfs, POOL, RUNOUTS, seed,
                   K.POP, K.HIGH, K.STRAIGHT, K.CATR, pool, eq)
    bb = float(v.bb)
    pot = v.pot / bb
    stacks_behind = [v.players[s].stack for s in opps]
    eff = min(v.hero.stack, max(stacks_behind)) / bb
    val = eq.copy()
    if eff > 0:          # chips behind: a flop is worth its bucket's measured value, not raw equity
        hands = np.vstack([hero[None, :], pool])
        lr = _last_raiser(v.acts[0])
        my_ctx = 2 if lr is None else (0 if lr == v.seat else 1)
        op_ctx = 2 if lr is None else (1 if lr == v.seat else 0)
        my_pos = 1 if V.hero_in_position(v) else 0
        sprb = int(X.spr_bucket(eff / max(pot, 1e-9)))
        board = np.zeros(5, np.int64)
        for t in range(10):
            board[:3] = auc[X.TRIPLES[t]]
            ec = X.eqr_classes(hands, board, 3, np.int64(0), K.POP, K.STRAIGHT, T.kick)
            for h in range(POOL + 1):
                pos, ctx = (my_pos, my_ctx) if h == 0 else (1 - my_pos, op_ctx)
                b = int(X.s_bucket(eq[h, t])) * X.NE + max(int(ec[h]), 0)
                val[h, t] = T.VFH[((pos * X.NCTX + ctx) * X.NSPR + sprb) * X.NB + b]
    # price of a flop-chip: the trained price, scaled by how many auctions the balance must last
    players = v.players_remaining or len(v.alive)
    hands_left = max(min(P["auction_table_hands"], v.hand_cap) - v.hand_number, 0)
    left = max(hands_left * min(0.9, P["auction_flop_rate"] * 8.0 / max(players, 2)), P["auction_min_left"])
    pace = min(max(left * P["auction_pay_target"], 0.2), 4.0)
    lam = T.lam * pace
    by_seat = v.flop_chips_by_seat
    rows = np.zeros((POOL, 5))
    marg = np.zeros(5)
    ob = np.zeros(5, np.int64)
    for j, s in enumerate(opps):
        bal = float(by_seat[s]) if s < len(by_seat) else 0.0
        if bal <= 0:
            continue
        lam_o = lam * T.bal_ref / bal
        for r in range(POOL):
            X.model_bid(val[1 + (r + 5 * j) % POOL], pot, T.kappa, bal, lam_o, marg, ob)
            rows[r] += ob
    out = np.zeros(5, np.int64)
    cand = np.zeros((16, 5), np.int64)
    X.best_bid(val[0], pot, float(balance), lam * T.bal_ref / balance, rows, np.ones(POOL), cand, out)
    return tuple(int(x) for x in out)


def warm(g):
    """Compile every kernel the hooks use (called once at import)."""
    if T is None:
        return
    M = K.COMBOS.shape[0]
    sc = np.linspace(0.0, 1.0, M)
    w = np.ones(M)
    pct = np.zeros(M)
    X.pct_vs(np.argsort(sc), sc, w, pct)
    board = np.array([0, 5, 10, 0, 0], np.int64)
    X.eqr_classes(K.COMBOS[:3], board, 3, np.int64(0), K.POP, K.STRAIGHT, T.kick)
    X.s_bucket(0.5), X.spr_bucket(3.0), X.fac_code(1.0, 3.0, 1, False), X.band_of(30.0)
    X.cell_index(0, 0, 0, 0, 0, X.ANY, X.ANY), X.pre_row(0, 0, 0, 0, 0)
    amt = np.zeros(X.NA)
    leg = np.zeros(X.NA, np.bool_)
    X.legal_post(3.0, 1.0, 20.0, 1, 1.0, 1.0, amt, leg)
    X.policy_rows(T.QH, 0, 1, leg, X.NA, T.tau_post, T.trim, np.zeros((1, X.NA)))
    X.policy_rows(T.QP, 0, 1, np.ones(X.NPA, np.bool_), X.NPA, T.tau_pre, T.trim, np.zeros((1, X.NPA)))
    pool = np.zeros((POOL, 2), np.int64)
    eq = np.zeros((POOL + 1, 10))
    cdfs = np.cumsum(np.ones((1, M)), axis=1)
    X.auction_pool(np.array([50, 51], np.int64), np.array([0, 5, 10, 15, 20], np.int64), g.deck, K.COMBOS,
                   cdfs, POOL, RUNOUTS, 1, K.POP, K.HIGH, K.STRAIGHT, K.CATR, pool, eq)
    ob = np.zeros(5, np.int64)
    X.model_bid(eq[1], 6.0, T.kappa, 500.0, T.lam, np.zeros(5), ob)
    X.best_bid(eq[0], 6.0, 500.0, T.lam, np.zeros((POOL, 5)), np.ones(POOL), np.zeros((16, 5), np.int64), ob)
