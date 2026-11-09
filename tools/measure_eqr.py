"""Measure equity realization (EQR) of the bot by simulation.

For every heads-up street start (flop, turn) that one of OUR seats reaches
with chips behind on both sides:
  X   = our chips at the end of the hand - our chips at the start of the street
  eq  = our exact all-in equity vs the opponent's ACTUAL hand (the simulator
        knows every hole card), enumerating the remaining board cards
  pot = pot at the start of the street
  s   = our showdown share on the actual runout (randomly completed if the
        hand ended early), used as a control variate: E[pot * s] = pot * eq
  R_bucket = sum(X - pot*s + pot*eq) / sum(pot*eq)

Preflop EQR: same, measured from the flop on, for heads-up flops after a
preflop raise and a call, with eq = preflop all-in equity (Monte Carlo).

Buckets and their definitions live in bot/pb/eqr.py (shared with the bot).
Sparse buckets are shrunk toward their parent (street, position) average with
a prior count of ``--prior-n``.

Usage:  python3 tools/measure_eqr.py --tables 2000 --procs 8
Output: bot/data/eqr_<variant name>.json  (+ a printed table)
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "stub_sdk"), os.path.join(ROOT, "bot"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from numba import njit  # noqa: E402

from pb import kernels as K  # noqa: E402

LINEUPS = {
    "field": ["C", "nit", "station", "maniac", "jammer", "minraiser", "random", "checkfold"],
    "lag": ["C", "maniac:0.35", "maniac:0.5", "nit", "station", "jammer", "maniac:0.2", "nit"],
    "self4": ["C", "C", "C", "C", "nit", "station", "maniac:0.35", "jammer"],
    "self8": ["C"] * 8,
    "strong7": ["C", "tag", "lag", "nashicm", "valuereg", "adaptive", "balanced", "bayes"],
    "strongmix": ["C", "C", "tag", "lag", "nashicm", "valuereg", "adaptive", "bayes"],
}


# ------------------------------------------------------------------ equity kernels
@njit(cache=False)
def exact_equity(h, v, H, board, nb, deck, board_total, rule, POP, HIGH, STRAIGHT, CATR):
    """Hero's equity vs one known hand, enumerating up to 2 remaining board cards."""
    used = np.int64(0)
    for i in range(H):
        used |= np.int64(1) << h[i]
        used |= np.int64(1) << v[i]
    for i in range(nb):
        used |= np.int64(1) << board[i]
    rem = np.zeros(deck.shape[0], np.int64)
    nr = 0
    for c in deck:
        if not (used >> c) & 1:
            rem[nr] = c
            nr += 1
    fb = np.zeros(board_total, np.int64)
    for i in range(nb):
        fb[i] = board[i]
    buf = np.zeros(K.MAXC, np.int64)
    need = board_total - nb
    win = 0.0
    cnt = 0.0
    if need == 0:
        a = K.eval_rule(h, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
        b = K.eval_rule(v, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
        return 1.0 if a > b else (0.5 if a == b else 0.0)
    if need == 1:
        for i in range(nr):
            fb[nb] = rem[i]
            a = K.eval_rule(h, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            b = K.eval_rule(v, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            win += 1.0 if a > b else (0.5 if a == b else 0.0)
            cnt += 1.0
        return win / cnt
    for i in range(nr):
        for j in range(i + 1, nr):
            fb[nb] = rem[i]
            fb[nb + 1] = rem[j]
            a = K.eval_rule(h, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            b = K.eval_rule(v, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            win += 1.0 if a > b else (0.5 if a == b else 0.0)
            cnt += 1.0
    return win / cnt


@njit(cache=False)
def mc_equity_known(h, v, H, board, nb, deck, board_total, n, seed, rule, POP, HIGH, STRAIGHT, CATR):
    """Monte Carlo equity vs one known hand (used preflop: 5 cards to come)."""
    np.random.seed(seed)
    used = np.int64(0)
    for i in range(H):
        used |= np.int64(1) << h[i]
        used |= np.int64(1) << v[i]
    for i in range(nb):
        used |= np.int64(1) << board[i]
    fb = np.zeros(board_total, np.int64)
    buf = np.zeros(K.MAXC, np.int64)
    nd = deck.shape[0]
    win = 0.0
    for s in range(n):
        m = used
        for i in range(nb):
            fb[i] = board[i]
        for i in range(nb, board_total):
            while True:
                c = deck[np.random.randint(0, nd)]
                if not (m >> c) & 1:
                    break
            m |= np.int64(1) << c
            fb[i] = c
        a = K.eval_rule(h, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
        b = K.eval_rule(v, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
        win += 1.0 if a > b else (0.5 if a == b else 0.0)
    return win / n


# ------------------------------------------------------------------ hand parsing
def _post_order(rec):
    """Seats in postflop order (first to act first) among players dealt in."""
    dealt = sorted(s for s in range(len(rec.stacks_start)) if rec.stacks_start[s] > 0)
    btn = rec.button
    after = [s for s in dealt if s > btn] + [s for s in dealt if s <= btn]
    return after


def measure_hand(rec, ours, game, rng):
    """Rows for each of our seats: (post_rows, pre_rows).

    post row: (street, ip, made_b, draw, spr_b, X, pot, eq, s)
    pre row:  (ip, key, X, pot, eq, s)
    """
    from pb import eqr as E
    from pb.view import cards_int

    hole = {int(s): cards_int(c) for s, c in rec.hole_cards.items()}
    board_all = cards_int(rec.board)
    contrib = defaultdict(int)
    round_bet = defaultdict(int)
    folded = set()
    street = 0
    n_raises_pre = 0
    snaps = {}  # street -> (pot, contrib copy, in_hand)
    for e in rec.events:
        k = e.kind
        s = e.seat
        if k in ("ante", "blind"):
            contrib[s] += e.amount
            if k == "blind":
                round_bet[s] += e.amount
        elif k == "board":
            street += 1
            round_bet = defaultdict(int)
            in_hand = [x for x in hole if x not in folded]
            snaps[street] = (sum(contrib.values()), dict(contrib), in_hand)
        elif k == "call":
            contrib[s] += e.amount
            round_bet[s] += e.amount
        elif k == "raise":
            inc = e.amount - round_bet[s]
            contrib[s] += inc
            round_bet[s] = e.amount
            if street == 0:
                n_raises_pre += 1
        elif k == "fold":
            folded.add(s)
    order = _post_order(rec)
    omaha = game.rule == K.RULE_OMAHA
    post_rows, pre_rows = [], []
    for st in (1, 2):
        if st not in snaps:
            continue
        pot, cb, in_hand = snaps[st]
        if len(in_hand) != 2:
            continue
        for me in in_hand:
            if me not in ours:
                continue
            vil = in_hand[0] if in_hand[1] == me else in_hand[1]
            my_behind = rec.stacks_start[me] - cb.get(me, 0)
            v_behind = rec.stacks_start[vil] - cb.get(vil, 0)
            if my_behind <= 0 or v_behind <= 0 or pot <= 0:
                continue
            nb = 3 if st == 1 else 4
            if len(board_all) < nb:
                continue
            board = board_all[:nb]
            h = np.array(hole[me], np.int64)
            v = np.array(hole[vil], np.int64)
            eq = exact_equity(h, v, game.H, np.array(board, np.int64), nb, game.deck, game.board_total,
                              game.rule, game.POP, game.HIGH, game.STRAIGHT, game.CATR)
            full = _complete(board_all, hole[me] + hole[vil], game, rng)
            s_res = exact_equity(h, v, game.H, np.array(full, np.int64), game.board_total, game.deck,
                                 game.board_total, game.rule, game.POP, game.HIGH, game.STRAIGHT, game.CATR)
            X = rec.stacks_end[me] - my_behind
            ip = order.index(me) > order.index(vil)
            made = E.hero_made_pct(game, hole[me], board)
            draw = E.has_draw(hole[me], board, omaha)
            spr = min(my_behind, v_behind) / pot
            post_rows.append((st, int(ip), E.made_bucket(made), int(draw), E.spr_bucket(spr),
                              float(X), float(pot), float(eq), float(s_res)))
            if st == 1 and n_raises_pre >= 1:
                eq_pre = mc_equity_known(h, v, game.H, np.zeros(0, np.int64), 0, game.deck, game.board_total,
                                         2000, int(rng.integers(1, 2**31 - 1)), game.rule,
                                         game.POP, game.HIGH, game.STRAIGHT, game.CATR)
                pre_rows.append((int(ip), E.pre_key(hole[me]), float(X), float(pot), float(eq_pre), float(s_res)))
    return post_rows, pre_rows


def _complete(board, dead, game, rng):
    """The actual board, completed at random if the hand ended early."""
    if len(board) >= game.board_total:
        return list(board[: game.board_total])
    used = set(board) | set(dead)
    rest = [c for c in game.deck.tolist() if c not in used]
    extra = rng.choice(rest, game.board_total - len(board), replace=False).tolist()
    return list(board) + [int(x) for x in extra]


# ------------------------------------------------------------------ workers
_W = {}


def _init_worker(fixed_samples):
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import agent as botmod
    from pb import brain, settings
    from sim import bots, engine
    P = copy.deepcopy(settings.P)
    if fixed_samples:
        P["mc_fixed_samples"] = fixed_samples
    _W.update(botmod=botmod, brain=brain, bots=bots, engine=engine, P=P)
    # compile the measurement kernels once
    g = brain.GAME
    h = g.deck[:g.H].copy()
    v = g.deck[g.H:2 * g.H].copy()
    b = g.deck[2 * g.H:2 * g.H + 3].copy()
    exact_equity(h, v, g.H, b, 3, g.deck, g.board_total, g.rule, g.POP, g.HIGH, g.STRAIGHT, g.CATR)
    mc_equity_known(h, v, g.H, b[:0], 0, g.deck, g.board_total, 10, 1, g.rule, g.POP, g.HIGH, g.STRAIGHT, g.CATR)


def _agent(spec, seed):
    if spec == "C":
        botmod, P = _W["botmod"], _W["P"]

        class Ours(botmod.Agent):
            def act(self, state, memory, _P=P):
                import poker_agent_sdk
                return _W["brain"].decide(state, memory, poker_agent_sdk, None, params=_P)
        return Ours()
    name, _, arg = spec.partition(":")
    if name in _W["bots"].BUILTIN_BOTS:
        cls = _W["bots"].BUILTIN_BOTS[name]
    else:
        from sim import strong_bots
        cls = strong_bots.STRONG_BOTS[name]
    return cls(seed=seed, aggression=float(arg)) if arg else cls(seed=seed)


def _play(job):
    lineup_name, seed = job
    lineup = LINEUPS[lineup_name]
    rot = seed % len(lineup)
    specs = lineup[rot:] + lineup[:rot]
    agents = [_agent(s, seed * 17 + i) for i, s in enumerate(specs)]
    res = _W["engine"].play_table(agents, seed=seed, timing=False, check_invariants=False,
                                  record_hands=True)
    ours = {i for i, s in enumerate(specs) if s == "C"}
    rng = np.random.default_rng(seed)
    post, pre = [], []
    for rec in res.hands or ():
        a, b = measure_hand(rec, ours, _W["brain"].GAME, rng)
        post.extend(a)
        pre.extend(b)
    return post, pre


# ------------------------------------------------------------------ aggregation
def _ratio(rows_X, rows_pot, rows_eq, rows_s):
    X = np.asarray(rows_X)
    pot = np.asarray(rows_pot)
    eq = np.asarray(rows_eq)
    s = np.asarray(rows_s)
    d = pot * eq
    y = X - pot * s + pot * eq
    D = d.sum()
    if D <= 0:
        return None, None
    R = y.sum() / D
    se = float(np.sqrt(((y - R * d) ** 2).sum()) / D) if len(y) > 1 else None
    return float(R), se


def aggregate(post, pre, prior_n):
    from pb import eqr as E
    out = {"post": {}, "pre": {}, "parent": {}}
    groups = defaultdict(list)
    parents = defaultdict(list)
    for r in post:
        st, ip, mb, dr, sb = r[:5]
        groups[E.post_key(st, ip, mb, dr, sb)].append(r)
        parents[f"{st}|{ip}"].append(r)
    par_R = {}
    for k, rows in parents.items():
        R, se = _ratio([r[5] for r in rows], [r[6] for r in rows], [r[7] for r in rows], [r[8] for r in rows])
        par_R[k] = R
        out["parent"][k] = {"R": R, "se": se, "n": len(rows)}
    for k, rows in sorted(groups.items()):
        R, se = _ratio([r[5] for r in rows], [r[6] for r in rows], [r[7] for r in rows], [r[8] for r in rows])
        st, ip = k.split("|")[:2]
        pr = par_R.get(f"{st}|{ip}") or 1.0
        n = len(rows)
        Rs = (n * R + prior_n * pr) / (n + prior_n) if R is not None else pr
        out["post"][k] = {"R": round(float(np.clip(Rs, 0.3, 2.0)), 4), "R_raw": R, "se": se, "n": n}
    pgroups = defaultdict(list)
    pparent = defaultdict(list)
    for r in pre:
        pgroups[E.pre_bucket_key(r[0], r[1])].append(r)
        pparent[str(r[0])].append(r)
    ppar = {}
    for k, rows in pparent.items():
        R, se = _ratio([r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows])
        ppar[k] = R
        out["parent"][f"pre|{k}"] = {"R": R, "se": se, "n": len(rows)}
    for k, rows in sorted(pgroups.items()):
        R, se = _ratio([r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows])
        pr = ppar.get(k.split("|")[0]) or 1.0
        n = len(rows)
        Rs = (n * R + prior_n * pr) / (n + prior_n) if R is not None else pr
        out["pre"][k] = {"R": round(float(np.clip(Rs, 0.3, 2.0)), 4), "R_raw": R, "se": se, "n": n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=2000, help="total tables (split across lineups)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--prior-n", type=float, default=30.0)
    ap.add_argument("--fixed-samples", type=int, default=250, help="0 = adaptive (time-based)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lineups", default=",".join(LINEUPS),
                    help="comma list of lineups to measure in (" + ", ".join(LINEUPS) + ")")
    args = ap.parse_args()
    from pb import settings
    name = settings.VARIANT.get("name", "holdem")
    out_path = args.out or os.path.join(ROOT, "bot", "data", f"eqr_{name}.json")
    names = [x for x in args.lineups.split(",") if x in LINEUPS]
    per = max(1, args.tables // len(names))
    jobs = [(ln, args.seed * 1_000_000 + i * 7 + j) for j, ln in enumerate(names) for i in range(per)]
    t = time.time()
    ctx = mp.get_context("spawn")
    post, pre = [], []
    with ctx.Pool(args.procs, initializer=_init_worker, initargs=(args.fixed_samples,)) as pool:
        for a, b in pool.imap_unordered(_play, jobs, chunksize=2):
            post.extend(a)
            pre.extend(b)
    table = aggregate(post, pre, args.prior_n)
    table["meta"] = {"variant": name, "tables": len(jobs), "post_samples": len(post), "pre_samples": len(pre),
                     "seconds": round(time.time() - t, 1), "lineups": names, "prior_n": args.prior_n,
                     "fixed_samples": args.fixed_samples}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(table, fh, indent=1)
    print(f"{len(jobs)} tables, {len(post)} postflop + {len(pre)} preflop samples in {time.time() - t:.0f}s -> {out_path}")
    print("parents:", json.dumps(table["parent"], indent=1))
    print("street|ip|made|draw|spr     n     R(shrunk)  R_raw   se")
    for k, rec in table["post"].items():
        raw = f"{rec['R_raw']:.3f}" if rec["R_raw"] is not None else "  -  "
        se = f"{rec['se']:.3f}" if rec["se"] is not None else "  -  "
        print(f"{k:24s} {rec['n']:6d}   {rec['R']:.3f}     {raw}   {se}")
    print("preflop ip|features         n     R(shrunk)  R_raw   se")
    for k, rec in table["pre"].items():
        raw = f"{rec['R_raw']:.3f}" if rec["R_raw"] is not None else "  -  "
        se = f"{rec['se']:.3f}" if rec["se"] is not None else "  -  "
        print(f"{k:24s} {rec['n']:6d}   {rec['R']:.3f}     {raw}   {se}")


if __name__ == "__main__":
    main()
