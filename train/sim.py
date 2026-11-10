"""Self-play simulator for the trainer (Numba).

One hand = preflop (8 seats, equal stacks, chip EV) -> sealed flop auction ->
heads-up flop / turn / river with the abstract bet sizes. Every decision is
looked up in the previous generation's EV tables. EVs for the next generation
are measured by letting one player deviate once per hand (a uniformly random
legal action) and recording what that decision earned.

Units: big blinds for chips, flop-chips for the auction.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from pb import absx as X
from pb import kernels as K

U = K.COMBOS
CLS = K.COMBO_CLASS
CNT = K.CLASS_COUNTS.astype(np.float64)
POP, HIGH, STRAIGHT, CATR = K.POP, K.HIGH, K.STRAIGHT, K.CATR
NA, NB, NPA = X.NA, X.NB, X.NPA
MAXREC = 64

# indices into the parameter vector
(P_TAU_POST, P_TAU_PRE, P_TRIM, P_EPS_POST, P_EPS_PRE, P_ONPOLICY, P_LAMBDA, P_KAPPA,
 P_KICK, P_NUT_TOP, P_BAL_LO, P_BAL_HI, P_BAL_REF, P_OPP_ROWS, P_BAND_W0, P_BAND_W1,
 P_BAND_W2, P_BAND_W3, P_PAY_UPDATE, P_N) = range(20)

# preflop sizes: open raise by band, re-raise multiple by band and level
OPEN_TO = np.array([0.0, 2.0, 2.2, 2.5])
RERAISE_X = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 3.3, 2.3]])
MAX_SMALL = np.array([0, 1, 2, 3])        # small raises allowed per hand, by band
BAND_LO = np.array([5.0, 10.0, 20.0, 40.0])
BAND_HI = np.array([10.0, 20.0, 40.0, 80.0])


@njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


@njit(cache=True)
def deal(deck, n):
    for i in range(n):
        j = i + np.random.randint(0, 52 - i)
        t = deck[i]
        deck[i] = deck[j]
        deck[j] = t


@njit(cache=True)
def sample_action(p, na):
    u = np.random.random()
    c = 0.0
    last = 0
    for a in range(na):
        if p[a] > 0.0:
            c += p[a]
            last = a
            if u < c:
                return a
    return last


@njit(cache=True)
def uniform_legal(leg, na):
    n = 0
    for a in range(na):
        if leg[a]:
            n += 1
    j = np.random.randint(0, n)
    for a in range(na):
        if leg[a]:
            if j == 0:
                return a
            j -= 1
    return 1


# ------------------------------------------------------------------------------
# Preflop
# ------------------------------------------------------------------------------
@njit(cache=True)
def play_preflop(stack, cls, QP, EQP, E169, P, explorer, rngs, out, rec_row, rec_a, rec_stk, rec_n):
    """Play the preflop round for 8 seats holding classes ``cls``.

    ``rngs[seat, class]`` ends as the weight of each class in that seat's range
    (product of the policy's probabilities of the actions it took).
    out: [0] kind (0 one player left, 1 all-in showdown, 2 heads-up flop, 3 multiway flop)
         [1] pot  [2] stack behind  [3] out-of-position seat  [4] in-position seat
         [5] last raiser (-1 none)  [6] pot type  [7] players in  [8..15] net result
    Returns the number of decisions recorded.
    """
    band = X.band_of(stack)
    tau = P[P_TAU_PRE]
    trim = P[P_TRIM]
    bet = np.zeros(8)
    alive = np.ones(8, np.int64)
    allin = np.zeros(8, np.int64)
    vol = np.zeros(8, np.int64)
    bet[6] = 0.5
    bet[7] = 1.0
    cur = 1.0
    L = 0
    nlimp = 0
    aggr = -1
    jam_live = False
    n_alive = 8
    to_act = 8
    i = 0
    leg = np.zeros(NPA, np.bool_)
    probs = np.zeros((X.NCLS, NPA))
    nrec = 0
    deviated = False
    for s in range(8):
        for c in range(X.NCLS):
            rngs[s, c] = 1.0
    while to_act > 0 and n_alive > 1:
        if alive[i] == 0 or allin[i] == 1:
            i = (i + 1) % 8
            continue
        to_call = cur - bet[i]
        if jam_live:
            sit = 6
        elif L == 0:
            sit = 0 if nlimp == 0 else 1
        elif L == 1:
            sit = 3 if vol[i] == 1 else 2
        else:
            sit = 4 if vol[i] == 2 else 5
        ag = 0 if aggr < 0 else (1 if aggr <= 2 else (2 if aggr <= 5 else 3))
        if L == 0:
            raise_to = OPEN_TO[band] + nlimp
        else:
            raise_to = RERAISE_X[band, min(L, 2)] * cur
        leg[0] = to_call > 0.0
        leg[1] = True
        leg[2] = (not jam_live) and L < MAX_SMALL[band] and raise_to < 0.5 * stack and raise_to > cur
        leg[3] = (not jam_live) and stack > cur and not (band == 3 and L == 0)
        base = X.pre_row(band, sit, i, ag, 0)
        X.policy_rows(QP, base, X.NCLS, leg, NPA, tau, trim, probs)
        a = sample_action(probs[cls[i]], NPA)
        dev = False
        if explorer == i and not deviated and np.random.random() < P[P_EPS_PRE]:
            a = uniform_legal(leg, NPA)
            deviated = True
            dev = True
        if (explorer < 0 or dev) and nrec < MAXREC:
            rec_row[nrec] = base + cls[i]
            rec_a[nrec] = a
            rec_stk[nrec] = stack - bet[i]
            rec_n[nrec] = i
            nrec += 1
        if not dev:
            for c in range(X.NCLS):
                rngs[i, c] *= probs[c, a]
        to_act -= 1
        if a == 0:
            alive[i] = 0
            n_alive -= 1
        elif a == 1:
            if L == 0 and to_call > 0.0:
                nlimp += 1
            bet[i] = min(cur, stack)
            if bet[i] >= stack:
                allin[i] = 1
            if vol[i] < 1:
                vol[i] = 1
        else:
            if a == 2:
                bet[i] = raise_to
            else:
                bet[i] = stack
                allin[i] = 1
                jam_live = True
            cur = bet[i]
            L += 1
            aggr = i
            vol[i] = 2
            to_act = 0
            for s in range(8):
                if s != i and alive[s] == 1 and allin[s] == 0:
                    to_act += 1
        i = (i + 1) % 8
    pot = 0.0
    for s in range(8):
        pot += bet[s]
        out[8 + s] = -bet[s]
    out[1] = pot
    out[5] = aggr
    pt = 0 if L == 0 else (1 if L == 1 else 2)
    out[6] = pt
    out[7] = n_alive
    if n_alive == 1:
        out[0] = 0
        for s in range(8):
            if alive[s] == 1:
                out[8 + s] += pot
        return nrec
    # who is last to act after the flop (the blinds act first)
    ip = -1
    oop = -1
    for s in range(8):
        if alive[s] == 1:
            if ip < 0 or (s + 2) % 8 > (ip + 2) % 8:
                ip = s
            if oop < 0 or (s + 2) % 8 < (oop + 2) % 8:
                oop = s
    out[3] = oop
    out[4] = ip
    any_allin = False
    for s in range(8):
        if alive[s] == 1 and allin[s] == 1:
            any_allin = True
    # showdown shares from the class-vs-class equities (product rule when multiway)
    tot = 0.0
    share = np.zeros(8)
    for s in range(8):
        if alive[s] == 1:
            e = 1.0
            for t in range(8):
                if t != s and alive[t] == 1:
                    e *= E169[cls[s], cls[t]]
            share[s] = e
            tot += e
    for s in range(8):
        share[s] /= tot
    if any_allin:
        out[0] = 1
        for s in range(8):
            if alive[s] == 1:
                out[8 + s] += share[s] * pot
        return nrec
    out[0] = 2 if n_alive == 2 else 3
    out[2] = stack - cur
    # equity x equity realization, re-centred so the shares add up to the pot
    ssum = 0.0
    for s in range(8):
        if alive[s] == 1:
            share[s] *= EQP[band, pt, 1 if s == ip else 0, cls[s]]
            ssum += share[s]
    for s in range(8):
        if alive[s] == 1:
            out[8 + s] += (share[s] - (ssum - 1.0) / n_alive) * pot
    return nrec


# ------------------------------------------------------------------------------
# Auction (heads-up)
# ------------------------------------------------------------------------------
@njit(cache=True)
def run_auction(holes, w, auc, pot, stack, ctx, band, bal, VFH, PAYL, coef, P, flop, stats, bk_sel):
    """Both players value the 10 flops, bid, and the flop is selected.

    w[p]: combo weights of player p's range (updated here by what p paid).
    Writes the selected flop (auction order) to ``flop`` and returns the dead mask.
    stats[p] = [paid, bid total, willingness in chips, value gained over the
    default flop, balance, pay bucket, flop bucket of the hand]."""
    dead5 = np.int64(0)
    for i in range(5):
        dead5 |= np.int64(1) << auc[i]
    M = U.shape[0]
    val = np.zeros((2, 10, M), np.float32)
    bks = np.zeros((2, 10, M), np.int8)
    board = np.zeros(5, np.int64)
    pct = np.zeros(M)
    bk = np.zeros(M, np.int8)
    sprb = X.spr_bucket(stack / pot)
    for t in range(10):
        for i in range(3):
            board[i] = auc[X.TRIPLES[t, i]]
        sc = X.board_scores(U, board, 3, dead5, coef, POP, HIGH, STRAIGHT, CATR)
        ec = X.eqr_classes(U, board, 3, dead5, POP, STRAIGHT, int(P[P_KICK]))
        order = np.argsort(sc)
        for p in range(2):
            X.pct_vs(order, sc, w[1 - p], pct)
            X.buckets_from(pct, ec, bk)
            base = ((p * X.NCTX + ctx[p]) * X.NSPR + sprb) * NB
            for k in range(M):
                bks[p, t, k] = bk[k]
                if bk[k] >= 0:
                    val[p, t, k] = VFH[base + bk[k]]
    R = int(P[P_OPP_ROWS])
    bids = np.zeros((2, 5), np.int64)
    opp_bids = np.zeros((R, 5))
    opp_w = np.ones(R)
    cand = np.zeros((16, 5), np.int64)
    marg = np.zeros(5)
    ob = np.zeros(5, np.int64)
    v = np.zeros(10)
    vo = np.zeros(10)
    cdf = np.zeros(M)
    hk = np.zeros(2, np.int64)
    for p in range(2):
        hk[p] = X.CIDX[holes[p, 0], holes[p, 1]]
    for p in range(2):
        q = 1 - p
        lam_p = P[P_LAMBDA] * P[P_BAL_REF] / max(bal[p], 1.0)
        lam_q = P[P_LAMBDA] * P[P_BAL_REF] / max(bal[q], 1.0)
        for t in range(10):
            v[t] = val[p, t, hk[p]]
        c = 0.0
        hm = (np.int64(1) << holes[p, 0]) | (np.int64(1) << holes[p, 1])
        for k in range(M):
            if bks[q, 0, k] >= 0 and not ((hm >> U[k, 0]) & 1) and not ((hm >> U[k, 1]) & 1):
                c += w[q][k]
            cdf[k] = c
        for r in range(R):
            k = min(np.searchsorted(cdf, np.random.random() * c), M - 1)
            for t in range(10):
                vo[t] = val[q, t, k]
            X.model_bid(vo, pot, P[P_KAPPA], bal[q], lam_q, marg, ob)
            for cc in range(5):
                opp_bids[r, cc] = ob[cc]
        gain, spent = X.best_bid(v, pot, bal[p], lam_p, opp_bids, opp_w, cand, bids[p])
        _, wl = X.willingness(v, pot, lam_p)
        stats[p, 1] = spent
        stats[p, 2] = wl
    tot = np.zeros(5)
    for cc in range(5):
        tot[cc] = bids[0, cc] + bids[1, cc]
    t = X.flop_of(tot)
    dead = dead5
    for i in range(3):
        flop[i] = auc[X.TRIPLES[t, i]]
    for p in range(2):
        paid = 0
        for i in range(3):
            paid += bids[p, X.TRIPLES[t, i]]
        pb = X.pay_bucket(paid, bal[p])
        stats[p, 0] = paid
        stats[p, 3] = val[p, t, hk[p]] - val[p, 0, hk[p]]
        stats[p, 4] = bal[p]
        stats[p, 5] = pb
        stats[p, 6] = bks[p, t, hk[p]]
        bk_sel[p] = bks[p, t, hk[p]]
        # what a seat paid narrows its range (not when it had nothing to bid with)
        if pb > 0 and P[P_PAY_UPDATE] > 0:
            base = ((band * 2 + p) * X.NPAY + pb) * NB
            s_new = 0.0
            for k in range(M):
                b = bks[p, t, k]
                if b >= 0:
                    s_new += w[p][k] * PAYL[base + b]
            if s_new > 0.0:
                for k in range(M):
                    b = bks[p, t, k]
                    w[p][k] = w[p][k] * PAYL[base + b] if b >= 0 else 0.0
    return dead


# ------------------------------------------------------------------------------
# Postflop (heads-up, player 0 out of position)
# ------------------------------------------------------------------------------
@njit(cache=True)
def play_post(holes, w, flop, dead, deck, dptr, pot0, stack0, pf_ctx, QH, coef, P, explorer,
              train_lo, train_hi, rec_row, rec_crow, rec_a, rec_pot, rec_stk, rec_pl, res):
    """Play flop, turn and river. Returns the number of decisions recorded.

    res: [0], [1] chips won or lost from the flop on (per player), [2], [3] the
    flop bucket of each player's hand, [4] SPR bucket at the flop, [5] streets seen."""
    M = U.shape[0]
    tau = P[P_TAU_POST]
    trim = P[P_TRIM]
    board = np.zeros(5, np.int64)
    for i in range(3):
        board[i] = flop[i]
    pot = pot0
    stack = np.array([stack0, stack0])
    ctx = np.array([pf_ctx[0], pf_ctx[1]])
    pct = np.zeros((2, M))
    bk = np.zeros((2, M), np.int8)
    amt = np.zeros(NA)
    leg = np.zeros(NA, np.bool_)
    probs = np.zeros((NB, NA))
    hk = np.array([X.CIDX[holes[0, 0], holes[0, 1]], X.CIDX[holes[1, 0], holes[1, 1]]])
    nrec = 0
    deviated = False
    winner = -1
    res[5] = 0
    dm = dead
    for i in range(3):
        dm |= np.int64(1) << board[i]
    for st in range(3):
        nb = 3 + st
        if st > 0:
            while True:
                c = deck[dptr]
                dptr += 1
                if not (dm >> c) & 1:
                    break
            board[nb - 1] = c
            dm |= np.int64(1) << c
        if stack[0] <= 0.0:
            continue                      # all-in: just run the board out
        res[5] = st + 1
        sc = X.board_scores(U, board, nb, dm, coef, POP, HIGH, STRAIGHT, CATR)
        ec = X.eqr_classes(U, board, nb, dm, POP, STRAIGHT, int(P[P_KICK]))
        order = np.argsort(sc)
        for p in range(2):
            for k in range(M):
                if sc[k] < 0.0:
                    w[p][k] = 0.0
        rv = np.array([X.ANY, X.ANY])
        nu = np.array([X.ANY, X.ANY])
        for p in range(2):
            X.pct_vs(order, sc, w[1 - p], pct[p])
            X.buckets_from(pct[p], ec, bk[p])
        if st == 0:
            for p in range(2):
                rv[p], nu[p] = X.range_stats(order, sc, w[p], w[1 - p], pct[p], P[P_NUT_TOP])
            res[2] = bk[0, hk[0]]
            res[3] = bk[1, hk[1]]
            res[4] = X.spr_bucket(stack0 / pot0)
        bet = np.zeros(2)
        actor = 0
        nr = 0
        last_inc = 0.0
        agg = -1
        checks = 0
        train = st >= train_lo and st <= train_hi
        while True:
            opp = 1 - actor
            to_call = bet[opp] - bet[actor]
            X.legal_post(pot, to_call, stack[actor], nr, last_inc, 1.0, amt, leg)
            fac = X.fac_code(to_call, pot, nr, stack[opp] <= 0.0)
            sprb = X.spr_bucket(stack[actor] / pot)
            cell = X.cell_index(st, actor, sprb, fac, ctx[actor], rv[actor], nu[actor])
            X.policy_rows(QH, cell * NB, NB, leg, NA, tau, trim, probs)
            hb = bk[actor, hk[actor]]
            a = sample_action(probs[hb], NA)
            dev = False
            if train and explorer == actor and not deviated and np.random.random() < P[P_EPS_POST]:
                a = uniform_legal(leg, NA)
                deviated = True
                dev = True
            if train and (explorer < 0 or dev) and nrec < MAXREC:
                rec_row[nrec] = cell * NB + hb
                rec_crow[nrec] = -1
                if st == 0:
                    rec_crow[nrec] = X.cell_index(st, actor, sprb, fac, ctx[actor], X.ANY, X.ANY) * NB + hb
                rec_a[nrec] = a
                rec_pot[nrec] = pot
                rec_stk[nrec] = stack[actor]
                rec_pl[nrec] = actor
                nrec += 1
            if not dev:
                s_new = 0.0
                for k in range(M):
                    if bk[actor, k] >= 0:
                        s_new += w[actor][k] * probs[bk[actor, k], a]
                if s_new > 0.0:
                    for k in range(M):
                        if bk[actor, k] >= 0:
                            w[actor][k] *= probs[bk[actor, k], a]
            if a == 0:
                winner = opp
                break
            if a == 1:
                stack[actor] -= amt[1]
                pot += amt[1]
                bet[actor] += amt[1]
                if to_call > 0.0:
                    break
                checks += 1
                if checks == 2:
                    break
            else:
                stack[actor] -= amt[a]
                pot += amt[a]
                bet[actor] += amt[a]
                last_inc = bet[actor] - bet[opp]
                nr += 1
                agg = actor
            actor = opp
        if winner >= 0:
            break
        if agg < 0:
            ctx[0] = 2
            ctx[1] = 2
        else:
            ctx[agg] = 0
            ctx[1 - agg] = 1
    if winner >= 0:
        stack[winner] += pot
    else:
        buf = np.zeros(K.MAXC, np.int64)
        v0 = K.eval_rule(holes[0], 2, board, 5, 0, buf, POP, HIGH, STRAIGHT, CATR)
        v1 = K.eval_rule(holes[1], 2, board, 5, 0, buf, POP, HIGH, STRAIGHT, CATR)
        if v0 > v1:
            stack[0] += pot
        elif v1 > v0:
            stack[1] += pot
        else:
            stack[0] += 0.5 * pot
            stack[1] += 0.5 * pot
    res[0] = stack[0] - stack0
    res[1] = stack[1] - stack0
    for i in range(nrec):
        rec_stk[i] = (stack[rec_pl[i]] - rec_stk[i]) / rec_pot[i]   # EV of the action, in pots
    return nrec


# ------------------------------------------------------------------------------
# Batch drivers
# ------------------------------------------------------------------------------
@njit(cache=True)
def pick_stack(P):
    u = np.random.random() * (P[P_BAND_W0] + P[P_BAND_W1] + P[P_BAND_W2] + P[P_BAND_W3])
    b = 3
    c = 0.0
    for i in range(4):
        c += P[P_BAND_W0 + i]
        if u < c:
            b = i
            break
    return BAND_LO[b] + np.random.random() * (BAND_HI[b] - BAND_LO[b])


@njit(cache=True)
def run_preflop(n, seed, QP, EQP, E169, P, QS, QN):
    """Preflop pass: measure the EV of every preflop action (in big blinds)."""
    np.random.seed(seed)
    deck = np.arange(52)
    cls = np.zeros(8, np.int64)
    rngs = np.zeros((8, X.NCLS))
    out = np.zeros(16)
    rec_row = np.zeros(MAXREC, np.int64)
    rec_a = np.zeros(MAXREC, np.int64)
    rec_stk = np.zeros(MAXREC)
    rec_n = np.zeros(MAXREC, np.int64)
    kinds = np.zeros(4)
    for _ in range(n):
        deal(deck, 16)
        for s in range(8):
            cls[s] = CLS[X.CIDX[deck[2 * s], deck[2 * s + 1]]]
        stack = pick_stack(P)
        explorer = -1 if np.random.random() < P[P_ONPOLICY] else np.random.randint(0, 8)
        nrec = play_preflop(stack, cls, QP, EQP, E169, P, explorer, rngs, out,
                            rec_row, rec_a, rec_stk, rec_n)
        kinds[int(out[0])] += 1
        for i in range(nrec):
            s = rec_n[i]
            # result of the hand from this decision on: chips at the end minus chips before acting
            g = (stack + out[8 + s]) - rec_stk[i]
            QS[rec_row[i], rec_a[i]] += g
            QN[rec_row[i], rec_a[i]] += 1.0
    return kinds


@njit(cache=True)
def run_post(n, seed, QP, EQP, E169, QH, VFH, PAYL, coef, P, train_lo, train_hi,
             QS, QN, VFS, VFN, EQS, EQE, EQN, PAYC, AST):
    """Postflop pass over ``n`` heads-up flops reached with the preflop policy.

    Streets train_lo..train_hi (0 flop, 1 turn, 2 river) have their action EVs
    measured. On hands with no deviation the flop values (VF), the preflop
    equity realization (EQ), the payment counts (PAYC) and auction statistics
    (AST) are recorded."""
    np.random.seed(seed)
    M = U.shape[0]
    deck = np.arange(52)
    cls = np.zeros(8, np.int64)
    rngs = np.zeros((8, X.NCLS))
    out = np.zeros(16)
    r1 = np.zeros(MAXREC, np.int64)
    r2 = np.zeros(MAXREC, np.int64)
    r3 = np.zeros(MAXREC)
    r4 = np.zeros(MAXREC, np.int64)
    rec_row = np.zeros(MAXREC, np.int64)
    rec_crow = np.zeros(MAXREC, np.int64)
    rec_a = np.zeros(MAXREC, np.int64)
    rec_pot = np.zeros(MAXREC)
    rec_stk = np.zeros(MAXREC)
    rec_pl = np.zeros(MAXREC, np.int64)
    holes = np.zeros((2, 2), np.int64)
    w = np.zeros((2, M))
    auc = np.zeros(5, np.int64)
    flop = np.zeros(3, np.int64)
    stats = np.zeros((2, 7))
    bk_sel = np.zeros(2, np.int64)
    res = np.zeros(6)
    bal = np.zeros(2)
    ctx = np.zeros(2, np.int64)
    seats = np.zeros(2, np.int64)
    done = 0
    tried = 0
    while done < n:
        tried += 1
        deal(deck, 52)
        for s in range(8):
            cls[s] = CLS[X.CIDX[deck[2 * s], deck[2 * s + 1]]]
        stack = pick_stack(P)
        band = X.band_of(stack)
        play_preflop(stack, cls, QP, EQP, E169, P, -1, rngs, out, r1, r2, r3, r4)
        if int(out[0]) != 2:
            continue
        done += 1
        seats[0] = int(out[3])
        seats[1] = int(out[4])
        pot = out[1]
        behind = out[2]
        aggr = int(out[5])
        pt = int(out[6])
        for p in range(2):
            s = seats[p]
            holes[p, 0] = deck[2 * s]
            holes[p, 1] = deck[2 * s + 1]
            ctx[p] = 2 if aggr < 0 else (0 if aggr == s else 1)
            bal[p] = np.floor(P[P_BAL_LO] + np.random.random() * (P[P_BAL_HI] - P[P_BAL_LO]))
            for k in range(M):
                w[p][k] = rngs[s, CLS[k]]
        # preflop equity against the range the opponent arrives with
        eq_pre = np.zeros(2)
        for p in range(2):
            q = seats[1 - p]
            num = 0.0
            den = 0.0
            for c in range(X.NCLS):
                x = rngs[q, c] * CNT[c]
                num += x * E169[cls[seats[p]], c]
                den += x
            eq_pre[p] = num / den if den > 0 else 0.5
        for i in range(5):
            auc[i] = deck[16 + i]
        dead = run_auction(holes, w, auc, pot, behind, ctx, band, bal, VFH, PAYL, coef, P,
                           flop, stats, bk_sel)
        explorer = -1 if np.random.random() < P[P_ONPOLICY] else np.random.randint(0, 2)
        nrec = play_post(holes, w, flop, dead, deck, 21, pot, behind, ctx, QH, coef, P, explorer,
                         train_lo, train_hi, rec_row, rec_crow, rec_a, rec_pot, rec_stk, rec_pl, res)
        for i in range(nrec):
            QS[rec_row[i], rec_a[i]] += rec_stk[i]
            QN[rec_row[i], rec_a[i]] += 1.0
            if rec_crow[i] >= 0:
                QS[rec_crow[i], rec_a[i]] += rec_stk[i]
                QN[rec_crow[i], rec_a[i]] += 1.0
        if explorer < 0:
            sprb = int(res[4])
            for p in range(2):
                share = res[p] / pot
                b = int(res[2 + p])
                i = ((p * X.NCTX + ctx[p]) * X.NSPR + sprb) * NB + b
                VFS[i] += share
                VFN[i] += 1.0
                j = ((band * X.NPT + pt) * 2 + p) * X.NCLS + cls[seats[p]]
                EQS[j] += share
                EQE[j] += eq_pre[p]
                EQN[j] += 1.0
                PAYC[((band * 2 + p) * X.NPAY + int(stats[p, 5])) * NB + int(stats[p, 6])] += 1.0
                AST[0] += 1.0
                AST[1] += stats[p, 0]            # chips paid
                AST[2] += stats[p, 1]            # chips bid
                AST[3] += max(stats[p, 2], 0.0)  # willingness
                AST[4] += stats[p, 3] * pot      # big blinds gained over the default flop
                AST[5] += stats[p, 0] / max(stats[p, 4], 1.0)
                AST[6] += 1.0 if stats[p, 0] <= 0 else 0.0
                AST[7 + int(stats[p, 5])] += 1.0
            AST[13 + band] += 1.0
            AST[17 + pt] += 1.0
            AST[20] += pot
            AST[21] += behind / pot
            AST[22 + int(res[5])] += 1.0
    AST[26] += tried
    AST[27] += done
    return done
