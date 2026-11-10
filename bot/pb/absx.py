"""Hand abstraction shared by the self-play trainer (train/) and the bot.

Postflop a hand is reduced to two factors:
  S  strength: the share of the opponent's range it beats on this board
     (fitted equity order on flop and turn, hand rank on the river), 10 buckets
  E  equity-realization class: 2 = nut draw, 1 = made hand at least top pair
     with a jack kicker, 0 = everything else (non-nut draws, weaker made hands)
Hands in the same (S, E) bucket are treated as the same hand.

A decision is looked up by a cell (street, position, SPR, what we face, who has
the initiative, and on the flop range-vs-range equity and nut advantage) plus
the bucket. Everything here is Numba so the trainer and the bot share one code
path. Hold'em (two hole cards) only.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from . import kernels as K

# ---- sizes of the abstraction ------------------------------------------------
NS = 10                 # strength buckets
NE = 3                  # equity-realization classes
NB = NS * NE
NA = 7                  # fold, check/call, bet 10/33/66/125% of the pot, all-in
FRACS = np.array([0.10, 0.33, 0.66, 1.25])
S_EDGES = np.array([0.15, 0.30, 0.45, 0.58, 0.70, 0.80, 0.88, 0.94, 0.98])
S_MID = np.array([0.075, 0.225, 0.375, 0.515, 0.64, 0.75, 0.84, 0.91, 0.96, 0.99])
NST, NPOS, NSPR, NFAC, NCTX, NRV, NNU = 3, 2, 5, 11, 3, 4, 4
ANY = 3                 # the "not split by this" slot of the RV and NU dimensions
NCELL = NST * NPOS * NSPR * NFAC * NCTX * NRV * NNU
NROW = NCELL * NB
SPR_EDGES = np.array([1.0, 2.5, 5.0, 10.0])
SPR_MID = np.array([0.6, 1.7, 3.6, 7.0, 15.0])
ALLIN_SPR = 5.0         # all-in is offered below this stack-to-pot ratio
MAX_RAISES = 3          # bets and raises per street before only call / fold / all-in remain

# ---- preflop -----------------------------------------------------------------
NBAND = 4               # effective stack: <10, 10-20, 20-40, 40+ big blinds
NSIT = 7                # unopened, vs limp, vs open, limped vs raise, raised vs re-raise, cold vs re-raise, vs jam
NSEAT = 8               # 0 = first to act ... 5 = button, 6 = small blind, 7 = big blind
NAG = 4                 # last raiser: none, seats 0-2, seats 3-5, blinds
NCLS = 169
NPA = 4                 # fold, call/limp/check, small raise, jam
NPROW = NBAND * NSIT * NSEAT * NAG * NCLS
NPT = 3                 # pot type: limped, single raised, re-raised

# ---- auction -----------------------------------------------------------------
NPAY = 6                # cannot bid, nothing, token (1-3), small (<=5%), medium (<=15%), large
TRIPLES = K.TRIPLES
TIDX = np.full((5, 5, 5), -1, np.int64)
for _i in range(10):
    TIDX[TRIPLES[_i, 0], TRIPLES[_i, 1], TRIPLES[_i, 2]] = _i

CIDX = np.full((52, 52), -1, np.int64)   # two cards -> index into kernels.COMBOS
for _k in range(K.COMBOS.shape[0]):
    CIDX[K.COMBOS[_k, 0], K.COMBOS[_k, 1]] = _k
    CIDX[K.COMBOS[_k, 1], K.COMBOS[_k, 0]] = _k


@njit(cache=False)
def band_of(stack_bb):
    if stack_bb < 10.0:
        return 0
    if stack_bb < 20.0:
        return 1
    if stack_bb < 40.0:
        return 2
    return 3


@njit(cache=False)
def s_bucket(p):
    for i in range(NS - 1):
        if p < S_EDGES[i]:
            return i
    return NS - 1


@njit(cache=False)
def spr_bucket(spr):
    for i in range(NSPR - 1):
        if spr < SPR_EDGES[i]:
            return i
    return NSPR - 1


@njit(cache=False)
def fac_code(to_call, pot, n_raises, opp_allin):
    """What we face: 0 nothing; else 1 + size class + 5 if it is a raise."""
    if to_call <= 0.0:
        return 0
    x = to_call / max(pot - to_call, 1e-9)
    if opp_allin:
        sz = 4
    elif x <= 0.2:
        sz = 0
    elif x <= 0.5:
        sz = 1
    elif x <= 0.9:
        sz = 2
    else:
        sz = 3
    return 1 + sz + (5 if n_raises >= 2 else 0)


@njit(cache=False)
def cell_index(st, pos, spr, fac, ctx, rv, nu):
    return (((((st * NPOS + pos) * NSPR + spr) * NFAC + fac) * NCTX + ctx) * NRV + rv) * NNU + nu


@njit(cache=False)
def pre_row(band, sit, seat, ag, cls):
    return (((band * NSIT + sit) * NSEAT + seat) * NAG + ag) * NCLS + cls


@njit(cache=False)
def pay_bucket(paid, balance):
    if balance <= 0:
        return 0
    if paid <= 0:
        return 1
    if paid <= 3:
        return 2
    f = paid / balance
    if f <= 0.05:
        return 3
    if f <= 0.15:
        return 4
    return 5


@njit(cache=False)
def eqr_classes(U, board, nb, dead, POP, STRAIGHT, kick_min):
    """Equity-realization class of every two-card hand on this board (-1 dead)."""
    M = U.shape[0]
    out = np.zeros(M, np.int8)
    bc = np.zeros(13, np.int64)
    bs = np.zeros(4, np.int64)
    bsm = np.zeros(4, np.int64)
    bm = 0
    top = -1
    for i in range(nb):
        r = board[i] >> 2
        s = board[i] & 3
        bc[r] += 1
        bs[s] += 1
        bsm[s] |= 1 << r
        bm |= 1 << r
        if r > top:
            top = r
    bstr = STRAIGHT[bm]
    for k in range(M):
        a = U[k, 0]
        b = U[k, 1]
        if ((dead >> a) & 1) or ((dead >> b) & 1):
            out[k] = -1
            continue
        ra = a >> 2
        rb = b >> 2
        sa = a & 3
        sb = b & 3
        full = bm | (1 << ra) | (1 << rb)
        strong = False
        nut = False
        for s in range(4):
            h = (1 if sa == s else 0) + (1 if sb == s else 0)
            if h == 0:
                continue
            if h + bs[s] >= 5:
                strong = True
            elif h + bs[s] == 4 and nb < 5:
                hr = 12
                while (bsm[s] >> hr) & 1:
                    hr -= 1
                if (sa == s and ra == hr) or (sb == s and rb == hr):
                    nut = True
        st = STRAIGHT[full]
        if st >= 0 and st > bstr:
            strong = True
        if ra == rb:
            if bc[ra] >= 1 or ra > top:          # set or better, or an overpair
                strong = True
        else:
            pa = bc[ra]
            pb = bc[rb]
            if pa >= 2 or pb >= 2 or (pa >= 1 and pb >= 1):   # trips or two pair
                strong = True
            elif pa >= 1:
                if ra == top and rb >= kick_min:
                    strong = True
            elif pb >= 1:
                if rb == top and ra >= kick_min:
                    strong = True
        if nb < 5 and st < 0 and not nut:
            good = 0
            for r in range(13):
                bit = 1 << r
                if full & bit:
                    continue
                mine = STRAIGHT[full | bit]
                if mine < 0 or STRAIGHT[bm | bit] >= 0:
                    continue
                bb = bm | bit
                best = -1
                for t in range(12, 3, -1):
                    if POP[(31 << (t - 4)) & ~bb] <= 2:
                        best = t
                        break
                if best < 0:
                    best = 3
                if mine >= best:
                    good += 1
            if good >= 2:                         # open-ended to the nut straight
                nut = True
        out[k] = 2 if nut else (1 if strong else 0)
    return out


@njit(cache=False)
def pct_vs(order, scores, w, out):
    """out[k] = share of the weight ``w`` that hand k beats (ties count half)."""
    M = scores.shape[0]
    tot = 0.0
    for k in range(M):
        if scores[k] >= 0.0:
            tot += w[k]
    i = 0
    below = 0.0
    while i < M:
        k = order[i]
        if scores[k] < 0.0:
            out[k] = -1.0
            i += 1
            continue
        j = i
        g = w[k]
        while j + 1 < M and scores[order[j + 1]] == scores[k]:
            j += 1
            g += w[order[j]]
        p = 0.5 if tot <= 0.0 else (below + 0.5 * g) / tot
        for t in range(i, j + 1):
            out[order[t]] = p
        below += g
        i = j + 1


@njit(cache=False)
def board_scores(U, board, nb, dead, coef, POP, HIGH, STRAIGHT, CATR):
    """Strength order of every hand: fitted equity on flop / turn, rank on the river."""
    if nb >= 5:
        v = K.strengths_all(U, 2, board, nb, dead, 0, POP, HIGH, STRAIGHT, CATR)
        return v.astype(np.float64)
    s, _ = K.combo_scores(U, 2, board, nb, dead, 0, coef, POP, HIGH, STRAIGHT, CATR)
    return s


@njit(cache=False)
def buckets_from(pct, ec, out):
    for k in range(pct.shape[0]):
        if pct[k] < 0.0 or ec[k] < 0:
            out[k] = -1
        else:
            out[k] = s_bucket(pct[k]) * NE + ec[k]


@njit(cache=False)
def range_stats(order, scores, w_me, w_opp, pct_me, nut_top):
    """(range-vs-range bucket, nut-advantage bucket) for the player owning ``w_me``.

    Range-vs-range equity: the weighted mean of our hands' strength against
    their range. Nut advantage: our share of the strongest ``nut_top`` of both
    ranges (each normalized)."""
    M = scores.shape[0]
    tm = 0.0
    to = 0.0
    acc = 0.0
    for k in range(M):
        if scores[k] >= 0.0:
            tm += w_me[k]
            to += w_opp[k]
            acc += w_me[k] * pct_me[k]
    if tm <= 0.0 or to <= 0.0:
        return 1, 1
    rvr = acc / tm
    am = 0.0
    ao = 0.0
    for i in range(M - 1, -1, -1):
        k = order[i]
        if scores[k] < 0.0:
            break
        am += w_me[k] / tm
        ao += w_opp[k] / to
        if 0.5 * (am + ao) >= nut_top:
            break
    share = 0.5 if am + ao <= 0.0 else am / (am + ao)
    rv = 0 if rvr < 0.47 else (1 if rvr <= 0.53 else 2)
    nu = 0 if share < 0.40 else (1 if share <= 0.60 else 2)
    return rv, nu


@njit(cache=False)
def legal_post(pot, to_call, stack, n_raises, last_inc, bb, amt, leg):
    """Legal abstract actions for the player to act. ``amt`` = chips to add."""
    for a in range(NA):
        leg[a] = False
        amt[a] = 0.0
    if to_call > 0.0:
        leg[0] = True
    leg[1] = True
    amt[1] = min(to_call, stack)
    if stack <= to_call + 1e-9:
        return
    allin = False
    if stack / max(pot, 1e-9) < ALLIN_SPR:
        leg[6] = True
        amt[6] = stack
        allin = True
    if n_raises >= MAX_RAISES:
        return
    floor = max(last_inc, bb)
    for i in range(4):
        add = FRACS[i] * (pot + to_call)
        if add < floor - 1e-9:
            continue
        tot = to_call + add
        if tot >= stack * 0.999:
            if not allin:
                leg[6] = True
                amt[6] = stack
                allin = True
            continue
        leg[2 + i] = True
        amt[2 + i] = tot


@njit(cache=False)
def policy_rows(QH, base, n, leg, na, tau, trim, out):
    """Action probabilities for rows base..base+n-1: softmax of the EVs over the
    legal actions; anything under ``trim`` is dropped (close hands mix, clear
    ones do not)."""
    for b in range(n):
        m = -1e30
        for a in range(na):
            if leg[a] and QH[base + b, a] > m:
                m = QH[base + b, a]
        s = 0.0
        for a in range(na):
            if leg[a]:
                out[b, a] = np.exp((QH[base + b, a] - m) / tau)
                s += out[b, a]
            else:
                out[b, a] = 0.0
        s2 = 0.0
        for a in range(na):
            p = out[b, a] / s
            if p < trim:
                p = 0.0
            out[b, a] = p
            s2 += p
        for a in range(na):
            out[b, a] /= s2


# ---- auction -------------------------------------------------------------------
@njit(cache=False)
def card_marginals(v, out):
    """Mean value of the flops holding a card minus the mean of those without it."""
    tot = 0.0
    for t in range(10):
        tot += v[t]
    for c in range(5):
        ins = 0.0
        for t in range(10):
            if TRIPLES[t, 0] == c or TRIPLES[t, 1] == c or TRIPLES[t, 2] == c:
                ins += v[t]
        out[c] = ins / 6.0 - (tot - ins) / 4.0


@njit(cache=False)
def split_bid(t, marg, spend, out):
    """Spread ``spend`` chips over the three cards of flop ``t``: one chip each
    (later cards first, they lose ties), the rest by the cards' marginal values."""
    for c in range(5):
        out[c] = 0
    left = int(spend)
    if left <= 0:
        return
    for i in range(2, -1, -1):
        if left > 0:
            out[TRIPLES[t, i]] = 1
            left -= 1
    ws = 0.0
    for i in range(3):
        ws += max(marg[TRIPLES[t, i]], 0.0)
    if left > 0 and ws > 0.0:
        used = 0
        for i in range(3):
            x = int(left * max(marg[TRIPLES[t, i]], 0.0) / ws)
            out[TRIPLES[t, i]] += x
            used += x
        out[TRIPLES[t, 2]] += left - used


@njit(cache=False)
def willingness(v, pot, lam):
    """(best flop, chips the best flop is worth over the average flop)."""
    best = 0
    mean = 0.0
    for t in range(10):
        mean += v[t]
        if v[t] > v[best]:
            best = t
    mean /= 10.0
    return best, (v[best] - mean) * pot / max(lam, 1e-9)


@njit(cache=False)
def model_bid(v, pot, kappa, balance, lam, marg, out):
    """The bid the opponent model assigns to a hand with flop values ``v``:
    a share ``kappa`` of what its best flop is worth, on that flop's cards."""
    best, wl = willingness(v, pot, lam)
    card_marginals(v, marg)
    split_bid(best, marg, min(int(kappa * wl), int(balance)), out)


@njit(cache=False)
def flop_of(tot):
    """Index (kernels.TRIPLES order) of the flop the summed bids select."""
    worst = -1
    worst2 = -1
    for _ in range(2):
        w = -1
        for c in range(4, -1, -1):          # later cards lose ties
            if c == worst:
                continue
            if w < 0 or tot[c] < tot[w]:
                w = c
        if worst < 0:
            worst = w
        else:
            worst2 = w
    sel = np.zeros(3, np.int64)
    n = 0
    for c in range(5):
        if c != worst and c != worst2:
            sel[n] = c
            n += 1
    return TIDX[sel[0], sel[1], sel[2]]


@njit(cache=False)
def best_bid(v, pot, balance, lam, opp_bids, opp_w, cand, out):
    """Best of a fixed set of candidate bids against weighted opponent bid rows.

    EV(bid) = sum over rows of P(row) x [value(flop) x pot - lam x chips paid].
    ``opp_bids[r]`` is the opponents' summed bid on each card in row r.
    Returns (EV gain over bidding nothing, chips bid)."""
    marg = np.zeros(5)
    card_marginals(v, marg)
    order = np.argsort(-v)
    best, wl = willingness(v, pot, lam)
    n = 0
    for c in range(5):
        cand[0, c] = 0
    n = 1
    levels = np.array([0.15, 0.3, 0.5, 0.8, 1.2])
    for j in range(2):
        t = order[j]
        split_bid(t, marg, min(3, int(balance)), cand[n])
        n += 1
        for li in range(5):
            split_bid(t, marg, min(int(levels[li] * wl), int(balance)), cand[n])
            n += 1
    bc = 0
    for c in range(5):
        if marg[c] > marg[bc]:
            bc = c
    for f in (0.2, 0.5):
        for c in range(5):
            cand[n, c] = 0
        cand[n, bc] = min(int(f * wl), int(balance))
        n += 1
    R = opp_bids.shape[0]
    tot = np.zeros(5)
    wsum = 0.0
    for r in range(R):
        wsum += opp_w[r]
    best_ev = -1e30
    ev0 = 0.0
    bi = 0
    for i in range(n):
        ev = 0.0
        for r in range(R):
            for c in range(5):
                tot[c] = cand[i, c] + opp_bids[r, c]
            t = flop_of(tot)
            paid = cand[i, TRIPLES[t, 0]] + cand[i, TRIPLES[t, 1]] + cand[i, TRIPLES[t, 2]]
            ev += opp_w[r] * (v[t] * pot - lam * paid)
        ev /= max(wsum, 1e-12)
        if i == 0:
            ev0 = ev
        if ev > best_ev + 1e-12:
            best_ev = ev
            bi = i
    s = 0
    for c in range(5):
        out[c] = cand[bi, c]
        s += cand[bi, c]
    return best_ev - ev0, s


@njit(cache=False)
def auction_pool(hero, auc, deck, U, cdfs, R, n_run, seed, POP, HIGH, STRAIGHT, CATR, pool, eq):
    """Runtime auction valuation (cheap enough for a 1 ms decision).

    Samples ``R`` opponent hands from the range cdfs (alternating opponents)
    and a few turn/river runouts, and returns for hero (row 0) and every
    sampled hand the share of the other hands it beats on each of the 10 flops.
    A sampled hand's opposition is the rest of the pool, which stands in for
    our range (the self-play prior: they hold what we would)."""
    np.random.seed(seed)
    base = np.int64(0)
    for i in range(2):
        base |= np.int64(1) << hero[i]
    for i in range(5):
        base |= np.int64(1) << auc[i]
    k = cdfs.shape[0]
    M = U.shape[0]
    for r in range(R):
        j = r % k
        tot = cdfs[j, M - 1]
        idx = 0
        for _ in range(30):
            idx = K._search(cdfs[j], np.random.random() * tot)
            if not ((base >> U[idx, 0]) & 1) and not ((base >> U[idx, 1]) & 1):
                break
        pool[r, 0] = U[idx, 0]
        pool[r, 1] = U[idx, 1]
    n = R + 1
    win = np.zeros((n, 10))
    cnt = np.zeros((n, 10))
    vals = np.zeros(n, np.int64)
    ok = np.zeros(n, np.bool_)
    fb = np.zeros(5, np.int64)
    buf = np.zeros(K.MAXC, np.int64)
    hand = np.zeros(2, np.int64)
    nd = deck.shape[0]
    for s in range(n_run):
        m = base
        for i in range(3, 5):
            while True:
                c = deck[np.random.randint(0, nd)]
                if not (m >> c) & 1:
                    break
            m |= np.int64(1) << c
            fb[i] = c
        run = (np.int64(1) << fb[3]) | (np.int64(1) << fb[4])
        ok[0] = True
        for h in range(1, n):
            ok[h] = not ((run >> pool[h - 1, 0]) & 1) and not ((run >> pool[h - 1, 1]) & 1)
        for t in range(10):
            for i in range(3):
                fb[i] = auc[TRIPLES[t, i]]
            for h in range(n):
                if ok[h]:
                    if h == 0:
                        hand[0] = hero[0]
                        hand[1] = hero[1]
                    else:
                        hand[0] = pool[h - 1, 0]
                        hand[1] = pool[h - 1, 1]
                    vals[h] = K.eval_rule(hand, 2, fb, 5, 0, buf, POP, HIGH, STRAIGHT, CATR)
            for h in range(n):
                if not ok[h]:
                    continue
                for j in range(n):
                    if j == h or not ok[j]:
                        continue
                    cnt[h, t] += 1.0
                    if vals[h] > vals[j]:
                        win[h, t] += 1.0
                    elif vals[h] == vals[j]:
                        win[h, t] += 0.5
    for h in range(n):
        for t in range(10):
            eq[h, t] = win[h, t] / cnt[h, t] if cnt[h, t] > 0 else 0.5
