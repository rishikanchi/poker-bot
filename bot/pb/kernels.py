"""Numba kernels: hand evaluation, equity against ranges, ICM.

Variant-agnostic by construction. A game is described by:
  * a deck (any subset of the 52 cards),
  * H hole cards per player,
  * a showdown rule: RULE_ANY (best 5 of hole + board, hold'em style) or
    RULE_OMAHA (exactly 2 hole cards + exactly 3 board cards),
  * ranking tables (STRAIGHT and CATR) so e.g. short deck can re-rank.

Cards are ints 0..51 with ``card = rank * 4 + suit`` (rank 0..12 = 2..A,
suit 0..3 = c, d, h, s).

A *range* is a weight vector over a fixed list of hands called the
*universe* (``U``, shape [M, H]). For hold'em the universe is all 1326
two-card combos, so ranges are exact. For bigger games it is a fixed random
sample of hands, so ranges are approximate but the same code runs.

Hand values are ints where bigger is better:
    rank_of_category << 20 | k1 << 16 | k2 << 12 | k3 << 8 | k4 << 4 | k5

Nothing is compiled ahead of time (the competition sandbox cannot load
compiled extensions); Numba compiles in memory when ``warmup`` runs at import.
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
from numba import njit

RULE_ANY = 0
RULE_OMAHA = 1
MAXC = 12  # max cards in one evaluation (hole + board)

# category ids (before re-ranking through CATR)
HIGHCARD, PAIR, TWOPAIR, TRIPS, STRAIGHT_C, FLUSH, FULLHOUSE, QUADS, SFLUSH = range(9)


def build_mask_tables(short_deck=False):
    """POP, HIGH, STRAIGHT over 13-bit rank masks; CATR category ranks."""
    nmask = 1 << 13
    pop = np.zeros(nmask, np.int64)
    high = np.full(nmask, -1, np.int64)
    straight = np.full(nmask, -1, np.int64)
    low_ace = 0b1111 if not short_deck else (0b1111 << 4)  # 2345 or 6789
    for m in range(nmask):
        pop[m] = bin(m).count("1")
        if m:
            high[m] = m.bit_length() - 1
        for t in range(12, 3, -1):
            need = 0b11111 << (t - 4)
            if m & need == need:
                straight[m] = t
                break
        else:
            wheel = (1 << 12) | low_ace
            if m & wheel == wheel:
                straight[m] = 3  # lowest straight
    catr = np.arange(9, dtype=np.int64)
    if short_deck:  # flush beats full house
        catr[FLUSH], catr[FULLHOUSE] = FULLHOUSE, FLUSH
    return pop, high, straight, catr


POP, HIGH, STRAIGHT, CATR = build_mask_tables(False)


# ----------------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------------
@njit(cache=False)
def _topk(mask, k, HIGH):
    """Pack the top k ranks of mask (k<=5) into nibbles, highest first."""
    out = 0
    for i in range(5):
        if i < k and mask:
            h = HIGH[mask]
            mask &= ~(1 << h)
            out = (out << 4) | h
        else:
            out = out << 4
    return out


@njit(cache=False)
def evaluate(cards, n, POP, HIGH, STRAIGHT, CATR):
    """Value of the best 5-card hand among cards[0:n] (n >= 5, any size)."""
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0
    s0 = 0
    s1 = 0
    s2 = 0
    s3 = 0
    for i in range(n):
        c = cards[i]
        bit = 1 << (c >> 2)
        s = c & 3
        if s == 0:
            s0 |= bit
        elif s == 1:
            s1 |= bit
        elif s == 2:
            s2 |= bit
        else:
            s3 |= bit
        if m3 & bit:
            m4 |= bit
        elif m2 & bit:
            m3 |= bit
        elif m1 & bit:
            m2 |= bit
        else:
            m1 |= bit
    f = 0
    if POP[s0] >= 5:
        f = s0
    if POP[s1] >= 5 and (f == 0 or STRAIGHT[s1] > STRAIGHT[f] or (STRAIGHT[f] < 0 and s1 > f)):
        f = s1
    if POP[s2] >= 5 and (f == 0 or STRAIGHT[s2] > STRAIGHT[f] or (STRAIGHT[f] < 0 and s2 > f)):
        f = s2
    if POP[s3] >= 5 and (f == 0 or STRAIGHT[s3] > STRAIGHT[f] or (STRAIGHT[f] < 0 and s3 > f)):
        f = s3
    best = -1
    if f:
        sf = STRAIGHT[f]
        if sf >= 0:
            return (CATR[8] << 20) | (sf << 16)
        best = (CATR[5] << 20) | _topk(f, 5, HIGH)
    cand = -1
    if m4:
        q = HIGH[m4]
        k = HIGH[m1 & ~(1 << q)]
        if k < 0:
            k = 0
        cand = (CATR[7] << 20) | (q << 16) | (k << 12)
    elif m3:
        t = HIGH[m3]
        rest = m2 & ~(1 << t)
        if rest:
            cand = (CATR[6] << 20) | (t << 16) | (HIGH[rest] << 12)
    if cand > best:
        best = cand
    if best >= 0:
        # flush / full house / quads found; nothing lower can beat them
        return best
    st = STRAIGHT[m1]
    if st >= 0:
        return (CATR[4] << 20) | (st << 16)
    if m3:
        t = HIGH[m3]
        return (CATR[3] << 20) | (t << 16) | (_topk(m1 & ~(1 << t), 2, HIGH) >> 4)
    if POP[m2] >= 2:
        p1 = HIGH[m2]
        p2 = HIGH[m2 & ~(1 << p1)]
        k = HIGH[m1 & ~((1 << p1) | (1 << p2))]
        if k < 0:
            k = 0
        return (CATR[2] << 20) | (p1 << 16) | (p2 << 12) | (k << 8)
    if m2:
        p = HIGH[m2]
        return (CATR[1] << 20) | (p << 16) | (_topk(m1 & ~(1 << p), 3, HIGH) >> 4)
    return (CATR[0] << 20) | _topk(m1, 5, HIGH)


@njit(cache=False)
def eval_rule(hole, H, board, nb, rule, buf, POP, HIGH, STRAIGHT, CATR):
    """Showdown value of hole[0:H] with board[0:nb] under the variant's rule.

    With fewer than 5 cards total the value of what is there is returned
    (used for "current made hand" before the river).
    """
    if rule == RULE_ANY or nb < 3:
        n = 0
        for i in range(H):
            buf[n] = hole[i]
            n += 1
        for i in range(nb):
            buf[n] = board[i]
            n += 1
        if n < 5:
            # pad with nothing: compare by pairs/high cards only
            return _partial(buf, n, HIGH)
        return evaluate(buf, n, POP, HIGH, STRAIGHT, CATR)
    # omaha: exactly two hole cards and exactly three board cards
    best = -1
    for a in range(H):
        for b in range(a + 1, H):
            for x in range(nb):
                for y in range(x + 1, nb):
                    for z in range(y + 1, nb):
                        buf[0] = hole[a]
                        buf[1] = hole[b]
                        buf[2] = board[x]
                        buf[3] = board[y]
                        buf[4] = board[z]
                        v = evaluate(buf, 5, POP, HIGH, STRAIGHT, CATR)
                        if v > best:
                            best = v
    return best


@njit(cache=False)
def _partial(buf, n, HIGH):
    """Crude value for fewer than 5 cards (pairs, then high cards)."""
    m1 = 0
    m2 = 0
    m3 = 0
    for i in range(n):
        bit = 1 << (buf[i] >> 2)
        if m2 & bit:
            m3 |= bit
        elif m1 & bit:
            m2 |= bit
        else:
            m1 |= bit
    if m3:
        return (3 << 20) | (HIGH[m3] << 16)
    if m2:
        return (1 << 20) | (HIGH[m2] << 16) | (_topk(m1 & ~m2, 3, HIGH) >> 4)
    return _topk(m1, 5, HIGH)


@njit(cache=False)
def omaha_pair_table(board, nb, dead, POP, HIGH, STRAIGHT, CATR):
    """T[a, b] = best value of hole pair (a, b) with exactly 3 of the board cards.

    Built once per board (C(nb,3) evaluations per live pair) so that any
    Omaha hand's value is just the max of T over its hole-card pairs.
    """
    T = np.full((52, 52), -1, np.int64)
    buf = np.zeros(5, np.int64)
    for a in range(52):
        if (dead >> a) & 1:
            continue
        for b in range(a):
            if (dead >> b) & 1:
                continue
            best = -1
            for x in range(nb):
                for y in range(x + 1, nb):
                    for z in range(y + 1, nb):
                        buf[0] = a
                        buf[1] = b
                        buf[2] = board[x]
                        buf[3] = board[y]
                        buf[4] = board[z]
                        v = evaluate(buf, 5, POP, HIGH, STRAIGHT, CATR)
                        if v > best:
                            best = v
            T[a, b] = best
            T[b, a] = best
    return T


@njit(cache=False)
def strengths_all(U, H, board, nb, dead, rule, POP, HIGH, STRAIGHT, CATR):
    """Value of every universe hand on this board; -1 where it hits a dead card."""
    M = U.shape[0]
    out = np.full(M, -1, np.int64)
    buf = np.zeros(MAXC, np.int64)
    hole = np.zeros(H, np.int64)
    if rule == RULE_OMAHA and nb >= 3:
        # dead cards of the board only; hands touching hero's cards are skipped below
        bdead = np.int64(0)
        for i in range(nb):
            bdead |= np.int64(1) << board[i]
        T = omaha_pair_table(board, nb, bdead, POP, HIGH, STRAIGHT, CATR)
        for k in range(M):
            hit = False
            for i in range(H):
                if (dead >> U[k, i]) & 1:
                    hit = True
                    break
            if hit:
                continue
            best = -1
            for i in range(H):
                for j in range(i + 1, H):
                    v = T[U[k, i], U[k, j]]
                    if v > best:
                        best = v
            out[k] = best
        return out
    for k in range(M):
        hit = False
        for i in range(H):
            c = U[k, i]
            if (dead >> c) & 1:
                hit = True
                break
            hole[i] = c
        if hit:
            continue
        out[k] = eval_rule(hole, H, board, nb, rule, buf, POP, HIGH, STRAIGHT, CATR)
    return out


NF = 10  # number of strength features (see hand_features)


@njit(cache=False)
def omaha_straight(hm, bm, POP):
    """Can two of the hole ranks ``hm`` and three of the board ranks ``bm``
    make a straight? (Omaha: exactly two hole cards play.)"""
    for t in range(4, 14):
        if t == 13:
            W = (1 << 12) | 15            # the wheel A-2-3-4-5
        else:
            W = 31 << (t - 4)
        no_board = W & ~bm                # ranks that must come from the hand
        no_hole = W & ~hm                 # ranks that must come from the board
        if (no_board & no_hole) == 0 and POP[no_board] <= 2 and POP[no_hole] <= 3:
            return True
    return False


@njit(cache=False)
def hand_features(U, H, board, nb, dead, rule, POP, HIGH, STRAIGHT, CATR):
    """Per-hand strength features on this board, for every universe hand.

    Returns (X, draw): X[k] = [1, p, p*p, fd, fd*p, oe, oe*p, gs, gs*p, bd]
      p   percentile of the current made hand among live hands (ties averaged)
      fd  flush draw (four to a flush using our hole cards)
      oe  open-ended straight draw (two or more ranks complete a straight)
      gs  gutshot (exactly one rank completes a straight)
      bd  backdoor flush draw (three to a flush; flop only)
    draw[k] = 1 for a strong draw (fd or oe). Dead hands get p = -1.
    On the final street only p is set.
    """
    M = U.shape[0]
    vals = strengths_all(U, H, board, nb, dead, rule, POP, HIGH, STRAIGHT, CATR)
    order = np.argsort(vals)
    live = 0
    for k in range(M):
        if vals[k] >= 0:
            live += 1
    X = np.zeros((M, NF))
    draw = np.zeros(M)
    for k in range(M):
        X[k, 1] = -1.0
    start = M - live
    i = start
    while i < M:
        j = i
        while j + 1 < M and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        p = ((i - start) + (j - start)) / 2.0 / max(live - 1, 1)
        for t in range(i, j + 1):
            X[order[t], 0] = 1.0
            X[order[t], 1] = p
            X[order[t], 2] = p * p
        i = j + 1
    if nb >= 5 or nb < 3:
        return X, draw
    bm = 0
    bs = np.zeros(4, np.int64)
    for i in range(nb):
        bm |= 1 << (board[i] >> 2)
        bs[board[i] & 3] += 1
    hs = np.zeros(4, np.int64)
    for k in range(M):
        p = X[k, 1]
        if p < 0:
            continue
        for s in range(4):
            hs[s] = 0
        hm = 0
        for i in range(H):
            c = U[k, i]
            hs[c & 3] += 1
            hm |= 1 << (c >> 2)
        best_cnt = 0
        for s in range(4):
            if rule == RULE_OMAHA:
                if hs[s] < 2:             # a flush needs two hole cards of the suit
                    continue
                cnt = 2 + min(bs[s], 3)
            else:
                cnt = hs[s] + bs[s]
            if hs[s] > 0 and cnt > best_cnt:
                best_cnt = cnt
        fd = 1.0 if best_cnt == 4 else 0.0
        bd = 1.0 if (best_cnt == 3 and nb == 3) else 0.0
        outs = 0
        full = bm | hm
        if rule == RULE_OMAHA:
            if not omaha_straight(hm, bm, POP):
                for r in range(13):
                    bit = 1 << r
                    if bm & bit:
                        continue
                    if omaha_straight(hm, bm | bit, POP):
                        outs += 1
        elif STRAIGHT[full] < 0:
            for r in range(13):
                bit = 1 << r
                if full & bit:
                    continue
                if STRAIGHT[full | bit] >= 0 and STRAIGHT[bm | bit] < 0:
                    outs += 1
        oe = 1.0 if outs >= 2 else 0.0
        gs = 1.0 if outs == 1 else 0.0
        X[k, 3] = fd
        X[k, 4] = fd * p
        X[k, 5] = oe
        X[k, 6] = oe * p
        X[k, 7] = gs
        X[k, 8] = gs * p
        X[k, 9] = bd
        if fd > 0 or oe > 0:
            draw[k] = 1.0
    return X, draw


@njit(cache=False)
def combo_scores(U, H, board, nb, dead, rule, coef, POP, HIGH, STRAIGHT, CATR):
    """Strength score in [0, 1] for every universe hand (-1 if dead), and a
    strong-draw flag.

    Final street: the made-hand percentile (hand rank). Earlier streets: the
    fitted estimate of the hand's equity against a random hand,
    score = coef[street] . features (plan: "sort their hands by equity"; the
    weights come from tools/fit_scores.py, not from hand-picked bonuses).
    With no fitted weights (all zero) the made-hand percentile is used.
    """
    X, draw = hand_features(U, H, board, nb, dead, rule, POP, HIGH, STRAIGHT, CATR)
    M = U.shape[0]
    out = np.empty(M)
    row = 0 if nb <= 3 else 1
    use = False
    if nb < 5 and nb >= 3:
        for f in range(NF):
            if coef[row, f] != 0.0:
                use = True
    for k in range(M):
        p = X[k, 1]
        if p < 0:
            out[k] = -1.0
        elif not use:
            out[k] = p
        else:
            s = 0.0
            for f in range(NF):
                s += coef[row, f] * X[k, f]
            out[k] = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
    return out, draw


@njit(cache=False)
def _search(cdf, u):
    lo = 0
    hi = cdf.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=False)
def mc_equity(hero, H, board, nb, board_total, deck, U, cdfs, k, n, seed, rule,
              POP, HIGH, STRAIGHT, CATR):
    """Monte Carlo equity of the hero hand against k opponents.

    cdfs[j] is the cumulative weight vector (length M) of opponent j's range
    over the universe U, already zero for hands that hit the board or hero's
    cards. Opponents are sampled jointly (no shared cards) and the rest of the
    board (up to board_total cards) is dealt from ``deck``.

    Returns per-sample arrays (n):
      vidx    - which universe hand the main opponent (j=0) held
      res_hu  - hero's result vs opponent 0 alone (1 win, .5 tie, 0 loss)
      res_all - hero's share of the pot vs everyone (multiway)
    """
    np.random.seed(seed)
    vidx = np.zeros(n, np.int64)
    res_hu = np.zeros(n)
    res_all = np.zeros(n)
    base = np.int64(0)
    for i in range(H):
        base |= np.int64(1) << hero[i]
    for i in range(nb):
        base |= np.int64(1) << board[i]
    fb = np.zeros(board_total, np.int64)
    vh = np.zeros((k, H), np.int64)
    vals = np.zeros(k, np.int64)
    buf = np.zeros(MAXC, np.int64)
    nd = deck.shape[0]
    for s in range(n):
        m = base
        idx0 = 0
        for j in range(k):
            tot = cdfs[j, cdfs.shape[1] - 1]
            idx = 0
            for _ in range(40):
                idx = _search(cdfs[j], np.random.random() * tot)
                ok = True
                for i in range(H):
                    if (m >> U[idx, i]) & 1:
                        ok = False
                        break
                if ok:
                    break
            for i in range(H):
                c = U[idx, i]
                m |= np.int64(1) << c
                vh[j, i] = c
            if j == 0:
                idx0 = idx
        for i in range(nb):
            fb[i] = board[i]
        for i in range(nb, board_total):
            if i == nb:
                # plan: stratify the runouts - the first undealt card walks through the
                # whole deck across the samples; a dead card falls back to a random draw
                # (so every live card stays equally likely: no bias toward the cards
                # that follow a dead one)
                c = deck[(s * 7919 + 3) % nd]
                while (m >> c) & 1:
                    c = deck[np.random.randint(0, nd)]
            else:
                while True:
                    c = deck[np.random.randint(0, nd)]
                    if not (m >> c) & 1:
                        break
            m |= np.int64(1) << c
            fb[i] = c
        hv = eval_rule(hero, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
        best = -1
        for j in range(k):
            v = eval_rule(vh[j], H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            vals[j] = v
            if v > best:
                best = v
        v0 = vals[0]
        res_hu[s] = 1.0 if hv > v0 else (0.5 if hv == v0 else 0.0)
        if hv > best:
            res_all[s] = 1.0
        elif hv == best:
            ties = 1
            for j in range(k):
                if vals[j] == best:
                    ties += 1
            res_all[s] = 1.0 / ties
        vidx[s] = idx0
    return vidx, res_hu, res_all


@njit(cache=False)
def preflop_equity_vs_random(U, H, board_total, deck, n, seed, rule, POP, HIGH, STRAIGHT, CATR):
    """Equity of every universe hand vs one random hand (used to rank hands
    in variants without a precomputed table)."""
    np.random.seed(seed)
    M = U.shape[0]
    out = np.zeros(M)
    fb = np.zeros(board_total, np.int64)
    vh = np.zeros(H, np.int64)
    hero = np.zeros(H, np.int64)
    buf = np.zeros(MAXC, np.int64)
    nd = deck.shape[0]
    for k in range(M):
        base = np.int64(0)
        for i in range(H):
            hero[i] = U[k, i]
            base |= np.int64(1) << U[k, i]
        win = 0.0
        for s in range(n):
            m = base
            for i in range(H):
                while True:
                    c = deck[np.random.randint(0, nd)]
                    if not (m >> c) & 1:
                        break
                m |= np.int64(1) << c
                vh[i] = c
            for i in range(board_total):
                while True:
                    c = deck[np.random.randint(0, nd)]
                    if not (m >> c) & 1:
                        break
                m |= np.int64(1) << c
                fb[i] = c
            a = eval_rule(hero, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            b = eval_rule(vh, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            if a > b:
                win += 1.0
            elif a == b:
                win += 0.5
        out[k] = win / n
    return out


@njit(cache=False)
def preflop_equity_vs_top(U, H, order, top, board_total, deck, n, seed, rule,
                          POP, HIGH, STRAIGHT, CATR):
    """Equity of every universe hand vs a hand drawn from the ``top`` best
    hands of ``order`` (used to rank hands against a mid-strength range)."""
    np.random.seed(seed)
    M = U.shape[0]
    out = np.zeros(M)
    fb = np.zeros(board_total, np.int64)
    vh = np.zeros(H, np.int64)
    hero = np.zeros(H, np.int64)
    buf = np.zeros(MAXC, np.int64)
    nd = deck.shape[0]
    for k in range(M):
        base = np.int64(0)
        for i in range(H):
            hero[i] = U[k, i]
            base |= np.int64(1) << U[k, i]
        win = 0.0
        for s in range(n):
            m = base
            vk = 0
            for _ in range(60):
                vk = order[np.random.randint(0, top)]
                ok = True
                for i in range(H):
                    if (m >> U[vk, i]) & 1:
                        ok = False
                        break
                if ok:
                    break
            for i in range(H):
                vh[i] = U[vk, i]
                m |= np.int64(1) << U[vk, i]
            for i in range(board_total):
                while True:
                    c = deck[np.random.randint(0, nd)]
                    if not (m >> c) & 1:
                        break
                m |= np.int64(1) << c
                fb[i] = c
            a = eval_rule(hero, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            b = eval_rule(vh, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
            if a > b:
                win += 1.0
            elif a == b:
                win += 0.5
        out[k] = win / n
    return out


@njit(cache=False)
def preflop_table(U, H, pre_pct, xs, nbins, n, board_total, deck, seed, rule,
                  POP, HIGH, STRAIGHT, CATR):
    """T[b, i] = equity of a hand in hero percentile bin b vs a villain drawn
    from the percentile band (xs[i-1], xs[i]] of the universe (xs[-1] = 0).

    Lets variants without a precomputed class table answer preflop equity
    questions with a lookup at decision time. Each band is sampled on its
    own, so equity against any band is an average of cells (no differences
    of noisy numbers).
    """
    np.random.seed(seed)
    M = U.shape[0]
    order = np.argsort(pre_pct)
    T = np.zeros((nbins, xs.shape[0]))
    fb = np.zeros(board_total, np.int64)
    buf = np.zeros(MAXC, np.int64)
    nd = deck.shape[0]
    hero = np.zeros(H, np.int64)
    vil = np.zeros(H, np.int64)
    for b in range(nbins):
        lo = int(b * M / nbins)
        hi = max(int((b + 1) * M / nbins), lo + 1)
        for i in range(xs.shape[0]):
            v_lo = int(xs[i - 1] * M) if i > 0 else 0
            v_hi = max(int(xs[i] * M), v_lo + 1)
            win = 0.0
            for s in range(n):
                while True:
                    hk = order[np.random.randint(lo, hi)]
                    vk = order[np.random.randint(v_lo, v_hi)]
                    m = np.int64(0)
                    ok = True
                    for j in range(H):
                        m |= np.int64(1) << U[hk, j]
                    for j in range(H):
                        if (m >> U[vk, j]) & 1:
                            ok = False
                        m |= np.int64(1) << U[vk, j]
                    if ok:
                        break
                for j in range(H):
                    hero[j] = U[hk, j]
                    vil[j] = U[vk, j]
                for t in range(board_total):
                    while True:
                        c = deck[np.random.randint(0, nd)]
                        if not (m >> c) & 1:
                            break
                    m |= np.int64(1) << c
                    fb[t] = c
                x = eval_rule(hero, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
                y = eval_rule(vil, H, fb, board_total, rule, buf, POP, HIGH, STRAIGHT, CATR)
                if x > y:
                    win += 1.0
                elif x == y:
                    win += 0.5
            T[b, i] = win / n
    return T


# ----------------------------------------------------------------------------
# ICM (Malmuth-Harville) via a DP over subsets of finishers
# ----------------------------------------------------------------------------
@njit(cache=False)
def icm(stacks, payouts):
    """Expected payout of each player given chip stacks.

    Players with stack <= 0 get 0 (their place is already settled).
    payouts[i] is the prize for finishing (i+1)-th among players with chips.
    Harville: P(i takes the next place | set M already placed) = s_i / (S - S_M).
    Q[M] = probability that exactly the players in M take the top |M| places.
    """
    n = stacks.shape[0]
    p = payouts.shape[0]
    out = np.zeros(n)
    alive = 0
    total = 0.0
    for i in range(n):
        if stacks[i] > 0:
            alive += 1
            total += stacks[i]
    if alive == 0:
        return out
    depth = min(p, alive)
    nm = 1 << n
    q = np.zeros(nm)
    ssum = np.zeros(nm)
    size = np.zeros(nm, np.int64)
    for msk in range(1, nm):
        low = msk & (-msk)
        j = 0
        while (low >> j) != 1:
            j += 1
        prev = msk ^ low
        ssum[msk] = ssum[prev] + max(stacks[j], 0.0)
        size[msk] = size[prev] + 1
    q[0] = 1.0
    for msk in range(nm):
        if msk != 0 and q[msk] == 0.0:
            continue
        sz = size[msk]
        if sz >= depth:
            continue
        rem = total - ssum[msk]
        if rem <= 0:
            continue
        for i in range(n):
            if (msk >> i) & 1 or stacks[i] <= 0:
                continue
            pr = q[msk] * stacks[i] / rem
            out[i] += pr * payouts[sz]
            q[msk | (1 << i)] += pr
    return out


@njit(cache=False)
def hero_values(rows, payouts, hero_i):
    """Hero's expected points for each row of end-of-hand stacks.

    payouts[i] is the prize for place i+1 among the players alive at the
    start of the hand (padded with zeros). A player who ends a row with no
    chips busts now: the busted players share the bottom places (mean of
    their prizes); everyone with chips splits the top places by ICM.
    """
    m, n = rows.shape
    out = np.zeros(m)
    st = np.zeros(n)
    for r in range(m):
        k = 0
        for i in range(n):
            st[i] = rows[r, i] if rows[r, i] > 0 else 0.0
            if st[i] > 0:
                k += 1
        if st[hero_i] <= 0:
            tot = 0.0
            cnt = 0
            for j in range(k, n):
                if j < payouts.shape[0]:
                    tot += payouts[j]
                cnt += 1
            out[r] = tot / cnt if cnt > 0 else 0.0
        else:
            out[r] = icm(st, payouts[:max(k, 1)])[hero_i]
    return out


# ----------------------------------------------------------------------------
# Offline: hold'em preflop class-vs-class equity (tools/build_tables.py)
# ----------------------------------------------------------------------------
@njit(cache=False)
def class_matrix(n_samples, seed, COMBOS, COMBO_CLASS, POP, HIGH, STRAIGHT, CATR):
    """E[i, j] = equity of hold'em class i vs class j (sampled combos and boards)."""
    np.random.seed(seed)
    members = np.full((169, 12), -1, np.int64)
    cnt = np.zeros(169, np.int64)
    for k in range(1326):
        c = COMBO_CLASS[k]
        members[c, cnt[c]] = k
        cnt[c] += 1
    E = np.zeros((169, 169))
    hand = np.zeros(7, np.int64)
    vh = np.zeros(7, np.int64)
    for i in range(169):
        for j in range(i, 169):
            win = 0.0
            for s in range(n_samples):
                while True:
                    a = members[i, np.random.randint(0, cnt[i])]
                    b = members[j, np.random.randint(0, cnt[j])]
                    ma = (np.int64(1) << COMBOS[a, 0]) | (np.int64(1) << COMBOS[a, 1])
                    mb = (np.int64(1) << COMBOS[b, 0]) | (np.int64(1) << COMBOS[b, 1])
                    if (ma & mb) == 0:
                        break
                m = ma | mb
                for t in range(5):
                    while True:
                        c = np.random.randint(0, 52)
                        if not (m >> c) & 1:
                            break
                    m |= np.int64(1) << c
                    hand[2 + t] = c
                    vh[2 + t] = c
                hand[0] = COMBOS[a, 0]
                hand[1] = COMBOS[a, 1]
                vh[0] = COMBOS[b, 0]
                vh[1] = COMBOS[b, 1]
                x = evaluate(hand, 7, POP, HIGH, STRAIGHT, CATR)
                y = evaluate(vh, 7, POP, HIGH, STRAIGHT, CATR)
                if x > y:
                    win += 1.0
                elif x == y:
                    win += 0.5
            e = win / n_samples
            E[i, j] = e
            E[j, i] = 1.0 - e
    return E


# ----------------------------------------------------------------------------
# Hold'em combos and the 169 starting-hand classes
# ----------------------------------------------------------------------------
def build_combos(deck):
    deck = sorted(deck, reverse=True)
    combos = []
    for i, a in enumerate(deck):
        for b in deck[i + 1:]:
            combos.append((a, b))  # higher card first
    return np.array(combos, np.int64)


COMBOS = build_combos(range(52))


def build_classes(combos):
    ranks = "23456789TJQKA"
    names, index = [], {}
    for hi in range(12, -1, -1):
        for lo in range(hi, -1, -1):
            if hi == lo:
                index[(hi, lo, 0)] = len(names)
                names.append(ranks[hi] * 2)
            else:
                index[(hi, lo, 1)] = len(names)
                names.append(ranks[hi] + ranks[lo] + "s")
                index[(hi, lo, 0)] = len(names)
                names.append(ranks[hi] + ranks[lo] + "o")
    cls = np.zeros(len(combos), np.int64)
    for k, (a, b) in enumerate(combos):
        ra, rb = a >> 2, b >> 2
        hi, lo = max(ra, rb), min(ra, rb)
        suited = 1 if (a & 3) == (b & 3) and hi != lo else 0
        cls[k] = index[(hi, lo, suited)]
    counts = np.bincount(cls, minlength=169).astype(np.int64)
    return cls, names, counts


COMBO_CLASS, CLASS_NAMES, CLASS_COUNTS = build_classes(COMBOS)


# ----------------------------------------------------------------------------
# Python wrappers bound to the default (52-card) tables - used by tests/tools
# ----------------------------------------------------------------------------
def eval_hand(cards):
    arr = np.asarray(cards, np.int64)
    return int(evaluate(arr, arr.shape[0], POP, HIGH, STRAIGHT, CATR))


def icm_values(stacks, payouts):
    return icm(np.asarray(stacks, np.float64), np.asarray(payouts, np.float64))
