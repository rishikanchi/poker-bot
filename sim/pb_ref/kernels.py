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


@njit(cache=False)
def combo_scores(U, H, board, nb, dead, rule, POP, HIGH, STRAIGHT, CATR):
    """Strength score in [0, 1] for every universe hand on this board (-1 if dead).

    Score = percentile of the current made hand among live hands, plus a draw
    bonus on non-final streets (flush draw, open-ender, gutshot, backdoor
    flush). Used to ORDER an opponent's range from strongest to weakest so we
    can model which part of it folds, calls or raises.
    """
    M = U.shape[0]
    vals = strengths_all(U, H, board, nb, dead, rule, POP, HIGH, STRAIGHT, CATR)
    order = np.argsort(vals)
    live = 0
    for k in range(M):
        if vals[k] >= 0:
            live += 1
    pct = np.full(M, -1.0)
    start = M - live
    i = start
    while i < M:
        j = i
        while j + 1 < M and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        p = ((i - start) + (j - start)) / 2.0 / max(live - 1, 1)
        for t in range(i, j + 1):
            pct[order[t]] = p
        i = j + 1
    if nb >= 5 or nb < 3:
        return pct
    bm = 0
    bs = np.zeros(4, np.int64)
    for i in range(nb):
        bm |= 1 << (board[i] >> 2)
        bs[board[i] & 3] += 1
    hs = np.zeros(4, np.int64)
    for k in range(M):
        if pct[k] < 0:
            continue
        for s in range(4):
            hs[s] = 0
        hm = 0
        for i in range(H):
            c = U[k, i]
            hs[c & 3] += 1
            hm |= 1 << (c >> 2)
        bonus = 0.0
        best_cnt = 0
        for s in range(4):
            if rule == RULE_OMAHA:
                cnt = min(hs[s], 2) + min(bs[s], 3)
            else:
                cnt = hs[s] + bs[s]
            if hs[s] > 0 and cnt > best_cnt:
                best_cnt = cnt
        if best_cnt == 4:
            bonus = 0.35
        elif best_cnt == 3 and nb == 3:
            bonus = 0.04
        full = bm | hm
        if STRAIGHT[full] < 0:
            outs = 0
            for r in range(13):
                bit = 1 << r
                if full & bit:
                    continue
                if STRAIGHT[full | bit] >= 0 and STRAIGHT[bm | bit] < 0:
                    outs += 1
            if outs >= 2:
                bonus = max(bonus, 0.25) + (0.1 if bonus >= 0.35 else 0.0)
            elif outs == 1:
                bonus = max(bonus, 0.10) + (0.05 if bonus >= 0.35 else 0.0)
        s = pct[k] + bonus * (1.0 - pct[k])
        pct[k] = min(s, 1.0)
    return pct


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
def mc_equity(hero, H, board, nb, board_total, deck, U, cdfs, k, n, main_scores, seed, rule,
              POP, HIGH, STRAIGHT, CATR):
    """Monte Carlo equity of the hero hand against k opponents.

    cdfs[j] is the cumulative weight vector (length M) of opponent j's range
    over the universe U, already zero for hands that hit the board or hero's
    cards. Opponents are sampled jointly (no shared cards) and the rest of the
    board (up to board_total cards) is dealt from ``deck``.

    Returns per-sample arrays (n):
      vscore  - main opponent's (j=0) strength score from main_scores
      res_hu  - hero's result vs opponent 0 alone (1 win, .5 tie, 0 loss)
      res_all - hero's share of the pot vs everyone (multiway)
    """
    np.random.seed(seed)
    vscore = np.zeros(n)
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
        vscore[s] = main_scores[idx0]
    return vscore, res_hu, res_all


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
def preflop_table(U, H, pre_pct, xs, nbins, n, board_total, deck, seed, rule,
                  POP, HIGH, STRAIGHT, CATR):
    """T[b, i] = equity of a hand in hero percentile bin b vs a villain drawn
    from the top xs[i] of the universe (by preflop percentile).

    Lets variants without a precomputed class table answer preflop equity
    questions with a lookup at decision time.
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
            top = max(int(xs[i] * M), 1)
            win = 0.0
            for s in range(n):
                while True:
                    hk = order[np.random.randint(lo, hi)]
                    vk = order[np.random.randint(0, top)]
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
