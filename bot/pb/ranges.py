"""Opponent range model for the current hand.

A range is a weight vector over the plug-in's universe of hands.

1. Preflop: each opponent's line (open, call, 3-bet, jam, limp, check)
   maps to a band of preflop percentiles, widened or narrowed by that
   player's observed frequencies (open ratio, 3-bet ratio, ...).
2. Postflop: every action keeps the region of the range the plan's region
   model assigns to it, with hands ordered by strength on that street
   (the plug-in's ``scores``: fitted equity before the river, hand rank on it):
     bet   -> the strongest part for value plus bluffs from the bottom
              (s / (1 + 2s) of the betting range; learned from showdowns on the river)
     raise -> the strongest part plus a slice of strong draws (river: bluffs)
     call  -> between the part that folds (bottom) and the part that raises (top)
     check -> between the part that would have bet for value and as a bluff
   Hands outside the region keep a small leak weight. The fractions come from
   the opponent model, so the same numbers predict how they respond to OUR
   bets (the region model in postflop.py).
"""
from __future__ import annotations

import numpy as np

from . import pushfold as PF
from .stats import Opp, min_weight
from .view import players_behind

def open_width(P, n_behind, n_alive):
    if n_alive <= 2 and n_behind >= 1:
        return P["open_width_hu"]      # heads-up button, first in
    return P["open_width"].get(int(n_behind), 0.13 if n_behind > 4 else 0.42)


def _clip_band(lo, hi):
    lo = float(min(max(lo, 0.0), 0.98))
    hi = float(min(max(hi, lo + 0.02), 1.0))
    return lo, hi


def _n_behind(v, seat):
    """Players who were still to act after ``seat`` in the preflop order."""
    order = v.pre_order
    if seat not in order:
        return 3
    return len([s for s in order[order.index(seat) + 1:] if s in v.alive])


def first_in_width(v, seat, opp, P):
    """Opening width for this seat's position, scaled by how often they open
    compared with a baseline player in the same spots (plan: "range width from
    their position, plus their observed raise frequency")."""
    w = open_width(P, _n_behind(v, seat), v.n_alive)
    if opp is not None:
        w *= max(opp.open_ratio(), 0.4)     # no upper cap: an always-raiser has every hand
    return min(w, 1.0)


def bb_defend_width(P, opener_w):
    """Plan sample ranges: the big blind calls ~25% vs a 15% open, ~45% vs a 45% open."""
    lo, hi = P["bb_defend"]
    return float(np.interp(opener_w, (0.15, 0.45), (lo, hi)))


def _min_share(a, opp, P, bet):
    """Share of the "tells nothing" reading for this bet / raise (plan: the
    "min" translation point; see stats.min_weight and Opp.min_size_wide)."""
    if a.allin or a.by <= 0:
        return 0.0
    m = min_weight(a.by, a.min_by, a.pot_before + a.facing, P["min_translate_bet"] if bet else None)
    return m * opp.min_size_wide() if m > 0 else 0.0


def preflop_band(v, seat, opp: Opp, P, opps=None):
    """Percentile band [lo, hi) for ``seat``'s preflop line this hand.

    Every width is a plan number or a solved chart value, scaled by what this
    opponent has shown (ratios and rates from stats.py):
      open            position width x open ratio
      3-bet / 4-bet   top ~4% x 3-bet ratio / top ~2.5%
      first-in jam    solved push/fold chart x jam ratio
      re-jam          solved re-shove chart (by the opener's width) x jam ratio
      call of a jam   solved calling chart x call ratio
      limp            the hands below their raising range, as wide as their limp rate
      call of an open below their 3-bet range: BB ~25-45% by opener width, others ~7%
      call of a 3-bet the part of their opening range between 4-betting and folding
      check (BB)      everything they would not have raised
    """
    n_raises = 0
    line = None
    prev_raiser = None
    raiser_before = None
    for a in v.acts[0]:
        if a.seat == seat and a.kind in ("raise", "call", "check"):
            line = (a.kind, n_raises, a.allin, a)
            raiser_before = prev_raiser
        if a.kind == "raise":
            n_raises += 1
            prev_raiser = a.seat
    if line is None:
        return 0.0, 1.0  # has not acted yet (e.g. a blind still to act)
    kind, nr, allin, act = line
    is_bb = seat == v.bb_seat
    p = v.players.get(seat)
    eff_bb = (p.start if p else 100 * v.bb) / max(v.bb, 1)
    own_open = first_in_width(v, seat, opp, P)

    def width_of(other):
        o = opps.get(other) if opps else None
        return first_in_width(v, other, o, P)

    if kind == "raise":
        if allin:
            if nr == 0:
                return _clip_band(0.0, opp.jam_width(PF.jam_width(eff_bb, max(1, _n_behind(v, seat)), P)))
            ow = width_of(raiser_before) if raiser_before is not None else P["prior_open"]
            return _clip_band(0.0, opp.jam_width(PF.rejam_width(eff_bb, ow, P)))
        if nr == 0:
            return _clip_band(0.0, own_open)
        # a minimum-size re-raise does not say "this is my re-raising range": it is
        # read (by its share m) as their whole continuing range, re-raises and calls
        m = _min_share(act, opp, P, bet=False)
        if nr == 1:
            w3 = opp.threebet_width()
            if m > 0:
                ow = width_of(raiser_before) if raiser_before is not None else P["prior_open"]
                wc = (bb_defend_width(P, ow) * max(opp.bb_defend_ratio(), 0.5) if is_bb
                      else P["prior_coldcall"] * max(opp.coldcall_ratio(), 0.5))
                w3 = w3 + m * wc
            return _clip_band(0.0, w3)
        if any(a.seat == seat and a.kind == "raise" and a is not act for a in v.acts[0]):
            # they opened and re-raised a 3-bet: the 4-betting share of their opening
            # range, as observed (prior: prior_4bet of the range, i.e. the plan's top ~2-3%)
            pf, pc, p4 = opp.response_to_3bet()
            return _clip_band(0.0, max(own_open * (p4 + m * pc), 0.01))
        return _clip_band(0.0, P["prior_4bet_width"] + m * opp.threebet_width())
    if kind == "call":
        facing_jam = _jam_faced(v, act)
        if facing_jam is not None:
            cr = float(np.clip(opp.call_jam_ratio(), 0.5, 3.0))
            jam_act, jam_nr = facing_jam
            e = min(eff_bb, v.players[jam_act.seat].start / max(v.bb, 1)) if jam_act.seat in v.players else eff_bb
            if jam_nr >= 1 and any(a.seat == seat and a.kind == "raise" for a in v.acts[0]):
                # they opened and called a re-shove: solved opener-calls-rejam chart
                return _clip_band(0.0, PF.rejam_call_width(e, own_open, P) * cr)
            return _clip_band(0.0, PF.call_width(e, is_bb, max(1, _n_behind(v, jam_act.seat)), P) * cr)
        if nr == 0:
            # limp: the next slice below the hands they raise, as wide as their limp
            # rate (counting the limp we are looking at as one more observation)
            return _clip_band(own_open, own_open + opp.limp_rate(extra=1))
        w3 = opp.threebet_width()
        if nr == 1:
            ow = width_of(raiser_before) if raiser_before is not None else P["prior_open"]
            if is_bb:
                wc = bb_defend_width(P, ow) * max(opp.bb_defend_ratio(), 0.5)
            else:
                wc = P["prior_coldcall"] * max(opp.coldcall_ratio(), 0.5)
            return _clip_band(w3, w3 + wc)
        if any(a.seat == seat and a.kind == "raise" for a in v.acts[0]):
            # they opened and called a 3-bet: between their 4-bets and their folds
            pf, pc, p4 = opp.response_to_3bet()
            return _clip_band(own_open * p4, own_open * (p4 + pc))
        return _clip_band(P["prior_4bet_width"], P["prior_4bet_width"] + w3)  # cold call of a 3-bet
    # check: only the big blind can check preflop; it would have raised its best hands
    return _clip_band(opp.pfr(), 1.0)


def _jam_faced(v, act):
    """(the all-in raise this call answered, raises before it) or None."""
    last = None
    n = 0
    n_before = 0
    for a in v.acts[act.street]:
        if a is act:
            break
        if a.kind == "raise":
            last = a
            n_before = n
            n += 1
    if last is not None and last.allin:
        return last, n_before
    return None


def _quantile(w, order):
    """Strength quantile of every hand inside the weighted range ``w``
    (0 = weakest .. 1 = strongest; the upper edge of each hand's slot)."""
    ww = w[order]
    tot = ww.sum()
    q = np.empty_like(w)
    if tot <= 0:
        q[order] = np.linspace(0.0, 1.0, w.size)
        return q
    cum = np.cumsum(ww) / tot
    cum[-1] = 1.0
    q[order] = cum
    return q


def keep(w, mask, leak):
    """Keep the hands in ``mask``; the rest keep a small ``leak`` weight
    (opponents do not follow the model exactly)."""
    return w * np.where(mask, 1.0, leak)


def narrow_call(w, order, fold, raise_, leak):
    """Plan region model: facing a bet their weakest ``fold`` share folds, their
    strongest ``raise_`` share raises, the middle calls. A call keeps the middle."""
    q = _quantile(w, order)
    return keep(w, (q > fold) & (q <= 1.0 - raise_), leak)


def _draw_slice(w, q, value, draws, share):
    """The strongest hands carrying a draw (outside the value part), up to
    ``share`` of the range."""
    out = np.zeros(w.size, bool)
    if draws is None or share <= 0:
        return out
    cand = (draws > 0) & ~value & (w > 0)
    if not cand.any():
        return out
    idx = np.nonzero(cand)[0]
    idx = idx[np.argsort(-q[idx])]            # strongest draws first
    cum = np.cumsum(w[idx]) / max(w.sum(), 1e-12)
    out[idx[cum <= share + 1e-12]] = True
    if not out.any():
        out[idx[0]] = True
    return out


def aggressive_region(w, order, freq, bluff_share, draws):
    """Hands that bet or raise with frequency ``freq``.

    The strongest freq * (1 - bluff) for value plus a bluff part of size
    freq * bluff. On the river the bluffs come from the bottom of the range
    (plan). Before the river the plan's range shapes are "value plus draws":
    the bluff part is the strongest draws, as far as the range has them.
    """
    q = _quantile(w, order)
    v = min(max(freq * (1.0 - bluff_share), 0.0), 1.0)
    b = min(max(freq * bluff_share, 0.0), 1.0)
    value = q > 1.0 - v
    if draws is None:
        return value | (q <= b)
    return value | _draw_slice(w, q, value, draws, b)


def narrow_bet(w, order, freq, bluff_share, leak, draws=None):
    """They bet: keep the betting region."""
    return keep(w, aggressive_region(w, order, freq, bluff_share, draws), leak)


def narrow_raise(w, order, freq, bluff_share, leak, draws=None):
    """They raised a bet: keep the raising region."""
    return keep(w, aggressive_region(w, order, freq, bluff_share, draws), leak)


def narrow_check(w, order, freq, bluff_share, leak, draws=None):
    """They checked: the hands they would have bet keep only weight ``leak``
    (plan: a checking range is capped, but players do check a share of their
    strong hands - the plan's own trap share is 10-20%)."""
    return keep(w, ~aggressive_region(w, order, freq, bluff_share, draws), leak)


def _bet_was_allin(v, st, act):
    """Was the bet this call answered an all-in?"""
    last = None
    for a in v.acts[st]:
        if a is act:
            break
        if a.kind == "raise":
            last = a
    return bool(last is not None and last.allin)


def _preflop_raiser(v):
    last = None
    for a in v.acts[0]:
        if a.kind == "raise":
            last = a.seat
    return last


def _size_mult(P, frac, opp=None):
    """Plan prior: "the larger the bet, the tighter the range you assume"
    (overbets and jams are usually tight). The frequency of a bet of s pots
    scales like 1 / (1 + s) - the same shape as the share of a range that can
    profitably continue against it - relative to the usual size.

    Plan: "a bot that always bets the same fraction gives you no information
    from size, so rely on its frequencies instead" -> no size effect then."""
    if opp is not None and opp.same_size_share() >= P["same_size_share"]:
        return 1.0
    ref = P["villain_bet_frac"]
    return (1.0 + ref) / (1.0 + max(float(frac), 0.0))


def call_floor(P, opp, w, q, sc, F, r):
    """Where their calling region starts, calibrated by showdowns (plan: showdown
    hands tell "how strong their calling range is", which sets thin value).

    w: their range weights; q: strength quantile of each hand inside it; sc:
    strength percentile among all live hands; F: fold share from the fold
    frequency; r: raise share. With nothing shown down this is F."""
    if sc is None or opp is None:
        return F
    thr = P["bluff_showdown_pct"]
    hi = max(1.0 - r, F + 1e-6)
    tot = float(w.sum())
    if tot <= 0:
        return F
    in_call = (q > F) & (q <= hi) & (w > 0)
    m = float(w[in_call].sum())
    if m <= 0:
        return F
    weak = sc < thr
    pred = float(w[in_call & weak].sum()) / m             # weak share the model predicts
    obs = opp.call_weak_share(pred)                        # shrunk toward that prediction
    if abs(obs - pred) < 1e-3:
        return F
    thr_q = float(w[weak & (w > 0)].sum()) / tot            # the weak / strong boundary as a quantile
    if obs >= 0.999:
        return 0.0
    Fp = (thr_q - obs * hi) / (1.0 - obs)                  # weak share of (Fp, hi] = obs, uniformly
    return float(min(max(Fp, 0.0), hi - 1e-6))


def bluff_prior(frac):
    """Plan: a balanced betting range has s / (1 + 2s) bluffs for a bet of s pots."""
    return frac / (1.0 + 2.0 * frac)


def seat_range(v, game, seat, opp, P, scores_by_street, orders, dead, draws_by_street=None, opps=None):
    """Range of ``seat`` now: preflop band, then narrowed street by street.

    ``dead`` holds only the cards nobody can have (the board), so this is the
    range as the opponent plays it; our own cards are removed afterwards
    (that is how blockers enter the probabilities).
    """
    lo, hi = preflop_band(v, seat, opp, P, opps)
    w = game.range_interval(lo, hi, dead)
    pfr = _preflop_raiser(v)
    leak = P["range_leak"]
    final_street = len(v.board) >= game.board_total
    for st in range(1, v.street + 1):
        sc = scores_by_street.get(st)
        if sc is None:
            continue
        w = w * (sc >= 0)
        order = orders[st]
        river = final_street and st == v.street
        draws = None if (river or draws_by_street is None) else draws_by_street.get(st)
        acted_before = False
        w_pre_check = None       # the range before a check on this street
        for a in v.acts[st]:
            if a.seat != seat:
                continue
            cbet_spot = st == 1 and seat == pfr and not acted_before
            if a.kind == "raise":
                if a.facing > 0:
                    # a check-raise comes from the range they checked with, strong hands included
                    base = w_pre_check if w_pre_check is not None else w
                    frac = (a.inc - a.facing) / max(a.pot_before + a.facing, 1)
                    f = opp.raise_vs_bet(st) * _size_mult(P, frac, opp)
                    w_new = narrow_raise(base, order, min(f, 0.95), opp.raise_bluff_share(), leak, draws)
                else:
                    frac = a.inc / max(a.pot_before, 1)
                    f = (opp.cbet() if cbet_spot else opp.bet_when_checked()) * _size_mult(P, frac, opp)
                    bs = opp.river_bluff_share(frac) if river else opp.bet_bluff_share(st, frac)
                    w_new = narrow_bet(w, order, min(f, 0.95), bs, leak, draws)
                # plan: a minimum-size bet or raise is cheap and often weak - by its
                # share m it leaves their range as it was
                m = _min_share(a, opp, P, bet=a.facing <= 0)
                w = w_new if m <= 0 else m * w + (1.0 - m) * w_new
                w_pre_check = None
            elif a.kind == "call":
                frac = a.facing / max(a.pot_before - a.facing, 1)
                F = opp.fold_to_bet(st, frac, allin=_bet_was_allin(v, st, a))
                r = opp.raise_vs_bet(st)
                w = narrow_call(w, order, call_floor(P, opp, w, _quantile(w, order), sc, F, r), r, leak)
                w_pre_check = None
            elif a.kind == "check":
                w_pre_check = w
                f = opp.cbet() if cbet_spot else opp.bet_when_checked()
                frac = opp.usual_bet_frac()
                bs = opp.river_bluff_share(frac) if river else opp.bet_bluff_share(st, frac)
                w = narrow_check(w, order, f, bs, P["range_leak_check"], draws)
            acted_before = True
    if w.sum() <= 1e-9:
        w = game.range_interval(lo, hi, dead)
        sc = scores_by_street.get(v.street)
        if sc is not None:
            w = w * (sc >= 0)
    return w


def build_ranges(v, game, opps, P, scores_by_street, draws_by_street, board_dead):
    """Unconditional range of every opponent still in the hand, plus the
    strength order of the universe on each street."""
    orders = {st: np.argsort(sc, kind="stable") for st, sc in scores_by_street.items() if sc is not None}
    ranges = {}
    for s, p in v.players.items():
        if s == v.seat or not p.in_hand:
            continue
        ranges[s] = seat_range(v, game, s, opps[s], P, scores_by_street, orders, board_dead,
                               draws_by_street, opps)
    return ranges, orders
