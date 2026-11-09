"""Opponent range model for the current hand.

A range is a weight vector over the plug-in's universe of hands.

1. Preflop: each opponent's line (open, call, 3-bet, jam, limp, check)
   maps to a band of preflop percentiles, widened or narrowed by that
   player's observed frequencies (open ratio, 3-bet ratio, ...).
2. Postflop: every action narrows the range by *strength on that street's
   board* (the plug-in's ``scores``):
     bet/raise -> keep the strong part plus a bluff slice from the bottom
     call      -> drop the part that would have folded (bottom) and most of
                  the part that would have raised (top)
     check     -> shrink (not remove) the part that would have bet
   The fractions come from the opponent model, so the same numbers predict
   how they respond to OUR bets (the "region model" in postflop.py).
"""
from __future__ import annotations

import numpy as np

from . import pushfold as PF
from .stats import Opp
from .view import players_behind

PUSH_W = ((5, 0.55), (8, 0.40), (10, 0.32), (13, 0.25), (16, 0.20), (20, 0.15), (30, 0.10), (50, 0.06), (100, 0.04))


def push_width(eff_bb):
    """Rough equilibrium-style open-shove width for an effective stack (bb)."""
    xs = [a for a, _ in PUSH_W]
    ys = [b for _, b in PUSH_W]
    return float(np.interp(eff_bb, xs, ys))


def call_jam_width(eff_bb, is_bb):
    """Rough calling width against an all-in of eff_bb big blinds."""
    w = float(np.interp(eff_bb, (5, 8, 10, 15, 20, 30, 50), (0.45, 0.30, 0.22, 0.14, 0.11, 0.08, 0.05)))
    return w * (1.3 if is_bb else 1.0)


def open_width(P, n_behind, n_alive):
    if n_alive <= 2:
        return P["open_width_hu"]
    return P["open_width"].get(int(n_behind), 0.13 if n_behind > 4 else 0.42)


def _clip_band(lo, hi):
    lo = float(min(max(lo, 0.0), 0.98))
    hi = float(min(max(hi, lo + 0.02), 1.0))
    return lo, hi


def preflop_band(v, seat, opp: Opp, P):
    """Percentile band [lo, hi) for ``seat``'s preflop line this hand."""
    n_raises = 0
    line = None  # (kind, n_raises_before, allin, pot_before, facing)
    order = v.pre_order
    for a in v.acts[0]:
        if a.seat == seat and a.kind in ("raise", "call", "check"):
            line = (a.kind, n_raises, a.allin, a.pot_before, a.facing, a)
        if a.kind == "raise":
            n_raises += 1
    is_bb = seat == v.bb_seat
    if line is None:
        # has not acted yet (e.g. a blind still to act): whole range
        return 0.0, 1.0
    kind, nr, allin, pot_before, facing, act = line
    p = v.players.get(seat)
    eff_bb = (p.start if p else 100 * v.bb) / max(v.bb, 1)
    if kind == "raise":
        if allin:
            try:
                i = order.index(seat)
                k = max(1, len([s for s in order[i + 1:] if s in v.alive]))
            except ValueError:
                k = 2
            w = PF.jam_width(eff_bb, k, P) * (0.6 if nr >= 1 else 1.0)
            w *= float(np.clip(opp.jam_ratio(), 0.6, 3.0))
            return _clip_band(0.0, w)
        if nr == 0:
            # how many players were behind the opener when they opened
            try:
                i = order.index(seat)
                n_behind = len([s for s in order[i + 1:] if s in v.alive])
            except ValueError:
                n_behind = 3
            w = open_width(P, n_behind, v.n_alive) * float(np.clip(opp.open_ratio(), 0.4, 3.5))
            return _clip_band(0.0, w)
        if nr == 1:
            return _clip_band(0.0, 0.06 * float(np.clip(opp.threebet_ratio(), 0.4, 4.0)))
        return _clip_band(0.0, 0.03)
    if kind == "call":
        if allin or (facing > 0 and _facing_allin(v, act)):
            k = _shover_behind(v, act)
            return _clip_band(0.0, PF.call_width(eff_bb, is_bb, k, P) * float(np.clip(opp.call_jam_ratio(), 0.5, 3.0)))
        if nr == 0:
            return _clip_band(0.06, 0.06 + 0.45 * float(np.clip(opp.vpip() / 0.25, 0.5, 2.5)))
        if nr == 1:
            w3 = 0.06 * float(np.clip(opp.threebet_ratio(), 0.4, 4.0))
            wc = (0.32 if is_bb else (0.22 if seat == v.sb_seat else 0.10)) * float(np.clip(opp.coldcall_ratio(), 0.5, 3.0))
            return _clip_band(min(w3, 0.08), min(w3, 0.08) + wc)
        return _clip_band(0.025, 0.11)
    # check: only the big blind can check preflop
    return _clip_band(0.10, 1.0)


def _shover_behind(v, act):
    """Players behind the all-in raiser this player called."""
    shover = None
    for a in v.acts[act.street]:
        if a is act:
            break
        if a.kind == "raise" and a.allin:
            shover = a.seat
    order = v.pre_order
    if shover is None or shover not in order:
        return 1
    i = order.index(shover)
    return max(1, len([s for s in order[i + 1:] if s in v.alive]))


def _facing_allin(v, act):
    """Was the bet this player called an all-in?"""
    for a in v.acts[act.street]:
        if a is act:
            break
        last = a
    else:
        return False
    try:
        return last.kind == "raise" and last.allin
    except UnboundLocalError:
        return False


def _soft_keep(cum, lo_q, hi_q, keep_out, width=0.03):
    """Weights multiplier: 1 inside quantile band [lo_q, hi_q], keep_out outside, smooth edges."""
    x = cum
    inside = 1.0 / (1.0 + np.exp(-(x - lo_q) / width)) * 1.0 / (1.0 + np.exp((x - hi_q) / width))
    return keep_out + (1.0 - keep_out) * inside


def narrow(w, scores, order, kind, frac_top, frac_bottom, keep_out):
    """Reweight range ``w`` by its strength quantiles on one street.

    ``order`` sorts the universe from weakest to strongest score. The band
    kept is [frac_bottom, 1 - frac_top] in cumulative-weight terms (bottom =
    weakest). ``keep_out`` is the weight left on hands outside the band.
    """
    ww = w[order]
    tot = ww.sum()
    if tot <= 0:
        return w
    cum = np.cumsum(ww) / tot  # 0 (weakest) .. 1 (strongest)
    mult = _soft_keep(cum, frac_bottom, 1.0 - frac_top, keep_out)
    out = np.empty_like(w)
    out[order] = ww * mult
    return out


def narrow_aggressive(w, order, freq, bluff_share, keep_out=0.04):
    """They bet/raised: keep their top ``freq*(1-bluff)`` plus bottom ``freq*bluff``."""
    ww = w[order]
    tot = ww.sum()
    if tot <= 0:
        return w
    cum = np.cumsum(ww) / tot
    vfrac = min(max(freq * (1.0 - bluff_share), 0.02), 0.98)
    bfrac = min(max(freq * bluff_share, 0.0), 0.5)
    value = 1.0 / (1.0 + np.exp(-(cum - (1.0 - vfrac)) / 0.03))
    bluff = 1.0 / (1.0 + np.exp((cum - bfrac) / 0.03)) if bfrac > 0.005 else 0.0
    mult = keep_out + (1.0 - keep_out) * np.clip(value + bluff, 0.0, 1.0)
    out = np.empty_like(w)
    out[order] = ww * mult
    return out


def _preflop_raiser(v):
    last = None
    for a in v.acts[0]:
        if a.kind == "raise":
            last = a.seat
    return last


def _size_mult(P, frac):
    """Plan prior: bigger bets come from stronger ranges (and smaller from wider)."""
    return 1.0 - P["size_strength"] * float(np.clip((frac - 0.66) / 0.6, -1.0, 1.0))


def seat_range(v, game, seat, opp, P, scores_by_street, orders, dead):
    """Range of ``seat`` now: preflop band, then narrowed street by street."""
    lo, hi = preflop_band(v, seat, opp, P)
    w = game.range_interval(lo, hi, dead)
    pfr = _preflop_raiser(v)
    for st in range(1, v.street + 1):
        sc = scores_by_street.get(st)
        if sc is None:
            continue
        w = w * (sc >= 0)
        order = orders[st]
        acted_before = False
        for a in v.acts[st]:
            if a.seat != seat:
                continue
            if a.kind == "raise":
                if a.facing > 0:
                    frac = (a.inc - a.facing) / max(a.pot_before + a.facing, 1)
                    f = opp.raise_vs_bet() * _size_mult(P, frac)
                    bs = opp.raise_bluff_share()
                else:
                    frac = a.inc / max(a.pot_before, 1)
                    cbet_spot = st == 1 and seat == pfr and not acted_before
                    f = (opp.cbet() if cbet_spot else opp.bet_when_checked()) * _size_mult(P, frac)
                    eq_share = frac / (1.0 + 2.0 * frac)
                    # river: equilibrium bluff share s/(1+2s), learned from showdowns.
                    # flop/turn: half of it as pure air; the other half are draws,
                    # which already rank high through the draw bonus in the scores.
                    bs = opp.river_bluff_share(frac) if st == 3 else 0.5 * eq_share
                w = narrow_aggressive(w, order, min(f, 0.95), bs)
            elif a.kind == "call":
                base = max(a.pot_before - a.facing, 1)
                frac = a.facing / base
                fold = opp.fold_to_bet(st, frac)
                rz = opp.raise_vs_bet()
                w = narrow(w, sc, order, "call", rz * 0.7, min(fold, 0.85), 0.05)
            elif a.kind == "check":
                cbet_spot = st == 1 and seat == pfr and not acted_before
                f = opp.cbet() if cbet_spot else opp.bet_when_checked()
                w = narrow(w, sc, order, "check", f * 0.6, 0.0, 0.35)
            acted_before = True
    if w.sum() <= 1e-9:
        w = game.range_interval(lo, hi, dead)
    return w


def build_ranges(v, game, mem, opps, P, scores_by_street, hero_dead):
    """Weights for every opponent still in the hand, narrowed through this street."""
    orders = {st: np.argsort(sc) for st, sc in scores_by_street.items() if sc is not None}
    ranges = {}
    for s, p in v.players.items():
        if s == v.seat or not p.in_hand:
            continue
        ranges[s] = seat_range(v, game, s, opps[s], P, scores_by_street, orders, hero_dead)
    return ranges, orders
