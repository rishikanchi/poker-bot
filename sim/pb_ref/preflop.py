"""Preflop decisions.

Hands are ranked once (plug-in ``hero_pre_pct``: 0 = best). Opponents'
ranges are percentile bands (ranges.preflop_band). Every option is scored
by expected value (EV) and the best one is played:

  fold  -> 0
  call  -> equity * EQR * (pot after calling) - cost of calling
  raise -> P(fold) * pot + P(call) * (EV when called) + P(re-raise) * (EV then)

EQR (equity realization) is how much of its raw equity a hand actually wins
once there is betting after the flop: >1 for hands that play well (suited,
connected, pairs, in position), <1 for hands that play badly.

Short stacks (<= push_fold_bb big blinds) and all-in spots use ICM points
instead of chips, with exact stacks and the table's payouts.
"""
from __future__ import annotations

from . import eqr as EQ
from . import icmtools as I
from . import pushfold as PF
from .ranges import call_jam_width, open_width, preflop_band
from .view import players_behind

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


class Ctx:
    """Everything a decision needs, bundled."""

    def __init__(self, v, game, opps, P, rng):
        self.v = v
        self.game = game
        self.opps = opps
        self.P = P
        self.rng = rng


def _features_eqr(game, hole, P):
    """Preflop EQR adjustment from simple, variant-agnostic hand features."""
    r = sorted((c >> 2 for c in hole), reverse=True)
    s = [c & 3 for c in hole]
    adj = 0.0
    if len(set(s)) < len(s):
        adj += P["eqr_suited"]
    gaps = [r[i] - r[i + 1] for i in range(len(r) - 1)]
    if any(0 < g <= 2 for g in gaps):
        adj += P["eqr_connected"]
    if any(g == 0 for g in gaps):
        adj += P["eqr_pair"]
    if len(set(s)) == len(s) and not any(g == 0 for g in gaps) and min(gaps) > 3 and r[0] < 12:
        adj += P["eqr_offsuit_junk"]
    return adj


def _pre_R(c, ip, extra=0.0):
    """Preflop EQR: prior from features (plan), blended with the measured table."""
    P = c.P
    prior = (P["eqr_pre_ip"] if ip else P["eqr_pre_oop"]) + _features_eqr(c.game, c.v.hole, P) + extra
    wm = P.get("eqr_measured_weight", 0.0) if P.get("use_measured_eqr") else 0.0
    if wm > 0:
        try:
            m = EQ.lookup_pre(ip, c.v.hole)
        except Exception:  # noqa: BLE001
            m = None
        if m is not None:
            return (1 - wm) * prior + wm * (min(max(m, 0.3), P["eqr_measured_cap"]) + extra)
    return prior


def _is_ip(v, villain):
    """Will hero act after villain on later streets?"""
    o = v.post_order
    if v.seat not in o or villain not in o:
        return False
    return o.index(v.seat) > o.index(villain)


def _raise_amount(v, total):
    lo, hi = v.legal.get(RAISE, (None, None))
    if lo is None:
        return None
    return int(max(lo, min(hi, int(total))))


def _jam(v):
    lo, hi = v.legal.get(RAISE, (None, None))
    return None if hi is None else int(hi)


def decide(c: Ctx):
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    bb = v.bb
    hole = v.hole
    pct = g.hero_pre_pct(hole)
    c.pct = pct
    acts = v.acts[0]
    raises = [a for a in acts if a.kind == RAISE]
    hero_raised = any(a.seat == v.seat and a.kind == RAISE for a in acts)
    others_in = [s for s in v.alive if s != v.seat and v.players[s].in_hand]
    if not others_in:
        return (CHECK, None) if CHECK in v.legal else (CALL, None)
    eff = min(hero.avail, max(v.players[s].avail for s in others_in))
    eff_bb = eff / bb
    c.eff_bb = eff_bb

    if not raises:
        return _unopened(c, pct, eff_bb)
    last = raises[-1]
    vil = last.seat
    if not hero_raised:
        return _facing_raise(c, pct, eff_bb, vil, last, len(raises))
    return _facing_reraise(c, pct, eff_bb, vil, last, len(raises))


# -----------------------------------------------------------------------------
def _unopened(c, pct, eff_bb):
    v, g, P = c.v, c.game, c.P
    bb = v.bb
    behind = players_behind(v, v.pre_order)
    limpers = [a.seat for a in v.acts[0] if a.kind == CALL and a.seat != v.seat]
    can_check = CHECK in v.legal
    if eff_bb <= P["push_fold_bb"] and RAISE in v.legal:
        ev_jam, ev_fold = shove_ev(c, behind + limpers)
        if can_check:
            # big blind option: checking keeps our equity in a cheap pot
            if ev_jam > ev_fold + 0.002 * sum(v.payouts):
                return RAISE, _jam(v)
            return CHECK, None
        if ev_jam > ev_fold:
            return RAISE, _jam(v)
        return FOLD, None
    n_behind = len(behind)
    if P.get("steal_ev", True) and 1 <= n_behind <= 2 and not limpers and RAISE in v.legal and all(
            s in (v.sb_seat, v.bb_seat) for s in behind):
        return _steal(c, pct, eff_bb, behind)
    width = open_width(P, n_behind, v.n_alive) * P.get("open_scale", 1.0)
    # plan: steal more when the players left to act fold a lot to opens
    blinds = [s for s in behind if s in (v.sb_seat, v.bb_seat)]
    if blinds and n_behind <= 3:
        f = sum(c.opps[s].blind_fold_vs_steal() for s in blinds) / len(blinds)
        width *= max(0.6, min(1.8, 1.0 + 2.0 * (f - 0.62)))
    if limpers:
        width *= P["iso_width_mult"]  # plan: isolate with the opening range plus a bit
    if pct < width and RAISE in v.legal:
        # plan: the 16-40bb band can open-jam; we jam the weaker part of the range
        # when shoving beats folding in ICM points, and raise the strong part
        if eff_bb <= P["open_jam_bb"] and pct > 0.5 * width:
            ev_jam, ev_fold = shove_ev(c, behind + limpers)
            if ev_jam > ev_fold:
                return RAISE, _jam(v)
        if limpers:
            total = (P["iso_size_bb"] + P["limp_iso_extra_bb"] * len(limpers)) * bb
        else:
            total = (P["open_size_bb"] if eff_bb > 40 else P["open_size_bb_short"]) * bb
        if total >= 0.4 * v.hero.avail:
            return RAISE, _jam(v)
        return RAISE, _raise_amount(v, total)
    if can_check:
        return CHECK, None
    # plan: raise or fold; over-limp only in position, cheaply, with hands that
    # play well multiway (pairs, suited connectors, suited aces)
    if limpers and CALL in v.legal and v.to_call <= bb and n_behind <= 2 and _plays_multiway(v.hole):
        return CALL, None
    return FOLD, None


def _plays_multiway(hole):
    if len(hole) != 2:
        return False
    a, b = hole
    ra, rb = a >> 2, b >> 2
    if ra == rb:
        return True
    suited = (a & 3) == (b & 3)
    return suited and (abs(ra - rb) <= 2 or max(ra, rb) == 12)


def _steal(c, pct, eff_bb, blinds):
    """Only the blinds are left: compare fold / raise by chip EV.

    DEVIATION from the plan's fixed widths, kept because it tested better
    (see README "Deviations"): heads-up the fixed rule over-folded.

    raise: P(all blinds fold) * pot
           + P(defended) * (equity vs defending range * EQR * pot after call - our chips added)
    The defending range is the top (1 - fold rate) of the main defender's hands.
    """
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    bb = v.bb
    pot = v.pot
    main = v.bb_seat if v.bb_seat in blinds else blinds[-1]
    vp = v.players[main]
    size_bb = P["open_size_bb"] if eff_bb > 40 else P["open_size_bb_short"]
    total = size_bb * bb
    if total >= 0.4 * hero.avail:
        total = hero.avail
    add = total - hero.bet
    pf = 1.0
    for s in blinds:
        pf *= c.opps[s].blind_fold_vs_steal()
    d_hi = max(1.0 - c.opps[main].blind_fold_vs_steal(), 0.08)
    eq_c = g.pre_equity(v.hole, 0.0, d_hi)
    ip = _is_ip(v, main)
    R = _pre_R(c, ip)
    v_add = min(total, vp.avail) - vp.bet
    ev_raise = pf * pot + (1 - pf) * (eq_c * R * (pot + add + v_add) - add)
    options = [(0.0, FOLD, None), (ev_raise, RAISE, _raise_amount(v, total) if total < hero.avail else _jam(v))]
    return _pick(c, options, pot)


def shove_ev(c, callers):
    """ICM value of shoving now vs folding, given who may call."""
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    hole = v.hole
    p_none = 1.0
    ev = 0.0
    seen = set()
    for s in callers:
        if s in seen or s == v.seat or not v.players[s].in_hand:
            continue
        seen.add(s)
        vil = v.players[s]
        eff_j = min(hero.start, vil.start) / v.bb
        w = PF.call_width(eff_j, s == v.bb_seat, max(1, len(callers)), P)
        w *= max(0.5, min(3.0, c.opps[s].call_jam_ratio()))
        if hero.start > vil.start:
            w *= 0.85  # they risk elimination: calling ranges tighten under ICM
        w = min(w, 1.0)
        eq = g.pre_equity(hole, 0.0, w)
        vw, vl, _ = I.allin_values(v, s)
        ev += p_none * w * (eq * vw + (1 - eq) * vl)
        p_none *= (1 - w)
    ev += p_none * I.value_all_fold(v)
    winner = v.bb_seat if v.bb_seat != v.seat else (callers[0] if callers else v.seat)
    return ev, I.value_fold(v, winner)


# -----------------------------------------------------------------------------
def _facing_raise(c, pct, eff_bb, vil, last, n_raises):
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    bb = v.bb
    lo, hi = preflop_band(v, vil, c.opps[vil], P)
    vp = v.players[vil]
    behind = players_behind(v, v.pre_order)
    callers = [a.seat for a in v.acts[0] if a.kind == CALL and a.seat != v.seat]
    to_call = v.to_call
    pot = v.pot
    commit_frac = (hero.bet + to_call) / max(hero.avail, 1)
    allin_spot = last.allin or vp.stack == 0 or commit_frac >= 0.45 or eff_bb <= P["push_fold_bb"]

    if allin_spot:
        return _allin_vs(c, vil, lo, hi, behind, last)

    # deep: fold / call / 3-bet (or 4-bet when n_raises >= 2) by chip EV
    eq = g.pre_equity(v.hole, lo, hi)
    for s in callers:
        if v.players[s].in_hand:
            clo, chi = preflop_band(v, s, c.opps[s], P)
            eq *= g.pre_equity(v.hole, clo, chi) * 1.15
    eq = min(eq, 0.95)
    ip = _is_ip(v, vil) and all(_is_ip(v, s) for s in callers if v.players[s].in_hand)
    R = _pre_R(c, ip) - 0.05 * len(callers)
    ev_call = eq * R * (pot + to_call) - to_call
    ev_call -= P["multiway_call_margin"] * len(behind) * pot
    rp = I.risk_premium(v, vil, P["icm_scale"])
    ev_call -= rp * commit_frac * (pot + to_call)

    ev_raise = -1e18
    raise_to = None
    if RAISE in v.legal:
        mult = 3.0 if ip else 4.0
        raise_to = last.total * mult + last.total * len(callers)
        jam = raise_to >= 0.45 * hero.avail or eff_bb <= (P["threebet_jam_bb"] if n_raises == 1 else P["fourbet_jam_bb"])
        if jam:
            raise_to = hero.avail
        add = raise_to - hero.bet
        # plan: P(fold), P(call), P(4-bet) from the opener's history (shrunk to priors) ...
        if n_raises == 1:
            pf, pc, p4 = c.opps[vil].response_to_3bet()
            # plan: deviate (bluff more) only as far as the evidence supports -
            # use a pessimistic fold estimate unless the hand is a value 3-bet
            if pct > P["threebet_value_pct"] and P["threebet_conf_z"] > 0:
                n = c.opps[vil].r[8] if c.opps[vil].r is not None else 0  # F3B_OPP
                sd = (pf * (1 - pf) / (n + P["k_rare"])) ** 0.5
                shift = min(pf - 0.05, P["threebet_conf_z"] * sd)
                pf -= shift
                pc += shift
        else:
            pf, pc, p4 = P["prior_fold_to_4bet"], 1.0 - P["prior_fold_to_4bet"] - 0.15, 0.15
        # ... and EVERYONE still to act has to fold too (plan: P(all fold) = product)
        others = [s for s in behind if s != vil] + [s for s in callers if v.players[s].in_hand]
        p_rest = (1.0 - P["behind_continue_vs_3bet"]) ** len(others)
        width = hi - lo
        if jam:
            # they get it in with the top (1 - pf) of their range
            p_all = pf * p_rest
            c_hi = lo + width * (1 - pf)
            eq_c = g.pre_equity(v.hole, lo, c_hi)
            v_add = min(raise_to, vp.avail) - vp.bet
            ev_called = eq_c * (pot + add + v_add) - add
            ev_raise = p_all * pot + (1 - p_all) * ev_called
            # plan: every all-in decision is checked in exact ICM points
            w_icm, l_icm, f_icm = I.allin_values(v, vil)
            icm_jam = p_all * I.value_all_fold(v) + (1 - p_all) * (eq_c * w_icm + (1 - eq_c) * l_icm)
            if icm_jam <= f_icm:
                ev_raise = -1e18
        else:
            p_all = pf * p_rest
            # a player behind who wakes up is treated like a 4-bet (we usually fold)
            p4_eff = p4 + pf * (1 - p_rest)
            pc_eff = max(0.0, 1.0 - p_all - p4_eff)
            c_lo = lo + width * p4
            c_hi = lo + width * (p4 + pc)
            eq_c = g.pre_equity(v.hole, c_lo, max(c_hi, c_lo + 1e-3))
            v_add = raise_to - vp.bet
            R3 = _pre_R(c, ip) + P["threebet_eqr_adj"]
            ev_called = eq_c * R3 * (pot + add + v_add) - add
            eq_4 = g.pre_equity(v.hole, lo, lo + max(width * p4, 1e-3))
            final = 2 * min(hero.avail, vp.avail) + (pot - hero.bet - vp.bet)
            ev_vs4 = max(-add, eq_4 * final - (hero.avail - hero.bet))
            ev_raise = p_all * pot + pc_eff * ev_called + p4_eff * ev_vs4
            ev_raise -= rp * commit_frac * pot
    options = [(0.0, FOLD, None), (ev_call, CALL, None)]
    if raise_to is not None:
        options.append((ev_raise, RAISE, _raise_amount(v, raise_to)))
    return _pick(c, options, pot)


def _facing_reraise(c, pct, eff_bb, vil, last, n_raises):
    """We raised earlier; someone re-raised. Fold, call, or move all-in."""
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    vp = v.players[vil]
    lo, hi = preflop_band(v, vil, c.opps[vil], P)
    behind = players_behind(v, v.pre_order)
    to_call = v.to_call
    pot = v.pot
    commit_frac = (hero.bet + to_call) / max(hero.avail, 1)
    if last.allin or vp.stack == 0 or commit_frac >= 0.35 or eff_bb <= P["fourbet_jam_bb"]:
        return _allin_vs(c, vil, lo, hi, behind, last)
    eq = g.pre_equity(v.hole, lo, hi)
    ip = _is_ip(v, vil)
    R = _pre_R(c, ip, -0.04)
    rp = I.risk_premium(v, vil, P["icm_scale"])
    ev_call = eq * R * (pot + to_call) - to_call - rp * commit_frac * (pot + to_call)
    options = [(0.0, FOLD, None), (ev_call, CALL, None)]
    if RAISE in v.legal:
        pf = P["prior_fold_to_4bet"] if n_raises == 2 else 0.25
        width = hi - lo
        eq_c = g.pre_equity(v.hole, lo, lo + width * (1 - pf))
        add = hero.avail - hero.bet
        v_add = min(hero.avail, vp.avail) - vp.bet
        ev_jam = pf * pot + (1 - pf) * (eq_c * (pot + add + v_add) - add)
        ev_jam -= rp * (1 - pf) * (pot + add + v_add)
        w_icm, l_icm, f_icm = I.allin_values(v, vil)
        if pf * I.value_all_fold(v) + (1 - pf) * (eq_c * w_icm + (1 - eq_c) * l_icm) > f_icm:
            options.append((ev_jam, RAISE, _jam(v)))  # plan: all-ins must win in ICM points
        # plan: deeper than ~40bb a 4-bet is ~2.3x the 3-bet, not a jam
        t4 = P["fourbet_size"] * last.total
        if n_raises == 2 and t4 < 0.45 * hero.avail:
            add4 = t4 - hero.bet
            v_add4 = t4 - vp.bet
            R4 = _pre_R(c, ip)
            ev_4 = pf * pot + (1 - pf) * (eq_c * R4 * (pot + add4 + v_add4) - add4)
            ev_4 -= rp * commit_frac * pot
            options.append((ev_4, RAISE, _raise_amount(v, t4)))
    return _pick(c, options, pot)


def _allin_vs(c, vil, lo, hi, behind, last):
    """Stack-off decision against ``vil`` in ICM points."""
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    vp = v.players[vil]
    vil_allin = last.allin or vp.stack == 0
    v_win, v_lose, v_fold = I.allin_values(v, vil)
    if vil_allin or RAISE not in v.legal:
        # call (or fold): they already have everything in
        eq = g.pre_equity(v.hole, lo, hi)
        if behind:
            eq -= P["multiway_call_margin"] * len(behind)
        ev_call = eq * v_win + (1 - eq) * v_lose
        if ev_call > v_fold:
            # cover them with players still to act: re-jam to push others out
            if RAISE in v.legal and behind and hero.avail > vp.avail:
                return RAISE, _jam(v)
            return CALL, None
        return FOLD, None
    # they are not all in: our shove makes them choose
    eff_v = min(hero.start, vp.start) / v.bb
    cw = PF.call_width(eff_v, vil == v.bb_seat, len(behind) + 1, P) * max(0.5, min(3.0, c.opps[vil].call_jam_ratio()))
    width = max(hi - lo, 1e-3)
    p_call = min(1.0, cw / width) if lo < 0.02 else min(1.0, cw / (hi))
    c_hi = lo + width * p_call
    eq_c = g.pre_equity(v.hole, lo, c_hi)
    v_all_fold = I.value_all_fold(v)
    ev_jam = (1 - p_call) * v_all_fold + p_call * (eq_c * v_win + (1 - eq_c) * v_lose)
    # calling (not jamming) is close to jamming in these spots; compare jam vs call vs fold
    eq = g.pre_equity(v.hole, lo, hi)
    ev_call = eq * v_win + (1 - eq) * v_lose  # pessimistic: assume stacks go in later
    best = max(ev_jam, ev_call, v_fold)
    if best == v_fold:
        return (CHECK, None) if CHECK in v.legal else (FOLD, None)
    if best == ev_jam:
        return RAISE, _jam(v)
    return CALL, None


def _pick(c, options, pot):
    """Best EV option; mix between near-ties so we are harder to read."""
    options = [o for o in options if o[1] in c.v.legal or (o[1] == FOLD and CHECK in c.v.legal)]
    if not options:
        return (CHECK, None) if CHECK in c.v.legal else (FOLD, None)
    options.sort(key=lambda o: -o[0])
    best = options[0]
    if len(options) > 1:
        second = options[1]
        if best[0] - second[0] < c.P["tie_margin"] * max(pot, 1) and c.rng.random() < 0.35:
            best = second
    kind, amount = best[1], best[2]
    if kind == FOLD and CHECK in c.v.legal:
        return CHECK, None
    if kind == RAISE and amount is None:
        return (CALL, None) if CALL in c.v.legal else (CHECK, None)
    return kind, amount
