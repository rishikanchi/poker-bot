"""Postflop decisions (flop, turn, river) by expected value.

The "region model": sort the main opponent's range from weakest to strongest
(plug-in ``scores``). Facing our bet of size s, the weakest fraction F folds,
the strongest fraction Rr raises, the middle calls. F and Rr come from the
opponent model (equilibrium priors shrunk toward what we have observed).
Our equity against each region is measured from the same equity samples.

All EVs are "chips won from now on" (folding = 0):
  check (last to act)  eq * R * pot
  check (they act)     P(check back) * eq_cb * R * pot
                       + P(bet) * max(fold 0, call, check-raise)
  bet b                P(all fold) * pot
                       + P(call) * (eq_call * R * (pot + 2b) - b)
                       + P(raise) * max(-b, eq_raise * (pot + 2T) - T)
  call c               eq * R * (pot + c) - c
  raise to T           P(fold) * pot + P(continue) * (eq_cont * R * final - added)
R = equity realization (EQR) for the streets still to come (1 on the river).
"""
from __future__ import annotations

import numpy as np

from . import eqr as EQ
from . import icmtools as I
from .preflop import CALL, CHECK, FOLD, RAISE, _pick
from . import view as V
from .view import hero_in_position, players_behind, yet_to_act


class Regions:
    """Equity vs slices of the main opponent's range, from one sampling pass."""

    def __init__(self, scores, w_main, vscore=None, res_hu=None, exact_beats=None):
        self.order = np.argsort(scores)
        ww = w_main[self.order]
        tot = ww.sum()
        self.cum = np.cumsum(ww) / tot if tot > 0 else np.linspace(0, 1, ww.size)
        self.sorted_scores = scores[self.order]
        self.ww = ww
        self.vscore = vscore
        self.res = res_hu
        self.beats = None if exact_beats is None else exact_beats[self.order]
        if self.res is not None and self.res.size:
            self.eq = float(self.res.mean())
        elif self.beats is not None and tot > 0:
            self.eq = float((self.beats * ww).sum() / tot)
        else:
            self.eq = 0.5

    def threshold(self, q):
        """Score value at cumulative-weight quantile q (0 weakest .. 1 strongest)."""
        i = int(np.searchsorted(self.cum, min(max(q, 0.0), 1.0)))
        i = min(i, self.sorted_scores.size - 1)
        return float(self.sorted_scores[i])

    def eq_between(self, q_lo, q_hi):
        """Hero's equity vs the opponent hands in strength quantiles [q_lo, q_hi]."""
        q_lo = max(0.0, q_lo)
        q_hi = min(1.0, q_hi)
        if q_hi <= q_lo + 1e-6:
            q_hi = min(1.0, q_lo + 0.02)
            q_lo = max(0.0, q_hi - 0.04)
        if self.beats is not None:
            m = (self.cum >= q_lo) & (self.cum <= q_hi + 1e-9)
            wsum = (self.ww * m).sum()
            if wsum <= 0:
                return self.eq
            return float((self.beats * self.ww * m).sum() / wsum)
        t_lo, t_hi = self.threshold(q_lo), self.threshold(q_hi)
        m = (self.vscore >= t_lo - 1e-9) & (self.vscore <= t_hi + 1e-9)
        n = int(m.sum())
        prior_n = 6.0
        # shrink toward a straight line between "beats everything weak" and the overall eq
        guess = self.eq * (1.4 - (q_lo + q_hi) * 0.8)
        guess = min(max(guess, 0.0), 1.0)
        return float((self.res[m].sum() + prior_n * guess) / (n + prior_n))


def hero_draw_flags(hole, board, rule_omaha=False):
    """(flush_draw, straight_outs) for hero on a non-final street."""
    if len(board) >= 5 or len(board) < 3:
        return False, 0
    hs = [0, 0, 0, 0]
    bs = [0, 0, 0, 0]
    for c in hole:
        hs[c & 3] += 1
    for c in board:
        bs[c & 3] += 1
    fd = False
    for s in range(4):
        cnt = (min(hs[s], 2) + min(bs[s], 3)) if rule_omaha else hs[s] + bs[s]
        if hs[s] > 0 and cnt == 4:
            fd = True
    bm = 0
    for c in board:
        bm |= 1 << (c >> 2)
    hm = bm
    for c in hole:
        hm |= 1 << (c >> 2)

    def straight(m):
        for t in range(12, 3, -1):
            need = 0b11111 << (t - 4)
            if m & need == need:
                return True
        return m & ((1 << 12) | 0b1111) == ((1 << 12) | 0b1111)

    outs = 0
    if not straight(hm):
        for r in range(13):
            b = 1 << r
            if not hm & b and straight(hm | b) and not straight(bm | b):
                outs += 1
    return fd, outs


def eqr(c, eq, ip, spr, street):
    """Equity realization for the rest of the hand.

    Prior (plan): 1.0 in position, 0.85 out of position, +0.1 strong draws /
    near-nut hands, -0.1 weak made hands, shrinking toward 1 at low SPR.
    Measured (plan): data/eqr_<variant>.json from tools/measure_eqr.py,
    blended with the prior (weight eqr_measured_weight) and capped, because it
    was measured against our caricature test bots, not the real field.
    """
    P = c.P
    if street >= 3:
        return 1.0
    r = P["eqr_ip"] if ip else P["eqr_oop"]
    fd, outs = hero_draw_flags(c.v.hole, c.v.board, c.game.rule == 1)
    draw = fd or outs >= 2
    if draw:
        r += P["eqr_draw"]
    if eq >= 0.75:
        r += P["eqr_strong"]
    elif 0.35 <= eq < 0.62 and not draw:
        r += P["eqr_weak_made"]
    full = P["eqr_spr_full"]
    r = 1.0 + (r - 1.0) * min(1.0, spr / full)
    prior = float(min(max(r, 0.6), 1.25))
    wm = P.get("eqr_measured_weight", 0.0) if P.get("use_measured_eqr") else 0.0
    if wm > 0:
        try:
            made = EQ.hero_made_pct(c.game, c.v.hole, c.v.board)
            m = EQ.lookup_post(street, ip, made, draw, spr)
        except Exception:  # noqa: BLE001
            m = None
        if m is not None:
            m = min(max(m, 0.3), P["eqr_measured_cap"])
            return (1 - wm) * prior + wm * m
    return prior


def decide(c, ranges, scores):
    """ranges: seat -> weights; scores: current-street strength scores."""
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    opp_seats = [s for s in ranges if v.players[s].in_hand]
    if not opp_seats:
        return (CHECK, None) if CHECK in v.legal else (CALL, None)
    street = v.street
    pot = max(v.pot, 1)
    to_call = v.to_call
    street_acts = v.acts[street]
    aggressors = [a.seat for a in street_acts if a.kind == RAISE and a.seat != v.seat]
    behind = [s for s in players_behind(v, v.post_order) if s in opp_seats]
    if aggressors and to_call > 0:
        main = aggressors[-1]
    elif behind:
        main = behind[0]
    else:
        main = opp_seats[0]
    others = [s for s in opp_seats if s != main]
    w_main = ranges[main]
    if w_main.sum() <= 0:
        w_main = (scores >= 0).astype(np.float64)
    weights = [w_main] + [ranges[s] if ranges[s].sum() > 0 else (scores >= 0) * 1.0 for s in others]

    # ---- equity ---------------------------------------------------------
    if street >= 3 or len(v.board) >= g.board_total:
        vals = getattr(c, "river_vals", None)
        if vals is None:
            vals = g.strengths(v.board, g.dead_mask(v.hole + v.board))
        hv = g.hand_value(v.hole, v.board)
        beats = np.where(vals < 0, 0.0, np.where(vals < hv, 1.0, np.where(vals == hv, 0.5, 0.0)))
        reg = Regions(scores, w_main, exact_beats=beats)
        eq_hu = reg.eq
        eq_all = eq_hu
        for w in weights[1:]:
            tw = w.sum()
            eq_all *= float((beats * w).sum() / tw) if tw > 0 else 1.0
    else:
        seed = V.stable_seed(v.table_id, v.hand_number, street, len(street_acts))
        vscore, res_hu, res_all = g.equity(v.hole, v.board, weights, c.n_samples, scores, seed)
        reg = Regions(scores, w_main, vscore=vscore, res_hu=res_hu)
        eq_hu = reg.eq
        eq_all = float(res_all.mean())
    mw = eq_all / eq_hu if eq_hu > 1e-6 else 1.0
    c.eq = eq_all

    ip = hero_in_position(v)
    to_act = [s for s in yet_to_act(v) if s in opp_seats and s != main]
    vstack = max(v.players[s].stack for s in opp_seats)
    eff = min(hero.stack, vstack)
    spr = eff / pot
    R = eqr(c, eq_all, ip, spr, street)
    om = c.opps[main]
    rp = I.risk_premium(v, main, P["icm_scale"]) if eff > 0 else 0.0
    hero_total = max(hero.stack + hero.bet, 1)
    # plan: the ICM risk premium grows as the hand cap approaches
    hands_left = max(v.hand_cap - v.hand_number, 0)
    rp *= 1.0 + P["cap_rp_boost"] * max(0.0, 1.0 - hands_left / max(P["cap_rp_hands"], 1))
    final_street = street >= 3 or len(v.board) >= g.board_total
    c.reg, c.w_main, c.main, c.final_street = reg, w_main, main, final_street

    options = []
    if to_call <= 0:
        # ---- check ------------------------------------------------------
        if not behind:
            ev_check = eq_all * R * pot
        else:
            bwc = om.bet_when_checked()
            B = min(0.66 * pot, v.players[main].stack, hero.stack)
            bs = om.river_bluff_share(0.66) if final_street else 0.5 * 0.66 / (1 + 2 * 0.66)
            v_frac = bwc * (1 - bs)
            b_frac = bwc * bs
            eq_cb = reg.eq_between(b_frac, 1 - v_frac) * mw
            eq_val = reg.eq_between(1 - v_frac, 1.0)
            eq_blf = reg.eq_between(0.0, b_frac)
            eq_b = ((v_frac * eq_val + b_frac * eq_blf) / max(bwc, 1e-6)) * mw
            Rb = 1.0 if B >= 0.9 * hero.stack else R
            ev_call_b = eq_b * Rb * (pot + 2 * B) - B
            # check-raise to call + raise_pot_frac * (pot after calling); jam if little left
            T = B + P["raise_pot_frac"] * (pot + 2 * B)
            if hero.stack - T < P["raise_jam_left_frac"] * (pot + 2 * T):
                T = hero.stack
            T = min(T, hero.stack)
            f2 = min(0.9, b_frac / max(bwc, 1e-6) + 0.25 * v_frac / max(bwc, 1e-6))
            fx = (T - B) / max(pot + B + T, 1)  # what they must add / pot they would win
            f2 = min(f2, fx / (1 + fx) + 0.1)
            ev_xr = f2 * (pot + B) + (1 - f2) * (eq_val * mw * (pot + 2 * T) - T) if T > B else -1e9
            ev_check = (1 - bwc) * eq_cb * R * pot + bwc * max(0.0, ev_call_b, ev_xr)
        options.append((ev_check, CHECK, None))
        # ---- bets: 33% / 66% / 125% (33/66 without the initiative), all-in at low SPR
        if RAISE in v.legal and hero.stack > 0:
            lo, hi = v.legal[RAISE]
            if _has_initiative(v):
                sizes = P["bet_sizes"].get(min(street, 3), (0.33, 0.66, 1.25))
            else:
                sizes = P["bet_sizes_no_initiative"]
            bets = []
            for sz in sizes:
                bet = int(sz * pot)
                if hero.stack - bet < P["jam_left_frac"] * (pot + 2 * bet):
                    bet = hero.stack
                bets.append(bet)
            if spr < P["allin_spr"]:
                bets.append(hero.stack)
            seen = set()
            for bet in bets:
                bet = min(bet, hero.stack, vstack) if vstack > 0 else min(bet, hero.stack)
                total = int(min(max(hero.bet + bet, lo), hi))
                if total in seen:
                    continue
                seen.add(total)
                bet = total - hero.bet
                frac = bet / pot
                if final_street and P["river_bluff_cap"] and not _river_bet_allowed(c, frac, eq_hu):
                    continue
                ev = _ev_bet(c, reg, om, others, street, pot, bet, frac, mw, R, eq_all, final_street)
                ev -= rp * (bet / hero_total) * (pot + 2 * bet)
                options.append((ev, RAISE, total))
    else:
        # ---- facing a bet: fold / call / raise --------------------------
        options.append((0.0, FOLD, None))
        call_amt = min(to_call, hero.stack)
        allin_call = to_call >= hero.stack
        Rc = 1.0 if allin_call else R
        ev_call = eq_all * Rc * (pot + call_amt) - call_amt
        ev_call -= P["multiway_call_margin"] * len(to_act) * pot
        ev_call -= rp * (call_amt / hero_total) * (pot + call_amt)
        options.append((ev_call, CALL, None))
        if RAISE in v.legal and not allin_call:
            lo, hi = v.legal[RAISE]
            # plan: raise to call + 0.75 * (pot after calling); all-in if < ~1 pot would be left
            T = hero.bet + to_call + P["raise_pot_frac"] * (pot + to_call)
            if (hero.bet + hero.stack) - T < P["raise_jam_left_frac"] * (pot + 2 * T):
                T = hero.bet + hero.stack
            T = int(min(max(T, lo), hi))
            add = T - hero.bet
            vp = v.players[main]
            v_add = max(0, min(T, vp.bet + vp.stack) - vp.bet)
            f2 = om.fold_to_raise(v_add / max(pot + add, 1))
            if len(others) > 0:
                f2 *= 0.85 ** len(others)
            eq_c2 = reg.eq_between(f2, 1.0) * mw
            R2 = 1.0 if T >= hero.bet + hero.stack else R
            ev_r = f2 * pot + (1 - f2) * (eq_c2 * R2 * (pot + add + v_add) - add)
            ev_r -= rp * (add / hero_total) * (pot + add + v_add)
            options.append((ev_r, RAISE, T))
    if P.get("postflop_icm_allin", True):
        options = _icm_filter(c, options, main, others, reg, mw, eq_all, pot, street)
    kind, amount = _pick(c, options, pot)
    # plan: protect the checking range - check a share of very strong hands
    if (kind == RAISE and to_call <= 0 and CHECK in v.legal and eq_all >= P["trap_eq"]
            and c.rng.random() < P["trap_share"]):
        return CHECK, None
    return kind, amount


def _icm_filter(c, options, main, others, reg, mw, eq_all, pot, street):
    """Plan: every all-in decision is checked in exact ICM points.

    all-in call : eq * V(win) + (1 - eq) * V(lose) must beat V(fold)
    all-in bet  : P(all fold) * V(win pot) + P(called) * (...) must beat the
                  alternative (fold / check), the check valued as V(fold) plus
                  its chip EV at the current ICM value of a chip.
    """
    v = c.v
    hero = v.hero
    allin_total = hero.bet + hero.stack
    try:
        w_icm, l_icm, f_icm = I.allin_values(v, main)
        v_win_pot = I.value_all_fold(v)
    except Exception:  # noqa: BLE001
        return options
    per_chip = (v_win_pot - f_icm) / max(pot, 1)
    ev_check = next((ev for ev, k, _ in options if k == CHECK), None)
    base = f_icm if ev_check is None else f_icm + per_chip * max(ev_check, 0.0)
    out = []
    for ev, kind, amt in options:
        if kind == CALL and v.to_call >= hero.stack:
            if eq_all * w_icm + (1 - eq_all) * l_icm <= f_icm:
                continue
        elif kind == RAISE and amt is not None and amt >= allin_total:
            om = c.opps[main]
            vp = v.players[main]
            add = amt - hero.bet
            if v.to_call > 0:
                v_add = max(0, min(amt, vp.bet + vp.stack) - vp.bet)
                F = om.fold_to_raise(v_add / max(pot + add, 1))
            else:
                F = om.fold_to_bet(street, add / max(pot, 1))
            for s in others:
                F *= c.opps[s].fold_to_bet(street, add / max(pot, 1))
            eq_c = min(1.0, reg.eq_between(F, 1.0) * mw)
            if F * v_win_pot + (1 - F) * (eq_c * w_icm + (1 - eq_c) * l_icm) <= base:
                continue
        out.append((ev, kind, amt))
    return out or options


def _has_initiative(v):
    """Hero made the last bet/raise of the previous street (preflop on the flop)."""
    prev = v.street - 1
    while prev >= 0:
        raises = [a.seat for a in v.acts[prev] if a.kind == RAISE]
        if raises:
            return raises[-1] == v.seat
        if prev == 0:
            return False
        prev -= 1
    return False


def _ev_bet(c, reg, om, others, street, pot, bet, frac, mw, R, eq_all, final_street):
    P = c.P
    f_point = om.fold_to_bet(street, frac)
    f_pess = om.fold_to_bet(street, frac, pessimistic=True)
    rr = om.raise_vs_bet()
    hero = c.v.hero
    best = -1e18
    for f_m in (f_point, f_pess):
        p_all_fold = f_m
        others_cont = 0.0
        for s in others:
            fo = c.opps[s].fold_to_bet(street, frac)
            p_all_fold *= fo
            others_cont = 1 - (1 - others_cont) * fo
        mwf = 1.0 + (mw - 1.0) * others_cont
        eq_c = reg.eq_between(f_m, 1 - rr) * mwf
        eq_r = reg.eq_between(1 - rr, 1.0) * mwf
        Tm = min(bet + P["raise_pot_frac"] * (pot + 2 * bet), hero.stack + hero.bet)
        Rc = 1.0 if bet >= hero.stack else R
        p_call = max(0.0, 1 - p_all_fold - rr)
        ev = (p_all_fold * pot
              + p_call * (eq_c * Rc * (pot + 2 * bet) - bet)
              + rr * max(-bet, eq_r * (pot + 2 * Tm) - Tm))
        # plan: fold equity on a LATER street for hands with some equity (draws,
        # overcards) - they call now and give up to the next bet; discounted.
        if not final_street and P["multistreet_lo"] <= eq_all <= P["multistreet_hi"] and bet < hero.stack:
            f_next = om.fold_to_bet(min(street + 1, 3), 0.66)
            pot_after = pot + 2 * bet
            b2 = min(0.66 * pot_after, hero.stack - bet)
            ev += P["multistreet_disc"] * p_call * max(
                0.0, f_next * pot_after - (1 - f_next) * (1 - eq_c) * b2)
        # bluffs (mostly fold equity) use the pessimistic fold estimate
        if f_m is f_point and eq_all >= 0.4:
            return ev
        best = ev
    return best


def _river_bet_allowed(c, frac, eq_hu):
    """River bluff rule from the plan.

    Value bets are left to EV. A hand with no showdown value may bet only if
    (1) the opponent's fold chance, given the cards we hold (blockers), beats
        the break-even rate alpha = s / (1 + s), and
    (2) it is among our best-blocking bluffs, with bluffs capped at
        s / (1 + 2s) of our betting range (value : bluff = (1 + s) : s).
    """
    P = c.P
    v, g = c.v, c.game
    reg, w_main, om = c.reg, c.w_main, c.opps[c.main]
    F = om.fold_to_bet(3, frac)
    rr = om.raise_vs_bet()
    eq_call = reg.eq_between(F, 1 - rr)
    if eq_call > 0.5 or eq_hu >= P["river_sdv_eq"]:
        return True  # value hand, or showdown value: let EV decide
    alpha = frac / (1.0 + frac)
    # villain fold region (weakest F of their range by strength)
    order = reg.order
    fold_mask = np.zeros(g.M, bool)
    fold_mask[order] = reg.cum <= F
    wf = w_main * fold_mask
    U = g.U
    card_t = np.zeros(52)
    card_f = np.zeros(52)
    for i in range(g.H):
        np.add.at(card_t, U[:, i], w_main)
        np.add.at(card_f, U[:, i], wf)
    Wt, Wf = w_main.sum(), wf.sum()
    if Wt <= 0:
        return False

    def pfold(cards):
        t = Wt - sum(card_t[x] for x in cards)
        f = Wf - sum(card_f[x] for x in cards)
        return f / t if t > 0 else F

    pf_hero = pfold(v.hole)
    if pf_hero <= alpha:
        return False
    if getattr(c, "deadline", None) is not None and V.time_ns() > c.deadline - 150_000:
        return True  # no time to size the cap: the alpha test alone (EV still decides)
    # our own river range, modelled the way we model opponents (equilibrium priors)
    from .ranges import seat_range
    from .stats import Opp
    bdead = g.dead_mask(v.board)
    hw = seat_range(v, g, v.seat, Opp(None, P), P, c.scores_by_street, c.orders, bdead)
    vals = g.strengths(v.board, bdead)
    live = vals >= 0
    hw = hw * live
    # share of the villain's calling region each of our hands beats
    call_mask = np.zeros(g.M, bool)
    call_mask[order] = (reg.cum > F) & (reg.cum <= 1 - rr)
    wc = w_main * call_mask
    def share_vs(wv):
        idx = np.argsort(vals)
        sv, cw = vals[idx], np.cumsum(wv[idx])
        tot = cw[-1] if cw.size else 0.0
        if tot <= 0:
            return np.zeros(g.M)
        pos = np.searchsorted(sv, vals, side="left")
        below = np.where(pos > 0, cw[np.maximum(pos - 1, 0)], 0.0)
        return below / tot
    share_call = share_vs(wc)
    share_full = share_vs(w_main)
    value_w = float((hw * (share_call > 0.5)).sum())
    budget = value_w * frac / (1.0 + frac)
    cand = (share_full < P["river_sdv_eq"]) & (hw > 0)
    if not cand.any():
        return True
    ci = np.nonzero(cand)[0]
    tsum = np.zeros(ci.size)
    fsum = np.zeros(ci.size)
    for i in range(g.H):
        tsum += card_t[U[ci, i]]
        fsum += card_f[U[ci, i]]
    pf_c = np.where(Wt - tsum > 0, (Wf - fsum) / np.maximum(Wt - tsum, 1e-12), F)
    ok = pf_c > alpha
    ci, pf_c = ci[ok], pf_c[ok]
    if ci.size == 0:
        return False
    srt = np.argsort(-pf_c)
    cw = np.cumsum(hw[ci][srt])
    if cw[-1] <= budget:
        return True  # every qualifying bluff fits under the cap
    k = int(np.searchsorted(cw, budget))
    if k <= 0:
        return False
    threshold = pf_c[srt][k - 1]
    return pf_hero >= threshold
