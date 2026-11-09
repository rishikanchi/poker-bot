"""Postflop decisions (flop, turn, river) by expected value.

The plan's region model: order the main opponent's range from weakest to
strongest (plug-in ``scores``: fitted equity before the river, hand rank on
the river). Facing our bet of size s, the weakest share F folds, the
strongest share r raises and the middle calls. F and r come from the
opponent model (equilibrium priors shrunk toward what we have observed).

Blockers: the regions are cut on the opponent's range as THEY play it; the
probability that they are in a region, and our equity against it, are taken
given the cards we hold.

All EVs are "chips won from now on" (folding = 0):
  check (we close the action)  eq * R * pot
  check (they act after us)    P(check back) * eq_cb * R * pot
                               + P(bet) * max(fold 0, call, check-raise)
  bet b                        P(all fold) * pot
                               + P(call) * (eq_call * R * (pot + 2b) - b)
                               + P(raise) * max(-b, eq_raise * (pot + 2T) - T)
  call c                       eq * R * (pot + c) - c
  raise to T                   P(all fold) * pot + P(continue) * (eq_cont * R * final - added)
R = equity realization (EQR) for the streets still to come (1 on the river).
A bet is "for value" when it wins more than half the time when called;
other bets may count on folds only as far as the evidence supports (a
pessimistic fold estimate). Every option carries the ICM risk premium for
the chips it puts at stake, and every all-in must win in exact ICM points.
"""
from __future__ import annotations

import numpy as np

from . import eqr as EQ
from . import icmtools as I
from . import view as V
from .preflop import CALL, CHECK, FOLD, RAISE, _pick
from .ranges import bluff_prior, call_floor
from .view import hero_in_position, players_behind, yet_to_act


class Regions:
    """The main opponent's range split by strength, with our blockers.

    order   universe hands from weakest to strongest on this street
    w_u     their range as they play it (only the board removed)
    live_h  1 for hands that do not use our hole cards
    Either ``beats`` (river: 1 / 0.5 / 0 per hand, exact) or Monte Carlo
    samples (vidx, res_hu, res_all) drawn from the range with our cards removed.
    """

    def __init__(self, order, w_u, live_h, prior_n, beats=None, mw_exact=1.0,
                 vidx=None, res_hu=None, res_all=None):
        ww = w_u[order]
        tot = ww.sum()
        M = ww.size
        self.cum = np.cumsum(ww) / tot if tot > 0 else np.linspace(1.0 / M, 1.0, M)
        self.cum[-1] = 1.0   # exact top edge (no float drift)
        wc = ww * live_h[order]
        self.cw = np.concatenate(([0.0], np.cumsum(wc)))
        self.tot_c = float(self.cw[-1])
        self.M = M
        self.prior_n = prior_n
        self.exact = beats is not None
        if self.exact:
            self.cb = np.concatenate(([0.0], np.cumsum(wc * beats[order])))
            self.mw = mw_exact
            self.eq_hu = float(self.cb[-1] / self.tot_c) if self.tot_c > 0 else 0.5
            self.eq_all = self.eq_hu * mw_exact
        else:
            q = np.empty(M)
            q[order] = self.cum
            qs = q[vidx]
            so = np.argsort(qs, kind="stable")
            self.qs = qs[so]
            hu, al = res_hu[so], res_all[so]
            self.ch = np.concatenate(([0.0], np.cumsum(hu)))
            self.ca = np.concatenate(([0.0], np.cumsum(al)))
            self.eq_hu = float(hu.mean()) if hu.size else 0.5
            self.eq_all = float(al.mean()) if al.size else 0.5
            # straight-line fit of our result on their strength quantile: the
            # prior for regions that hold only a few samples
            xm = float(self.qs.mean()) if hu.size else 0.5
            var = float(((self.qs - xm) ** 2).mean()) if hu.size else 0.0
            if var > 1e-9:
                self.b_hu = float(((self.qs - xm) * (hu - self.eq_hu)).mean() / var)
                self.b_al = float(((self.qs - xm) * (al - self.eq_all)).mean() / var)
            else:
                self.b_hu = self.b_al = 0.0
            self.a_hu = self.eq_hu - self.b_hu * xm
            self.a_al = self.eq_all - self.b_al * xm

    def _i(self, q):
        if q >= 1.0:
            return self.M
        if q <= 0.0:
            return 0
        return int(np.searchsorted(self.cum, q, side="right"))

    def mass(self, lo, hi):
        """P(their hand is in strength quantiles (lo, hi] | our cards)."""
        if self.tot_c <= 0 or hi <= lo:
            return 0.0
        return float(self.cw[self._i(hi)] - self.cw[self._i(lo)]) / self.tot_c

    def eq(self, lo, hi, p_multi=0.0):
        """Our equity against their hands in quantiles (lo, hi].

        p_multi: probability that the other opponents are still in the pot
        (0 = heads-up against the main opponent, 1 = everyone)."""
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        if self.exact:
            a, b = self._i(lo), self._i(hi)
            wsum = self.cw[b] - self.cw[a]
            if wsum <= 0:
                e = self.eq_hu
            else:
                e = float((self.cb[b] - self.cb[a]) / wsum)
            return e * (1.0 + (self.mw - 1.0) * p_multi)
        a = 0 if lo <= 0.0 else int(np.searchsorted(self.qs, lo, side="right"))
        b = self.qs.size if hi >= 1.0 else int(np.searchsorted(self.qs, hi, side="right"))
        n = b - a
        mid = 0.5 * (lo + hi)
        k = self.prior_n
        e_hu = (self.ch[b] - self.ch[a] + k * min(max(self.a_hu + self.b_hu * mid, 0.0), 1.0)) / (n + k)
        if p_multi <= 0:
            return float(e_hu)
        e_al = (self.ca[b] - self.ca[a] + k * min(max(self.a_al + self.b_al * mid, 0.0), 1.0)) / (n + k)
        return float((1.0 - p_multi) * e_hu + p_multi * e_al)


def hero_draw_flags(hole, board, rule_omaha=False):
    """(flush_draw, straight_outs, nut) for hero on a non-final street; ``nut``
    says the draw is to the nuts (the plan separates nut draws from the rest)."""
    fd, outs, nut = _hero_draw_flags(hole, board, rule_omaha)
    return fd, outs, nut


def _nut_flush_draw(hole, board, suit):
    """Do we hold the highest card of ``suit`` that is not on the board?"""
    on_board = {c >> 2 for c in board if (c & 3) == suit}
    ours = max((c >> 2 for c in hole if (c & 3) == suit), default=-1)
    best_out = max((r for r in range(13) if r not in on_board), default=-1)
    return ours >= best_out


def _hero_draw_flags(hole, board, rule_omaha=False):
    if len(board) >= 5 or len(board) < 3:
        return False, 0, False
    hs = [0, 0, 0, 0]
    bs = [0, 0, 0, 0]
    for c in hole:
        hs[c & 3] += 1
    for c in board:
        bs[c & 3] += 1
    fd = False
    nut = False
    for s in range(4):
        if rule_omaha:
            cnt = 2 + min(bs[s], 3) if hs[s] >= 2 else 0   # a flush needs two hole cards of the suit
        else:
            cnt = hs[s] + bs[s]
        if hs[s] > 0 and cnt == 4:
            fd = True
            nut = nut or _nut_flush_draw(hole, board, s)
    bm = 0
    for c in board:
        bm |= 1 << (c >> 2)
    hm = bm
    hole_m = 0
    for c in hole:
        hm |= 1 << (c >> 2)
        hole_m |= 1 << (c >> 2)
    if rule_omaha:
        # exactly two hole cards play: count the ranks that complete a straight
        from .kernels import omaha_straight
        from .game import get_game
        POP = get_game().POP
        outs = 0
        if not omaha_straight(hole_m, bm, POP):
            for r in range(13):
                b = 1 << r
                if not bm & b and omaha_straight(hole_m, bm | b, POP):
                    outs += 1
        return fd, outs, nut           # Omaha straight draws: nut status not resolved (flush only)

    def straight(m):
        for t in range(12, 3, -1):
            need = 0b11111 << (t - 4)
            if m & need == need:
                return True
        return m & ((1 << 12) | 0b1111) == ((1 << 12) | 0b1111)

    def top(m):                                   # highest straight top rank in mask m, or -1
        for t in range(12, 3, -1):
            need = 0b11111 << (t - 4)
            if m & need == need:
                return t
        return 3 if m & ((1 << 12) | 0b1111) == ((1 << 12) | 0b1111) else -1

    outs = 0
    nut_outs = 0
    if not straight(hm):
        for r in range(13):
            b = 1 << r
            if not hm & b and straight(hm | b) and not straight(bm | b):
                outs += 1
                # the nut straight with this card: the highest window with 3+ board ranks
                bm2 = bm | b
                best = -1
                for t in range(12, 3, -1):
                    need = 0b11111 << (t - 4)
                    if bin(bm2 & need).count("1") >= 3:
                        best = t
                        break
                if best < 0 and bin(bm2 & ((1 << 12) | 0b1111)).count("1") >= 3:
                    best = 3
                if top(hm | b) >= best:
                    nut_outs += 1
    if outs >= 2 and nut_outs >= 2:
        nut = True
    return fd, outs, nut


def eqr(c, eq, ip, spr, final_street, n_extra=0, first=False):
    """Equity realization for the rest of the hand (plan priors).

    1.0 in position, 0.85 out of position; +0.1 for strong draws and near-nut
    hands, -0.1 for weak made hands; multiway +-0.05 per extra opponent
    (strong hands and draws gain, the rest lose), last to act +0.05, first
    to act -0.05; all of it shrinking toward 1 as the SPR falls.
    """
    P = c.P
    if final_street:
        return 1.0
    r = P["eqr_ip"] if ip else (P["eqr_oop_turn"] if c.v.street >= 2 else P["eqr_oop"])
    fd, outs, nut = hero_draw_flags(c.v.hole, c.v.board, c.game.rule == 1)
    draw = fd or outs >= 2
    strong = eq >= P["eqr_strong_eq"]
    # plan: nut draws and near-nut hands realize more than their equity; non-nut
    # draws and weak made hands less
    if strong or (draw and nut):
        r += max(P["eqr_draw"] if (draw and nut) else 0.0, P["eqr_strong"] if strong else 0.0)
    elif draw:
        r += P["eqr_nonnut_draw"]
    elif P["eqr_weak_lo"] <= eq < P["eqr_weak_hi"]:
        r += P["eqr_weak_made"]
    if n_extra > 0:
        r += (1.0 if (draw or strong) else -1.0) * P["eqr_multiway"] * n_extra
        if ip:
            r += P["eqr_multiway_pos"]
        elif first:
            r -= P["eqr_multiway_pos"]
    r = 1.0 + (r - 1.0) * min(1.0, spr / P["eqr_spr_full"])
    prior = float(min(max(r, 0.6), 1.3))
    wm = P.get("eqr_measured_weight", 0.0) if P.get("use_measured_eqr") else 0.0
    if wm > 0:
        try:
            made = EQ.hero_made_pct(c.game, c.v.hole, c.v.board)
            m = EQ.lookup_post(1 if len(c.v.board) <= 3 else 2, ip, made, draw, spr)
        except Exception:  # noqa: BLE001
            m = None
        if m is not None:
            return (1 - wm) * prior + wm * min(max(m, 0.3), P["eqr_measured_cap"])
    return prior


def decide(c, ranges, scores):
    """ranges: seat -> range as that opponent plays it (board removed only);
    scores: strength of every universe hand on this street."""
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
        # we close the action: model the first opponent who can still respond
        actors = [s for s in opp_seats if v.players[s].can_act]
        main = actors[0] if actors else opp_seats[0]
    others = [s for s in opp_seats if s != main]
    order = c.orders[street]
    live_h = g.live(g.dead_mask(v.hole)).astype(np.float64)
    live_board = (scores >= 0).astype(np.float64)
    w_u = ranges[main]
    if (w_u * live_h).sum() <= 0:
        w_u = live_board
    others_c = []
    for s in others:
        w = ranges[s] * live_h
        others_c.append(w if w.sum() > 0 else live_board * live_h)
    final_street = street >= 3 or len(v.board) >= g.board_total

    # ---- equity ---------------------------------------------------------
    if final_street:
        vals = getattr(c, "river_vals", None)
        if vals is None:
            vals = g.strengths(v.board, g.dead_mask(v.board))
        hv = g.hand_value(v.hole, v.board)
        beats = np.where(vals < 0, 0.0, np.where(vals < hv, 1.0, np.where(vals == hv, 0.5, 0.0)))
        mw = 1.0
        for w in others_c:
            mw *= float((beats * w).sum() / w.sum())
        reg = Regions(order, w_u, live_h, P["region_prior_n"], beats=beats, mw_exact=mw)
    else:
        seed = V.stable_seed(v.table_id, v.hand_number, street, len(street_acts))
        vidx, res_hu, res_all = g.equity(v.hole, v.board, [w_u * live_h] + others_c, c.n_samples, seed)
        reg = Regions(order, w_u, live_h, P["region_prior_n"], vidx=vidx, res_hu=res_hu, res_all=res_all)
    eq_all = reg.eq_all
    c.eq = eq_all
    pm = 1.0 if others else 0.0   # the other opponents are still in the pot

    ip = hero_in_position(v)
    in_hand_order = [s for s in v.post_order if v.players[s].in_hand]
    first = bool(in_hand_order) and in_hand_order[0] == v.seat
    to_act = [s for s in yet_to_act(v) if s in opp_seats and s != main]
    vp = v.players[main]
    vstack = max(v.players[s].stack for s in opp_seats)
    eff = min(hero.stack, vstack)
    eff_main = min(hero.stack, vp.stack)   # the most that can still go in against the main opponent
    spr = eff / pot
    R = eqr(c, eq_all, ip, spr, final_street, len(others), first)
    om = c.opps[main]
    icm = c.icm
    lean = V.time_ns() > getattr(c, "hard_deadline", float("inf"))   # plan: simpler policy when time runs out
    _fd, _outs, hero_nut = hero_draw_flags(v.hole, v.board, g.rule == 1) if not final_street else (False, 0, False)
    hero_draw = _fd or _outs >= 2
    scale = c.icm_scale        # plan: the ICM risk premium grows as the hand cap approaches
    allin_total = hero.bet + hero.stack
    win, lose, v_fold, v_allfold = icm.allin(main)
    per_chip = (v_allfold - v_fold) / pot   # ICM points per chip of the pot, for comparisons

    def others_fold(frac):
        p = 1.0
        for s in others:
            p *= c.opps[s].fold_to_bet(street, frac, allin=frac * pot >= hero.stack)
        return p

    options = []
    if to_call <= 0:
        # ---- check ------------------------------------------------------
        if not behind:
            ev_check = eq_all * R * pot
        else:
            bwc = om.bet_when_checked()
            bf = om.usual_bet_frac()                       # plan: their usual size
            B = min(bf * pot, vp.stack, hero.stack)
            bs = om.river_bluff_share(bf) if final_street else om.bet_bluff_share(street, bf)
            v_frac = bwc * (1 - bs)          # they bet their strongest v_frac for value ...
            b_frac = bwc * bs                # ... and their weakest b_frac as bluffs (plan)
            p_v = reg.mass(1 - v_frac, 1.0)
            p_b = reg.mass(0.0, b_frac)
            p_bet = p_v + p_b
            eq_cb = reg.eq(b_frac, 1 - v_frac, pm)
            ev_vs_bet = 0.0
            if p_bet > 0 and B > 0:
                eq_val = reg.eq(1 - v_frac, 1.0, pm)
                eq_blf = reg.eq(0.0, b_frac, pm)
                eq_b = (p_v * eq_val + p_b * eq_blf) / p_bet
                T = B + P["raise_pot_frac"] * (pot + 2 * B)     # plan: raise to call + 0.75 pot
                if eff_main - T < P["raise_jam_left_frac"] * (pot + 2 * T):
                    T = eff_main
                T = min(T, eff_main)
                prem = icm.premiums(main, [(B, B), (T, T), (eff_main, eff_main)], scale)
                Rb = 1.0 if B >= eff_main else R
                ev_call_b = eq_b * Rb * (pot + 2 * B) - B - prem[0] * (pot + 2 * B)
                ev_xr = -1e18
                if T > B and not lean:
                    # plan: their response to the raise comes from the region model
                    # applied to their betting range: its weakest share folds
                    fx = (T - B) / max(pot + B + T, 1)
                    fold_u = om.fold_to_raise(fx) * bwc     # as a share of their whole range
                    if fold_u <= b_frac:
                        p_cont = p_v + reg.mass(fold_u, b_frac)
                        eq_cont = ((p_v * eq_val + reg.mass(fold_u, b_frac) * reg.eq(fold_u, b_frac, pm))
                                   / max(p_cont, 1e-12))
                    else:
                        cut = 1 - v_frac + (fold_u - b_frac)
                        p_cont = reg.mass(cut, 1.0)
                        eq_cont = reg.eq(cut, 1.0, pm)
                    p_fold_xr = max(p_bet - p_cont, 0.0)
                    Rx = 1.0 if T >= eff_main else R
                    called = eq_cont * Rx * (pot + 2 * T) - T - prem[1] * (pot + 2 * T)
                    # plan: their re-raise of our check-raise (the top of their betting
                    # range, all-in): we fold, or get it in with the all-in risk premium
                    p_3b = min(om.raise_vs_bet(street) * p_bet, p_cont) if T < eff_main else 0.0
                    if p_3b > 0:
                        eq_top = reg.eq(1.0 - p_3b, 1.0, pm)
                        allin_branch = max(-T, (eq_top - prem[2]) * (pot + 2 * eff_main) - (eff_main - T))
                        called = ((p_cont - p_3b) * called + p_3b * allin_branch) / max(p_cont, 1e-12)
                    ev_xr = (p_fold_xr * (pot + B) + p_cont * called) / p_bet
                ev_vs_bet = max(0.0, ev_call_b, ev_xr)
            ev_check = (1 - p_bet) * eq_cb * R * pot + p_bet * ev_vs_bet
        options.append((ev_check, CHECK, None))
        # ---- bets: 33% / 66% / 125% (33/66 without the initiative or multiway), all-in at low SPR
        if RAISE in v.legal and hero.stack > 0:
            lo, hi = v.legal[RAISE]
            if lean:
                sizes = (P["villain_bet_frac"],)
            elif others:
                sizes = P["bet_sizes_no_initiative"]      # plan multiway: 33-66%, no big polarized overbets
            elif _has_initiative(v):
                sizes = P["bet_sizes"].get(min(street, 3), (0.33, 0.66, 1.25))
            else:
                sizes = P["bet_sizes_no_initiative"]
            bets = []
            for sz in sizes:
                bet = int(sz * pot)
                if hero.stack - bet < P["jam_left_frac"] * (pot + 2 * bet):
                    bet = hero.stack                 # plan: < half a pot behind -> all-in
                bets.append(bet)
            if not final_street and eq_all >= P["eqr_strong_eq"] and not lean:
                # plan: geometric sizing with very strong hands - the same pot fraction
                # on each remaining street gets the stacks in by the river
                n_left = 3 if street == 1 else 2
                f_geo = ((1.0 + 2.0 * spr) ** (1.0 / n_left) - 1.0) / 2.0
                if sizes[0] < f_geo < spr:
                    bets.append(int(f_geo * pot))
            if spr < P["allin_spr"]:
                bets.append(hero.stack)              # plan: all-in on the menu at low SPR
            totals = []
            for bet in bets:
                bet = min(bet, hero.stack, vstack) if vstack > 0 else min(bet, hero.stack)
                total = int(min(max(hero.bet + bet, lo), hi))
                if total not in totals:
                    totals.append(total)
            pairs = [(t - hero.bet, min(t - hero.bet, vp.stack)) for t in totals]
            # their raise-to against each size (plan: call + 0.75 pot, all-in on the river
            # or when that would leave < 1 pot), so its ICM premium is priced too
            raise_pairs = []
            for t in totals:
                b_eff = min(t - hero.bet, vp.stack) if vp.stack > 0 else t - hero.bet
                left_t = min(hero.stack, vp.stack) if vp.stack > 0 else hero.stack
                Tm = min(b_eff + P["raise_pot_frac"] * (pot + 2 * b_eff), left_t)
                if final_street or left_t - Tm < pot + 2 * Tm:
                    Tm = left_t
                raise_pairs.append((Tm, Tm))
            prem_all = icm.premiums(main, pairs + raise_pairs, scale) if pairs else []
            prem = prem_all[:len(pairs)]
            prem_r = prem_all[len(pairs):]
            for total, pr, pr_r in zip(totals, prem, prem_r):
                bet = total - hero.bet
                ev, F, eq_c = _ev_bet(c, reg, om, others, street, pot, bet, R, pr, final_street, pm, vp.stack,
                                      w_u, order, scores, pr_r)
                if final_street and P["river_bluff_cap"] and not _river_bluff_ok(c, reg, order, w_u, om, bet / pot, eq_c, F):
                    continue   # plan: river bluffs only with the best blockers, capped at s/(1+2s) of our betting range
                if others and eq_c <= 0.5 and not (hero_draw and hero_nut):
                    continue   # plan multiway: bluff almost never; semi-bluff with nut draws only
                if total >= allin_total and P["postflop_icm_allin"]:
                    # plan: an all-in must beat the alternative in exact ICM points
                    p_af = reg.mass(0.0, F) * others_fold(bet / pot)
                    base = v_fold + per_chip * max(ev_check, 0.0)
                    if p_af * v_allfold + (1 - p_af) * (eq_c * win + (1 - eq_c) * lose) <= base:
                        continue
                options.append((ev, RAISE, total))
    else:
        # ---- facing a bet: fold / call / raise --------------------------
        options.append((0.0, FOLD, None))
        call_amt = min(to_call, hero.stack)
        allin_call = to_call >= hero.stack
        raise_to = None
        if RAISE in v.legal and not allin_call and vp.stack > 0 and not lean:   # a raise needs someone who can respond
            lo, hi = v.legal[RAISE]
            T = hero.bet + to_call + P["raise_pot_frac"] * (pot + to_call)   # plan: call + 0.75 pot
            if allin_total - T < P["raise_jam_left_frac"] * (pot + 2 * T):
                T = allin_total                                              # plan: < 1 pot behind -> all-in
            raise_to = int(min(max(T, lo), hi))
        v_total = vp.bet + vp.stack                    # chips beyond this come straight back to us
        add = (min(raise_to, v_total) - hero.bet) if raise_to is not None else 0
        v_add = max(0, min(raise_to, v_total) - vp.bet) if raise_to is not None else 0
        prem = icm.premiums(main, [(call_amt, 0)] + ([(add, v_add)] if raise_to is not None else []), scale)
        Rc = 1.0 if allin_call else R
        ev_call = eq_all * Rc * (pot + call_amt) - call_amt
        ev_call -= P["multiway_call_margin"] * len(to_act) * pot      # plan: margin per player still to act
        ev_call -= prem[0] * (pot + call_amt)
        if not (allin_call and P["postflop_icm_allin"] and eq_all * win + (1 - eq_all) * lose <= v_fold):
            options.append((ev_call, CALL, None))
        if raise_to is not None:
            price = v_add / max(pot + add, 1)
            R2 = 1.0 if raise_to >= min(allin_total, v_total) else R
            of = others_fold(add / pot)

            def ev_raise(f_main):
                p_fm = reg.mass(0.0, f_main)
                p_all = p_fm * of
                eq_c = reg.eq(f_main, 1.0, pm * (1.0 - of))
                final = pot + add + v_add
                return p_all * pot + (1 - p_all) * (eq_c * R2 * final - add - prem[1] * final), p_all, eq_c

            ev_r, p_all, eq_c2 = ev_raise(om.fold_to_raise(price))
            if eq_c2 <= 0.5:   # not for value: pessimistic fold estimate (plan)
                ev_r, p_all, eq_c2 = ev_raise(om.fold_to_raise(price, pessimistic=True))
            ok = True
            if raise_to >= allin_total and P["postflop_icm_allin"]:
                ok = p_all * v_allfold + (1 - p_all) * (eq_c2 * win + (1 - eq_c2) * lose) > v_fold
            if ok and others and eq_c2 < P["eqr_strong_eq"]:
                ok = False      # plan multiway: raise only very strong hands or the nuts
            if ok and final_street and P["river_bluff_cap"] and eq_c2 <= 0.5:
                # plan: river bluff-raises follow the same rule, s = the raise relative to the pot after calling
                ok = _river_bluff_ok(c, reg, order, w_u, om, price, eq_c2, om.fold_to_raise(price), raise_=True)
            if ok:
                options.append((ev_r, RAISE, raise_to))
    kind, amount = _pick(c, options, pot)
    # plan: protect the checking range - check a share of very strong hands
    if (kind == RAISE and to_call <= 0 and CHECK in v.legal and eq_all >= P["trap_eq"]
            and c.rng.random() < P["trap_share"]):
        return CHECK, None
    return kind, amount


def _river_bluff_ok(c, reg, order, w_main, om, frac, eq_c, F, raise_=False):
    """Plan's river bluff rule. Value bets are left to EV. A hand with no showdown
    value may bet (or raise) s pots only if
      (1) the opponent's fold chance, given the cards we hold (blockers), beats
          the break-even rate alpha = s / (1 + s), and
      (2) it is among our best-blocking bluffs, with bluffs capped at
          s / (1 + 2s) of our betting range (value : bluff = (1 + s) : s).
    ``F`` is the share of their range that folds; ``eq_c`` our equity when called.
    """
    P = c.P
    v, g = c.v, c.game
    if eq_c > 0.5 or reg.eq_all >= P["river_sdv_eq"]:
        return True                                   # value, or showdown value: EV decides
    alpha = frac / (1.0 + frac)
    q = np.empty(g.M)
    q[order] = reg.cum                                # strength quantile of every hand in their range
    fold_mask = q <= F
    wf = w_main * fold_mask
    U = g.U
    card_t = np.zeros(52)
    card_f = np.zeros(52)
    for i in range(g.H):
        np.add.at(card_t, U[:, i], w_main)
        np.add.at(card_f, U[:, i], wf)
    Wt, Wf = float(w_main.sum()), float(wf.sum())
    if Wt <= 0:
        return False

    def pfold(cards):                                 # P(they fold | the cards we hold)
        t = Wt - sum(card_t[x] for x in cards)
        f = Wf - sum(card_f[x] for x in cards)
        return f / t if t > 0 else F

    pf_hero = pfold(v.hole)
    if pf_hero <= alpha:
        return False
    if getattr(c, "deadline", None) is not None and V.time_ns() > c.deadline - 150_000:
        return True                                   # no time to size the cap: the alpha test alone
    # our own river range, modelled the way we model opponents (equilibrium priors)
    from .ranges import seat_range
    from .stats import Opp
    bdead = g.dead_mask(v.board)
    hw = seat_range(v, g, v.seat, Opp(None, P), P, c.scores_by_street, c.orders, bdead, c.draws_by_street)
    vals = c.river_vals if getattr(c, "river_vals", None) is not None else g.strengths(v.board, bdead)
    hw = hw * (vals >= 0)
    if not raise_:
        rr = om.raise_vs_bet(3)
        call_mask = (q > F) & (q <= 1.0 - rr)         # the part of their range that calls
    else:
        call_mask = q > F
    wc = w_main * call_mask
    idx = np.argsort(vals)
    sv = vals[idx]

    def share_vs(wv):                                 # share of ``wv`` each of our hands beats
        cw = np.cumsum(wv[idx])
        tot = cw[-1] if cw.size else 0.0
        if tot <= 0:
            return np.zeros(g.M)
        pos = np.searchsorted(sv, vals, side="left")
        return np.where(pos > 0, cw[np.maximum(pos - 1, 0)], 0.0) / tot

    value_w = float((hw * (share_vs(wc) > 0.5)).sum())
    budget = value_w * frac / (1.0 + frac)            # bluffs : value = s : (1 + s)
    cand = (share_vs(w_main) < P["river_sdv_eq"]) & (hw > 0)
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
    srt = np.argsort(-pf_c)                           # best blockers first
    cw = np.cumsum(hw[ci][srt])
    if cw[-1] <= budget:
        return True
    k = int(np.searchsorted(cw, budget))
    if k <= 0:
        return False
    return pf_hero >= pf_c[srt][k - 1]


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


def _ev_bet(c, reg, om, others, street, pot, bet, R, prem, final_street, pm, v_stack,
            w_main=None, order=None, scores=None, prem_raise=0.0):
    """EV of betting ``bet`` (plan formula). Returns (EV, fold share used, equity when called).

    v_stack: what the main opponent has left. Chips we bet beyond it come
    straight back, so against them the bet is min(bet, v_stack)."""
    P = c.P
    hero = c.v.hero
    allin_bet = bet >= hero.stack
    of = 1.0
    extra_callers = 0.0
    for s in others:
        of *= c.opps[s].fold_to_bet(street, bet / pot, allin=allin_bet)   # plan: everyone has to fold
        # plan: once someone has called, the players after them get a better price
        extra_callers += 1.0 - c.opps[s].fold_to_bet(street, bet / (pot + bet), allin=allin_bet)
    if v_stack > 0:
        bet = min(bet, v_stack)
    frac = bet / pot
    left = min(hero.stack, v_stack) if v_stack > 0 else hero.stack
    q_main = None
    if w_main is not None and order is not None and scores is not None:
        q_main = np.empty(reg.M)
        q_main[order] = reg.cum
    rr = om.raise_vs_bet(street) if left > bet else 0.0     # nobody raises an all-in
    # their raise-to: call + 0.75 pot, or all-in on the river / when that would leave < 1 pot (plan)
    Tm = min(bet + P["raise_pot_frac"] * (pot + 2 * bet), left)
    if final_street or left - Tm < pot + 2 * Tm:
        Tm = left
    Rc = 1.0 if bet >= left else R

    def ev_with(F):
        r_eff = min(rr, max(1.0 - F, 0.0))
        p_fold = reg.mass(0.0, F)               # given our cards (blockers)
        p_raise = reg.mass(1.0 - r_eff, 1.0)
        p_all = p_fold * of
        p_called = max(0.0, 1.0 - p_all - p_raise)
        p_multi = pm * (1.0 - of)
        # plan: showdowns calibrate how strong their calling range is (thin value)
        Fc = call_floor(P, om, w_main, q_main, scores, F, r_eff) if q_main is not None else F
        eq_c = reg.eq(Fc, 1.0 - r_eff, p_multi)
        eq_r = reg.eq(1.0 - r_eff, 1.0, p_multi)
        # plan: the pot when called counts the callers we expect, not just one
        final = pot + 2 * bet + extra_callers * min(bet, hero.stack)
        Rr = 1.0 if Tm >= left else R                       # plan: x R for the streets to come, 1 if all-in
        ev = (p_all * pot
              + p_called * (eq_c * Rc * final - bet - prem * final)
              + p_raise * max(-bet, (eq_r * Rr - prem_raise) * (pot + 2 * Tm) - Tm))
        # plan: fold equity on the NEXT street for hands with some equity
        # (they call now and give up to the next bet), discounted
        if not final_street and P["multistreet_lo"] <= reg.eq_all <= P["multistreet_hi"] and bet < left:
            f_next = om.fold_to_bet(min(street + 1, 3), P["villain_bet_frac"])
            b2 = min(P["villain_bet_frac"] * final, left - bet)
            ev += P["multistreet_disc"] * p_called * max(0.0, f_next * final - (1 - f_next) * (1 - eq_c) * b2)
        return ev, eq_c

    F = om.fold_to_bet(street, frac, allin=allin_bet)
    ev, eq_c = ev_with(F)
    if eq_c <= 0.5:
        # plan: a bet that is not for value (it does not win more than half the
        # time when called) may count on folds only as far as the evidence
        # supports: use the pessimistic end of the fold estimate
        F = om.fold_to_bet(street, frac, pessimistic=True, allin=allin_bet)
        ev, eq_c = ev_with(F)
    return ev, F, eq_c
