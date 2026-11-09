"""Preflop decisions.

Hands are ranked once (plug-in ``hero_pre_pct``: 0 = best). Opponents'
ranges are percentile bands (ranges.preflop_band). Every option is scored
by expected value (EV) and the best one is played:

  fold  -> 0
  call  -> equity * EQR * (pot after calling) - cost of calling
  raise -> P(all fold) * pot + P(call) * (EV when called) + P(re-raise) * (EV then)

EQR (equity realization) is how much of its raw equity a hand actually wins
once there is betting after the flop: >1 for hands that play well (suited,
connected, pairs, in position), <1 for hands that play badly.

Plan rules used here:
  * unopened pots are raise-or-fold by position width; <= push_fold_bb is
    shove-or-fold;
  * every all-in decision is compared in exact ICM points at the real stacks
    and payouts; other options carry the ICM risk premium for the chips they
    put at stake;
  * probabilities of an opponent folding / calling / re-raising are taken
    given the cards we hold (blockers);
  * a raise that would leave less than about one pot behind is an all-in;
  * a non-value raise uses a pessimistic estimate of how often they fold.
"""
from __future__ import annotations

from . import eqr as EQ
from . import icmtools as I
from . import pushfold as PF
from . import view as V
from .ranges import bb_defend_width, first_in_width, open_width, preflop_band  # noqa: F401
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
        # plan: the ICM risk premium grows as the hand cap approaches (every street)
        hands_left = max(v.hand_cap - v.hand_number, 0)
        self.icm_scale = P["icm_scale"] * (1.0 + P["cap_rp_boost"] * max(0.0, 1.0 - hands_left / max(P["cap_rp_hands"], 1)))
        if getattr(game, "open_pct", None) is None:
            game.open_rank_init(lambda hand: _features_eqr(game, hand, P))


# ----------------------------------------------------------------------------- helpers
def _features_eqr(game, hole, P):
    """Preflop EQR adjustment from simple, variant-agnostic hand features
    (plan: suited +, connected +, pair +, offsuit weak kicker -)."""
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


def _plays_multiway(hole):
    """Plan: hands that win big multiway pots - pairs (sets), suited connectors, suited aces."""
    n = len(hole)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = hole[i], hole[j]
            ra, rb = a >> 2, b >> 2
            if ra == rb:
                return True
            if (a & 3) == (b & 3) and (abs(ra - rb) <= 2 or max(ra, rb) == 12):
                return True
    return False


def pre_R(c, ip, spot_adj=0.0, n_extra=0, spr=None):
    """Preflop EQR prior for our hand.

    base (in / out of position) + hand features + spot adjustment (3-bet pots)
    +- eqr_multiway per extra opponent (plan: nut-type hands gain multiway,
    the rest lose), shrinking toward 1 as the stack-to-pot ratio falls (plan).
    """
    P = c.P
    r = (P["eqr_pre_ip"] if ip else P["eqr_pre_oop"]) + _features_eqr(c.game, c.v.hole, P) + spot_adj
    if n_extra > 0:
        r += (1.0 if _plays_multiway(c.v.hole) else -1.0) * P["eqr_multiway"] * n_extra
    wm = P.get("eqr_measured_weight", 0.0) if P.get("use_measured_eqr") else 0.0
    if wm > 0:
        try:
            m = EQ.lookup_pre(ip, c.v.hole)
        except Exception:  # noqa: BLE001
            m = None
        if m is not None:
            r = (1 - wm) * r + wm * (min(max(m, 0.3), P["eqr_measured_cap"]) + spot_adj)
    if spr is not None:
        r = 1.0 + (r - 1.0) * min(1.0, max(spr, 0.0) / P["eqr_spr_full"])
    return r


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


def _can_jam(v):
    """Is a real all-in on offer? (No-limit: always. Pot-limit: only when the
    largest allowed raise covers the effective stack.)"""
    lo, hi = v.legal.get(RAISE, (None, None))
    if hi is None:
        return False
    opp = max((v.players[s].avail for s in v.alive if s != v.seat and v.players[s].in_hand), default=0)
    return hi >= min(v.hero.avail, opp)


def _seed(v, salt=0):
    return V.stable_seed(v.table_id, v.hand_number, len(v.history), salt)


def _call_prob_eq(c, band, w):
    """A player whose range is ``band`` calls our shove with the top ``w`` of
    all hands. Returns (P(call | our cards), our equity when called)."""
    g = c.game
    lo, hi = band
    c_hi = min(hi, max(w, lo))
    m_call, m_fold = g.band_probs([lo, c_hi, hi], c.dead)
    tot = m_call + m_fold
    if tot <= 0 or m_call <= 0:
        return 0.0, 0.5
    return m_call / tot, g.pre_equity(c.v.hole, lo, max(c_hi, lo + 1e-3))


# ----------------------------------------------------------------------------- entry
def decide(c: Ctx):
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    c.pct = g.hero_pre_pct(v.hole)
    c.dead = g.dead_mask(v.hole)
    acts = v.acts[0]
    raises = [a for a in acts if a.kind == RAISE]
    others_in = [s for s in v.alive if s != v.seat and v.players[s].in_hand]
    if not others_in:
        return (CHECK, None) if CHECK in v.legal else (CALL, None)
    # plan: effective stack = the smaller of ours and the largest stack still in
    # (unopened pot), or the raiser's stack when facing a raise
    eff = min(hero.avail, max(v.players[s].avail for s in others_in))
    if raises and raises[-1].seat in v.players and raises[-1].seat != v.seat:
        eff = min(hero.avail, v.players[raises[-1].seat].avail)
    c.eff_bb = eff / v.bb
    c.icm = I.IcmCtx(v)
    if not raises:
        return _unopened(c)
    return _facing_raise(c, raises)


# ----------------------------------------------------------------------------- unopened pot
def _unopened(c):
    v, g, P = c.v, c.game, c.P
    bb = v.bb
    hero = v.hero
    behind = players_behind(v, v.pre_order)
    limpers = [a.seat for a in v.acts[0] if a.kind == CALL and a.seat != v.seat and v.players[a.seat].in_hand]
    can_check = CHECK in v.legal
    if RAISE not in v.legal and not can_check:
        return _call_or_fold(c, limpers)
    if c.eff_bb <= P["push_fold_bb"] and _can_jam(v):
        # plan: shove or fold, in ICM points
        ev_jam, v_fold, v_allfold = shove_value(c, behind + limpers)
        alt = _check_value(c, limpers, v_fold, v_allfold) if can_check else v_fold
        if ev_jam > alt:
            return RAISE, _jam(v)
        return (CHECK, None) if can_check else (FOLD, None)
    n_behind = len(behind)
    width = open_width(P, n_behind, v.n_alive) * P["open_scale"]
    blinds = [s for s in behind if s in (v.sb_seat, v.bb_seat)]
    if blinds and n_behind - len(blinds) <= 1:
        # plan: steal more when the blinds fold a lot (observed rate / baseline rate)
        f = sum(c.opps[s].blind_fold_vs_steal() for s in blinds) / len(blinds)
        width *= max(0.6, min(1.8, f / P["prior_blind_fold"]))
    if behind:
        # plan: "tighten when they re-raise a lot" (their 3-bet rate / the baseline rate)
        tb = sum(c.opps[s].threebet_width() for s in behind) / len(behind)
        width *= max(0.6, min(1.5, P["prior_3bet"] / max(tb, 1e-6)))
    if limpers:
        width *= P["iso_width_mult"]  # plan: isolate with the opening range plus a bit
    # plan: deep opens are ordered by equity x playability (JTs above A2o), and
    # hands at the edge of the range are mixed
    pct_open = g.hero_open_pct(v.hole)
    opens = pct_open < width or (abs(pct_open - width) < P["tie_margin"] and c.rng.random() < 0.5)
    if opens and RAISE in v.legal:
        if limpers:
            total = (P["iso_size_bb"] + P["limp_iso_extra_bb"] * len(limpers)) * bb
        else:
            total = (P["open_size_bb"] if c.eff_bb > 40 else P["open_size_bb_short"]) * bb
        add = total - hero.bet
        # plan: a raise that leaves less than about one pot behind is an all-in
        if c.eff_bb * bb - total < P["commit_left_frac"] * (v.pot + 2 * add):
            return RAISE, _jam(v)
        if not limpers and c.eff_bb <= P["open_jam_bb"] and _can_jam(v):
            # plan 16-40bb band: fold / small open / open-jam - the jam is taken when it
            # is worth more in ICM points than the small open
            ev_jam, v_fold, v_allfold = shove_value(c, behind)
            if ev_jam > _open_value(c, behind, total, v_fold, v_allfold):
                return RAISE, _jam(v)
        return RAISE, _raise_amount(v, total)
    if can_check:
        return CHECK, None
    # plan: raise or fold; over-limp only in position with hands that play well multiway
    in_position = v.seat not in (v.sb_seat, v.bb_seat) and len(blinds) == n_behind
    if limpers and CALL in v.legal and v.to_call <= bb and in_position and _plays_multiway(v.hole):
        return CALL, None
    return FOLD, None


def _call_or_fold(c, limpers):
    """Nobody has raised and only call or fold is offered (the players left
    are all-in from the blinds, or calling takes our last chips): there is no
    more betting, so compare calling and folding in exact ICM points (plan)."""
    v, g, P = c.v, c.game, c.P
    if CALL not in v.legal:
        return FOLD, None
    opp = [s for s in v.alive if s != v.seat and v.players[s].in_hand
           and (v.players[s].allin or s in limpers or s == v.bb_seat)]
    if not opp:
        opp = [s for s in v.alive if s != v.seat and v.players[s].in_hand]
    bands = [preflop_band(v, s, c.opps[s], P, c.opps) for s in opp]
    eq = g.pre_equity_multi(v.hole, bands, _seed(v, 4))
    vil = max(opp, key=lambda s: v.players[s].committed)
    v_fold, _, win, lose, _ = c.icm.outcomes(vil, [(min(v.to_call, v.hero.stack), 0)])
    if eq * win[0] + (1 - eq) * lose[0] > v_fold:
        return CALL, None
    return FOLD, None


def shove_value(c, callers):
    """ICM value of shoving now, of folding, and of everyone folding to us.

    First-caller model (plan): players act in turn; each calls with the top
    part of their range given by the solved chart, scaled by their observed
    calling ratio, tighter if we cover them. Probabilities use our blockers.
    """
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    v_fold = c.icm.outcomes(v.bb_seat if v.bb_seat != v.seat else (callers[0] if callers else None), [])[0]  # noqa: E501
    v_allfold = None
    p_none = 1.0
    ev = 0.0
    seen = set()
    k = max(1, len(callers))
    for s in callers:
        if s in seen or s == v.seat or s not in v.players or not v.players[s].in_hand:
            continue
        seen.add(s)
        vil = v.players[s]
        eff_j = min(hero.start, vil.start) / v.bb
        w = PF.call_width(eff_j, s == v.bb_seat, k, P)
        w *= max(0.5, min(3.0, c.opps[s].call_jam_ratio()))
        if hero.start > vil.start:
            w *= P["cover_call_mult"]  # plan: a player we would knock out calls tighter
        w = min(w, 1.0)
        band = preflop_band(v, s, c.opps[s], P, c.opps)
        pc, eq = _call_prob_eq(c, band, w)
        win, lose, _, af = c.icm.allin(s)
        v_allfold = af
        ev += p_none * pc * (eq * win + (1 - eq) * lose)
        p_none *= (1 - pc)
    if v_allfold is None:
        v_allfold = c.icm.outcomes(None, [])[1]
    ev += p_none * v_allfold
    return ev, v_fold, v_allfold


def _open_value(c, behind, total, v_fold, v_allfold):
    """ICM value of a small open: everyone folds (we take the pot) or we see a
    flop against the first continuing range with our equity x EQR, the same
    conversion ``_check_value`` uses."""
    v, g, P = c.v, c.game, c.P
    p_all = 1.0
    conts = []
    for s in behind:
        o = c.opps[s]
        if s in (v.sb_seat, v.bb_seat):
            cont = 1.0 - o.blind_fold_vs_steal()
        else:
            cont = o.threebet_width() + P["prior_coldcall"] * max(o.coldcall_ratio(), 0.5)
        cont = min(max(cont, 0.02), 1.0)
        conts.append(cont)
        p_all *= 1.0 - cont
    if not conts:
        return v_allfold
    eq = g.pre_equity(v.hole, 0.0, max(conts))
    pot_after = v.pot + 2 * (total - v.hero.bet)
    R = pre_R(c, False, 0.0, spr=max(c.eff_bb * v.bb - total, 0) / max(pot_after, 1))
    share = p_all + (1.0 - p_all) * min(max(eq * R, 0.0), 1.0)
    return v_fold + share * (v_allfold - v_fold)


def _check_value(c, limpers, v_fold, v_allfold):
    """Big blind option when short: the value of checking and seeing a flop.

    We expect to win about equity * EQR of the pot; in points that is the
    share of the way from losing the pot (v_fold) to winning it (v_allfold).
    """
    v, g, P = c.v, c.game, c.P
    bands = [preflop_band(v, s, c.opps[s], P, c.opps) for s in limpers] or [(0.0, 1.0)]
    eq = g.pre_equity_multi(v.hole, bands, _seed(v, 1))
    ip = all(_is_ip(v, s) for s in limpers) if limpers else False
    R = pre_R(c, ip, 0.0, n_extra=max(len(bands) - 1, 0))
    share = min(max(eq * R, 0.0), 1.0)
    return v_fold + share * (v_allfold - v_fold)


# ----------------------------------------------------------------------------- facing a raise
def _facing_raise(c, raises):
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    last = raises[-1]
    vil = last.seat
    n_raises = len(raises)
    band = preflop_band(v, vil, c.opps[vil], P, c.opps)
    lo, hi = band
    vp = v.players[vil]
    # who else is involved: players who called this raise, and players who still act after us
    after_last = False
    callers = []
    for a in v.acts[0]:
        if a is last:
            after_last = True
            continue
        if after_last and a.kind == CALL and a.seat != v.seat and v.players[a.seat].in_hand:
            callers.append(a.seat)
    to_act = [s for s in V.yet_to_act(v) if s != vil and s not in callers]
    to_call = v.to_call
    pot = v.pot
    left_after_call = hero.avail - hero.bet - to_call
    committed = left_after_call < P["commit_left_frac"] * (pot + to_call)
    short = c.eff_bb <= P["push_fold_bb"] or committed
    # shove-or-fold needs a real all-in on offer (pot-limit games may not have one;
    # then the raise is priced below like any other raise)
    if last.allin or vp.stack == 0 or (short and (_can_jam(v) or RAISE not in v.legal)):
        return _allin_vs(c, vil, band, last, n_raises, callers, to_act)

    # ---- deep: fold / call / re-raise by chip EV with ICM premiums -------------
    bands = [band] + [preflop_band(v, s, c.opps[s], P, c.opps) for s in callers]
    eq = g.pre_equity_multi(v.hole, bands, _seed(v, 2))
    ip = _is_ip(v, vil) and all(_is_ip(v, s) for s in callers)
    in_3bet_pot = n_raises >= 2
    R = pre_R(c, ip, P["threebet_eqr_adj"] if in_3bet_pot else 0.0, n_extra=len(callers),
              spr=left_after_call / max(pot + to_call, 1))

    raise_to = None
    add = v_add = 0
    jam = False
    x_eff = min(hero.avail, vp.avail)     # the most that can go in between us and the raiser
    if RAISE in v.legal:
        r_hi = v.legal[RAISE][1]
        if n_raises == 1:
            raise_to = last.total * (3.0 if ip else 4.0) + last.total * len(callers)  # plan: 3x IP, 4x OOP, +1x per caller
        else:
            raise_to = P["fourbet_size"] * last.total                                  # plan: ~2.3x
        raise_to = min(raise_to, hero.avail, r_hi)
        add = min(raise_to, x_eff) - hero.bet           # chips the raiser cannot match come back
        v_add = max(min(raise_to, vp.avail) - vp.bet, 0)
        jam_depth = P["threebet_jam_bb"] if n_raises == 1 else P["fourbet_jam_bb"]
        jam = (c.eff_bb <= jam_depth
               or x_eff - raise_to < P["commit_left_frac"] * (pot + add + v_add))
        if jam:
            raise_to = min(hero.avail, r_hi)
            jam = raise_to >= x_eff     # pot-limit: a capped raise is priced as a raise, not as an all-in
            add = min(raise_to, x_eff) - hero.bet
            v_add = max(min(raise_to, vp.avail) - vp.bet, 0)

    pairs = [(to_call, 0)] + ([(add, v_add), c.icm.allin_pair(vil)] if raise_to is not None else [])
    prem = c.icm.premiums(vil, pairs, c.icm_scale)   # call | our raise | all the chips
    ev_call = eq * R * (pot + to_call) - to_call
    ev_call -= P["multiway_call_margin"] * len(to_act) * pot   # plan: margin per player still to act
    ev_call -= prem[0] * (pot + to_call)
    options = [(0.0, FOLD, None), (ev_call, CALL, None)]

    if raise_to is not None:
        # plan: their fold / call / re-raise response comes from what they have done
        # when re-raised, shrunk to the starting values for this level of raise
        pf, pc, p4 = c.opps[vil].response_to_3bet(level=n_raises)
        n_obs = c.opps[vil].n_3bet_faced()
        rest = [s for s in v.alive if s not in (v.seat, vil) and v.players[s].can_act]
        cont = g.band_probs([0.0, P["behind_continue_vs_3bet"]], c.dead)[0]
        p_rest = (1.0 - cont) ** len(rest)   # plan: everyone still in has to fold too
        width = hi - lo
        win, lose, v_fold, v_allfold = c.icm.allin(vil)

        def ev_raise(pf_, pc_, p4_):
            # their range, strongest first: re-raise | call | fold - given our cards (blockers)
            e4 = lo + width * p4_
            ec = lo + width * (p4_ + pc_)
            m4, mc, mf = g.band_probs([lo, e4, ec, hi], c.dead)
            tot = m4 + mc + mf
            if tot <= 0:
                return -1e18, 0.0
            m4, mc, mf = m4 / tot, mc / tot, mf / tot
            if jam:
                p_all = mf * p_rest
                eq_c = g.pre_equity(v.hole, lo, max(ec, lo + 1e-3))
                # plan: an all-in must win in exact ICM points
                if p_all * v_allfold + (1 - p_all) * (eq_c * win + (1 - eq_c) * lose) <= v_fold:
                    return -1e18, eq_c
                return p_all * pot + (1 - p_all) * ((eq_c - prem[1]) * (pot + add + v_add) - add), eq_c
            p_all = mf * p_rest
            p4_eff = m4 + mf * (1 - p_rest)   # someone behind waking up is treated like a re-raise
            eq_c = g.pre_equity(v.hole, e4, max(ec, e4 + 1e-3))
            R3 = pre_R(c, ip, P["threebet_eqr_adj"], spr=max(x_eff - raise_to, 0) / max(pot + add + v_add, 1))
            ev_called = eq_c * R3 * (pot + add + v_add) - add - prem[1] * (pot + add + v_add)
            eq_4 = g.pre_equity(v.hole, lo, max(e4, lo + 1e-3))
            final = 2 * x_eff + (pot - hero.bet - vp.bet)
            # they re-raise: we fold, or get it all in - which carries the all-in risk premium
            ev_vs4 = max(-add, (eq_4 - prem[2]) * final - (x_eff - hero.bet))
            return p_all * pot + mc * ev_called + p4_eff * ev_vs4, eq_c

        ev_r, eq_c = ev_raise(pf, pc, p4)
        if eq_c <= 0.5 and P["threebet_conf_z"] > 0 and ev_r > -1e17:
            # plan: a raise that is not for value may rely on folds only as far as
            # the evidence supports - use the pessimistic end of the fold estimate
            sd = (pf * (1 - pf) / (n_obs + P["k_rare"])) ** 0.5
            shift = max(0.0, min(pf - 0.05, P["threebet_conf_z"] * sd))
            ev_r, _ = ev_raise(pf - shift, pc + shift, p4)
        options.append((ev_r, RAISE, _jam(v) if jam else _raise_amount(v, raise_to)))
    return _pick(c, options, pot)


def _allin_vs(c, vil, band, last, n_raises, callers, to_act):
    """Stack-off decision against ``vil`` in exact ICM points (plan)."""
    v, g, P = c.v, c.game, c.P
    hero = v.hero
    vp = v.players[vil]
    lo, hi = band
    vil_allin = last.allin or vp.stack == 0
    win, lose, v_fold, v_allfold = c.icm.allin(vil)
    if vil_allin or RAISE not in v.legal:
        # call or fold: they already have everything in (others who called are in the pot too)
        bands = [band] + [preflop_band(v, s, c.opps[s], P, c.opps) for s in callers]
        eq = g.pre_equity_multi(v.hole, bands, _seed(v, 3))
        eq -= P["multiway_call_margin"] * len(to_act)   # plan: margin per player still to act
        ev_call = eq * win + (1 - eq) * lose
        if ev_call > v_fold:
            # plan: when we cover and others can still act, re-jam to shut them out -
            # but a player behind who covers US may call the re-jam for our whole
            # stack, so that branch is priced too (first-caller chain)
            if RAISE in v.legal and to_act and hero.avail > vp.avail:
                p_none = 1.0
                ev_rj = 0.0
                k = max(1, len(to_act))
                for s in to_act:
                    if s not in v.players or not v.players[s].in_hand:
                        continue
                    o = v.players[s]
                    w = PF.call_width(min(hero.start, o.start) / v.bb, s == v.bb_seat, k, P)
                    w *= max(0.5, min(3.0, c.opps[s].call_jam_ratio()))
                    if hero.start > o.start:
                        w *= P["cover_call_mult"]
                    band_s = preflop_band(v, s, c.opps[s], P, c.opps)
                    pc, _ = _call_prob_eq(c, band_s, min(w, 1.0))
                    lo_s, hi_s = band_s
                    eq3 = g.pre_equity_multi(v.hole, [band, (lo_s, min(hi_s, max(w, lo_s + 1e-3)))], _seed(v, 5))
                    w2, l2, _, _ = c.icm.allin(s)
                    ev_rj += p_none * pc * (eq3 * w2 + (1 - eq3) * l2)
                    p_none *= 1 - pc
                ev_rj += p_none * ev_call
                if ev_rj >= ev_call:
                    return RAISE, _jam(v)
            return (CALL, None) if CALL in v.legal else (CHECK, None)
        return (CHECK, None) if CHECK in v.legal else (FOLD, None)
    # they are not all-in: shove or fold (plan: no flat calls when short / committed)
    eff_v = min(hero.start, vp.start) / v.bb
    ratio = max(0.5, min(3.0, c.opps[vil].call_jam_ratio()))
    if n_raises == 1:
        cw = PF.rejam_call_width(eff_v, hi, P) * ratio      # the opener's solved call of a re-shove
    else:
        cw = PF.call_width(eff_v, vil == v.bb_seat, 1, P) * ratio
    if hero.start > vp.start:
        cw *= P["cover_call_mult"]
    p_call, eq_c = _call_prob_eq(c, band, cw)
    ev = p_call * (eq_c * win + (1 - eq_c) * lose)
    p_none = 1.0 - p_call
    k = max(1, len(to_act) + len(callers))
    for s in list(to_act) + list(callers):   # anyone else may call too (first-caller model)
        if s not in v.players or not v.players[s].in_hand:
            continue
        o = v.players[s]
        w = PF.call_width(min(hero.start, o.start) / v.bb, s == v.bb_seat, k, P)
        w *= max(0.5, min(3.0, c.opps[s].call_jam_ratio()))
        if hero.start > o.start:
            w *= P["cover_call_mult"]
        pc, eq = _call_prob_eq(c, preflop_band(v, s, c.opps[s], P, c.opps), min(w, 1.0))
        w2, l2, _, _ = c.icm.allin(s)
        ev += p_none * pc * (eq * w2 + (1 - eq) * l2)
        p_none *= (1 - pc)
    ev += p_none * v_allfold
    if ev > v_fold:
        return RAISE, _jam(v)
    return (CHECK, None) if CHECK in v.legal else (FOLD, None)


def _pick(c, options, pot):
    """Best EV option; mix between near-ties so we are harder to read (plan)."""
    options = [o for o in options if o[1] in c.v.legal or (o[1] == FOLD and CHECK in c.v.legal)]
    if not options:
        return (CHECK, None) if CHECK in c.v.legal else (FOLD, None)
    options.sort(key=lambda o: -o[0])
    best = options[0]
    if len(options) > 1:
        second = options[1]
        if best[0] - second[0] < c.P["tie_margin"] * max(pot, 1) and c.rng.random() < c.P["tie_mix_prob"]:
            best = second
    kind, amount = best[1], best[2]
    if kind == FOLD and CHECK in c.v.legal:
        return CHECK, None
    if kind == RAISE and amount is None:
        return (CALL, None) if CALL in c.v.legal else (CHECK, None)
    return kind, amount
