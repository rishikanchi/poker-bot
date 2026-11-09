"""Seven stronger test opponents (not simple heuristics).

Every bot reasons with real equity: preflop strength percentiles and equity
vs ranges from the variant plug-in (bot/pb/game.py), Monte Carlo equity
postflop vs ranges it estimates for its opponents, pot odds, solved
push/fold charts at short stacks, and ICM for all-ins. They differ in style
and in how they model opponents, so our bot meets several kinds of
competent play. All work in any community-card variant the plug-in supports
(hold'em, Omaha) and respect pot-limit through the legal action bounds.

  TAG          tight-aggressive, equity-driven; solved push/fold <= 12bb
  LAG          loose-aggressive: wide opens, light 3-bets, frequent barrels
  NashICM      push/fold Nash charts up to 20bb, ICM-checked all-in calls,
               tight deep play
  Adaptive     tracks each opponent (VPIP, PFR, aggression, fold to bets) and
               exploits: no bluffs vs stations, steals vs nits, calls down maniacs
  Balanced     game-theory frequencies: defends by minimum defence frequency
               and bluffs at s/(1+2s) of its betting range, estimated by
               sampling its OWN range's equities
  ValueReg     tight, value-heavy, geometric sizing with strong hands, rarely bluffs
  BayesReader  Bayesian range per opponent (likelihood update after every
               action) and EV-maximising bet choice with observed fold rates

They use a frozen copy of our calculator modules (sim/pb_ref) only as a
calculator and legality layer; their strategies are independent of the bot's
decision code, and editing bot/pb does not change them.
"""
from __future__ import annotations

import math
import os
import random
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SIM = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_ROOT, "stub_sdk"), _SIM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import poker_agent_sdk as SDK  # noqa: E402
from poker_agent_sdk import Agent  # noqa: E402

# The opponents use a FROZEN copy of the bot's calculator modules (sim/pb_ref,
# data in sim/data), so changing bot/pb never changes how the opponents play.
from pb_ref import actions as A  # noqa: E402
from pb_ref import icmtools as I  # noqa: E402
from pb_ref import pushfold as PF  # noqa: E402
from pb_ref import view as V  # noqa: E402
from pb_ref.game import get_game  # noqa: E402
from pb_ref.postflop import hero_draw_flags  # noqa: E402

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"
POS_OPEN = {1: 0.40, 2: 0.42, 3: 0.27, 4: 0.21, 5: 0.18, 6: 0.16, 7: 0.14, 8: 0.13, 9: 0.12}
DEBUG = bool(os.environ.get("PB_DEBUG"))


class Tracker:
    """Streaming opponent counters from state.events (per table, in self)."""

    def __init__(self):
        self.s = {}
        self.street = 0
        self.high = 0
        self.bets = {}
        self.vp = set()
        self.pr = set()

    def row(self, seat):
        r = self.s.get(seat)
        if r is None:
            r = self.s[seat] = {"hands": 0, "vpip": 0, "pfr": 0, "aggr": 0, "calls": 0,
                                "faced": 0, "folded": 0, "seen": 0}
        return r

    def feed(self, events):
        for e in events:
            ev = V.norm_event(e)
            k, s = ev.kind, ev.seat
            if k == "hand":
                for seat in self.vp:
                    self.row(seat)["vpip"] += 1
                for seat in self.pr:
                    self.row(seat)["pfr"] += 1
                self.vp, self.pr = set(), set()
                self.street, self.high, self.bets = 0, 0, {}
            elif k == "blind":
                self.bets[s] = self.bets.get(s, 0) + ev.amount
                self.high = max(self.high, self.bets[s])
                self.row(s)["hands"] += 1
            elif k == "board":
                self.street += 1
                self.high, self.bets = 0, {}
            elif k in ("fold", "check", "call", "raise"):
                r = self.row(s)
                if self.street == 0 and k in ("call", "raise"):
                    self.vp.add(s)
                    if k == "raise":
                        self.pr.add(s)
                cur = self.bets.get(s, 0)
                facing = self.high > cur
                if self.street > 0:
                    r["seen"] += 1
                    if facing:
                        r["faced"] += 1
                        if k == "fold":
                            r["folded"] += 1
                    if k == "raise":
                        r["aggr"] += 1
                    elif k == "call":
                        r["calls"] += 1
                if k == "call":
                    self.bets[s] = cur + ev.amount
                elif k == "raise":
                    self.bets[s] = ev.amount
                    self.high = max(self.high, ev.amount)

    def stats(self, seat):
        r = self.s.get(seat)
        if not r:
            return {"vpip": 0.25, "pfr": 0.15, "af": 1.5, "ftb": 0.45, "n": 0}
        h = max(r["hands"], 1)
        return {"vpip": (r["vpip"] + 2.5) / (h + 10), "pfr": (r["pfr"] + 1.5) / (h + 10),
                "af": (r["aggr"] + 1.5) / (r["calls"] + 1.0),
                "ftb": (r["folded"] + 4.5) / (r["faced"] + 10), "n": h}


class Strong(Agent):
    """Shared machinery: parsing, ranges, equity, push/fold, legal actions."""

    label = "Strong"
    open_mult = 1.0
    open_bb = 2.2
    tb_value = 0.04       # 3-bet for value: top fraction of hands
    tb_bluff = 0.0        # probability of a light 3-bet with a playable hand
    call_cap = 0.45       # never flat a raise with hands weaker than this percentile
    pf_bb = 12.0          # push/fold at or below
    cbet = 0.55
    value_eq = 0.62
    semi = 0.5
    bluff_river = 0.10
    barrel = 0.35
    call_margin = 0.03
    raise_eq = 0.82
    sizes = (0.66, 0.66, 0.75)  # flop, turn, river (fraction of pot)
    n_mc = 240

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.g = get_game()
        self.track = Tracker()

    # ------------------------------------------------------------------ entry
    def act(self, state, memory):
        v = V.build_view(state)
        try:
            self.track.feed(getattr(state, "events", ()) or ())
            if len(v.hole) != self.g.H or v.phase != "betting":
                return A.automatic(SDK, v), memory
            kind, amt = self.preflop(v) if not v.board else self.postflop(v)
            return A.to_action(SDK, v, kind, amt), memory
        except Exception:
            if DEBUG:
                raise
            return A.automatic(SDK, v), memory

    # ------------------------------------------------------------- helpers
    def pct(self, v):
        return self.g.hero_pre_pct(v.hole)

    def eff_bb(self, v):
        others = [p for s, p in v.players.items() if s != v.seat and p.in_hand]
        if not others or v.hero is None:
            return 100.0
        return min(v.hero.avail, max(p.avail for p in others)) / max(v.bb, 1)

    @staticmethod
    def jam(v):
        lo, hi = v.legal.get(RAISE, (None, None))
        return (RAISE, int(hi)) if hi is not None else ((CALL, None) if CALL in v.legal else (CHECK, None))

    @staticmethod
    def raise_to(v, total):
        lo, hi = v.legal.get(RAISE, (None, None))
        if lo is None:
            return (CALL, None) if CALL in v.legal else (CHECK, None)
        return RAISE, int(min(max(int(total), int(lo)), int(hi)))

    @staticmethod
    def passive(v):
        return (CHECK, None) if CHECK in v.legal else (FOLD, None)

    def bet_frac(self, v, frac):
        """Bet/raise of ``frac`` x pot (total for the round); jam when it nearly commits us."""
        hero = v.hero
        add = frac * max(v.pot + v.to_call, 1) + v.to_call
        if hero.stack - add < 0.5 * (v.pot + 2 * add):
            return self.jam(v)
        return self.raise_to(v, hero.bet + add)

    def preflop_line(self, v):
        raises = [a for a in v.acts[0] if a.kind == RAISE]
        limpers = [a.seat for a in v.acts[0] if a.kind == CALL and not raises]
        hero_raised = any(a.seat == v.seat and a.kind == RAISE for a in v.acts[0])
        return raises, limpers, hero_raised

    def open_width(self, v, n_behind):
        if v.n_alive <= 2:
            return min(1.0, 0.8 * self.open_mult)
        return min(1.0, POS_OPEN.get(n_behind, 0.13) * self.open_mult)

    def villain_band(self, v, seat):
        """Simple preflop band for an opponent from their line this hand."""
        nr = 0
        line = None
        for a in v.acts[0]:
            if a.seat == seat and a.kind in (RAISE, CALL, CHECK):
                line = (a.kind, nr, a.allin)
            if a.kind == RAISE:
                nr += 1
        st = self.track.stats(seat)
        loose = float(np.clip(st["vpip"] / 0.25, 0.5, 3.0))
        aggro = float(np.clip(st["pfr"] / 0.15, 0.5, 4.0))
        if line is None:
            return 0.0, 1.0
        kind, nr, allin = line
        p = v.players.get(seat)
        eff = (p.start if p else 100 * v.bb) / max(v.bb, 1)
        if kind == RAISE:
            if allin:
                return 0.0, min(1.0, PF.jam_width(eff, 2) * (0.6 if nr else 1.0) * aggro)
            if nr == 0:
                return 0.0, min(1.0, 0.20 * aggro)
            if nr == 1:
                return 0.0, min(1.0, 0.06 * aggro)
            return 0.0, 0.03
        if kind == CALL:
            if nr == 0:
                return 0.08, min(1.0, 0.08 + 0.45 * loose)
            if nr == 1:
                return 0.04, min(1.0, 0.04 + (0.35 if seat == v.bb_seat else 0.15) * loose)
            return 0.02, 0.10
        return 0.10, 1.0

    def ranges(self, v, likelihood=False):
        """Weights per opponent still in the hand, narrowed by postflop actions."""
        g = self.g
        dead = g.dead_mask(v.hole + v.board)
        boards = {1: v.board[:3], 2: v.board[:4], 3: v.board[:5]}
        sc = {st: g.scores(boards[st], g.dead_mask(v.hole + boards[st])) for st in range(1, v.street + 1)}
        out = {}
        for s, p in v.players.items():
            if s == v.seat or not p.in_hand:
                continue
            lo, hi = self.villain_band(v, s)
            w = g.range_interval(lo, hi, dead)
            af = self.track.stats(s)["af"]
            for st in range(1, v.street + 1):
                score = sc[st]
                w = w * (score >= 0)
                for a in v.acts[st]:
                    if a.seat != s:
                        continue
                    w = w * self.action_likelihood(a, score, af, likelihood)
            if w.sum() <= 0:
                w = g.range_interval(lo, hi, dead)
            out[s] = w
        return out, sc

    @staticmethod
    def action_likelihood(a, score, af, smooth):
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
        shift = 0.08 * float(np.clip(af - 1.5, -1.5, 3.0))  # aggressive players bet wider
        if a.kind == RAISE:
            c = (0.72 if a.facing > 0 else 0.55) - shift
            l = sig((score - c) / 0.07) + (0.12 if not smooth else 0.15) * sig((0.2 - score) / 0.07)
        elif a.kind == CALL:
            l = sig((score - 0.35 + shift / 2) / 0.08) * (1 - 0.5 * sig((score - 0.88) / 0.05))
        elif a.kind == CHECK:
            l = 1.0 - 0.5 * sig((score - 0.75) / 0.07)
        else:
            l = np.ones_like(score)
        return np.where(score >= 0, l, 0.0)

    def equity(self, v, ranges, sc):
        w = [ranges[s] for s in ranges]
        if not w:
            return 1.0, None
        seed = V.stable_seed(self.label, v.table_id, v.hand_number, len(v.history))
        main_scores = sc.get(v.street, np.zeros(self.g.M))
        _, hu, al = self.g.equity(v.hole, v.board, w, self.n_mc, main_scores, seed)
        return float(al.mean()), hu

    # ------------------------------------------------------------- preflop
    def preflop(self, v):
        pct = self.pct(v)
        eff = self.eff_bb(v)
        raises, limpers, hero_raised = self.preflop_line(v)
        if eff <= self.pf_bb:
            return self.pushfold(v, pct, eff, raises)
        behind = V.players_behind(v, v.pre_order)
        if not raises:
            width = self.open_width(v, len(behind))
            if limpers:
                width *= 1.1
            if pct < width and RAISE in v.legal:
                return self.raise_to(v, (self.open_bb + len(limpers)) * v.bb)
            if CHECK in v.legal:
                return CHECK, None
            return FOLD, None
        last = raises[-1]
        vil = last.seat
        lo, hi = self.villain_band(v, vil)
        to_call, pot = v.to_call, v.pot
        need = to_call / max(pot + to_call, 1)
        ip = V.hero_in_position(v)
        if not hero_raised and len(raises) == 1:
            if pct < self.tb_value or (pct < 0.30 and self.rng.random() < self.tb_bluff):
                mult = 3.0 if ip else 4.0
                total = last.total * mult
                if eff <= 25 or total > 0.4 * v.hero.avail:
                    return self.jam(v)
                return self.raise_to(v, total)
            eq = self.g.pre_equity(v.hole, lo, hi)
            R = 0.95 if ip else 0.80
            if pct < self.call_cap and eq * R >= need + 0.02:
                return CALL, None
            return self.passive(v)
        # facing a 3-bet or more
        eq = self.g.pre_equity(v.hole, lo, hi)
        if pct < 0.025 or eq >= 0.55:
            return self.jam(v)
        if eff > 40 and eq * 0.85 >= need and pct < 0.08:
            return CALL, None
        return self.passive(v)

    def pushfold(self, v, pct, eff, raises):
        behind = V.players_behind(v, v.pre_order)
        if not raises:
            if pct < PF.jam_width(eff, max(1, len(behind))) and RAISE in v.legal:
                return self.jam(v)
            return self.passive(v)
        vil = raises[-1].seat
        lo, hi = self.villain_band(v, vil)
        eq = self.g.pre_equity(v.hole, lo, hi)
        if self.icm_aware:
            w, l, f = I.allin_values(v, vil)
            ok = eq * w + (1 - eq) * l > f
        else:
            need = v.to_call / max(v.pot + v.to_call, 1)
            ok = eq >= need
        if ok:
            return self.jam(v) if RAISE in v.legal and behind else ((CALL, None) if CALL in v.legal else self.jam(v))
        return self.passive(v)

    icm_aware = False

    # ------------------------------------------------------------ postflop
    def postflop(self, v):
        ranges, sc = self.ranges(v)
        eq, _ = self.equity(v, ranges, sc)
        return self.post_decide(v, eq, ranges, sc)

    def was_aggressor(self, v, street):
        acts = v.acts[street]
        raises = [a.seat for a in acts if a.kind == RAISE]
        return bool(raises) and raises[-1] == v.seat

    def post_decide(self, v, eq, ranges, sc):
        street = v.street
        fd, outs = hero_draw_flags(v.hole, v.board, self.g.rule == 1)
        draw = fd or outs >= 2
        size = self.sizes[min(street, 3) - 1]
        n_opp = len(ranges)
        r = self.rng.random()
        if v.to_call <= 0:
            if eq >= self.value_eq + 0.04 * (n_opp - 1):
                return self.bet_frac(v, size)
            if draw and eq >= 0.30 and r < self.semi and n_opp <= 2:
                return self.bet_frac(v, size)
            if n_opp == 1 and street == 1 and self.was_aggressor(v, 0) and r < self.cbet:
                return self.bet_frac(v, 0.4)
            if n_opp == 1 and street == 2 and self.was_aggressor(v, 1) and eq >= 0.3 and r < self.barrel:
                return self.bet_frac(v, size)
            if n_opp == 1 and street == 3 and eq < 0.2 and r < self.bluff_river:
                return self.bet_frac(v, size)
            return CHECK, None
        need = v.to_call / max(v.pot + v.to_call, 1)
        oop = 0.0 if V.hero_in_position(v) else 0.03
        if eq >= self.raise_eq and RAISE in v.legal:
            return self.bet_frac(v, 0.75)
        if draw and street < 3 and eq >= need and r < self.semi * 0.3 and RAISE in v.legal:
            return self.bet_frac(v, 0.75)
        if eq >= need + self.call_margin + oop:
            return CALL, None
        return FOLD, None


class TAG(Strong):
    label = "TAG"


class LAG(Strong):
    label = "LAG"
    open_mult = 1.5
    open_bb = 2.5
    tb_value = 0.06
    tb_bluff = 0.25
    call_cap = 0.6
    cbet = 0.8
    value_eq = 0.55
    semi = 0.8
    bluff_river = 0.3
    barrel = 0.65
    call_margin = -0.01
    raise_eq = 0.75
    sizes = (0.75, 0.85, 1.0)


class NashICM(Strong):
    label = "NashICM"
    pf_bb = 20.0
    icm_aware = True
    open_mult = 0.9
    value_eq = 0.65
    bluff_river = 0.05
    call_margin = 0.05


class ValueReg(Strong):
    label = "ValueReg"
    open_mult = 0.8
    tb_value = 0.03
    call_cap = 0.3
    cbet = 0.35
    value_eq = 0.66
    semi = 0.25
    bluff_river = 0.0
    barrel = 0.15
    call_margin = 0.06
    raise_eq = 0.85
    icm_aware = True

    def post_decide(self, v, eq, ranges, sc):
        # geometric sizing with very strong hands: get the stacks in by the river
        if v.to_call <= 0 and eq >= 0.80 and len(ranges) == 1:
            streets_left = max(1, 4 - v.street)
            S = min(v.hero.stack, max(v.players[s].stack for s in ranges))
            P = max(v.pot, 1)
            f = ((1 + 2 * S / P) ** (1 / streets_left) - 1) / 2
            return self.bet_frac(v, float(np.clip(f, 0.33, 1.5)))
        return super().post_decide(v, eq, ranges, sc)


class Adaptive(Strong):
    label = "Adaptive"

    def _kind(self, seat):
        st = self.track.stats(seat)
        if st["n"] < 8:
            return "unknown"
        if st["vpip"] > 0.45 and st["ftb"] < 0.35:
            return "station"
        if st["af"] > 3.0 and st["pfr"] > 0.3:
            return "maniac"
        if st["vpip"] < 0.15:
            return "nit"
        return "reg"

    def preflop(self, v):
        behind = V.players_behind(v, v.pre_order)
        kinds = [self._kind(s) for s in behind]
        self.open_mult = 1.4 if kinds and all(k == "nit" for k in kinds) else (0.85 if "maniac" in kinds else 1.0)
        return super().preflop(v)

    def post_decide(self, v, eq, ranges, sc):
        main = next(iter(ranges)) if ranges else None
        for a in reversed(v.acts[v.street]):
            if a.kind == RAISE and a.seat != v.seat:
                main = a.seat
                break
        k = self._kind(main) if main is not None else "unknown"
        base = {n: getattr(Strong, n) for n in ("value_eq", "bluff_river", "call_margin", "semi", "cbet", "sizes")}
        for n, val in base.items():
            setattr(self, n, val)
        if k == "station":
            self.value_eq, self.bluff_river, self.semi, self.cbet = 0.55, 0.0, 0.2, 0.3
            self.sizes = (0.85, 1.0, 1.1)
        elif k == "maniac":
            self.call_margin, self.value_eq, self.bluff_river = -0.06, 0.75, 0.0
            if v.to_call <= 0 and eq >= 0.7 and self.rng.random() < 0.6:
                return CHECK, None  # let the maniac bet into us
        elif k == "nit":
            st = self.track.stats(main)
            if v.to_call > 0 and eq < 0.6:
                return FOLD, None  # nits' aggression is value
            self.cbet, self.bluff_river = 0.85, 0.35 if st["ftb"] > 0.5 else 0.1
        return super().post_decide(v, eq, ranges, sc)


class Balanced(Strong):
    """Frequency-based play estimated by sampling its own range's equities."""

    label = "Balanced"
    open_mult = 1.1
    tb_value = 0.04
    tb_bluff = 0.3
    n_own = 24
    n_own_mc = 50

    def own_range_eqs(self, v, ranges):
        """Equities of hands from our own estimated range on this board."""
        g = self.g
        lo, hi = self.villain_band(v, v.seat)
        dead = g.dead_mask(v.board)
        w = g.range_interval(lo, hi, dead)
        if w.sum() <= 0:
            return np.array([0.5])
        idx = np.random.default_rng(V.stable_seed(v.table_id, v.hand_number, v.street)).choice(
            g.M, size=self.n_own, p=w / w.sum())
        wlist = [ranges[s] for s in ranges]
        out = []
        for k in idx:
            hole = [int(x) for x in g.U[k]]
            if any((1 << c) & g.dead_mask(v.board) for c in hole):
                continue
            ww = [x * g.live(g.dead_mask(hole + v.board)) for x in wlist]
            if any(x.sum() <= 0 for x in ww):
                continue
            _, _, al = g.equity(hole, v.board, ww, self.n_own_mc, np.zeros(g.M), int(k))
            out.append(al.mean())
        return np.array(out) if out else np.array([0.5])

    def post_decide(self, v, eq, ranges, sc):
        own = self.own_range_eqs(v, ranges)
        r = self.rng.random()
        street = v.street
        size = self.sizes[min(street, 3) - 1]
        if v.to_call <= 0:
            v_thr = float(np.quantile(own, 0.70))
            if eq >= max(v_thr, 0.5):
                return self.bet_frac(v, size)
            b_thr = float(np.quantile(own, 0.30))
            if eq <= b_thr and len(ranges) == 1:
                # bluff:value = s : (1 + s) with equal-size candidate pools
                if r < size / (1 + size):
                    return self.bet_frac(v, size)
            return CHECK, None
        s = v.to_call / max(v.pot - v.to_call, 1)
        mdf = 1.0 / (1.0 + s)
        cont_thr = float(np.quantile(own, 1.0 - mdf))
        raise_thr = float(np.quantile(own, 0.92))
        if eq >= raise_thr and RAISE in v.legal and eq > 0.6:
            return self.bet_frac(v, 0.75)
        if eq >= cont_thr or eq >= v.to_call / max(v.pot + v.to_call, 1) + 0.1:
            return CALL, None
        return FOLD, None


class BayesReader(Strong):
    """Bayesian ranges (likelihood per action) + EV-max bet choice with observed folds."""

    label = "BayesReader"
    sizes_menu = (0.5, 0.75, 1.0)

    def postflop(self, v):
        ranges, sc = self.ranges(v, likelihood=True)
        eq, hu = self.equity(v, ranges, sc)
        pot = max(v.pot, 1)
        hero = v.hero
        if not ranges:
            return self.passive(v)
        main = list(ranges)[0]
        for a in reversed(v.acts[v.street]):
            if a.kind == RAISE and a.seat != v.seat:
                main = a.seat
        ftb = self.track.stats(main)["ftb"]
        n_opp = len(ranges)
        if v.to_call <= 0:
            best = (eq * pot, CHECK, None)
            for s in self.sizes_menu:
                b = min(s * pot, hero.stack)
                F = min(0.9, ftb * (0.8 + 0.4 * s)) ** n_opp
                # equity when called: vs the stronger part of the range
                eq_c = max(0.0, eq - 0.15 * (1 - eq) * (F > 0.3))
                ev = F * pot + (1 - F) * (eq_c * (pot + 2 * b) - b)
                if ev > best[0]:
                    best = (ev, RAISE, s)
            if best[1] == RAISE:
                return self.bet_frac(v, best[2])
            return CHECK, None
        c = min(v.to_call, hero.stack)
        ev_call = eq * (pot + c) - c
        ev_raise = -1e9
        if RAISE in v.legal and eq > 0.7:
            add = c + 0.75 * (pot + c)
            F2 = 0.35
            ev_raise = F2 * pot + (1 - F2) * ((eq - 0.1) * (pot + 2 * add) - add)
        if ev_raise > max(ev_call, 0):
            return self.bet_frac(v, 0.75)
        if ev_call > 0:
            return CALL, None
        return FOLD, None


STRONG_BOTS = {
    "tag": TAG, "lag": LAG, "nashicm": NashICM, "valuereg": ValueReg,
    "adaptive": Adaptive, "balanced": Balanced, "bayes": BayesReader,
}
