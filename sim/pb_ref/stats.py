"""Opponent statistics: counters per seat, updated from completed hands.

Each counter pair is (opportunities, occurrences). Estimates are shrunk
toward an equilibrium-style prior (team design):

    p_hat = (occurrences + k * p_prior) / (opportunities + k)

so with no data we play the default, and we deviate as evidence builds.
``k`` is how many observations it takes to move halfway from the prior.
"""
from __future__ import annotations

import numpy as np

# counter indices --------------------------------------------------------------
HANDS, VPIP, PFR = 0, 1, 2
OPEN_OPP, OPEN = 3, 4            # first in (no raise yet) -> raised
TB_OPP, TB, CC = 5, 6, 7         # faced exactly one raise -> re-raised / cold-called
F3B_OPP, F3B = 8, 9              # raised, then faced a re-raise -> folded
JAM = 10                         # preflop all-in raises (per HANDS)
CJ_OPP, CJ = 11, 12              # faced a preflop all-in -> called
N_SIZES = 3                      # our menu: 33%, 66%, 125% of the pot
FTB_OPP0 = 13                    # + (street-1)*3 + size bucket: faced a bet
FTB0 = 22                        # same layout: folded to it
RVB_OPP, RVB = 31, 32            # faced a postflop bet -> raised
BU_OPP, BU = 33, 34              # postflop, no bet in front, not the c-bettor -> bet
SAWF, WTSD = 35, 36              # saw the flop -> reached showdown
RB_SHOWN, RB_BLUFF = 37, 38      # bet/raised the river and showed -> was weak
BVS_OPP, BVS_FOLD = 39, 40       # in a blind facing a lone open -> folded
FTR_OPP, FTR = 41, 42            # bet postflop, got raised -> folded
LIMP = 43                        # limped first in (per OPEN_OPP)
CB_OPP, CB = 44, 45              # preflop raiser, flop checked to / first in -> c-bet
RZ_SHOWN, RZ_BLUFF = 46, 47      # raised a postflop bet and showed -> was weak
C3B, R4B = 48, 49                # raised, faced a re-raise -> called / re-raised (per F3B_OPP)
K_STATS = 50

MENU = (0.33, 0.66, 1.25)


def size_bucket(frac):
    """Map a bet (fraction of pot) onto our menu with the pseudo-harmonic rule.

    Between menu sizes A < x < B, x counts as A with probability
    p(A) = (B - x)(1 + A) / ((B - A)(1 + x)); we take the likelier side.
    """
    if frac <= MENU[0]:
        return 0
    if frac >= MENU[-1]:
        return len(MENU) - 1
    for i in range(len(MENU) - 1):
        a, b = MENU[i], MENU[i + 1]
        if a <= frac <= b:
            pa = (b - frac) * (1 + a) / ((b - a) * (1 + frac))
            return i if pa >= 0.5 else i + 1
    return len(MENU) - 1


def replay_hand(events, mem, hero_seat, game, starting_stack):
    """Update ``mem.stats`` and ``mem.last_stacks`` from one finished hand.

    events: list of (kind, seat, amount, cards) with cards as ints.
    """
    S = mem.stats
    start = {s: int(mem.last_stacks[i]) for i, s in enumerate(mem.seats)}
    committed = {}
    round_bet = {}
    high = 0
    pot = 0
    street = 0
    n_raises = 0
    callers_since_raise = 0
    last_raiser = None
    last_raise_allin = False
    first_done = set()
    voluntarily = set()
    raised_pre = {}            # seat -> raise count index when they last raised
    f3b_counted = set()
    blinds = []
    folded = {}
    board = []
    shown = {}
    river_aggr = set()
    raisers_post = set()
    cb_done = set()
    postflop_bettor = None     # last seat to bet/raise on this street
    last_bet_size = 0.0
    final_stacks = {}

    def sl(seat):
        return mem.slot(seat)

    for kind, seat, amount, cards in events:
        if kind == "ante":
            committed[seat] = committed.get(seat, 0) + amount
            pot += amount
            continue
        if kind == "blind":
            committed[seat] = committed.get(seat, 0) + amount
            round_bet[seat] = round_bet.get(seat, 0) + amount
            pot += amount
            high = max(high, round_bet[seat])
            blinds.append(seat)
            continue
        if kind == "board":
            street = min(street + 1, 3)
            board.extend(cards)
            round_bet = {}
            high = 0
            postflop_bettor = None
            continue
        if kind == "show":
            shown[seat] = list(cards)
            continue
        if kind == "stack":
            final_stacks[seat] = amount
            continue
        if kind not in ("fold", "check", "call", "raise"):
            continue
        i = sl(seat)
        track = i >= 0 and seat != hero_seat
        cur = round_bet.get(seat, 0)
        facing = max(0, high - cur)
        if kind == "call":
            inc = amount
            total = cur + inc
        elif kind == "raise":
            total = amount
            inc = max(0, total - cur)
        else:
            inc = 0
            total = cur
        allin = (committed.get(seat, 0) + inc) >= start.get(seat, starting_stack) and inc > 0
        if street == 0:
            if track and seat not in first_done:
                S[i, HANDS] += 1
                if n_raises == 0:
                    S[i, OPEN_OPP] += 1
                    if kind == "raise":
                        S[i, OPEN] += 1
                    elif kind == "call":
                        S[i, LIMP] += 1
                elif n_raises == 1:
                    S[i, TB_OPP] += 1
                    if kind == "raise":
                        S[i, TB] += 1
                    elif kind == "call":
                        S[i, CC] += 1
                    if seat in blinds and callers_since_raise == 0:
                        S[i, BVS_OPP] += 1
                        if kind == "fold":
                            S[i, BVS_FOLD] += 1
            if track and seat in raised_pre and raised_pre[seat] < n_raises and seat not in f3b_counted:
                S[i, F3B_OPP] += 1
                f3b_counted.add(seat)
                if kind == "fold":
                    S[i, F3B] += 1
                elif kind == "call":
                    S[i, C3B] += 1
                elif kind == "raise":
                    S[i, R4B] += 1
            if track and last_raise_allin and facing > 0:
                S[i, CJ_OPP] += 1
                if kind in ("call", "raise"):
                    S[i, CJ] += 1
            if kind in ("call", "raise"):
                voluntarily.add(seat)
            if track and kind == "raise" and allin:
                S[i, JAM] += 1
            first_done.add(seat)
            if kind == "raise":
                n_raises += 1
                raised_pre[seat] = n_raises
                last_raiser = seat
                last_raise_allin = allin
                callers_since_raise = 0
            elif kind == "call":
                callers_since_raise += 1
        else:
            if track:
                if facing > 0:
                    b = size_bucket(last_bet_size)
                    S[i, FTB_OPP0 + (street - 1) * N_SIZES + b] += 1
                    S[i, RVB_OPP] += 1
                    if kind == "fold":
                        S[i, FTB0 + (street - 1) * N_SIZES + b] += 1
                    if kind == "raise":
                        S[i, RVB] += 1
                        raisers_post.add(seat)
                    if postflop_bettor is not None and postflop_bettor != seat and \
                            committed_this_street_bet(seat, round_bet) > 0:
                        S[i, FTR_OPP] += 1
                        if kind == "fold":
                            S[i, FTR] += 1
                elif street == 1 and seat == last_raiser and seat not in cb_done:
                    cb_done.add(seat)
                    S[i, CB_OPP] += 1
                    if kind == "raise":
                        S[i, CB] += 1
                else:
                    S[i, BU_OPP] += 1
                    if kind == "raise":
                        S[i, BU] += 1
            if kind == "raise":
                pot_before_bet = pot + (high - round_bet.get(seat, 0) if high > 0 else 0)
                last_bet_size = (total - high) / max(pot_before_bet, 1)
                postflop_bettor = seat
                if street == 3:
                    river_aggr.add(seat)
        if kind == "fold":
            folded[seat] = street
        round_bet[seat] = total
        committed[seat] = committed.get(seat, 0) + inc
        pot += inc
        if total > high:
            high = total

    # per-hand flags
    for seat in voluntarily:
        i = sl(seat)
        if i >= 0 and seat != hero_seat:
            S[i, VPIP] += 1
            if seat in raised_pre:
                S[i, PFR] += 1
    if street >= 1:
        for seat in first_done:
            i = sl(seat)
            if i < 0 or seat == hero_seat:
                continue
            # saw the flop = did not fold preflop (we only know fold street roughly)
            if folded.get(seat, 9) >= 1:
                S[i, SAWF] += 1
                if seat in shown:
                    S[i, WTSD] += 1
    # river aggression that got shown down: was it a bluff?
    if (river_aggr or raisers_post) and len(board) >= 3 and game is not None:
        bdead = game.dead_mask(board)
        vals = None
        for seat in river_aggr | raisers_post:
            i = sl(seat)
            if i < 0 or seat == hero_seat or seat not in shown or len(shown[seat]) != game.H:
                continue
            if vals is None:
                vals = game.strengths(board, bdead)
                live = vals[vals >= 0]
            v = game.hand_value(shown[seat], board)
            pct = float((live < v).mean()) if live.size else 0.5
            if seat in river_aggr:
                S[i, RB_SHOWN] += 1
                if pct < 0.45:
                    S[i, RB_BLUFF] += 1
            if seat in raisers_post:
                S[i, RZ_SHOWN] += 1
                if pct < 0.45:
                    S[i, RZ_BLUFF] += 1
    for seat, st in final_stacks.items():
        i = sl(seat)
        if i >= 0:
            mem.last_stacks[i] = st


def committed_this_street_bet(seat, round_bet):
    return round_bet.get(seat, 0)


class Opp:
    """Shrunk estimates for one opponent seat."""

    def __init__(self, row, P):
        self.r = row
        self.P = P

    def rate(self, num, den, prior, k=None):
        if k is None:
            k = self.P["k_common"]
        n = float(self.r[den]) if self.r is not None else 0.0
        x = float(self.r[num]) if self.r is not None else 0.0
        return (x + k * prior) / (n + k)

    def low_bound(self, num, den, prior, k=None, z=0.5):
        """Pessimistic estimate (for bluffs): p_hat - z * sd."""
        p = self.rate(num, den, prior, k)
        n = (float(self.r[den]) if self.r is not None else 0.0) + (k or self.P["k_common"])
        return max(0.0, p - z * (p * (1 - p) / n) ** 0.5)

    # preflop
    def open_ratio(self):
        return self.rate(OPEN, OPEN_OPP, 0.22) / 0.22

    def threebet_ratio(self):
        return self.rate(TB, TB_OPP, 0.07, self.P["k_rare"]) / 0.07

    def coldcall_ratio(self):
        return self.rate(CC, TB_OPP, 0.12) / 0.12

    def response_to_3bet(self):
        """(fold, call, re-raise) when their raise is re-raised; plan: estimate
        all three from table history, shrunk toward the priors."""
        k = self.P["k_rare"]
        p4 = self.rate(R4B, F3B_OPP, self.P["prior_4bet"], k)
        pc = self.rate(C3B, F3B_OPP, self.P["prior_call_3bet"], k)
        pf = max(0.05, 1.0 - p4 - pc)
        t = pf + pc + p4
        return pf / t, pc / t, p4 / t

    def fold_to_3bet(self):
        return self.rate(F3B, F3B_OPP, self.P["prior_fold_to_3bet"], self.P["k_rare"])

    def jam_ratio(self):
        return self.rate(JAM, HANDS, 0.03, self.P["k_rare"]) / 0.03

    def call_jam_ratio(self):
        return self.rate(CJ, CJ_OPP, 0.25, self.P["k_rare"]) / 0.25

    def blind_fold_vs_steal(self):
        return self.rate(BVS_FOLD, BVS_OPP, 0.62)

    def limp_rate(self):
        return self.rate(LIMP, OPEN_OPP, 0.03)

    def vpip(self):
        return self.rate(VPIP, HANDS, 0.25)

    # postflop
    def fold_to_bet(self, street, frac, pessimistic=False):
        st = min(max(street, 1), 3)
        b = size_bucket(frac)
        prior = frac / (1.0 + frac)  # equilibrium: defend 1/(1+s) of the range
        j = (st - 1) * N_SIZES + b
        if pessimistic:
            return self.low_bound(FTB0 + j, FTB_OPP0 + j, prior, z=self.P["bluff_conf_z"])
        return self.rate(FTB0 + j, FTB_OPP0 + j, prior)

    def raise_vs_bet(self):
        return self.rate(RVB, RVB_OPP, self.P["prior_raise_vs_bet"], self.P["k_rare"])

    def bet_when_checked(self):
        return self.rate(BU, BU_OPP, self.P["prior_bet_when_checked"])

    def fold_to_raise(self, frac=None):
        """Folds when their bet is raised. Prior (plan): a balanced player folds
        about s / (1 + s), s = what they must add / the pot they would win."""
        if frac is None or not self.P.get("raise_fold_price_prior", True):
            prior = self.P["prior_fold_to_raise"]
        else:
            prior = frac / (1.0 + frac)
        return self.rate(FTR, FTR_OPP, prior, self.P["k_rare"])

    def cbet(self):
        return self.rate(CB, CB_OPP, self.P["prior_cbet"])

    def river_bluff_share(self, frac=0.66):
        """Bluffs in their river betting range. Prior: equilibrium s / (1 + 2s)."""
        return self.rate(RB_BLUFF, RB_SHOWN, frac / (1.0 + 2.0 * frac), self.P["k_rare"])

    def raise_bluff_share(self):
        """Bluffs among their raises. Prior starts low (~10%), updated from showdowns."""
        return self.rate(RZ_BLUFF, RZ_SHOWN, self.P["prior_raise_bluff_share"], self.P["k_rare"])
