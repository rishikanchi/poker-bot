"""Opponent statistics: counters per seat, updated from completed hands.

Each counter pair is (opportunities, occurrences). Estimates are shrunk
toward an equilibrium-style prior (team design):

    p_hat = (occurrences + k * p_prior) / (opportunities + k)

so with no data we play the default, and we deviate as evidence builds.
``k`` is how many observations it takes to move halfway from the prior.
"""
from __future__ import annotations

import numpy as np

from . import pushfold as PF

# counter indices --------------------------------------------------------------
HANDS, VPIP, PFR = 0, 1, 2
OPEN_OPP, OPEN = 3, 4            # first in, deeper than push/fold -> raised
TB_OPP, TB, CC = 5, 6, 7         # faced exactly one raise -> re-raised / cold-called
F3B_OPP, F3B = 8, 9              # raised, then faced a re-raise -> folded
JAM = 10                         # first-in all-in raises at chart depth (<= 25bb)
CJ_OPP, CJ = 11, 12              # faced a preflop all-in -> called
N_SIZES = 5                      # PLAN translation points for bets faced: 33 / 66 / 100 / 125% of the pot, all-in
FTB_OPP0 = 13                    # + (street-1)*5 + size bucket: faced a bet
FTB0 = 28                        # same layout: folded to it
RVB_OPP, RVB = 43, 44            # faced a postflop bet -> raised
BU_OPP, BU = 45, 46              # postflop, checked to (not the c-bet spot) -> bet
SAWF, WTSD = 47, 48              # saw the flop -> reached showdown
RB_SHOWN, RB_BLUFF = 49, 50      # bet/raised the river and showed -> was weak
BVS_OPP, BVS_FOLD = 51, 52       # in a blind facing a lone open -> folded
FTR_OPP, FTR = 53, 54            # had chips in this street, got raised -> folded
LIMP = 55                        # limped first in (per LIMP_OPP)
CB_OPP, CB = 56, 57              # preflop raiser, flop checked to / first in -> c-bet
RZ_SHOWN, RZ_BLUFF = 58, 59      # raised a postflop bet and showed -> was not a strong hand then
C3B, R4B = 60, 61                # raised, faced a re-raise -> called / re-raised (per F3B_OPP)
OPEN_EXP = 62                    # x100: opens a baseline player would make in the same spots
JAM_EXP = 63                     # x100: first-in jams the solved chart expects in the same spots
CJ_EXP = 64                      # x100: calls of all-ins the solved chart expects
LIMP_OPP = 65                    # first-in opportunities at any depth
JAM_D_OPP, JAM_D = 66, 67        # first in, deeper than the chart (> 25bb) -> open-jammed
CCN_OPP, CCN = 68, 69            # NOT the big blind, faced exactly one raise -> cold-called
MZ_SHOWN, MZ_WEAK = 70, 71       # made a minimum-size bet/raise and showed -> was a weak hand then
BSZ0 = 72                        # + size bucket: postflop bets / raises THEY made, by size (5)
CALL_SHOWN, CALL_WEAK = 77, 78   # called a postflop bet and showed -> was a weak hand on that board
BETF_N, BETF_SUM = 79, 80        # postflop bets they made: count, sum of sizes in hundredths of the pot
B_SHOWN0, B_WEAK0 = 81, 84       # + street-1: bet (no bet faced) on that street and showed -> weak then
RBS_SHOWN0, RBS_BLUFF0 = 87, 92  # + size bucket: river bet/raise shown down, by size (5 each)
K_STATS = 97

MENU = (0.33, 0.66, 1.0, 1.25)   # PLAN: opponents' bets translate onto 33 / 66 / 100 / 125 / all-in
ALLIN_BUCKET = 4


def size_bucket(frac, allin=False):
    """Map a bet (fraction of pot) onto the translation points with the
    pseudo-harmonic rule; an all-in is its own point.

    Between points A < x < B, x counts as A with probability
    p(A) = (B - x)(1 + A) / ((B - A)(1 + x)); we take the likelier side.
    """
    if allin:
        return ALLIN_BUCKET
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


def min_weight(by, min_by, pot_call, upper):
    """How much a bet / raise counts as "the minimum" (plan: opponents' bets
    are translated onto min, 33%, 75%, 125%, all-in with the pseudo-harmonic
    rule between points; raises onto min-raise, 0.75 pot, pot, all-in).

    by: the bet, or how far the raise went over the bet it faced; min_by: the
    smallest legal amount; pot_call: the pot after calling; upper: for a bet,
    the next translation point (a fraction of that pot); None for a raise.
    Bets: 1 = exactly the minimum, falling to 0 at the next point (0 as well
    when the minimum itself is that big). Raises: 1 for a min-raise, else 0.
    """
    if pot_call <= 0 or by <= 0:
        return 0.0
    if upper is None:                  # a raise: it is a min-raise (a "click-back") or it is not
        return 1.0 if by <= min_by * 1.001 else 0.0
    x = by / pot_call
    a = min_by / pot_call
    if a >= upper or (by + 1) / pot_call >= upper:     # (+1 chip: a bet rounded down from the point is at it)
        return 0.0
    if x <= a:
        return 1.0
    return float((upper - x) * (1 + a) / ((upper - a) * (1 + x)))


def _open_width(P, n_behind, n_alive):
    if n_alive <= 2:
        return P["open_width_hu"]
    return P["open_width"].get(int(n_behind), 0.13 if n_behind > 4 else 0.42)


def replay_hand(events, mem, hero_seat, game, starting_stack, P=None):
    """Update ``mem.stats`` and ``mem.last_stacks`` from one finished hand.

    events: list of (kind, seat, amount, cards) with cards as ints.
    """
    if P is None:
        from . import settings
        P = settings.P
    S = mem.stats
    start = {s: int(mem.last_stacks[i]) for i, s in enumerate(mem.seats)}
    committed = {}
    round_bet = {}
    high = 0
    pot = 0
    street = 0
    n_raises = 0
    n_limps = 0
    checks_street = 0          # checks so far on the current postflop street
    callers_since_raise = 0
    last_raiser = None
    last_raise_allin = False
    shover_k = 1               # players behind the last all-in raiser
    first_done = set()
    voluntarily = set()
    raised_pre = {}            # seat -> raise count index when they last raised
    f3b_counted = set()
    blinds = []
    folded = {}
    board = []
    shown = {}
    river_aggr = set()
    raise_board = {}           # seat -> board length when they first raised a postflop bet
    cb_done = set()
    last_by = 0                # smallest legal bet / raise-by on this street (0 = one big blind)
    min_board = {}             # seat -> board length when they made a minimum-size bet/raise
    postflop_bettor = None     # last seat to bet/raise on this street
    last_bet_size = 0.0
    last_bet_allin = False
    call_board = {}            # seat -> board length when they first called a postflop bet
    bet_board = {}             # seat -> (board length, street) of their first postflop bet (no bet faced)
    river_bucket = {}          # seat -> size bucket of their river bet / raise
    final_stacks = {}
    bb = 0
    dealt_in = 0               # seats that posted or acted this hand

    def sl(seat):
        return mem.slot(seat)

    n_alive = sum(1 for s in mem.seats if start.get(s, 0) > 0)
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
            bb = max(bb, amount)
            blinds.append(seat)
            continue
        if kind == "board":
            street = min(street + 1, 3)
            board.extend(cards)
            round_bet = {}
            high = 0
            last_by = 0
            postflop_bettor = None
            checks_street = 0
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
        if kind == "raise":
            by = max(0, total - high)
            min_by = max(last_by, bb, 1)
            # minimum-size bets and re-raises (not opens, whose size is normal, and not all-ins)
            if track and not allin and (street >= 1 or n_raises >= 1):
                upper = P["min_translate_bet"] if (street >= 1 and facing == 0) else None
                if min_weight(by, min_by, pot + facing, upper) >= 0.5:
                    min_board.setdefault(seat, len(board))
            if by >= max(last_by, bb):
                last_by = by
        if street == 0:
            first = seat not in first_done
            n_behind = max(1, n_alive - len(first_done) - 1)
            eff_bb = start.get(seat, starting_stack) / max(bb, 1)
            if track and first:
                S[i, HANDS] += 1
                if n_raises == 0 and n_limps == 0:
                    # truly first in (a big blind checking its option or an
                    # over-limp is not an opening opportunity)
                    S[i, LIMP_OPP] += 1
                    if kind == "call":
                        S[i, LIMP] += 1
                    if eff_bb > P["push_fold_bb"]:
                        S[i, OPEN_OPP] += 1
                        S[i, OPEN_EXP] += int(round(100 * _open_width(P, n_behind, n_alive)))
                        if kind == "raise":
                            S[i, OPEN] += 1
                    # all-ins. Within the solved chart's depth: compare with what the
                    # chart expects in the same spots. Deeper than that nobody
                    # "should" open-jam, so the frequency itself is the evidence.
                    if eff_bb <= 25.0:
                        S[i, JAM_EXP] += int(round(100 * PF.jam_width(eff_bb, n_behind, P)))
                        if kind == "raise" and allin:
                            S[i, JAM] += 1
                    else:
                        S[i, JAM_D_OPP] += 1
                        if kind == "raise" and allin:
                            S[i, JAM_D] += 1
                elif n_raises == 1:
                    S[i, TB_OPP] += 1
                    if kind == "raise":
                        S[i, TB] += 1
                    elif kind == "call":
                        S[i, CC] += 1
                    if not (blinds and seat == blinds[-1]):   # the big blind's defends are a different range
                        S[i, CCN_OPP] += 1
                        if kind == "call":
                            S[i, CCN] += 1
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
                e = min(start.get(seat, starting_stack), start.get(last_raiser, starting_stack)) / max(bb, 1)
                S[i, CJ_EXP] += int(round(100 * PF.call_width(e, len(blinds) >= 1 and seat == blinds[-1],
                                                               shover_k, P)))
                if kind in ("call", "raise"):
                    S[i, CJ] += 1
            if kind in ("call", "raise"):
                voluntarily.add(seat)
            first_done.add(seat)
            if kind == "raise":
                n_raises += 1
                raised_pre[seat] = n_raises
                last_raiser = seat
                last_raise_allin = allin
                if allin:
                    shover_k = n_behind
                callers_since_raise = 0
            elif kind == "call":
                callers_since_raise += 1
                if n_raises == 0:
                    n_limps += 1
        else:
            if track:
                if facing > 0 and round_bet.get(seat, 0) > 0:
                    # they already had chips in this street and were raised
                    S[i, FTR_OPP] += 1
                    if kind == "fold":
                        S[i, FTR] += 1
                elif facing > 0:
                    b = size_bucket(last_bet_size, last_bet_allin)
                    S[i, FTB_OPP0 + (street - 1) * N_SIZES + b] += 1
                    S[i, RVB_OPP] += 1
                    if kind == "fold":
                        S[i, FTB0 + (street - 1) * N_SIZES + b] += 1
                    if kind == "raise":
                        S[i, RVB] += 1
                        raise_board.setdefault(seat, len(board))
                    if kind == "call":
                        call_board.setdefault(seat, len(board))
                elif street == 1 and seat == last_raiser and seat not in cb_done:
                    cb_done.add(seat)
                    S[i, CB_OPP] += 1
                    if kind == "raise":
                        S[i, CB] += 1
                elif checks_street > 0:
                    # checked to (the plan's "bets when checked to")
                    S[i, BU_OPP] += 1
                    if kind == "raise":
                        S[i, BU] += 1
            if kind == "check":
                checks_street += 1
            if kind == "raise":
                pot_before_bet = pot + (high - round_bet.get(seat, 0) if high > 0 else 0)
                was_bet = facing == 0
                last_bet_size = (total - high) / max(pot_before_bet, 1)
                last_bet_allin = allin
                if track:
                    S[i, BSZ0 + size_bucket(last_bet_size, allin)] += 1   # the sizes this player uses
                    if was_bet:
                        S[i, BETF_N] += 1                                  # their usual bet size
                        S[i, BETF_SUM] += int(round(100 * min(last_bet_size, 3.0)))
                        bet_board.setdefault(seat, (len(board), street))
                postflop_bettor = seat
                if street == 3:
                    river_aggr.add(seat)
                    river_bucket[seat] = size_bucket(last_bet_size, allin)
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
            if folded.get(seat, 9) >= 1:
                S[i, SAWF] += 1
                if seat in shown:
                    S[i, WTSD] += 1
    # PLAN: each time a bet or raise reaches showdown, record whether it was a
    # bluff. River bets are judged on the final board; a raise is judged on the
    # board it was made on ("strong hand then" vs draw / air).
    if min_board and game is not None:
        # PLAN: "watch whether a particular opponent only min-raises with strong hands"
        thr = P["bluff_showdown_pct"]
        for seat, nb in min_board.items():
            i = sl(seat)
            if i < 0 or seat == hero_seat or seat not in shown or len(shown[seat]) != game.H:
                continue
            if nb >= 3:
                vals = game.strengths(board[:nb], game.dead_mask(board[:nb]))
                live = vals[vals >= 0]
                pct = float((live < game.hand_value(shown[seat], board[:nb])).mean()) if live.size else 0.5
            else:
                pct = 1.0 - game.hero_pre_pct(shown[seat])     # preflop: 1 = the best starting hand
            S[i, MZ_SHOWN] += 1
            if pct < thr:
                S[i, MZ_WEAK] += 1
    if (river_aggr or raise_board or call_board or bet_board) and len(board) >= 3 and game is not None:
        thr = P["bluff_showdown_pct"]
        pct_cache = {}

        def pct_on(cards, nb):
            key = nb
            if key not in pct_cache:
                vals = game.strengths(board[:nb], game.dead_mask(board[:nb]))
                pct_cache[key] = vals[vals >= 0]
            live = pct_cache[key]
            v = game.hand_value(cards, board[:nb])
            return float((live < v).mean()) if live.size else 0.5

        for seat in set(river_aggr) | set(raise_board):
            i = sl(seat)
            if i < 0 or seat == hero_seat or seat not in shown or len(shown[seat]) != game.H:
                continue
            if seat in river_aggr:
                S[i, RB_SHOWN] += 1
                b = river_bucket.get(seat, 1)
                S[i, RBS_SHOWN0 + b] += 1
                if pct_on(shown[seat], len(board)) < thr:
                    S[i, RB_BLUFF] += 1
                    S[i, RBS_BLUFF0 + b] += 1
            if seat in bet_board:
                nb, st_b = bet_board[seat]
                if st_b <= 2:                      # flop / turn bets, judged on the board they were made on
                    S[i, B_SHOWN0 + st_b - 1] += 1
                    if pct_on(shown[seat], nb) < thr:
                        S[i, B_WEAK0 + st_b - 1] += 1
            if seat in raise_board:
                S[i, RZ_SHOWN] += 1
                if pct_on(shown[seat], raise_board[seat]) < thr:
                    S[i, RZ_BLUFF] += 1
        # PLAN: showdowns also say how strong their calling range is (thin value)
        for seat, nb in call_board.items():
            i = sl(seat)
            if i < 0 or seat == hero_seat or seat not in shown or len(shown[seat]) != game.H:
                continue
            S[i, CALL_SHOWN] += 1
            if pct_on(shown[seat], nb) < thr:
                S[i, CALL_WEAK] += 1
    for seat, st in final_stacks.items():
        i = sl(seat)
        if i >= 0:
            mem.last_stacks[i] = st
    if final_stacks:
        # a seat with no end-of-hand stack has busted: it must not count as a
        # player in later hands (positions and expected frequencies depend on it)
        for i, seat in enumerate(mem.seats):
            if seat not in final_stacks:
                mem.last_stacks[i] = 0


class Opp:
    """Shrunk estimates for one opponent seat.

    rate:  p_hat = (occurrences + k * prior) / (opportunities + k)        (plan)
    ratio: (events + k_ratio) / (expected events + k_ratio), where "expected"
           is what a baseline player would have done in the same spots; it
           multiplies the baseline width for the spot at hand.
    """

    def __init__(self, row, P):
        self.r = row
        self.P = P

    def n(self, idx):
        return float(self.r[idx]) if self.r is not None else 0.0

    def rate(self, num, den, prior, k=None, extra=0):
        """``extra`` adds an observation we are looking at right now (the
        action that is still on the table and not yet in the counters)."""
        if k is None:
            k = self.P["k_common"]
        return (self.n(num) + extra + k * prior) / (self.n(den) + extra + k)

    def ratio(self, num, exp100):
        k = self.P["k_ratio"]
        return (self.n(num) + k) / (self.n(exp100) / 100.0 + k)

    def low_bound(self, num, den, prior, k=None, z=0.5):
        """Pessimistic estimate (for bluffs): p_hat - z * sd."""
        p = min(max(self.rate(num, den, prior, k), 0.0), 1.0)
        n = self.n(den) + (k or self.P["k_common"])
        return max(0.0, p - z * (p * (1 - p) / n) ** 0.5)

    # preflop
    def open_ratio(self):
        return self.ratio(OPEN, OPEN_EXP)

    def threebet_width(self):
        """Share of hands they re-raise an open with: the observed rate, shrunk
        toward the plan's ~4% (facing a raise is a common spot, so the common k)."""
        return self.rate(TB, TB_OPP, self.P["prior_3bet"])

    def coldcall_ratio(self):
        """How much more often than the baseline they call an open when NOT in the big blind."""
        p0 = self.P["prior_coldcall"]
        return self.rate(CCN, CCN_OPP, p0) / p0

    def bb_defend_ratio(self):
        """How much more often than the baseline they continue in a blind against a lone open."""
        p0 = 1.0 - self.P["prior_blind_fold"]
        return (1.0 - self.blind_fold_vs_steal()) / p0

    def response_to_3bet(self, level=1):
        """(fold, call, re-raise) when their raise is re-raised; plan: estimate
        all three from table history, shrunk toward the priors.

        level 1: their open is 3-bet. level 2+: their 3-bet (or 4-bet ...) is
        re-raised. The evidence is the same - what this player does when a
        raise of theirs is re-raised - only the starting values differ."""
        k = self.P["k_rare"]
        if level <= 1:
            p4_0, pc_0 = self.P["prior_4bet"], self.P["prior_call_3bet"]
        else:
            p4_0 = self.P["prior_5bet"]
            pc_0 = max(0.0, 1.0 - self.P["prior_fold_to_4bet"] - p4_0)
        p4 = self.rate(R4B, F3B_OPP, p4_0, k)
        pc = self.rate(C3B, F3B_OPP, pc_0, k)
        pf = max(0.05, 1.0 - p4 - pc)
        t = pf + pc + p4
        return pf / t, pc / t, p4 / t

    def n_3bet_faced(self):
        return self.n(F3B_OPP)

    def jam_width(self, chart_w):
        """Width of their all-in range, given the solved chart width for the spot.

        Two pieces of evidence: how often they shove compared with the chart in
        chart-depth spots (a ratio that scales the chart), and how often they
        open-shove deep stacks, where the chart says never (that frequency is a
        direct lower bound on their shoving range)."""
        ratio = min(max(self.ratio(JAM, JAM_EXP), 0.6), 3.0)
        deep = self.n(JAM_D) / (self.n(JAM_D_OPP) + self.P["k_common"])
        return min(max(chart_w * ratio, deep), 1.0)

    def call_jam_ratio(self):
        return self.ratio(CJ, CJ_EXP)

    def blind_fold_vs_steal(self):
        return self.rate(BVS_FOLD, BVS_OPP, self.P["prior_blind_fold"])

    def limp_rate(self, extra=0):
        return self.rate(LIMP, LIMP_OPP, self.P["prior_limp"], extra=extra)

    def vpip(self):
        return self.rate(VPIP, HANDS, self.P["prior_vpip"])

    def pfr(self):
        return self.rate(PFR, HANDS, self.P["prior_pfr"])

    # postflop
    def fold_to_bet(self, street, frac, pessimistic=False, allin=False):
        """P(they fold) to a bet of ``frac`` pots on ``street``.

        Plan: "one overall fold-to-bet plus a per-street offset", with a larger k
        for the rare later-street cells: the cells of this size on every street
        are pooled first (prior: defend 1/(1+s) of the range), and the street's own
        cell is shrunk toward that pooled rate."""
        st = min(max(street, 1), 3)
        b = size_bucket(frac, allin)
        prior = frac / (1.0 + frac)
        k = self.P["k_common"]
        num = sum(self.n(FTB0 + (s - 1) * N_SIZES + b) for s in (1, 2, 3))
        den = sum(self.n(FTB_OPP0 + (s - 1) * N_SIZES + b) for s in (1, 2, 3))
        p_pool = (num + k * prior) / (den + k)
        j = (st - 1) * N_SIZES + b
        k2 = self.P["k_rare"] if st >= 2 else k
        n_cell = self.n(FTB_OPP0 + j)
        p = min(max((self.n(FTB0 + j) + k2 * p_pool) / (n_cell + k2), 0.0), 1.0)
        if pessimistic:
            z = self.P["bluff_conf_z"]
            return max(0.0, p - z * (p * (1 - p) / (n_cell + k2)) ** 0.5)
        return p

    def same_size_share(self):
        """Share of this player's postflop bets that use their favourite size
        (plan: "a bot that always bets the same fraction gives you no
        information from size, so rely on its frequencies instead")."""
        counts = [self.n(BSZ0 + b) for b in range(N_SIZES)]
        tot = sum(counts)
        return max(counts) / tot if tot >= self.P["same_size_min_bets"] else 0.0

    def call_weak_share(self, prior):
        """Share of the hands they showed down after calling a postflop bet that
        were weak on that board, shrunk toward what the region model predicts."""
        return self.rate(CALL_WEAK, CALL_SHOWN, prior, self.P["k_rare"])

    def raise_vs_bet(self, street=None):
        prior = self.P["prior_raise_vs_bet_river"] if street == 3 else self.P["prior_raise_vs_bet"]
        return self.rate(RVB, RVB_OPP, prior, self.P["k_rare"])

    def bet_when_checked(self):
        return self.rate(BU, BU_OPP, self.P["prior_bet_when_checked"])

    def fold_to_raise(self, frac=None, pessimistic=False):
        """Folds when their bet is raised. Prior (plan): a balanced player folds
        about s / (1 + s), s = what they must add / the pot they would win."""
        if frac is None or not self.P.get("raise_fold_price_prior", True):
            prior = self.P["prior_fold_to_raise"]
        else:
            prior = frac / (1.0 + frac)
        if pessimistic:
            return self.low_bound(FTR, FTR_OPP, prior, self.P["k_rare"], self.P["bluff_conf_z"])
        return self.rate(FTR, FTR_OPP, prior, self.P["k_rare"])

    def cbet(self):
        return self.rate(CB, CB_OPP, self.P["prior_cbet"])

    def usual_bet_frac(self):
        """Their usual bet size as a fraction of the pot (plan: "their usual size,
        or the 75% translation point" until we have seen their bets)."""
        n = self.n(BETF_N)
        k = self.P["k_common"]
        return (self.n(BETF_SUM) / 100.0 + k * self.P["villain_bet_frac"]) / (n + k)

    def bet_bluff_share(self, street, frac):
        """Bluff share of their flop / turn bets, learned from showdowns, shrunk to
        the balanced s/(1+2s) (plan: "adjusted by what they've shown down")."""
        st = min(max(street, 1), 2)
        prior = frac / (1.0 + 2.0 * frac)
        return self.rate(B_WEAK0 + st - 1, B_SHOWN0 + st - 1, prior, self.P["k_rare"])

    def river_bluff_share(self, frac=0.66):
        """Bluffs in their river betting range. Prior: equilibrium s / (1 + 2s)."""
        # pooled over sizes first, then this size's own cell (plan: record the size
        # next to the hand they showed)
        k = self.P["k_rare"]
        pooled = self.rate(RB_BLUFF, RB_SHOWN, frac / (1.0 + 2.0 * frac), k)
        b = size_bucket(frac)
        return self.rate(RBS_BLUFF0 + b, RBS_SHOWN0 + b, pooled, k)

    def min_size_wide(self):
        """How far a minimum-size bet or raise from this player "tells nothing"
        (1 = not at all informative ... 0 = they only do it with strong hands).

        Plan: min-bets are cheap and often weak, so continue wide; watch whether
        this opponent only min-raises strong hands - then the size does tell us.
        With no information a shown hand is weak ``bluff_showdown_pct`` of the time."""
        thr = self.P["bluff_showdown_pct"]
        weak = self.rate(MZ_WEAK, MZ_SHOWN, thr)
        return float(min(max(weak / thr, 0.0), 1.0))

    def raise_bluff_share(self):
        """Share of their raises that were not strong hands when made (bluffs on
        the river, draws earlier). Prior starts low (~10%), updated from showdowns."""
        return self.rate(RZ_BLUFF, RZ_SHOWN, self.P["prior_raise_bluff_share"], self.P["k_rare"])
