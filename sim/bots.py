"""Simple opponents for the local simulator. Every bot only returns offered actions.

Each takes an optional ``seed`` so tables are reproducible. ``BUILTIN_BOTS`` maps
CLI names to classes (see sim/run.py).
"""
from __future__ import annotations

import math
import os
import random
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STUB = os.path.join(_REPO, "stub_sdk")
if _STUB not in sys.path:
    sys.path.insert(0, _STUB)
if _REPO not in sys.path:
    sys.path.insert(1, _REPO)

from poker_agent_sdk import Action, Agent  # noqa: E402

from sim.evaluator import RANKS, STRAIGHT, cards_int, category, evaluate, evaluate_omaha  # noqa: E402

_RANK_IDX = {r: i for i, r in enumerate(RANKS)}


# --------------------------------------------------------------------------- helpers

def high_bet(state) -> int:
    return max(s.bet for s in state.seats)


def my_bet(state) -> int:
    return state.seats[state.seat].bet


def check_or_fold(state) -> Action:
    return Action.check() if state.can("check") else Action.fold()


def call_or_check(state) -> Action:
    return Action.call() if state.can("call") else Action.check()


def clamp_raise(la, total) -> int:
    return max(la.min_amount, min(la.max_amount, int(total)))


def pot_raise_to(state, frac: float = 1.0) -> int:
    """Raise-to total for a raise of ``frac`` times the pot after calling."""
    hb = high_bet(state)
    owed = hb - my_bet(state)
    return int(hb + frac * (state.pot + owed))


def chen_score(c1: str, c2: str) -> int:
    """Bill Chen's preflop formula (rounded up)."""
    r1, r2 = _RANK_IDX[c1[0].upper()], _RANK_IDX[c2[0].upper()]
    hi, lo = max(r1, r2), min(r1, r2)
    pts = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(hi, (hi + 2) / 2.0)
    if r1 == r2:
        return int(math.ceil(max(pts * 2, 5)))
    score = pts
    if c1[1].lower() == c2[1].lower():
        score += 2
    gap = hi - lo - 1
    score -= (0, 1, 2, 4)[gap] if gap < 4 else 5
    if gap <= 1 and hi < 10:  # both below a queen
        score += 1
    return int(math.ceil(score))


def _chen_threshold(frac: float) -> int:
    """Smallest Chen score T so that hands with score >= T are about ``frac`` of all combos."""
    cards = [r + s for r in RANKS for s in "cdhs"]
    scores = []
    for i in range(52):
        for j in range(i + 1, 52):
            scores.append(chen_score(cards[i], cards[j]))
    total = len(scores)
    best_t, best_err = None, None
    for t in sorted(set(scores)):
        f = sum(1 for s in scores if s >= t) / total
        err = abs(f - frac)
        if best_err is None or err < best_err:
            best_t, best_err = t, err
    return best_t


NIT_CHEN = _chen_threshold(0.10)      # ~top 10% of hands
JAMMER_CHEN = _chen_threshold(0.30)   # ~top 30% of hands


def hand_score(cards) -> int:
    """Chen score for 2 cards; for Omaha-style hands, the best 2-card Chen score
    among the hole cards (+2 if the hand holds two suited pairs of cards)."""
    if len(cards) == 2:
        return chen_score(*cards)
    best = max(chen_score(a, b) for i, a in enumerate(cards) for b in cards[i + 1:])
    suits = [c[1].lower() for c in cards]
    if sum(1 for s in set(suits) if suits.count(s) >= 2) >= 2:
        best += 2
    return best


_THRESH_CACHE = {}


def score_threshold(frac: float, n_cards: int) -> int:
    """Score cutoff for about the top ``frac`` of hands with ``n_cards`` hole cards."""
    if n_cards == 2:
        return NIT_CHEN if frac == 0.10 else (JAMMER_CHEN if frac == 0.30 else _chen_threshold(frac))
    key = (frac, n_cards)
    if key not in _THRESH_CACHE:
        rng = random.Random(12345)
        deck = [r + s for r in RANKS for s in "cdhs"]
        scores = sorted((hand_score(rng.sample(deck, n_cards)) for _ in range(4000)), reverse=True)
        _THRESH_CACHE[key] = scores[max(0, int(frac * len(scores)) - 1)]
    return _THRESH_CACHE[key]


def has_pair_or_better(hole, board) -> bool:
    """Pair or better that uses a hole card (pocket pair, paired a board card, or straight+)."""
    hr = [c[0] for c in hole]
    if len(hole) != 2:  # Omaha-style: approximate with the exactly-2-hole-cards rule
        if len(set(hr)) < len(hr):
            return True
        br = {c[0] for c in board}
        if any(r in br for r in hr):
            return True
        if len(board) >= 3:
            return category(evaluate_omaha(cards_int(list(hole)), cards_int(list(board)))) >= STRAIGHT
        return False
    if hr[0] == hr[1]:
        return True
    br = {c[0] for c in board}
    if hr[0] in br or hr[1] in br:
        return True
    if len(board) >= 3:
        return category(evaluate(cards_int(list(hole) + list(board)))) >= STRAIGHT
    return False


class _Bot(Agent):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)


# --------------------------------------------------------------------------- bots

class CheckOrFold(_Bot):
    """Check when free, otherwise fold."""

    def act(self, state, memory):
        return check_or_fold(state), memory


class MinRaiser(_Bot):
    """Raise to the minimum whenever possible, else call/check."""

    def act(self, state, memory):
        la = state.can("raise")
        if la is not None:
            return Action.raise_to(la.min_amount), memory
        return call_or_check(state), memory


class CallingStation(_Bot):
    """Call or check every time; never raise or fold."""

    def act(self, state, memory):
        return call_or_check(state), memory


class Maniac(_Bot):
    """70% of decisions raise about the size of the pot, otherwise call/check."""

    def __init__(self, seed=None, aggression: float = 0.7):
        super().__init__(seed)
        self.aggression = aggression

    def act(self, state, memory):
        la = state.can("raise")
        if la is not None and self.rng.random() < self.aggression:
            return Action.raise_to(clamp_raise(la, pot_raise_to(state, 1.0))), memory
        return call_or_check(state), memory


class Nit(_Bot):
    """Preflop: raise about the top 10% (Chen score), fold the rest.
    Postflop: bet/raise ~2/3 pot with pair or better, check/fold without."""

    def act(self, state, memory):
        la = state.can("raise")
        if not state.board:
            if hand_score(state.cards) >= score_threshold(0.10, len(state.cards)):
                if la is not None:
                    hb = high_bet(state)
                    big = state.blinds.big
                    target = 3 * big if hb <= big else 3 * hb
                    return Action.raise_to(clamp_raise(la, target)), memory
                return call_or_check(state), memory
            return check_or_fold(state), memory
        if has_pair_or_better(state.cards, state.board):
            if la is not None:
                return Action.raise_to(clamp_raise(la, pot_raise_to(state, 0.66))), memory
            return call_or_check(state), memory
        return check_or_fold(state), memory


class Jammer(_Bot):
    """Preflop: all-in with about the top 30% (Chen score), else check/fold.
    Postflop: check/fold."""

    def act(self, state, memory):
        if not state.board and hand_score(state.cards) >= score_threshold(0.30, len(state.cards)):
            la = state.can("raise")
            if la is not None:
                return Action.raise_to(la.max_amount), memory
            return call_or_check(state), memory
        return check_or_fold(state), memory


class RandomLegal(_Bot):
    """Uniformly random offered action; raise amount uniform in [min, max]."""

    def act(self, state, memory):
        la = self.rng.choice(state.legal_actions)
        if la.kind == "raise":
            return Action.raise_to(self.rng.randint(la.min_amount, la.max_amount)), memory
        return Action(la.kind), memory


class Scripted(Agent):
    """Test helper: plays ``script`` step by step, then ``then`` ("checkfold" or "call").

    A step is an Action, a kind string ("fold", "check", "call", "min" = min raise,
    "allin" = raise to max else call), ("raise", total), or a callable
    ``(state, memory) -> (action, memory)`` (may raise, return bad memory, ...).
    Every received (state, memory) is recorded in ``self.states`` / ``self.memories``.
    """

    def __init__(self, script=(), then: str = "checkfold", seed=None):
        self.script = list(script)
        self.then = then
        self.states = []
        self.memories = []

    def act(self, state, memory):
        self.states.append(state)
        self.memories.append(memory)
        if self.script:
            step = self.script.pop(0)
            if callable(step):
                return step(state, memory)
            return self._resolve(step, state), memory
        if self.then == "call":
            return call_or_check(state), memory
        return check_or_fold(state), memory

    @staticmethod
    def _resolve(step, state):
        if isinstance(step, Action):
            return step
        if isinstance(step, tuple):
            return Action(step[0], step[1] if len(step) > 1 else None)
        if step == "min":
            la = state.can("raise")
            return Action.raise_to(la.min_amount) if la else call_or_check(state)
        if step == "allin":
            la = state.can("raise")
            if la is not None:
                return Action.raise_to(la.max_amount)
            return call_or_check(state)
        return Action(step)


BUILTIN_BOTS = {
    "checkfold": CheckOrFold, "check_or_fold": CheckOrFold, "checkorfold": CheckOrFold,
    "minraiser": MinRaiser, "minraise": MinRaiser,
    "station": CallingStation, "callingstation": CallingStation, "calling_station": CallingStation,
    "maniac": Maniac,
    "nit": Nit,
    "jammer": Jammer,
    "random": RandomLegal, "randomlegal": RandomLegal, "random_legal": RandomLegal,
}

__all__ = [
    "CheckOrFold", "MinRaiser", "CallingStation", "Maniac", "Nit", "Jammer", "RandomLegal",
    "Scripted", "BUILTIN_BOTS", "chen_score", "NIT_CHEN", "JAMMER_CHEN",
]
