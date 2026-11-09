"""Cards and a pure-Python 5/6/7-card hold'em hand evaluator.

Card ints: ``rank * 4 + suit`` with rank 0..12 = 2..A and suit 0..3 = c, d, h, s
(the same encoding as bot/pb/view.py). Strings look like "As", "Td", "2c".

``evaluate(cards)`` returns an int; bigger is better, equal means a tie. Layout:
``category << 20 | r1 << 16 | r2 << 12 | r3 << 8 | r4 << 4 | r5`` where the r's are
the ranks that matter for that category, most significant first.
"""
from __future__ import annotations

RANKS = "23456789TJQKA"
SUITS = "cdhs"

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = ("high card", "pair", "two pair", "trips", "straight", "flush",
                  "full house", "quads", "straight flush")

CARD_STR = tuple(RANKS[c >> 2] + SUITS[c & 3] for c in range(52))
_CARD_INT = {s: i for i, s in enumerate(CARD_STR)}


def card_int(s: str) -> int:
    """'As' -> 51. Accepts lowercase ranks and '10' for ten."""
    s = s.strip()
    if len(s) == 3 and s[:2] == "10":
        s = "T" + s[2]
    try:
        return _CARD_INT[s[0].upper() + s[1].lower()]
    except (KeyError, IndexError):
        raise ValueError(f"bad card {s!r}") from None


def cards_int(cs) -> list:
    return [card_int(c) for c in cs]


def card_str(c: int) -> str:
    return CARD_STR[c]


# STRAIGHT_HIGH[mask13] = rank of the straight's top card, or -1. Wheel (A2345) -> 3.
_WHEEL = (1 << 12) | 0b1111
STRAIGHT_HIGH = [-1] * 8192
for _m in range(8192):
    for _hi in range(12, 3, -1):
        if (_m >> (_hi - 4)) & 0x1F == 0x1F:
            STRAIGHT_HIGH[_m] = _hi
            break
    else:
        if _m & _WHEEL == _WHEEL:
            STRAIGHT_HIGH[_m] = 3
del _m, _hi


def _pack(cat: int, ranks) -> int:
    v = cat
    for i in range(5):
        v = (v << 4) | (ranks[i] if i < len(ranks) else 0)
    return v


def evaluate(cards) -> int:
    """Strength of the best 5-card hand in ``cards`` (5 to 7 card ints)."""
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    rmask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suits[c & 3] |= 1 << r
        rmask |= 1 << r
    # With at most 7 cards a flush excludes quads and full houses, so check it first.
    for sm in suits:
        if sm.bit_count() >= 5:
            hi = STRAIGHT_HIGH[sm]
            if hi >= 0:
                return _pack(STRAIGHT_FLUSH, (hi,))
            top = []
            for r in range(12, -1, -1):
                if (sm >> r) & 1:
                    top.append(r)
                    if len(top) == 5:
                        break
            return _pack(FLUSH, top)
    quads = []
    trips = []
    pairs = []
    singles = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 1:
            singles.append(r)
        elif n == 2:
            pairs.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 4:
            quads.append(r)
    if quads:
        q = quads[0]
        kick = -1
        for r in range(12, -1, -1):
            if r != q and counts[r]:
                kick = r
                break
        return _pack(QUADS, (q, max(kick, 0)))
    if trips:
        t = trips[0]
        p = trips[1] if len(trips) > 1 else -1
        if pairs and pairs[0] > p:
            p = pairs[0]
        if p >= 0:
            return _pack(FULL_HOUSE, (t, p))
    hi = STRAIGHT_HIGH[rmask]
    if hi >= 0:
        return _pack(STRAIGHT, (hi,))
    if trips:
        return _pack(TRIPS, [trips[0]] + singles[:2])
    if len(pairs) >= 2:
        kick = singles[0] if singles else -1
        if len(pairs) > 2 and pairs[2] > kick:
            kick = pairs[2]
        return _pack(TWO_PAIR, (pairs[0], pairs[1], max(kick, 0)))
    if pairs:
        return _pack(PAIR, [pairs[0]] + singles[:3])
    return _pack(HIGH_CARD, singles[:5])


def evaluate_str(cards) -> int:
    return evaluate(cards_int(cards))


def category(score: int) -> int:
    return score >> 20


def describe(score: int) -> str:
    return CATEGORY_NAMES[score >> 20]


# --------------------------------------------------------------------------- variants

#: hole cards per player for each supported variant id
VARIANT_HOLE_CARDS = {"nlhe": 2, "plo4": 4, "plo5": 5, "plo6": 6}
#: matching plug-in name for our bot (bot/pb/settings.py reads PB_VARIANT)
VARIANT_PB_NAME = {"nlhe": "holdem", "plo4": "omaha4", "plo5": "omaha5", "plo6": "omaha6"}


def evaluate_omaha(hole, board) -> int:
    """Omaha showdown value: best hand using EXACTLY 2 hole cards and EXACTLY 3 board cards."""
    from itertools import combinations
    best = -1
    board = list(board)
    for h in combinations(hole, 2):
        for b in combinations(board, 3):
            v = evaluate(list(h) + list(b))
            if v > best:
                best = v
    return best


def evaluate_variant(hole, board, omaha: bool) -> int:
    return evaluate_omaha(hole, board) if omaha else evaluate(list(hole) + list(board))
