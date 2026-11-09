"""Adapter: turn the SDK's ``state`` into a plain ``View`` the strategy uses.

This is the ONLY module that knows the SDK's object shapes. If the real kit
names something differently, fix it here (``_get`` already accepts
attributes or dict keys). Everything is recomputed from ``state.history``
where possible, because history semantics are documented precisely:

  blind/ante: chips posted     call: chips added
  raise: raiser's TOTAL bet for the betting round
  board: newly dealt cards     fold/check: no chips
"""
from __future__ import annotations

import zlib

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def card_int(c) -> int:
    s = str(c).strip()
    if len(s) == 3 and s[:2] == "10":
        s = "T" + s[2]
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[-1].lower())


def cards_int(cs):
    out = []
    for c in cs or ():
        try:
            out.append(card_int(c))
        except (ValueError, IndexError):
            pass
    return out


def _int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def blinds_of(b):
    """(small, big, ante) from whatever shape the SDK uses."""
    if b is None:
        return None
    for names in (("small", "big", "ante"), ("sb", "bb", "ante"),
                  ("small_blind", "big_blind", "ante")):
        sm, bg = _get(b, names[0]), _get(b, names[1])
        if sm is not None and bg is not None:
            return _int(sm), _int(bg), _int(_get(b, names[2], 0))
    try:
        seq = list(b)
        if len(seq) >= 2:
            return _int(seq[0]), _int(seq[1]), _int(seq[2]) if len(seq) > 2 else 0
    except TypeError:
        pass
    return None


class Ev:
    __slots__ = ("kind", "seat", "amount", "cards")

    def __init__(self, kind, seat, amount, cards):
        self.kind = kind
        self.seat = seat
        self.amount = amount
        self.cards = cards


def norm_event(e) -> Ev:
    seat = _get(e, "seat")
    return Ev(str(_get(e, "kind", "")), None if seat is None else _int(seat, -1),
              _int(_get(e, "amount", 0)), tuple(_get(e, "cards", ()) or ()))


class P:
    """One seat at the table, for the current hand."""
    __slots__ = ("seat", "stack", "bet", "status", "committed", "start", "folded",
                 "busted", "allin")

    def __init__(self, seat, stack, bet, status):
        self.seat = seat
        self.stack = stack
        self.bet = bet
        self.status = status
        self.committed = 0
        self.start = 0
        self.folded = status == "folded"
        self.busted = status == "busted"
        self.allin = status == "all_in" or (stack <= 0 and not self.busted and not self.folded)

    @property
    def in_hand(self):
        return not self.folded and not self.busted

    @property
    def can_act(self):
        return self.in_hand and not self.allin and self.stack > 0

    @property
    def avail(self):
        """Chips this player can still have in front of them this round (bet + behind)."""
        return self.bet + self.stack


class Act:
    __slots__ = ("seat", "kind", "total", "inc", "pot_before", "facing", "allin", "street")

    def __init__(self, seat, kind, total, inc, pot_before, facing, allin, street):
        self.seat = seat
        self.kind = kind          # fold / check / call / raise
        self.total = total        # round total after the action
        self.inc = inc            # chips added by this action
        self.pot_before = pot_before
        self.facing = facing      # chips this player faced before acting
        self.allin = allin
        self.street = street


class View:
    pass


def build_view(state) -> View:
    v = View()
    cfg = _get(state, "config")
    v.cfg = cfg
    v.seat = _int(_get(state, "seat", _get(cfg, "seat", 0)))
    v.hole = cards_int(_get(state, "cards", ()))
    v.board = cards_int(_get(state, "board", ()))
    v.phase = str(_get(state, "phase", "betting"))
    v.to_call = _int(_get(state, "to_call", 0))
    v.state_pot = _int(_get(state, "pot", 0))
    v.button = _int(_get(state, "button", -1), -1)
    v.hand_number = _int(_get(state, "hand_number", 0))
    v.hand_cap = _int(_get(cfg, "hand_cap", 200), 200)
    v.players_remaining = _int(_get(state, "players_remaining", 0))
    v.decision_ms = float(_get(cfg, "decision_ms", 1.0) or 1.0)
    v.time_bank_ms = float(_get(state, "time_bank_ms", _get(cfg, "time_bank_ms", 0.0)) or 0.0)
    v.memory_bytes = _int(_get(cfg, "memory_bytes", 65536), 65536)
    v.variant_id = str(_get(cfg, "variant", "") or "")
    v.table_id = str(_get(cfg, "table_id", "") or "")
    pay = _get(cfg, "payouts", None) or (50, 30, 20)
    try:
        v.payouts = [float(x) for x in pay]
    except TypeError:
        v.payouts = [50.0, 30.0, 20.0]

    # legal actions -> {kind: (min, max)}
    v.legal = {}
    for a in _get(state, "legal_actions", ()) or ():
        k = str(_get(a, "kind", ""))
        v.legal[k] = (_get(a, "min_amount"), _get(a, "max_amount"))
    v.legal_raw = list(_get(state, "legal_actions", ()) or ())

    # seats
    v.players = {}
    for s in _get(state, "seats", ()) or ():
        p = P(_int(_get(s, "seat", -1), -1), _int(_get(s, "stack", 0)), _int(_get(s, "bet", 0)),
              str(_get(s, "status", "active")))
        v.players[p.seat] = p
    v.hero = v.players.get(v.seat)

    # blinds
    bl = blinds_of(_get(state, "blinds"))
    hist = [norm_event(e) for e in (_get(state, "history", ()) or ())]
    v.history = hist
    blind_posts = [(e.seat, e.amount) for e in hist if e.kind == "blind"]
    if bl is None or bl[1] <= 0:
        bb = max((a for _, a in blind_posts), default=0)
        bl = (bb // 2, bb, 0)
    v.sb, v.bb, v.ante = bl
    if v.bb <= 0:
        v.bb = 1
    v.sb_seat = v.bb_seat = None
    if blind_posts:
        if len(blind_posts) >= 2:
            v.sb_seat, v.bb_seat = blind_posts[0][0], blind_posts[1][0]
        else:
            v.bb_seat = blind_posts[0][0]

    _replay_history(v)
    _order(v)
    return v


def _replay_history(v):
    """Rebuild committed chips and the action list of this hand from history."""
    street = 0
    round_bet = {}
    committed = {}
    acts = [[], [], [], []]
    high = 0
    pot = 0
    for e in v.history:
        k = e.kind
        s = e.seat
        if k in ("blind", "ante"):
            committed[s] = committed.get(s, 0) + e.amount
            pot += e.amount
            if k == "blind":
                round_bet[s] = round_bet.get(s, 0) + e.amount
                high = max(high, round_bet[s])
        elif k == "board":
            street = min(street + 1, 3)
            round_bet = {}
            high = 0
        elif k in ("fold", "check", "call", "raise"):
            cur = round_bet.get(s, 0)
            facing = max(0, high - cur)
            if k == "call":
                inc = e.amount
                total = cur + inc
            elif k == "raise":
                total = e.amount
                inc = max(0, total - cur)
            else:
                inc = 0
                total = cur
            p = v.players.get(s)
            allin = p is not None and p.stack == 0 and p.status == "all_in" and inc > 0
            acts[street].append(Act(s, k, total, inc, pot, facing, allin, street))
            round_bet[s] = total
            committed[s] = committed.get(s, 0) + inc
            pot += inc
            if total > high:
                high = total
    v.acts = acts
    for s, p in v.players.items():
        p.committed = committed.get(s, 0)
        p.start = p.stack + p.committed
    v.hist_pot = pot
    v.pot = max(pot, v.state_pot)
    # street from the board length (hold'em: 0, 3, 4, 5 cards)
    nb = len(v.board)
    v.street = 0 if nb == 0 else (1 if nb <= 3 else (2 if nb == 4 else 3))
    # mark all-in raises now that start stacks are known
    for st in acts:
        for a in st:
            p = v.players.get(a.seat)
            if p is not None and a.kind in ("raise", "call"):
                # all-in if this action put in everything the player started with
                a.allin = a.allin or (p.stack == 0 and p.committed > 0 and _last_action_of(acts, a.seat) is a)


def _last_action_of(acts, seat):
    last = None
    for st in acts:
        for a in st:
            if a.seat == seat:
                last = a
    return last


def _order(v):
    """Seat order for this hand (alive seats only), plus useful positions."""
    seats = sorted(s for s, p in v.players.items() if not p.busted)
    v.alive = seats
    v.n_alive = len(seats)
    if not seats:
        v.pre_order = v.post_order = []
        return
    def after(x):
        later = [s for s in seats if s > x]
        return later[0] if later else seats[0]
    btn = v.button if v.button in seats else (v.sb_seat if v.sb_seat in seats else seats[0])
    if v.bb_seat is None or v.bb_seat not in seats:
        if len(seats) == 2:
            v.sb_seat, v.bb_seat = btn, after(btn)
        else:
            v.sb_seat = after(btn)
            v.bb_seat = after(v.sb_seat)
    # preflop: first to act is the seat after the big blind
    start = after(v.bb_seat)
    order = [start]
    while len(order) < len(seats):
        order.append(after(order[-1]))
    v.pre_order = order
    # postflop: first alive seat after the button
    start = after(btn)
    order = [start]
    while len(order) < len(seats):
        order.append(after(order[-1]))
    v.post_order = order
    v.btn = btn


def players_behind(v, street_order):
    """Seats able to act that sit after hero in this street's order (no wrap)."""
    if v.seat not in street_order:
        return []
    i = street_order.index(v.seat)
    out = []
    for s in street_order[i + 1:]:
        p = v.players.get(s)
        if p is None or not p.can_act:
            continue
        out.append(s)
    return out


def yet_to_act(v):
    """Opponents who still have to act in this betting round after our action."""
    inhand = [p for p in v.players.values() if p.in_hand]
    high = max((p.bet for p in inhand), default=0)
    order = v.pre_order if v.street == 0 else v.post_order
    if v.seat not in order:
        return []
    i = order.index(v.seat)
    out = []
    for s in order[i + 1:] + order[:i]:
        p = v.players.get(s)
        if p is None or s == v.seat or not p.can_act:
            continue
        if p.bet < high or not acted_this_round(v, s):
            out.append(s)
    return out


def hero_in_position(v):
    """Hero acts last postflop among players still in the hand."""
    live = [s for s in v.post_order if v.players[s].in_hand and (v.players[s].can_act or s == v.seat)]
    return bool(live) and live[-1] == v.seat


def acted_this_round(v, seat):
    return any(a.seat == seat for a in v.acts[v.street])


def stable_seed(*parts) -> int:
    """Deterministic 31-bit seed (Python's hash() is salted per process)."""
    return zlib.crc32(repr(parts).encode()) & 0x7FFFFFFF


def time_ns():
    import time
    return time.perf_counter_ns()
