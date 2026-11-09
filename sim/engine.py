"""No-limit hold'em sit-and-go table engine.

Produces exactly the State / Event / SeatState / LegalAction / TableConfig /
BlindLevel objects of ``stub_sdk/poker_agent_sdk`` so a bot sees the same shapes
it will see in the competition.

Main entry points:
    play_table(agents, seed=0, **options) -> TableResult      # one full table
    Table(agents, seed=0, **options)                           # step hand by hand
        .play_hand() -> HandRecord | None,  .run() -> TableResult

Rules: see README-style notes in the module-level docstring of each method and
the task spec; the short version is standard NLHE with TDA-style incomplete
all-in raises, side pots, odd chips to the first winner left of the button,
blinds rising every ``hands_per_level`` hands and an ICM split at ``hand_cap``.
"""
from __future__ import annotations

import math
import os
import random
import sys
import traceback
from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Callable, List, Optional, Sequence

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STUB = os.path.join(_REPO, "stub_sdk")
if _STUB not in sys.path:
    sys.path.insert(0, _STUB)
if _REPO not in sys.path:
    sys.path.insert(1, _REPO)

from poker_agent_sdk import (  # noqa: E402
    BlindLevel, Event, LegalAction, SeatState, State, TableConfig,
)

from sim.evaluator import (CARD_STR, VARIANT_HOLE_CARDS, card_int, evaluate,  # noqa: E402
                           evaluate_omaha)

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"
ACTIVE, FOLDED, ALL_IN, BUSTED = "active", "folded", "all_in", "busted"
FAIL_REASONS = ("exception", "illegal", "timeout")

DEFAULT_BLINDS = tuple(BlindLevel(s, b, 0) for s, b in (
    (10, 20), (15, 30), (25, 50), (40, 80), (60, 120), (100, 200), (150, 300),
    (250, 500), (400, 800), (600, 1200), (1000, 2000)))
DEFAULT_PAYOUTS = (50, 30, 20)

_LA_FOLD = LegalAction(FOLD)
_LA_CHECK = LegalAction(CHECK)


# --------------------------------------------------------------------------- results

@dataclass
class SeatStats:
    name: str
    decisions: int = 0
    failures: dict = field(default_factory=lambda: {r: 0 for r in FAIL_REASONS})
    times_ns: list = field(default_factory=list)       # every act() duration, in order
    tracebacks: list = field(default_factory=list)     # first few exception tracebacks
    illegal_examples: list = field(default_factory=list)
    bank_ms: float = 0.0                               # bank left at the end of the table
    min_bank_ms: float = 0.0                           # lowest bank seen

    @property
    def failed(self) -> int:
        return sum(self.failures.values())


@dataclass
class HandRecord:
    hand_number: int
    button: int
    sb: int
    bb: int
    blinds: BlindLevel
    stacks_start: list
    hole_cards: dict          # seat -> tuple of H hole-card strings
    board: tuple
    events: list              # this hand's events, in order
    stacks_end: list
    busted: list              # seats busted this hand (worse place first)


@dataclass
class TableResult:
    seed: int
    table_id: str
    players: int
    names: list
    hands_played: int
    points: list              # per seat
    places: list              # per seat, 1 = best
    final_stacks: list
    bust_order: list          # seats, first bust first
    ended_by: str             # "winner" | "hand_cap"
    stats: list               # SeatStats per seat
    hands: Optional[list] = None   # HandRecord per hand when record_hands=True


# --------------------------------------------------------------------------- ICM

def icm(stacks: Sequence[float], payouts: Sequence[float]) -> list:
    """Malmuth-Harville expected payout per player (subset DP, exact)."""
    k = len(stacks)
    pay = [float(payouts[p]) if p < len(payouts) else 0.0 for p in range(k)]
    depth = 0
    for p in range(k):
        if pay[p] != 0.0:
            depth = p + 1
    ev = [0.0] * k
    if depth == 0:
        return ev
    total = float(sum(stacks))
    prob = {0: (1.0, total)}  # mask of already-placed players -> (probability, chips left)
    for place in range(depth):
        nxt = {}
        for mask, (p, rem) in prob.items():
            left = [j for j in range(k) if not (mask >> j) & 1]
            for j in left:
                pj = stacks[j] / rem if rem > 0 else 1.0 / len(left)
                if pj <= 0.0:
                    continue
                q = p * pj
                ev[j] += q * pay[place]
                m2 = mask | (1 << j)
                old = nxt.get(m2)
                nxt[m2] = (q + (old[0] if old else 0.0), rem - stacks[j])
        prob = nxt
    return ev


def _normalize_blinds(blinds) -> tuple:
    if blinds is None:
        return DEFAULT_BLINDS
    out = []
    for b in blinds:
        if isinstance(b, BlindLevel):
            out.append(b)
        else:
            t = tuple(b)
            out.append(BlindLevel(int(t[0]), int(t[1]), int(t[2]) if len(t) > 2 else 0))
    if not out:
        raise ValueError("empty blind schedule")
    return tuple(out)


# --------------------------------------------------------------------------- table

class Table:
    """One sit-and-go table.

    Options (all keyword):
      starting_stack, blinds (BlindLevel or (sb, bb[, ante]) list), hands_per_level,
      hand_cap, payouts, decision_ms, time_bank_ms, memory_bytes, table_id,
      strict_timing (no bank: any act() over decision_ms is a timeout),
      timing (False = never time out), record_hands, max_tracebacks, names.
    Test hooks:
      stacks  -- per-seat starting stacks instead of starting_stack
      button  -- first hand's button seat instead of a random one
      deals   -- list (index = hand_number - 1) or callable(hand_number) giving
                 {"hole": {seat: ("As", "Kd")}, "board": [...up to 5]} or None;
                 unspecified cards are dealt from the shuffled deck.
      check_invariants -- assert chip conservation after every hand (default True)
    Variants:
      variant -- "nlhe" (default), "plo4", "plo5", "plo6": hole cards per player
                 2/4/5/6; Omaha showdowns use exactly 2 hole + exactly 3 board cards
      betting -- "no_limit" or "pot_limit" (default: no_limit for nlhe, pot_limit for plo*).
                 Pot limit: max raise-to = high bet + pot (all chips in, current bets
                 included) + the amount to call, capped by the stack.
    """

    def __init__(self, agents: Sequence, seed: int = 0, *, starting_stack: int = 2000,
                 blinds=None, hands_per_level: int = 25, hand_cap: int = 200,
                 payouts: Sequence[float] = DEFAULT_PAYOUTS, decision_ms: float = 1.0,
                 time_bank_ms: float = 100.0, memory_bytes: int = 65536,
                 table_id: Optional[str] = None, strict_timing: bool = False,
                 timing: bool = True, record_hands: bool = False, max_tracebacks: int = 5,
                 names: Optional[Sequence[str]] = None, stacks: Optional[Sequence[int]] = None,
                 button: Optional[int] = None, deals=None, check_invariants: bool = True,
                 variant: str = "nlhe", betting: Optional[str] = None):
        n = len(agents)
        if not 2 <= n <= 10:
            raise ValueError("need 2..10 agents")
        if variant not in VARIANT_HOLE_CARDS:
            raise ValueError(f"unknown variant {variant!r}; use one of {sorted(VARIANT_HOLE_CARDS)}")
        self.variant = variant
        self.H = VARIANT_HOLE_CARDS[variant]
        self.omaha = variant.startswith("plo")
        if betting is None:
            betting = "pot_limit" if self.omaha else "no_limit"
        if betting not in ("no_limit", "pot_limit"):
            raise ValueError("betting must be 'no_limit' or 'pot_limit'")
        self.betting = betting
        self.pot_limit = betting == "pot_limit"
        if n * self.H + 5 > 52:
            raise ValueError(f"{n} players x {self.H} hole cards + 5 board cards exceeds the deck")
        self.n = n
        self.agents = list(agents)
        self.seed = seed
        self.rng = random.Random(seed)
        self.blind_schedule = _normalize_blinds(blinds)
        self.hands_per_level = int(hands_per_level)
        self.hand_cap = int(hand_cap)
        self.payouts = tuple(payouts)
        self.decision_ms = float(decision_ms)
        self.strict_timing = bool(strict_timing)
        self.timing = bool(timing)
        bank = 0.0 if self.strict_timing else float(time_bank_ms)
        self.memory_bytes = int(memory_bytes)
        self.table_id = table_id if table_id is not None else f"t{seed}"
        self.record_hands = record_hands
        self.max_tracebacks = max_tracebacks
        self.check_invariants = check_invariants
        self.deals = deals
        self._first_button = button
        self.names = list(names) if names is not None else [type(a).__name__ for a in agents]

        self.configs = [TableConfig(
            table_id=self.table_id, seat=i, variant=self.variant, players=n,
            starting_stack=int(starting_stack), blinds=self.blind_schedule,
            hands_per_level=self.hands_per_level, hand_cap=self.hand_cap,
            payouts=self.payouts, decision_ms=self.decision_ms, time_bank_ms=bank,
            memory_bytes=self.memory_bytes) for i in range(n)]

        self.stacks = [int(s) for s in stacks] if stacks is not None else [int(starting_stack)] * n
        if len(self.stacks) != n:
            raise ValueError("stacks must have one entry per seat")
        self.total_chips = sum(self.stacks)
        self.busted = [s <= 0 for s in self.stacks]
        self.places: List[Optional[int]] = [None] * n
        self.bust_order: List[int] = []
        self.bank = [bank] * n
        self.memory = [b""] * n
        self.cursor = [0] * n
        self.events: List[Event] = []
        self.stats = [SeatStats(name=self.names[i], bank_ms=bank, min_bank_ms=bank)
                      for i in range(n)]
        self.hand_number = 0
        self.button: Optional[int] = None
        self.finished = False
        self.ended_by = ""
        self.points: List[float] = [0.0] * n
        self.hands: Optional[list] = [] if record_hands else None
        # per-hand state
        self.status = [BUSTED if b else ACTIVE for b in self.busted]
        self.bets = [0] * n
        self.contrib = [0] * n
        self.hole: dict = {}
        self.hole_str: dict = {}
        self.board: list = []
        self.board_str: tuple = ()
        self.hand_start = 0
        self.level_idx = 0
        self.level = self.blind_schedule[0]
        self.alive_count = n
        self.hands_until_next = self.hands_per_level

    # ------------------------------------------------------------------ helpers
    def _next_alive(self, seat: int) -> int:
        n = self.n
        for k in range(1, n + 1):
            i = (seat + k) % n
            if not self.busted[i]:
                return i
        raise RuntimeError("no alive seat")

    def _emit(self, kind, seat=None, amount=None, cards=()):
        self.events.append(Event(kind, seat, amount, cards))

    def _post(self, seat: int, amount: int, kind: str, to_bet: bool):
        amt = min(amount, self.stacks[seat])
        if amt <= 0:
            return
        self.stacks[seat] -= amt
        self.contrib[seat] += amt
        if to_bet:
            self.bets[seat] += amt
        if self.stacks[seat] == 0:
            self.status[seat] = ALL_IN
        self._emit(kind, seat, amt)

    def _live_count(self) -> int:
        return sum(1 for s in self.status if s == ACTIVE or s == ALL_IN)

    # ------------------------------------------------------------------ hand
    def play_hand(self) -> Optional[HandRecord]:
        if self.finished:
            raise RuntimeError("table finished")
        n = self.n
        self.hand_number += 1
        hn = self.hand_number
        alive = [i for i in range(n) if not self.busted[i]]
        self.alive_count = len(alive)
        hpl = self.hands_per_level
        self.level_idx = min((hn - 1) // hpl, len(self.blind_schedule) - 1)
        self.level = lvl = self.blind_schedule[self.level_idx]
        self.hands_until_next = hpl - ((hn - 1) % hpl)

        if self.button is None:
            fb = self._first_button
            self.button = fb if (fb is not None and not self.busted[fb]) else self.rng.choice(alive)
        else:
            self.button = self._next_alive(self.button)
        btn = self.button
        if len(alive) == 2:
            sb = btn
        else:
            sb = self._next_alive(btn)
        bb = self._next_alive(sb)

        # ---- deal
        deck = list(range(52))
        self.rng.shuffle(deck)
        spec = None
        if self.deals is not None:
            spec = self.deals(hn) if callable(self.deals) else (
                self.deals[hn - 1] if hn - 1 < len(self.deals) else None)
        fixed_hole = {}
        fixed_board: list = []
        if spec:
            for seat, cs in (spec.get("hole") or {}).items():
                fixed_hole[int(seat)] = [card_int(c) for c in cs]
            fixed_board = [card_int(c) for c in (spec.get("board") or ())]
            used = set(fixed_board)
            for cs in fixed_hole.values():
                used.update(cs)
            deck = [c for c in deck if c not in used]
        order = [(btn + k) % n for k in range(1, n + 1)]  # left of button first
        self.hole = {}
        for i in order:
            if self.busted[i]:
                continue
            cs = fixed_hole.get(i)
            # hold'em: (deck.pop(), deck.pop()) exactly as before
            self.hole[i] = tuple(cs[:self.H]) if cs else tuple(deck.pop() for _ in range(self.H))
        board_cards = list(fixed_board[:5])
        while len(board_cards) < 5:
            board_cards.append(deck.pop())
        self.hole_str = {i: tuple(CARD_STR[c] for c in cs) for i, cs in self.hole.items()}

        # ---- reset per-hand state
        stacks_start = list(self.stacks)
        self.status = [BUSTED if self.busted[i] else ACTIVE for i in range(n)]
        self.bets = [0] * n
        self.contrib = [0] * n
        self.board = []
        self.board_str = ()
        self.hand_start = len(self.events)
        self._emit("hand", None, hn)
        if lvl.ante > 0:
            for i in order:
                if not self.busted[i]:
                    self._post(i, lvl.ante, "ante", to_bet=False)
        self._post(sb, lvl.small, "blind", to_bet=True)
        self._post(bb, lvl.big, "blind", to_bet=True)

        # ---- betting
        self._betting_round(True, bb)
        for k in (3, 1, 1):
            if self._live_count() <= 1:
                break
            new = board_cards[len(self.board):len(self.board) + k]
            self.board.extend(new)
            self.board_str = tuple(CARD_STR[c] for c in self.board)
            self._emit("board", None, None, tuple(CARD_STR[c] for c in new))
            self.bets = [0] * n
            self._betting_round(False, bb)

        self._showdown(order)

        for i in range(n):
            if self.stacks[i] > 0:
                self._emit("stack", i, self.stacks[i])

        if self.check_invariants:
            if sum(self.stacks) != self.total_chips or min(self.stacks) < 0:
                raise AssertionError(
                    f"chip conservation broken in hand {hn}: {self.stacks} total {self.total_chips}")

        # ---- busts and table end
        newly = [i for i in alive if self.stacks[i] == 0]
        if newly:
            tiebreak = {i: self.rng.random() for i in newly}
            newly.sort(key=lambda i: (stacks_start[i], tiebreak[i]))
            for idx, i in enumerate(newly):
                self.places[i] = self.alive_count - idx
                self.busted[i] = True
            self.bust_order.extend(newly)
        remaining = [i for i in range(n) if not self.busted[i]]
        if len(remaining) == 1:
            self.places[remaining[0]] = 1
            self._finish("winner")
        elif hn >= self.hand_cap:
            self._finish("hand_cap")

        rec = None
        if self.record_hands:
            rec = HandRecord(
                hand_number=hn, button=btn, sb=sb, bb=bb, blinds=lvl,
                stacks_start=stacks_start, hole_cards=dict(self.hole_str),
                board=self.board_str, events=self.events[self.hand_start:],
                stacks_end=list(self.stacks), busted=list(newly))
            self.hands.append(rec)
        return rec

    # ------------------------------------------------------------------ betting
    def _betting_round(self, preflop: bool, bb: int):
        n = self.n
        status = self.status
        stacks = self.stacks
        if preflop:
            high = max(self.bets)
            ptr = (bb + 1) % n
        else:
            high = 0
            ptr = (self.button + 1) % n
        min_inc = self.level.big
        acted_at: List[Optional[int]] = [None] * n
        while True:
            bets = self.bets
            live = 0
            active = []
            for i in range(n):
                st = status[i]
                if st == ACTIVE:
                    active.append(i)
                    live += 1
                elif st == ALL_IN:
                    live += 1
            if live <= 1 or not active:
                return
            if len(active) == 1 and bets[active[0]] >= high:
                return
            seat = -1
            for k in range(n):
                i = (ptr + k) % n
                if status[i] == ACTIVE and (acted_at[i] is None or bets[i] < high):
                    seat = i
                    break
            if seat < 0:
                return

            stack = stacks[seat]
            bet = bets[seat]
            raw_to_call = high - bet
            call_amt = min(raw_to_call, stack)
            if raw_to_call <= 0:
                legal = [_LA_CHECK]
            else:
                legal = [_LA_FOLD, LegalAction(CALL, call_amt, call_amt)]
            reopened = acted_at[seat] is None or (high - acted_at[seat]) >= min_inc
            others = False
            for i in active:
                if i != seat:
                    others = True
                    break
            if stack > raw_to_call and reopened and others:
                max_to = bet + stack
                if self.pot_limit:
                    # call, then raise the size of the pot after the call
                    pot_cap = high + sum(self.contrib) + raw_to_call
                    if pot_cap < max_to:
                        max_to = pot_cap
                min_to = high + min_inc
                if max_to < min_to:
                    min_to = max_to
                legal.append(LegalAction(RAISE, min_to, max_to))

            kind, amount = self._decide(seat, tuple(legal), max(call_amt, 0))

            if kind == FOLD:
                status[seat] = FOLDED
                self._emit(FOLD, seat)
            elif kind == CHECK:
                acted_at[seat] = high
                self._emit(CHECK, seat)
            elif kind == CALL:
                stacks[seat] -= call_amt
                bets[seat] += call_amt
                self.contrib[seat] += call_amt
                if stacks[seat] == 0:
                    status[seat] = ALL_IN
                acted_at[seat] = high
                self._emit(CALL, seat, call_amt)
            else:  # RAISE to `amount` (validated)
                add = amount - bet
                stacks[seat] -= add
                bets[seat] = amount
                self.contrib[seat] += add
                if stacks[seat] == 0:
                    status[seat] = ALL_IN
                inc = amount - high
                if inc >= min_inc:
                    min_inc = inc
                high = amount
                acted_at[seat] = amount
                self._emit(RAISE, seat, amount)
            ptr = (seat + 1) % n

    # ------------------------------------------------------------------ decisions
    def _build_state(self, seat: int, legal: tuple, to_call: int) -> State:
        n = self.n
        stacks, bets, status = self.stacks, self.bets, self.status
        seats = tuple(SeatState(i, stacks[i], bets[i], status[i]) for i in range(n))
        ev = self.events
        return State(
            config=self.configs[seat], seat=seat, cards=self.hole_str[seat],
            board=self.board_str, pot=sum(self.contrib), to_call=to_call,
            legal_actions=legal, history=tuple(ev[self.hand_start:]), phase="betting",
            seats=seats, button=self.button, blind_level=self.level_idx, blinds=self.level,
            hands_until_next_level=self.hands_until_next, hand_number=self.hand_number,
            players_remaining=self.alive_count, bust_order=tuple(self.bust_order),
            events=tuple(ev[self.cursor[seat]:]), time_bank_ms=self.bank[seat])

    @staticmethod
    def _validate(action, legal: tuple):
        """(kind, amount) if legal else None. Floats floor; out-of-range is illegal."""
        kind = getattr(action, "kind", None)
        la = None
        for a in legal:
            if a.kind == kind:
                la = a
                break
        if la is None:
            return None
        if kind != RAISE:
            return kind, None
        amt = getattr(action, "amount", None)
        if isinstance(amt, bool):
            return None
        if isinstance(amt, int):
            v = amt
        elif isinstance(amt, float):
            if not math.isfinite(amt):
                return None
            v = int(math.floor(amt))
        else:
            return None  # includes numpy integer types: be stricter than needed
        if v < la.min_amount or v > la.max_amount:
            return None
        return RAISE, v

    def _decide(self, seat: int, legal: tuple, to_call: int):
        ev_index = len(self.events)
        state = self._build_state(seat, legal, to_call)
        st = self.stats[seat]
        st.decisions += 1
        mem = self.memory[seat]
        err = None
        ret = None
        act = self.agents[seat].act
        t0 = perf_counter_ns()
        try:
            ret = act(state, mem)
            elapsed = perf_counter_ns() - t0
        except Exception as e:  # timestamp first: formatting the traceback is not the bot's time
            elapsed = perf_counter_ns() - t0
            err = e
        st.times_ns.append(elapsed)

        action = new_mem = None
        if err is None:
            try:
                action, new_mem = ret
                if not isinstance(new_mem, bytes):
                    raise TypeError(f"memory must be bytes, got {type(new_mem).__name__}")
            except Exception as e:
                err = e

        timed_out = False
        if self.timing:
            ms = elapsed / 1e6
            if self.strict_timing:
                timed_out = ms > self.decision_ms
            else:
                excess = ms - self.decision_ms
                if excess > 0:
                    self.bank[seat] -= excess
                    if self.bank[seat] < 0:
                        timed_out = True
                        self.bank[seat] = 0.0
            st.bank_ms = self.bank[seat]
            if self.bank[seat] < st.min_bank_ms:
                st.min_bank_ms = self.bank[seat]

        fallback = (CHECK, None) if any(a.kind == CHECK for a in legal) else (FOLD, None)
        if err is not None:
            st.failures["exception"] += 1
            if len(st.tracebacks) < self.max_tracebacks:
                tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
                st.tracebacks.append(f"hand {self.hand_number} seat {seat}:\n{tb}")
            return fallback
        if timed_out:
            st.failures["timeout"] += 1
            return fallback
        # successful call (legal or not): keep memory, advance the event cursor
        self.memory[seat] = new_mem[:self.memory_bytes]
        self.cursor[seat] = ev_index
        res = self._validate(action, legal)
        if res is None:
            st.failures["illegal"] += 1
            if len(st.illegal_examples) < self.max_tracebacks:
                st.illegal_examples.append(
                    f"hand {self.hand_number} seat {seat}: returned "
                    f"{getattr(action, 'kind', action)!r} {getattr(action, 'amount', None)!r}; "
                    f"legal={[(a.kind, a.min_amount, a.max_amount) for a in legal]}")
            return fallback
        return res

    # ------------------------------------------------------------------ showdown
    def _build_pots(self, contenders: list) -> list:
        contrib = self.contrib
        levels = sorted(set(contrib[i] for i in contenders))
        pots = []
        prev = 0
        for lv in levels:
            amount = 0
            for c in contrib:
                if c > prev:
                    amount += (c if c < lv else lv) - prev
            if amount > 0:
                pots.append([amount, [i for i in contenders if contrib[i] >= lv]])
            prev = lv
        leftover = sum(c - prev for c in contrib if c > prev)
        if leftover:  # cannot happen (the top contributor never folds); keep chips anyway
            if pots:
                pots[-1][0] += leftover
            else:
                pots.append([leftover, list(contenders)])
        return pots

    def _showdown(self, order: list):
        status = self.status
        contenders = [i for i in order if status[i] == ACTIVE or status[i] == ALL_IN]
        ranks = {}
        if len(contenders) >= 2:
            board = self.board
            for i in contenders:
                self._emit("show", i, None, self.hole_str[i])
                ranks[i] = (evaluate_omaha(self.hole[i], board) if self.omaha
                            else evaluate(list(self.hole[i]) + board))
        pos = {seat: k for k, seat in enumerate(order)}
        for amount, elig in self._build_pots(contenders):
            if len(elig) == 1:
                winners = elig
            else:
                best = max(ranks[i] for i in elig)
                winners = [i for i in elig if ranks[i] == best]
            winners.sort(key=pos.__getitem__)
            share, rem = divmod(amount, len(winners))
            for k, w in enumerate(winners):
                a = share + (1 if k < rem else 0)
                if a:
                    self.stacks[w] += a
                    self._emit("win", w, a)

    # ------------------------------------------------------------------ end
    def _finish(self, how: str):
        self.finished = True
        self.ended_by = how
        n = self.n
        pay = self.payouts
        remaining = [i for i in range(n) if not self.busted[i]]
        if how == "hand_cap" and len(remaining) > 1:
            evs = icm([self.stacks[i] for i in remaining], pay)
            for i, e in zip(remaining, evs):
                self.points[i] = e
            ranked = sorted(remaining, key=lambda i: (-self.stacks[i], i))
            for p, i in enumerate(ranked, start=1):
                self.places[i] = p
        for i in range(n):
            if how != "hand_cap" or self.busted[i] or len(remaining) == 1:
                p = self.places[i]
                self.points[i] = float(pay[p - 1]) if p is not None and p - 1 < len(pay) else 0.0

    def run(self) -> TableResult:
        while not self.finished:
            self.play_hand()
        return self.result()

    def result(self) -> TableResult:
        return TableResult(
            seed=self.seed, table_id=self.table_id, players=self.n, names=list(self.names),
            hands_played=self.hand_number, points=list(self.points), places=list(self.places),
            final_stacks=list(self.stacks), bust_order=list(self.bust_order),
            ended_by=self.ended_by, stats=self.stats, hands=self.hands)


def play_table(agents: Sequence, seed: int = 0, **options) -> TableResult:
    """Play one full table (until one player has every chip or ``hand_cap`` hands).

    ``agents``: one bot instance per seat (seat i = agents[i]); pass fresh instances.
    ``options``: any keyword accepted by ``Table`` (starting_stack, blinds,
    hands_per_level, hand_cap, payouts, decision_ms, time_bank_ms, memory_bytes,
    table_id, strict_timing, timing, record_hands, max_tracebacks, names, ...).
    """
    return Table(agents, seed, **options).run()


def percentiles_ms(times_ns: Sequence[int], qs=(50, 90, 99, 99.9)) -> dict:
    """Summary of decision times in milliseconds (nearest-rank percentiles)."""
    if not times_ns:
        return {"n": 0}
    xs = sorted(times_ns)
    out = {"n": len(xs)}
    for q in qs:
        k = min(len(xs) - 1, max(0, int(math.ceil(q / 100.0 * len(xs))) - 1))
        out[f"p{q:g}"] = xs[k] / 1e6
    out["max"] = xs[-1] / 1e6
    out["mean"] = sum(xs) / len(xs) / 1e6
    return out


__all__ = [
    "Table", "TableResult", "SeatStats", "HandRecord", "play_table", "icm",
    "percentiles_ms", "DEFAULT_BLINDS", "DEFAULT_PAYOUTS",
]
