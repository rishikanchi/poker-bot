"""Local stand-in for the competition's ``poker_agent_sdk``.

It mirrors the interface described on the No Limit '26 "Getting started" page so
the bot and the local simulator can run tonight. It is NOT uploaded: the real
kit ships its own ``poker_agent_sdk``. When the kit arrives, delete this folder
from your PYTHONPATH and run the bot against the real package; the only file in
the bot that depends on these shapes is ``bot/pb/view.py``.

Documented facts this stub follows:
  * ``Agent.act(state, memory) -> (Action, bytes)``
  * ``Action.fold() / .check() / .call() / .raise_to(total)`` and ``Action(kind, amount)``
  * ``state.can(kind)`` -> legal action with ``min_amount`` / ``max_amount`` or None
  * events: kind, seat (None for hand/board), amount, cards
  * seats: seat, stack, bet, status (active/folded/all_in/busted), shown_cards
Guesses (unknown until the kit arrives): the exact type of ``blinds`` and of the
``config.blinds`` schedule, and whether ``state.pot`` includes current bets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

FOLD = "fold"
CHECK = "check"
CALL = "call"
RAISE = "raise"


@dataclass(frozen=True)
class LegalAction:
    kind: str
    min_amount: Optional[int] = None
    max_amount: Optional[int] = None


class Action:
    __slots__ = ("kind", "amount")

    def __init__(self, kind: str, amount: Optional[int] = None):
        self.kind = kind
        self.amount = amount

    @classmethod
    def fold(cls) -> "Action":
        return cls(FOLD)

    @classmethod
    def check(cls) -> "Action":
        return cls(CHECK)

    @classmethod
    def call(cls) -> "Action":
        return cls(CALL)

    @classmethod
    def raise_to(cls, total: int) -> "Action":
        return cls(RAISE, total)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Action({self.kind!r}, {self.amount!r})"


@dataclass(frozen=True)
class Event:
    kind: str
    seat: Optional[int] = None
    amount: Optional[int] = None
    cards: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SeatState:
    seat: int
    stack: int
    bet: int
    status: str
    shown_cards: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BlindLevel:
    small: int
    big: int
    ante: int = 0


@dataclass(frozen=True)
class TableConfig:
    table_id: str
    seat: int
    variant: str
    players: int
    starting_stack: int
    blinds: Tuple[BlindLevel, ...]
    hands_per_level: int
    hand_cap: int
    payouts: Tuple[float, ...]
    decision_ms: float = 1.0
    time_bank_ms: float = 100.0
    memory_bytes: int = 65536


@dataclass
class State:
    config: TableConfig
    seat: int
    cards: Tuple[str, ...]
    board: Tuple[str, ...]
    pot: int  # stub: every chip committed this hand, current-round bets included
    to_call: int
    legal_actions: Tuple[LegalAction, ...]
    history: Tuple[Event, ...]  # this hand only
    phase: str
    seats: Tuple[SeatState, ...]
    button: int
    blind_level: int
    blinds: BlindLevel
    hands_until_next_level: int
    hand_number: int
    players_remaining: int
    bust_order: Tuple[int, ...]
    events: Tuple[Event, ...]  # everything public since this seat's last turn
    time_bank_ms: float = 100.0
    variant: object = None

    def can(self, kind: str) -> Optional[LegalAction]:
        for a in self.legal_actions:
            if a.kind == kind:
                return a
        return None


class Agent:
    """Base class. Subclasses implement ``act``."""

    def act(self, state: State, memory: bytes):  # pragma: no cover - interface
        raise NotImplementedError


__all__ = [
    "Action", "Agent", "BlindLevel", "Event", "LegalAction", "SeatState",
    "State", "TableConfig", "FOLD", "CHECK", "CALL", "RAISE",
]
