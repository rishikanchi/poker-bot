"""Decision pipeline for one ``act`` call, with safety at every step.

  memory bytes -> Memory (never raises; bad bytes -> fresh)
  state.events -> pending hand events; each finished hand updates opponent stats
  state        -> View (view.py is the only SDK-shape-aware code)
  View         -> preflop.decide or postflop.decide -> (kind, amount)
  (kind, amount) -> legal SDK Action (actions.py)
  Memory       -> bytes (kept under the table's memory limit)

If anything fails, we still return a legal action and the best memory we
have. The time budget comes from ``state.config.decision_ms``.
"""
from __future__ import annotations

import random
import time

import numpy as np

from . import actions as A
from . import auction as Au
from . import trained as Tr
from . import game as G
from . import icmtools as I
from . import postflop as Po
from . import preflop as Pf
from . import ranges as Rg
from . import settings
from . import stats as St
from . import view as V
from .memory import Memory

P = settings.P
GAME = None
DEBUG = bool(__import__("os").environ.get("PB_DEBUG")) or __import__("os").path.exists(
    __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__file__)), "PB_DEBUG"))
# local testing only: env PB_DEBUG=1, or an empty file named PB_DEBUG next to agent.py (never pack it)  # local testing: surface strategy errors
COST = {"us_per_sample": 0.25, "us_per_opp": 0.12, "scores_us": 90.0}
DRAW_KEY = 10   # memory cache key offset for the draw flags of a street


def init():
    """Build the variant plug-in, compile kernels, calibrate timing, and play a
    few practice decisions so no code path is cold on the first real hand."""
    global GAME
    GAME = G.warmup()
    _calibrate(GAME)
    if any(P.get(k) for k in ("use_trained_preflop", "use_trained_postflop", "use_trained_auction")):
        try:
            if Tr.load():
                Tr.warm(GAME)
        except Exception:  # noqa: BLE001 - without tables the hooks return None
            Tr.T = None
    try:
        warm_decisions()
    except Exception:  # noqa: BLE001 - warmup must never stop the import
        pass
    return GAME


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeAction:
    def __init__(self, kind, amount=None):
        self.kind, self.amount = kind, amount

    @classmethod
    def fold(cls):
        return cls("fold")

    @classmethod
    def check(cls):
        return cls("check")

    @classmethod
    def call(cls):
        return cls("call")

    @classmethod
    def raise_to(cls, t):
        return cls("raise", t)


def warm_decisions(rounds=3):
    """Run synthetic preflop/flop/turn/river decisions through ``decide``."""
    g = GAME
    sdk = _NS(Action=_FakeAction)
    rng = np.random.default_rng(3)
    names = "23456789TJQKA"
    suits = "cdhs"
    def cs(c):
        return names[c >> 2] + suits[c & 3]
    blob = b""
    for r in range(rounds):
        cards = [int(x) for x in rng.choice(g.deck, g.H * 3 + 5, replace=False)]
        hole = [cs(c) for c in cards[: g.H]]
        board = [cs(c) for c in cards[g.H * 3: g.H * 3 + 5]]
        ev = lambda k, s, a=0, c=(): _NS(kind=k, seat=s, amount=a, cards=tuple(c))  # noqa: E731
        hist = [ev("hand", None, 5 + r), ev("blind", 0, 10), ev("blind", 1, 20), ev("raise", 2, 50),
                ev("call", 3, 50), ev("fold", 4), ev("fold", 5), ev("fold", 0), ev("call", 1, 30)]
        for nb, extra in ((0, []), (3, [ev("board", None, 0, board[:3]), ev("check", 1), ev("raise", 2, 60)]),
                          (4, [ev("call", 3, 60), ev("call", 1, 60), ev("board", None, 0, board[3:4]),
                               ev("check", 1)]),
                          (5, [ev("raise", 2, 150), ev("call", 3, 150), ev("call", 1, 150),
                               ev("board", None, 0, board[4:5]), ev("check", 1), ev("raise", 2, 300)])):
            hist = hist + extra
            facing = 50 if nb == 0 else (60 if nb == 3 else (0 if nb == 4 else 300))
            seats = [_NS(seat=i, stack=1500, bet=0, status="active") for i in range(6)]
            for i in (0, 4, 5):
                seats[i].status = "folded"
            seats[2].bet = facing
            legal = ([_NS(kind="fold", min_amount=None, max_amount=None),
                      _NS(kind="call", min_amount=facing, max_amount=facing),
                      _NS(kind="raise", min_amount=2 * facing, max_amount=1500)] if facing else
                     [_NS(kind="check", min_amount=None, max_amount=None),
                      _NS(kind="raise", min_amount=20, max_amount=1500)])
            cfg = _NS(table_id="warm", seat=3, variant="", players=6, starting_stack=2000,
                      hand_cap=200, payouts=(50, 30, 20), decision_ms=1.0, time_bank_ms=100.0,
                      memory_bytes=65536)
            state = _NS(config=cfg, seat=3, cards=tuple(hole), board=tuple(board[:nb]), pot=400,
                        to_call=facing, legal_actions=legal, history=tuple(hist), phase="betting",
                        seats=seats, button=5, blinds=(10, 20, 0), hand_number=5 + r,
                        players_remaining=6, events=tuple(hist), time_bank_ms=100.0)
            _, blob = decide(state, blob, sdk)


def _calibrate(g):
    rng = np.random.default_rng(0)
    cards = [int(x) for x in rng.choice(g.deck, g.H + 4, replace=False)]
    hole, board = cards[: g.H], cards[g.H:g.H + 3]
    t = time.perf_counter_ns()
    for _ in range(5):
        g.scores(board, g.dead_mask(board))
    COST["scores_us"] = (time.perf_counter_ns() - t) / 5e3
    w = g.range_interval(0, 0.5, g.dead_mask(hole + board))
    n = 200
    best1 = best2 = 1e18
    for _ in range(5):
        t = time.perf_counter_ns()
        g.equity(hole, board, [w], n, 1)
        best1 = min(best1, time.perf_counter_ns() - t)
        t = time.perf_counter_ns()
        g.equity(hole, board, [w, w], n, 1)
        best2 = min(best2, time.perf_counter_ns() - t)
    COST["us_per_sample"] = best1 / 1e3 / n
    COST["us_per_opp"] = max(0.0, (best2 - best1) / 1e3 / n)


def _now():
    return time.perf_counter_ns()


def ingest(mem, events, hero_seat, game, starting_stack, P=None):
    """Buffer the hand in progress; replay each finished hand into the stats."""
    for e in events:
        ev = V.norm_event(e)
        cards = tuple(V.cards_int(ev.cards))
        item = (ev.kind, ev.seat, ev.amount, cards)
        if ev.kind == "hand":
            if mem.pending:
                try:
                    St.replay_hand(mem.pending, mem, hero_seat, game, starting_stack, P)
                except Exception:  # noqa: BLE001 - stats are best effort
                    pass
            mem.pending = [item]
        else:
            mem.pending.append(item)


def decide(state, memory, sdk, t0=None, params=None):
    """``params`` overrides settings.P (local tuning only; production uses the default)."""
    Pm = P if params is None else params
    t0 = t0 or _now()
    mem = Memory.decode(memory)
    try:
        v = V.build_view(state)
    except Exception:  # noqa: BLE001
        return _fallback(sdk, state), memory
    budget_ns = v.decision_ms * 1e6 * Pm["budget_frac"]
    if v.time_bank_ms < Pm["low_bank_ms"]:
        budget_ns *= 0.5  # bank running low at this table: play the cheap version
    deadline = t0 + budget_ns
    starting_stack = V._int(V._get(v.cfg, "starting_stack", 0)) or max(
        (p.start for p in v.players.values()), default=1000)
    try:
        mem.ensure_seats(sorted(v.players), starting_stack)
        ingest(mem, V._get(state, "events", ()) or (), v.seat, GAME, starting_stack, Pm)
    except Exception:  # noqa: BLE001
        mem = Memory()
    kind, amount = None, None
    try:
        kind, amount = _strategy(v, mem, deadline, Pm, t0)
    except Exception:  # noqa: BLE001
        if DEBUG:
            raise
        kind, amount = None, None
    try:
        action = A.to_action(sdk, v, kind, amount) if kind else A.automatic(sdk, v)
    except Exception:  # noqa: BLE001
        action = _fallback(sdk, state)
    try:
        blob = mem.encode(v.memory_bytes)
    except Exception:  # noqa: BLE001
        blob = memory if isinstance(memory, (bytes, bytearray)) else b""
    return action, bytes(blob)


def _strategy(v, mem, deadline, P, t0=None):
    g = GAME
    if g is not None:
        # Boston: the auction cards left off the flop are dead for this hand
        mask = 0
        for x in getattr(v, "dead_cards", ()):
            mask |= 1 << int(x)
        g.extra_dead = mask
    auction_phase = v.phase == "auction" and "bid" in v.legal
    if not auction_phase and (v.phase != "betting" or not set(v.legal) <= {"fold", "check", "call", "raise"}):
        return None, None  # unknown phase or action kind: automatic choice
    if g is None or len(v.hole) != g.H or v.hero is None:
        return (None, None) if auction_phase else _safe_mode(v)
    opps = {s: St.Opp(mem.stats[mem.slot(s)] if mem.slot(s) >= 0 else None, P) for s in v.players}
    rng = random.Random(V.stable_seed(v.table_id, v.hand_number, len(v.history)))
    c = Pf.Ctx(v, g, opps, P, rng)
    if auction_phase:
        bids = Tr.auction(c, P) if P.get("use_trained_auction") else None
        return "bid", (bids if bids is not None else Au.plan(c, P))
    if v.street == 0 or not v.board:
        got = Tr.preflop(c) if P.get("use_trained_preflop") else None
        return got if got is not None else Pf.decide(c)
    # postflop: strength of every hand on each street so far. Only the board is
    # removed here (ranges are built as the opponents play them); our own cards
    # are removed afterwards, which is how blockers enter. Cached per hand.
    if mem.cache_hand != v.hand_number:
        mem.cache_hand = v.hand_number
        mem.cache = {}
    boards = {1: v.board[:3], 2: v.board[:4], 3: v.board[:5]}
    scores = {}
    draws = {}
    c.river_vals = None
    final = len(v.board) >= g.board_total
    for st in range(1, v.street + 1):
        b = boards.get(st, v.board)
        if final and st == v.street:
            # river: one strength pass gives the ordering and exact equity
            vals = g.strengths(b, g.dead_mask(b))
            c.river_vals = vals
            scores[st] = _pct_from_values(vals)
            draws[st] = None
            continue
        sc, dr = mem.cache.get(st), mem.cache.get(st + DRAW_KEY)
        if sc is not None and dr is not None and sc.size == g.M and dr.size == g.M:
            scores[st], draws[st] = sc, dr
        else:
            scores[st], draws[st] = g.scores(b, g.dead_mask(b))
            mem.cache[st], mem.cache[st + DRAW_KEY] = scores[st], draws[st]
    ranges, orders = Rg.build_ranges(v, g, opps, P, scores, draws, g.dead_mask(v.board))
    if P.get("use_trained_postflop"):
        got = Tr.postflop(c, ranges, scores[v.street])
        if got is not None:
            return got
    c.orders = orders
    c.scores_by_street = scores
    c.draws_by_street = draws
    c.deadline = deadline
    c.hard_deadline = (t0 or _now()) + v.decision_ms * 1e6 * P["lean_at"]   # past this, play the lean version
    if P.get("mc_fixed_samples"):
        c.hard_deadline = float("inf")       # reproducible runs (tuning, A/B): never the lean version
    c.icm = I.IcmCtx(v)
    # sample count from the time left
    left_us = (deadline - _now()) / 1e3 - 60.0
    n_opp = max(1, len([s for s in ranges if v.players[s].in_hand]))
    per = COST["us_per_sample"] + COST["us_per_opp"] * (n_opp - 1)
    n = int(left_us * 0.75 / max(per, 1e-3))
    lo = P["mc_samples_min"] if g.exact else P["mc_samples_min_sampled"]
    c.n_samples = int(min(max(n, lo), P["mc_samples_max"]))
    if P.get("mc_fixed_samples"):
        c.n_samples = int(P["mc_fixed_samples"])  # reproducible runs (tuning)
    return Po.decide(c, ranges, scores[v.street])


def _pct_from_values(vals):
    """Percentile (ties averaged) of each live hand's value; -1 for dead hands."""
    out = np.full(vals.shape[0], -1.0)
    live = vals >= 0
    lv = vals[live]
    if lv.size == 0:
        return out
    _, inv, counts = np.unique(lv, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    avg = start + (counts - 1) / 2.0
    out[live] = avg[inv] / max(lv.size - 1, 1)
    return out


def _safe_mode(v):
    """Variant we have no plug-in for: play legally and cautiously."""
    if "check" in v.legal:
        return "check", None
    if "call" in v.legal and v.to_call <= max(v.bb, 0.05 * (v.pot or 1)):
        return "call", None
    return "fold", None


def _fallback(sdk, state):
    try:
        if state.can("check"):
            return sdk.Action.check()
        if state.can("fold"):
            return sdk.Action.fold()
        first = state.legal_actions[0]
        return sdk.Action(first.kind, first.min_amount)
    except Exception:  # noqa: BLE001
        return sdk.Action.fold()
