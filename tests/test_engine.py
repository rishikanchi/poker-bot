"""Tests for the local simulator (sim/). Run: python3 -m pytest tests/test_engine.py -q"""
from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter
from itertools import combinations

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "stub_sdk"))
sys.path.insert(0, _REPO)

from poker_agent_sdk import Action, Agent, Event, LegalAction  # noqa: E402

from sim.bots import (  # noqa: E402
    BUILTIN_BOTS, CallingStation, CheckOrFold, RandomLegal, Scripted, check_or_fold,
)
from sim.engine import Table, icm, play_table  # noqa: E402
from sim.evaluator import (  # noqa: E402
    FLUSH, FULL_HOUSE, HIGH_CARD, PAIR, QUADS, STRAIGHT, STRAIGHT_FLUSH, TRIPS, TWO_PAIR,
    card_int, category, evaluate, evaluate_str,
)


def ev(s: str) -> int:
    return evaluate_str(s.split())


# =========================================================================== evaluator

def test_categories():
    assert category(ev("As Ks Qs Js Ts 2c 3d")) == STRAIGHT_FLUSH
    assert category(ev("9c 9d 9h 9s Ac 2d 3h")) == QUADS
    assert category(ev("Kc Kd Kh 2s 2c 7d 8h")) == FULL_HOUSE
    assert category(ev("2h 7h 9h Jh Kh As Ad")) == FLUSH
    assert category(ev("5c 6d 7h 8s 9c Kd Kh")) == STRAIGHT
    assert category(ev("Qc Qd Qh 2s 7c 9d 4h")) == TRIPS
    assert category(ev("Qc Qd 7h 7s 2c 9d 4h")) == TWO_PAIR
    assert category(ev("Qc Qd 7h 6s 2c 9d 4h")) == PAIR
    assert category(ev("Qc Jd 7h 6s 2c 9d 4h")) == HIGH_CARD


def test_category_order():
    hands = ["Qc Jd 7h 6s 2c 9d 4h", "Qc Qd 7h 6s 2c 9d 4h", "Qc Qd 7h 7s 2c 9d 4h",
             "Qc Qd Qh 2s 7c 9d 4h", "5c 6d 7h 8s 9c Kd Kh", "2h 7h 9h Jh Kh As Ad",
             "Kc Kd Kh 2s 2c 7d 8h", "9c 9d 9h 9s Ac 2d 3h", "9s Ks Qs Js Ts 2c 3d",
             "As Ks Qs Js Ts 2c 3d"]
    vals = [ev(h) for h in hands]
    assert vals == sorted(vals) and len(set(vals)) == len(vals)


def test_wheel():
    wheel = ev("Ac 2d 3h 4s 5c Kd Qh")
    assert category(wheel) == STRAIGHT
    assert ev("2c 3d 4h 5s 6c Kd Qh") > wheel                 # six-high straight beats the wheel
    assert ev("Tc Jd Qh Ks Ac 2d 3h") > ev("2c 3d 4h 5s 6c Kd Qh")
    assert wheel > ev("Ac Ad Ah 2s 3c 7d 8h")                 # any straight beats trips
    steel = ev("Ah 2h 3h 4h 5h Kc Kd")
    assert category(steel) == STRAIGHT_FLUSH
    assert ev("2h 3h 4h 5h 6h Kc Kd") > steel
    assert category(ev("Ac 2d 3h 4s Kc 9d 8h")) == HIGH_CARD  # A234 + K is not a straight
    assert category(ev("Jc Qd Kh As 2c 3d 4h")) == HIGH_CARD  # no wrap-around straight


def test_kickers():
    board = "Ah Kd 7c 4s 2h "
    assert ev(board + "Qc 3d") > ev(board + "Jc 3d")          # AK Q vs AK J
    board = "Kh Kd 7c 7s 2h "
    assert ev(board + "Ac 3d") > ev(board + "Qc 3d")          # two pair, ace kicker
    assert ev(board + "3c 3d") == ev(board + "3h 3s")         # KK77 with a 3 kicker each: tie
    # three pairs: best two pairs plus best remaining card
    assert ev("Ac Ad Kc Kd Qc Qd 2h") > ev("Ac Ad Kc Kd Jc Jd 9h")
    assert ev("Ac Ad Kc Kd Qc 2d 3h") == ev("Ah As Kh Ks Qd 4d 5h")
    # board plays: exact tie
    assert ev("Ah Kh Qh Jh 9c 2c 3d") == ev("Ah Kh Qh Jh 9c 2d 4s")
    # quads kicker
    assert ev("9c 9d 9h 9s Ac 2d 3h") > ev("9c 9d 9h 9s Kc 2d 3h")
    # flush compares all five cards
    assert ev("Ah 9h 7h 5h 3h Kc Qd") > ev("Ah 9h 7h 5h 2h Kc Qd")
    # pair kickers down to the third
    assert ev("8c 8d Ah Kd 5c 3s 2h") > ev("8c 8d Ah Kd 4c 3s 2h")
    assert ev("8c 8d Ah Kd 5c 4s 2h") == ev("8h 8s Ad Kc 5h 3d 2c")  # only the top 3 kickers count


def test_flush_vs_straight():
    board = "4h 5h 6h 7c 9d "
    straight = ev(board + "8c Tc")
    flush = ev(board + "Ah 2h")
    assert category(straight) == STRAIGHT and category(flush) == FLUSH
    assert flush > straight
    assert category(ev("4h 5h 6h 7c 8d Kh 2h")) == FLUSH     # both present -> flush
    assert category(ev("4h 5h 6h 7h 8h Kh 2h")) == STRAIGHT_FLUSH


def test_full_house_over_trips():
    assert ev("Kc Kd Kh 2s 2c 7d 8h") > ev("Ac Ad Ah 2s 3c 7d 8h")
    assert category(ev("Kc Kd Kh 2s 2c 2d 8h")) == FULL_HOUSE  # two sets make a boat
    assert ev("Kc Kd Kh 2s 2c 2d 8h") > ev("Qc Qd Qh As Ac 7d 8h")  # trips rank decides first
    assert ev("Kc Kd Kh As Ac 7d 8h") > ev("Kc Kd Kh Qs Qc 7d 8h")
    assert ev("Kc Kd Kh As Ac 2d 2h") == ev("Kc Kd Kh As Ac 7d 8h")  # best pair only


def _naive5(cards):
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    flush = len({c & 3 for c in cards}) == 1
    uniq = sorted(set(ranks), reverse=True)
    sh = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            sh = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            sh = 3
    groups = sorted(Counter(ranks).items(), key=lambda kv: (-kv[1], -kv[0]))
    shape = [g[1] for g in groups]
    ordered = [g[0] for g in groups]
    if sh is not None and flush:
        return (8, sh)
    if shape == [4, 1]:
        return (7, *ordered)
    if shape == [3, 2]:
        return (6, *ordered)
    if flush:
        return (5, *ranks)
    if sh is not None:
        return (4, sh)
    if shape == [3, 1, 1]:
        return (3, *ordered)
    if shape == [2, 2, 1]:
        return (2, *ordered)
    if shape == [2, 1, 1, 1]:
        return (1, *ordered)
    return (0, *ranks)


def test_matches_bruteforce_best_of_21():
    rng = random.Random(7)
    hands = []
    for _ in range(2500):
        hands.append(rng.sample(range(52), 7))
    # bias toward made hands: a few hundred with forced pairs / flush draws
    for _ in range(500):
        r = rng.randrange(13)
        s = rng.randrange(4)
        base = {r * 4 + s, r * 4 + (s + 1) % 4}
        suited = [x * 4 + s for x in rng.sample(range(13), 5)]
        pool = list(base | set(suited))
        while len(pool) < 7:
            c = rng.randrange(52)
            if c not in pool:
                pool.append(c)
        hands.append(pool[:7])
    for h in hands:
        n7 = max(_naive5(c) for c in combinations(h, 5))
        assert category(evaluate(h)) == n7[0], h
    scored = [(max(_naive5(c) for c in combinations(h, 5)), evaluate(h)) for h in hands]
    scored.sort()
    for (na, ea), (nb, eb) in zip(scored, scored[1:]):
        if na == nb:
            assert ea == eb
        else:
            assert ea < eb
    # 5- and 6-card inputs too
    for _ in range(500):
        for k in (5, 6):
            h = rng.sample(range(52), k)
            assert category(evaluate(h)) == max(_naive5(c) for c in combinations(h, 5))[0]


# =========================================================================== helpers

class Recorder(Agent):
    """Logs (seat, street, legal kinds) to a shared list, then delegates to a script."""

    def __init__(self, log, script=(), then="call"):
        self.log = log
        self.inner = Scripted(script, then=then)

    def act(self, state, memory):
        self.log.append((state.seat, len(state.board), tuple(a.kind for a in state.legal_actions)))
        return self.inner.act(state, memory)


def events_of(table, kinds):
    return [(e.kind, e.seat, e.amount) for e in table.events if e.kind in kinds]


# =========================================================================== pots

def test_side_pots_three_all_ins():
    # button 0, SB 1, BB 2, UTG 3. Shorter stacks hold better hands.
    stacks = [1000, 100, 300, 600]
    deals = [{"hole": {0: ("7h", "2d"), 1: ("As", "Ad"), 2: ("Ks", "Kd"), 3: ("Qs", "Qd")},
              "board": ["2c", "5d", "9h", "Js", "3c"]}]
    agents = [Scripted(["allin"]), Scripted(["call"]), Scripted(["call"]), Scripted(["allin"])]
    t = Table(agents, seed=1, stacks=stacks, button=0, deals=deals, hand_cap=1)
    t.play_hand()
    assert [e for e in events_of(t, {"raise"})] == [("raise", 3, 600), ("raise", 0, 1000)]
    assert events_of(t, {"call"}) == [("call", 1, 90), ("call", 2, 280)]
    assert events_of(t, {"win"}) == [("win", 1, 400), ("win", 2, 600), ("win", 3, 600), ("win", 0, 400)]
    assert [e.seat for e in t.events if e.kind == "show"] == [1, 2, 3, 0]
    assert t.stacks == [400, 400, 600, 600]
    assert [e.cards for e in t.events if e.kind == "board"] == [("2c", "5d", "9h"), ("Js",), ("3c",)]
    # seat 1 (SB) facing the shove: may only fold or call its last 90 chips
    st = agents[1].states[0]
    assert st.legal_actions == (LegalAction("fold"), LegalAction("call", 90, 90))
    assert st.to_call == 90 and st.pot == 10 + 20 + 600 + 1000


def test_multi_bust_places_by_starting_stack():
    stacks = [1000, 100, 300, 600]
    deals = [{"hole": {0: ("As", "Ad"), 1: ("7h", "2d"), 2: ("Ks", "Kd"), 3: ("Qs", "Qd")},
              "board": ["2c", "5d", "9h", "Js", "3c"]}]
    agents = [Scripted(["allin"]), Scripted(["call"]), Scripted(["call"]), Scripted(["allin"])]
    t = Table(agents, seed=1, stacks=stacks, button=0, deals=deals)
    t.play_hand()
    assert t.finished and t.ended_by == "winner"
    res = t.result()
    assert res.places == [1, 4, 3, 2]
    assert res.bust_order == [1, 2, 3]
    assert res.points == [50.0, 0.0, 20.0, 30.0]
    assert res.final_stacks == [2000, 0, 0, 0]
    # only non-busted seats get a stack event
    assert events_of(t, {"stack"}) == [("stack", 0, 2000)]


def test_split_pot_odd_chip_left_of_button():
    # blinds 5/10, button 0 limps, SB folds (5 dead), BB checks; board is a royal flush
    deals = [{"hole": {0: ("2c", "3d"), 1: ("4c", "5d"), 2: ("2d", "3c")},
              "board": ["As", "Ks", "Qs", "Js", "Ts"]}]
    agents = [Scripted(["call"]), Scripted(["fold"]), Scripted(["check"])]
    t = Table(agents, seed=3, stacks=[1000, 1000, 1000], button=0, deals=deals,
              blinds=[(5, 10)], hand_cap=1)
    t.play_hand()
    assert events_of(t, {"win"}) == [("win", 2, 13), ("win", 0, 12)]
    assert t.stacks == [1002, 995, 1003]
    assert [e.seat for e in t.events if e.kind == "show"] == [2, 0]


def test_uncontested_win_shows_nothing():
    agents = [Scripted([("raise", 60)]), Scripted(["fold"]), Scripted(["fold"])]
    t = Table(agents, seed=3, button=0, hand_cap=1)
    t.play_hand()
    assert not [e for e in t.events if e.kind == "show"]
    assert events_of(t, {"win"}) == [("win", 0, 90)]
    assert not [e for e in t.events if e.kind == "board"]


# =========================================================================== betting rules

def test_min_raise_and_legal_action_shapes():
    log = []
    agents = [Recorder(log, ["fold"]), Recorder(log, ["fold"]), Recorder(log, ["call"]),
              Recorder(log, [("raise", 60)]), Recorder(log, [("raise", 150)]),
              Recorder(log, ["call"])]
    t = Table(agents, seed=5, button=0, hand_cap=1)
    t.play_hand()
    s3 = agents[3].inner.states[0]
    assert s3.legal_actions == (LegalAction("fold"), LegalAction("call", 20, 20),
                                LegalAction("raise", 40, 2000))
    assert s3.to_call == 20 and s3.pot == 30
    s4 = agents[4].inner.states[0]
    assert s4.can("raise").min_amount == 100          # 60 + last full raise (40)
    s5 = agents[5].inner.states[0]
    assert s5.can("raise").min_amount == 240          # 150 + 90
    assert s5.can("call").min_amount == 150
    # history/events start with this hand's hand event; pot includes current bets
    assert s5.history[0] == Event("hand", None, 1)
    assert s5.pot == 10 + 20 + 60 + 150


def test_big_blind_option_has_no_fold():
    agents = [Scripted(["call"]), Scripted(["call"]), Scripted(["check"])]
    t = Table(agents, seed=5, button=0, hand_cap=1)
    t.play_hand()
    bb_state = agents[2].states[0]
    assert bb_state.to_call == 0
    assert bb_state.legal_actions == (LegalAction("check"), LegalAction("raise", 40, 2000))


def test_incomplete_all_in_raise_does_not_reopen():
    # 3-handed: button 0 opens to 100, SB shoves 150 total (+50 < 80), BB calls.
    agents = [Scripted([("raise", 100), "call"]), Scripted(["allin"]), Scripted(["call"])]
    t = Table(agents, seed=2, stacks=[2000, 150, 2000], button=0, hand_cap=1)
    t.play_hand()
    sb_state = agents[1].states[0]
    assert sb_state.can("raise") == LegalAction("raise", 150, 150)  # all-in for less than a min raise
    bb_state = agents[2].states[0]
    assert bb_state.can("raise").min_amount == 230    # not yet acted: may raise, 150 + 80
    btn_again = agents[0].states[1]
    assert tuple(a.kind for a in btn_again.legal_actions) == ("fold", "call")
    assert btn_again.can("call").min_amount == 50


def test_full_all_in_raise_reopens():
    agents = [Scripted([("raise", 100), "call"]), Scripted(["allin"]), Scripted(["call"])]
    t = Table(agents, seed=2, stacks=[2000, 200, 2000], button=0, hand_cap=1)
    t.play_hand()
    assert agents[2].states[0].can("raise").min_amount == 300    # 200 + 100
    btn_again = agents[0].states[1]
    assert btn_again.can("raise") == LegalAction("raise", 300, 2000)


def test_no_raise_when_everyone_else_is_all_in():
    agents = [Scripted(["allin"]), Scripted(["fold"]), Scripted(["call"])]
    t = Table(agents, seed=2, stacks=[500, 2000, 3000], button=0, hand_cap=1)
    t.play_hand()
    bb = agents[2].states[0]
    assert tuple(a.kind for a in bb.legal_actions) == ("fold", "call")


def test_heads_up_action_order():
    log = []
    agents = [Recorder(log, ["call"], then="call"), Recorder(log, ["check"], then="call")]
    t = Table(agents, seed=4, button=0, hand_cap=1)
    t.play_hand()
    assert events_of(t, {"blind"}) == [("blind", 0, 10), ("blind", 1, 20)]  # button posts SB
    order = [(s, b) for s, b, _ in log]
    assert order == [(0, 0), (1, 0), (1, 3), (0, 3), (1, 4), (0, 4), (1, 5), (0, 5)]


def test_three_handed_action_order():
    log = []
    agents = [Recorder(log), Recorder(log), Recorder(log)]   # everyone calls / checks
    t = Table(agents, seed=4, button=0, hand_cap=1)
    t.play_hand()
    order = [(s, b) for s, b, _ in log]
    # preflop: left of BB (= button 0), SB 1, BB 2; postflop starts left of button
    assert order[:3] == [(0, 0), (1, 0), (2, 0)]
    assert order[3:6] == [(1, 3), (2, 3), (0, 3)]


def test_short_all_in_blinds():
    # SB has 5 chips, BB has 15: both all-in posting; button to act
    agents = [Scripted(["call"]), Scripted([]), Scripted([])]
    t = Table(agents, seed=9, stacks=[2000, 5, 15], button=0, hand_cap=1)
    t.play_hand()
    assert events_of(t, {"blind"}) == [("blind", 1, 5), ("blind", 2, 15)]
    st = agents[0].states[0]
    assert st.to_call == 15 and st.can("raise") is None   # nobody left who could call a raise
    assert len(agents[0].states) == 1                      # no decisions once all-in runout
    assert sum(t.stacks) == 2020


# =========================================================================== failures and memory

def test_illegal_raise_amount_folds_when_facing_bet():
    agents = [Scripted([lambda s, m: (Action.raise_to(39), b"kept")]),
              Scripted(["call"]), Scripted(["check"])]
    t = Table(agents, seed=6, button=0, hand_cap=1)
    t.play_hand()
    st = t.stats[0]
    assert st.failures == {"exception": 0, "illegal": 1, "timeout": 0}
    assert ("fold", 0, None) in events_of(t, {"fold"})
    assert t.memory[0] == b"kept"         # illegal still stores memory


def test_illegal_actions_check_when_free_and_float_floor():
    # BB option: raise above max is illegal -> check. Button: 40.9 floors to a legal 40.
    agents = [Scripted([("raise", 40.9)]), Scripted(["call"]),
              Scripted([("raise", 10 ** 9), "call"])]
    t = Table(agents, seed=6, button=0, hand_cap=1)
    t.play_hand()
    assert t.stats[0].failures["illegal"] == 0
    assert ("raise", 0, 40) in events_of(t, {"raise"})
    # BB faced to_call 20 after the raise, so its huge raise is illegal -> fold
    assert t.stats[2].failures["illegal"] == 1
    assert ("fold", 2, None) in events_of(t, {"fold"})

    agents = [Scripted(["call"]), Scripted(["call"]), Scripted([("raise", 10 ** 9)])]
    t = Table(agents, seed=6, button=0, hand_cap=1)
    t.play_hand()
    assert t.stats[2].failures["illegal"] == 1
    assert ("check", 2, None) in events_of(t, {"check"})


def test_unoffered_kind_and_bad_types_are_illegal():
    import numpy as np
    bad = [Action.check(),                      # check while facing the big blind
           Action.raise_to(np.int64(40)),       # numpy int (rejected: stricter than needed)
           Action.raise_to(None), Action("bet", 40), None, Action.raise_to(float("nan"))]
    for a in bad:
        agents = [Scripted([lambda s, m, a=a: (a, m)]), Scripted(["call"]), Scripted(["check"])]
        t = Table(agents, seed=6, button=0, hand_cap=1)
        t.play_hand()
        assert t.stats[0].failures == {"exception": 0, "illegal": 1, "timeout": 0}, a
        assert ("fold", 0, None) in events_of(t, {"fold"})
    # a bare tuple instead of (action, memory) is an exception, not an illegal action
    agents = [Scripted([lambda s, m: Action.fold()]), Scripted(["call"]), Scripted(["check"])]
    t = Table(agents, seed=6, button=0, hand_cap=1)
    t.play_hand()
    assert t.stats[0].failures["exception"] == 1


class MemProbe(Agent):
    """Check/folds; 2nd call raises, 4th returns non-bytes memory."""

    def __init__(self):
        self.calls = []

    def act(self, state, memory):
        n = len(self.calls) + 1
        self.calls.append((state.events, memory, state.hand_number))
        if n == 2:
            raise RuntimeError("boom")
        if n == 4:
            return check_or_fold(state), "not bytes"
        return check_or_fold(state), memory + bytes([n])


def test_event_cursor_and_memory_semantics():
    probe = MemProbe()
    t = Table([probe, CallingStation(), CallingStation()], seed=11, hand_cap=30, timing=False)
    t.run()
    calls = probe.calls
    assert len(calls) >= 6
    # first turn: everything since the table started
    assert calls[0][0][0] == Event("hand", None, 1)
    assert calls[0][1] == b""
    assert calls[1][1] == b"\x01"
    # exception on call 2: memory kept, and the same events are delivered again
    assert calls[2][1] == b"\x01"
    assert calls[2][0][:len(calls[1][0])] == calls[1][0]
    # call 4 returned non-bytes memory -> counted as exception, memory unchanged
    assert calls[4][1] == b"\x01\x03"
    assert calls[4][0][:len(calls[3][0])] == calls[3][0]
    assert t.stats[0].failures["exception"] == 2
    assert len(t.stats[0].tracebacks) == 2 and "boom" in t.stats[0].tracebacks[0]
    # concatenating events of successful turns reproduces the table log with no gaps
    ok = [c[0] for i, c in enumerate(calls) if i not in (1, 3)]
    flat = [e for evs in ok for e in evs]
    assert flat == t.events[:len(flat)]


def test_memory_truncated_to_memory_bytes():
    seen = []

    def step(state, memory):
        seen.append(memory)
        return check_or_fold(state), b"x" * 10

    agents = [Scripted([step] * 50), CallingStation(), CallingStation()]
    t = Table(agents, seed=1, memory_bytes=4, hand_cap=5)
    t.run()
    assert seen[0] == b"" and all(m == b"xxxx" for m in seen[1:])


class Slow(Agent):
    def __init__(self, ms):
        self.ms = ms
        self.mem_seen = []

    def act(self, state, memory):
        self.mem_seen.append(memory)
        end = time.perf_counter() + self.ms / 1000.0
        while time.perf_counter() < end:
            pass
        return check_or_fold(state), memory + b"."


def test_strict_timing_and_timing_off():
    slow = Slow(2.0)
    t = Table([slow, CallingStation()], seed=1, hand_cap=3, strict_timing=True)
    t.run()
    st = t.stats[0]
    assert st.decisions > 0 and st.failures["timeout"] == st.decisions
    assert t.configs[0].time_bank_ms == 0.0
    assert all(m == b"" for m in slow.mem_seen)         # timeouts never store memory

    slow = Slow(2.0)
    t = Table([slow, CallingStation()], seed=1, hand_cap=3, strict_timing=True, timing=False)
    t.run()
    assert t.stats[0].failed == 0


def test_time_bank_drains_then_times_out():
    slow = Slow(5.0)
    t = Table([slow, CallingStation(), CallingStation()], seed=2, hand_cap=4,
              time_bank_ms=10.0, decision_ms=1.0)
    t.run()
    st = t.stats[0]
    assert st.decisions >= 4
    # ~4 ms over per decision: two fit in the 10 ms bank, every later one times out
    assert st.failures["timeout"] == st.decisions - 2
    assert st.bank_ms == 0.0
    assert len(st.times_ns) == st.decisions


# =========================================================================== full tables

def _replay_hand(h, n):
    """Rebuild every seat's end stack from the event stream alone."""
    stack = list(h.stacks_start)
    rb = [0] * n
    for e in h.events:
        if e.kind in ("ante", "blind"):
            stack[e.seat] -= e.amount
            if e.kind == "blind":
                rb[e.seat] += e.amount
        elif e.kind == "call":
            stack[e.seat] -= e.amount
            rb[e.seat] += e.amount
        elif e.kind == "raise":
            stack[e.seat] -= e.amount - rb[e.seat]
            rb[e.seat] = e.amount
        elif e.kind == "board":
            rb = [0] * n
        elif e.kind == "win":
            stack[e.seat] += e.amount
    return stack


class Checked(Agent):
    """Wraps a bot and records every State invariant violation."""

    def __init__(self, inner, problems):
        self.inner = inner
        self.problems = problems
        self.seen = []

    def act(self, state, memory):
        p = self.problems
        self.seen.extend(state.events)
        h = state.history
        if not h or h[0].kind != "hand" or h[0].amount != state.hand_number:
            p.append("history does not start with hand event")
        if list(h) != self.seen[len(self.seen) - len(h):]:
            p.append("history is not the tail of delivered events")
        kinds = [a.kind for a in state.legal_actions]
        if kinds != [k for k in ("fold", "check", "call", "raise") if k in kinds]:
            p.append(f"bad legal order {kinds}")
        if ("check" in kinds) == ("call" in kinds):
            p.append(f"check/call both or neither {kinds}")
        if ("check" in kinds) == ("fold" in kinds):
            p.append(f"check with fold {kinds}")
        if (state.to_call == 0) != ("check" in kinds):
            p.append("to_call vs check mismatch")
        me = state.seats[state.seat]
        if me.status != "active" or me.stack <= 0:
            p.append("acting seat not active")
        c = state.can("call")
        if c and not (c.min_amount == c.max_amount == state.to_call == min(c.min_amount, me.stack)):
            p.append("bad call amounts")
        r = state.can("raise")
        hb = max(s.bet for s in state.seats)
        if r and not (hb < r.min_amount <= r.max_amount == me.bet + me.stack):
            p.append(f"bad raise range {r} hb={hb}")
        committed = sum(e.amount for e in h if e.kind in ("ante", "blind", "call"))
        rb = {}
        for e in h:
            if e.kind == "blind":
                rb[e.seat] = rb.get(e.seat, 0) + e.amount
            elif e.kind == "call":
                rb[e.seat] = rb.get(e.seat, 0) + e.amount
            elif e.kind == "raise":
                committed += e.amount - rb.get(e.seat, 0)
                rb[e.seat] = e.amount
            elif e.kind == "board":
                rb = {}
        if committed != state.pot:
            p.append(f"pot {state.pot} != committed {committed}")
        if any(rb.get(s.seat, 0) != s.bet for s in state.seats):
            p.append("seat bets disagree with history")
        board = tuple(c for e in h if e.kind == "board" for c in e.cards)
        if board != state.board or len(board) not in (0, 3, 4, 5):
            p.append("board mismatch")
        if len(state.cards) != 2 or state.phase != "betting":
            p.append("bad cards/phase")
        if state.config.seat != state.seat or len(state.seats) != state.config.players:
            p.append("bad config")
        return self.inner.act(state, memory)


def test_random_legal_tables_conserve_chips_every_hand():
    problems = []
    hands = 0
    seed = 0
    while hands < 400:
        n = 8
        agents = [Checked(RandomLegal(seed=seed * 10 + i), problems) for i in range(n)]
        t = Table(agents, seed=seed, record_hands=True)
        res = t.run()
        for h in res.hands:
            assert sum(h.stacks_end) == 2000 * n
            assert _replay_hand(h, n) == h.stacks_end
            stack_ev = {e.seat: e.amount for e in h.events if e.kind == "stack"}
            assert stack_ev == {i: s for i, s in enumerate(h.stacks_end) if s > 0}
        assert sorted(res.places) == list(range(1, n + 1))
        assert abs(sum(res.points) - 100.0) < 1e-9
        assert all(st.failed == 0 for st in res.stats), [st.tracebacks for st in res.stats]
        hands += res.hands_played
        seed += 1
    assert not problems, problems[:5]


def test_all_builtin_bots_full_tables():
    problems = []
    classes = sorted(set(BUILTIN_BOTS.values()), key=lambda c: c.__name__)
    for seed in range(15):
        rng = random.Random(seed)
        lineup = [rng.choice(classes) for _ in range(8)]
        agents = [Checked(c(seed=seed * 8 + i), problems) for i, c in enumerate(lineup)]
        res = play_table(agents, seed=seed, record_hands=True)
        for h in res.hands:
            assert _replay_hand(h, 8) == h.stacks_end
        assert all(st.failed == 0 for st in res.stats), [st.tracebacks for st in res.stats]
        assert abs(sum(res.points) - 100.0) < 1e-9
    assert not problems, problems[:5]


def test_hand_cap_icm_split_and_blind_levels():
    agents = [Checked(CheckOrFold(), []) for _ in range(8)]
    t = Table(agents, seed=3, record_hands=True)
    res = t.run()
    assert res.hands_played == 200 and res.ended_by == "hand_cap"
    assert abs(sum(res.points) - 100.0) < 1e-9
    assert sorted(res.places) == list(range(1, 9))
    alive = [i for i in range(8) if res.final_stacks[i] > 0]
    expect = icm([res.final_stacks[i] for i in alive], (50, 30, 20))
    assert [round(res.points[i], 9) for i in alive] == [round(x, 9) for x in expect]
    assert res.hands[0].blinds.big == 20 and res.hands[24].blinds.big == 20
    assert res.hands[25].blinds.big == 30 and res.hands[199].blinds.big == 500
    # nobody busts at a check/fold table, so the button moves one seat per hand
    assert all(h.stacks_end.count(0) == 0 for h in res.hands)
    for a, b in zip(res.hands, res.hands[1:]):
        assert b.button == (a.button + 1) % 8
        assert (b.sb, b.bb) == ((b.button + 1) % 8, (b.button + 2) % 8)
    # deterministic for a given seed
    res2 = play_table([CheckOrFold() for _ in range(8)], seed=3)
    assert res2.final_stacks == res.final_stacks and res2.points == res.points


def test_state_fields_level_and_bank():
    agents = [Scripted(then="call") for _ in range(3)]
    t = Table(agents, seed=8, hand_cap=30, hands_per_level=10, timing=False)
    t.run()
    for st in agents[0].states:
        assert st.blind_level == (st.hand_number - 1) // 10
        assert st.hands_until_next_level == 10 - (st.hand_number - 1) % 10
        assert st.blinds == st.config.blinds[st.blind_level]
        assert st.time_bank_ms == 100.0
        assert st.players_remaining == sum(1 for s in st.seats if s.status != "busted")


def test_icm_values():
    assert icm([1000, 1000], (50, 30, 20)) == [40.0, 40.0]
    a, b = icm([3000, 1000], (50, 30, 20))
    assert abs(a - 45.0) < 1e-9 and abs(b - 35.0) < 1e-9
    xs = icm([500] * 3, (50, 30, 20))
    assert all(abs(x - 100 / 3) < 1e-9 for x in xs)
    xs = icm([100, 200, 300, 400, 500], (50, 30, 20))
    assert abs(sum(xs) - 100) < 1e-9 and xs == sorted(xs)


def test_card_parsing():
    assert card_int("2c") == 0 and card_int("As") == 51 and card_int("Td") == 33
    assert card_int("10h") == card_int("Th")


def test_short_all_ins_that_add_up_to_a_full_raise_reopen():
    # TDA rule: a player who already acted may re-raise only if the total raise it now
    # faces is at least the last full raise (here 80). 100 -> 150 -> 220 adds up to 120.
    def run(last_stack):
        agents = [Scripted(["call"]), Scripted(["fold"]), Scripted(["fold"]),
                  Scripted([("raise", 100), "call"]), Scripted(["allin"]), Scripted(["allin"])]
        t = Table(agents, seed=1, stacks=[2000, 2000, 2000, 2000, 150, last_stack],
                  button=0, hand_cap=1)
        t.play_hand()
        return agents[3].states[1]
    st = run(220)
    assert st.can("raise") is not None and st.can("raise").min_amount == 300
    st = run(170)                       # only +70 in total: no re-raise
    assert tuple(a.kind for a in st.legal_actions) == ("fold", "call")


def test_antes_posted_before_blinds_and_not_in_round_bets():
    agents = [Scripted(then="call") for _ in range(3)]
    t = Table(agents, seed=1, button=0, blinds=[(10, 20, 5)], hand_cap=1, record_hands=True)
    t.play_hand()
    kinds = [e.kind for e in t.events[:6]]
    assert kinds == ["hand", "ante", "ante", "ante", "blind", "blind"]
    st = agents[0].states[0]
    assert st.pot == 15 + 30 and st.to_call == 20
    assert [s.bet for s in st.seats] == [0, 10, 20]
    assert _replay_hand(t.hands[0], 3) == t.hands[0].stacks_end
    assert sum(t.stacks) == 6000
