"""Hand-built states through the full bot: legal actions, memory round-trips, timing."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stub_sdk"))
sys.path.insert(0, os.path.join(ROOT, "bot"))

import poker_agent_sdk as sdk  # noqa: E402
from poker_agent_sdk import BlindLevel, Event, LegalAction, SeatState, State, TableConfig  # noqa: E402

import agent as bot  # noqa: E402

CFG = TableConfig("t1", 0, "nlhe", 6, 2000, (BlindLevel(10, 20),), 25, 200, (50, 30, 20))


def mk(seat, cards, board, seats, history, legal, to_call, button=5, events=(), hand=3):
    pot = sum(e.amount for e in history if e.kind in ("blind", "ante", "call")) + sum(
        0 for _ in ())
    # raise amounts are totals; compute pot from seat bets + previous streets roughly
    return State(config=CFG, seat=seat, cards=tuple(cards), board=tuple(board), pot=pot,
                 to_call=to_call, legal_actions=tuple(legal), history=tuple(history),
                 phase="betting", seats=tuple(seats), button=button, blind_level=0,
                 blinds=BlindLevel(10, 20), hands_until_next_level=20, hand_number=hand,
                 players_remaining=len(seats), bust_order=(), events=tuple(events) or tuple(history))


def seat_states(stacks, bets, statuses=None):
    return [SeatState(i, stacks[i], bets[i], (statuses or ["active"] * len(stacks))[i]) for i in range(len(stacks))]


def check_legal(state, action):
    la = state.can(action.kind)
    assert la is not None, (action.kind, state.legal_actions)
    if action.kind == "raise":
        assert isinstance(action.amount, int) and la.min_amount <= action.amount <= la.max_amount, action.amount


def run(state, mem=b""):
    a = bot.Agent()
    t = time.perf_counter_ns()
    act, mem2 = a.act(state, mem)
    dt = (time.perf_counter_ns() - t) / 1e6
    check_legal(state, act)
    assert isinstance(mem2, bytes)
    return act, mem2, dt


H = [Event("hand", None, 3), Event("blind", 0, 10), Event("blind", 1, 20)]


def test_unopened_utg_aces_raise():
    # 6-max: button 5, SB 0, BB 1, hero seat 2 (UTG) with AA
    st = mk(2, ["Ah", "As"], [], seat_states([1990, 1980, 2000, 2000, 2000, 2000], [10, 20, 0, 0, 0, 0]), H,
            [LegalAction("fold"), LegalAction("call", 20, 20), LegalAction("raise", 40, 2000)], 20)
    act, mem, dt = run(st)
    assert act.kind == "raise", act
    print("AA UTG:", act, f"{dt:.3f}ms")


def test_unopened_utg_trash_fold():
    st = mk(2, ["7h", "2c"], [], seat_states([1990, 1980, 2000, 2000, 2000, 2000], [10, 20, 0, 0, 0, 0]), H,
            [LegalAction("fold"), LegalAction("call", 20, 20), LegalAction("raise", 40, 2000)], 20)
    act, mem, dt = run(st)
    assert act.kind == "fold", act


def test_short_stack_push():
    # hero on the button with 10bb and A9o, folded to us
    hist = H + [Event("fold", 2), Event("fold", 3), Event("fold", 4)]
    st = mk(5, ["Ad", "9c"], [], seat_states([1990, 1980, 1500, 1500, 1500, 200], [10, 20, 0, 0, 0, 0],
                                             ["active", "active", "folded", "folded", "folded", "active"]),
            hist, [LegalAction("fold"), LegalAction("call", 20, 20), LegalAction("raise", 40, 200)], 20)
    act, mem, dt = run(st)
    assert act.kind == "raise" and act.amount == 200, act
    print("10bb A9o BTN:", act, f"{dt:.3f}ms")


def test_flop_facing_bet_and_memory_roundtrip():
    hist = H + [Event("fold", 2), Event("fold", 3), Event("raise", 4, 50), Event("fold", 5), Event("fold", 0),
                Event("call", 1, 30), Event("board", None, 0, ("Ks", "7d", "2c")), Event("check", 1),
                Event("raise", 4, 60)]
    # hero = BB (seat 1) facing a 60 c-bet on K72 with K9s; pot = 10 + 50 + 50 + 60 = 170
    st = mk(1, ["Kh", "9h"], ["Ks", "7d", "2c"],
            seat_states([1990, 1950, 1500, 1500, 1890, 2000], [0, 0, 0, 0, 60, 0],
                        ["folded", "active", "folded", "folded", "active", "folded"]),
            hist, [LegalAction("fold"), LegalAction("call", 60, 60), LegalAction("raise", 120, 1950)], 60)
    act, mem, dt = run(st)
    assert act.kind in ("call", "raise"), act
    print("BB K9s vs cbet:", act, f"{dt:.3f}ms", "mem", len(mem), "bytes")
    # second call with returned memory must parse and still be legal
    act2, mem2, dt2 = run(st, mem)
    print("  again with memory:", act2, f"{dt2:.3f}ms")
    # garbage memory must not break anything
    act3, mem3, _ = run(st, b"\x00garbage" * 50)


def test_river_check_with_air():
    hist = H + [Event("fold", 2), Event("fold", 3), Event("raise", 4, 50), Event("fold", 5), Event("fold", 0),
                Event("call", 1, 30), Event("board", None, 0, ("Ks", "7d", "2c")), Event("check", 1),
                Event("check", 4), Event("board", None, 0, ("9s",)), Event("check", 1), Event("check", 4),
                Event("board", None, 0, ("3h",))]
    st = mk(1, ["5h", "4h"], ["Ks", "7d", "2c", "9s", "3h"],
            seat_states([1990, 1950, 1500, 1500, 1950, 2000], [0, 0, 0, 0, 0, 0],
                        ["folded", "active", "folded", "folded", "active", "folded"]),
            hist, [LegalAction("check"), LegalAction("raise", 20, 1950)], 0)
    act, mem, dt = run(st)
    print("river 5h4h (wheel draw missed? no: 5-4 on K7293 = 5-high straight? A missing) ->", act, f"{dt:.3f}ms")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("smoke ok")


def test_call_or_fold_only_when_blind_is_all_in():
    """Small blind vs a big blind who is all-in from posting: only call/fold is
    offered and nobody has raised. A premium hand must call, not fold."""
    stacks = [0, 1990, 5000, 4000, 3000, 2000]
    bets = [12, 10, 0, 0, 0, 0]
    statuses = ["all_in", "active", "folded", "folded", "folded", "folded"]
    hist = [Event("hand", None, 3), Event("blind", 1, 10), Event("blind", 0, 12),
            Event("fold", 2, 0), Event("fold", 3, 0), Event("fold", 4, 0), Event("fold", 5, 0)]
    legal = [LegalAction("fold"), LegalAction("call", 2, 2)]
    st = mk(1, ("As", "Ah"), (), seat_states(stacks, bets, statuses), hist, legal, 2, button=5)
    action, _, _ = run(st)
    assert action.kind == "call", action.kind


def test_pot_limit_raise_is_not_priced_as_a_shove():
    """With 12bb and a raise capped at 3.5bb (pot-limit) no all-in is on offer."""
    from pb import preflop as PRE, view as V
    stacks = [240, 240, 240, 240, 230, 220]
    bets = [0, 0, 0, 0, 10, 20]
    hist = [Event("hand", None, 3), Event("blind", 4, 10), Event("blind", 5, 20)]
    capped = mk(0, ("As", "Ah"), (), seat_states(stacks, bets), hist,
                [LegalAction("fold"), LegalAction("call", 20, 20), LegalAction("raise", 40, 70)], 20, button=3)
    full = mk(0, ("As", "Ah"), (), seat_states(stacks, bets), hist,
              [LegalAction("fold"), LegalAction("call", 20, 20), LegalAction("raise", 40, 240)], 20, button=3)
    assert not PRE._can_jam(V.build_view(capped)) and PRE._can_jam(V.build_view(full))
    action, _, _ = run(capped)
    check_legal(capped, action)
    assert action.kind == "raise"


def test_calls_a_minimum_bet_with_huge_pot_odds():
    """Heads-up, 88 bb in the pot after a preflop raising war; the opponent bets
    one big blind on every street. King-high must not fold the river for 1 bb."""
    board = ["6s", "5c", "4c", "4s", "Ts"]
    hist = [Event("hand", None, 3), Event("blind", 0, 10), Event("blind", 1, 20),
            Event("raise", 0, 44), Event("raise", 1, 68), Event("raise", 0, 156), Event("raise", 1, 244),
            Event("raise", 0, 562), Event("raise", 1, 880), Event("call", 0, 318),
            Event("board", None, 0, tuple(board[:3])), Event("raise", 1, 20), Event("call", 0, 20),
            Event("board", None, 0, tuple(board[3:4])), Event("raise", 1, 20), Event("call", 0, 20),
            Event("board", None, 0, tuple(board[4:5])), Event("raise", 1, 20)]
    seats = [SeatState(0, 3340 - 920, 0, "active"), SeatState(1, 4660 - 940, 20, "active")]
    legal = [LegalAction("fold"), LegalAction("call", 20, 20), LegalAction("raise", 40, 3340 - 920)]
    cfg2 = TableConfig("t9", 0, "nlhe", 2, 4000, (BlindLevel(10, 20),), 25, 200, (50, 30, 20))
    st = State(config=cfg2, seat=0, cards=("Kd", "Qh"), board=tuple(board), pot=1820, to_call=20,
               legal_actions=tuple(legal), history=tuple(hist), phase="betting", seats=tuple(seats), button=0,
               blind_level=0, blinds=BlindLevel(10, 20), hands_until_next_level=20, hand_number=3,
               players_remaining=2, bust_order=(), events=tuple(hist))
    action, _, _ = run(st)
    assert action.kind in ("call", "raise"), action.kind
