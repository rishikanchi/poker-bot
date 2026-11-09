"""State parsing: all-in flags, orders, effective stacks."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stub_sdk"))
sys.path.insert(0, os.path.join(ROOT, "bot"))

from poker_agent_sdk import BlindLevel, Event, LegalAction, SeatState, State, TableConfig  # noqa: E402

from pb import view as V  # noqa: E402

CFG = TableConfig("t", 0, "nlhe", 6, 2000, (BlindLevel(10, 20),), 25, 200, (50, 30, 20))


def state(seat, seats, history, legal, to_call, button=5, board=()):
    return State(config=CFG, seat=seat, cards=("Ah", "Kd"), board=tuple(board), pot=0, to_call=to_call,
                 legal_actions=tuple(legal), history=tuple(history), phase="betting", seats=tuple(seats),
                 button=button, blind_level=0, blinds=BlindLevel(10, 20), hands_until_next_level=5,
                 hand_number=4, players_remaining=len(seats), bust_order=(), events=tuple(history))


def test_allin_flag_only_on_the_action_that_went_all_in():
    # seat 2 opens to 50 with 400 behind, seat 3 3-bets to 150, seat 2 shoves 400 total
    hist = [Event("hand", None, 4), Event("blind", 0, 10), Event("blind", 1, 20),
            Event("raise", 2, 50), Event("raise", 3, 150), Event("fold", 4), Event("fold", 5),
            Event("fold", 0), Event("fold", 1), Event("raise", 2, 400)]
    seats = [SeatState(0, 1990, 0, "folded"), SeatState(1, 1980, 0, "folded"), SeatState(2, 0, 400, "all_in"),
             SeatState(3, 1850, 150, "active"), SeatState(4, 2000, 0, "folded"), SeatState(5, 2000, 0, "folded")]
    v = V.build_view(state(3, seats, hist, [LegalAction("fold"), LegalAction("call", 250, 250)], 250))
    s2 = [a for a in v.acts[0] if a.seat == 2]
    assert [a.allin for a in s2] == [False, True], "the open was not all-in; only the shove was"
    assert v.players[2].start == 400 and v.pot == 10 + 20 + 400 + 150
    assert v.pre_order == [2, 3, 4, 5, 0, 1] and v.post_order == [0, 1, 2, 3, 4, 5]


def test_heads_up_orders():
    hist = [Event("hand", None, 4), Event("blind", 5, 10), Event("blind", 0, 20)]
    seats = [SeatState(0, 1980, 20, "active"), SeatState(1, 0, 0, "busted"), SeatState(2, 0, 0, "busted"),
             SeatState(3, 0, 0, "busted"), SeatState(4, 0, 0, "busted"), SeatState(5, 1990, 10, "active")]
    v = V.build_view(state(5, seats, hist, [LegalAction("fold"), LegalAction("call", 10, 10),
                                           LegalAction("raise", 40, 2000)], 10, button=5))
    assert v.alive == [0, 5] and v.sb_seat == 5 and v.bb_seat == 0
    assert v.pre_order == [5, 0], "button/small blind acts first preflop heads-up"
    assert v.post_order == [0, 5], "big blind acts first postflop heads-up"


if __name__ == "__main__":
    test_allin_flag_only_on_the_action_that_went_all_in(); test_heads_up_orders()
    print("view tests ok")


def test_minimum_size_translation_point():
    """Plan: opponents' bets are translated onto min / 33% / ...; a minimum-size
    bet tells little. A raise is a min-raise or it is not."""
    from pb.stats import min_weight
    # bets (upper point 33% of the pot after calling)
    assert min_weight(20, 20, 1800, 0.33) == 1.0          # 1 bb into 90 bb: the minimum
    assert min_weight(594, 20, 1800, 0.33) == 0.0         # a 33% bet (rounded down by a chip)
    assert 0.0 < min_weight(200, 20, 1800, 0.33) < 1.0    # in between: split
    assert min_weight(20, 20, 40, 0.33) == 0.0            # tiny pot: the minimum is already half the pot
    # raises
    assert min_weight(26, 26, 200, None) == 1.0           # raise-by equals the bet: a min-raise
    assert min_weight(60, 26, 200, None) == 0.0


def test_actions_know_the_minimum_raise():
    from pb import view as V
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_icm as T
    from poker_agent_sdk import Event, SeatState
    hist = [Event("hand", None, 9), Event("blind", 0, 10), Event("blind", 1, 20), Event("raise", 0, 44),
            Event("call", 1, 24), Event("board", None, 0, ("Ks", "7d", "2c")),
            Event("raise", 1, 26), Event("raise", 0, 52), Event("raise", 1, 150)]
    seats = [SeatState(0, 1904, 52, "active"), SeatState(1, 1806, 150, "active")]
    v = V.build_view(T.mk(0, seats, hist, 98, 0))
    bet, minraise, big = v.acts[1]
    assert (bet.by, bet.min_by) == (26, 20)               # a bet of 26 when the minimum was one big blind
    assert (minraise.by, minraise.min_by) == (26, 26)     # raise by exactly the bet: the minimum
    assert (big.by, big.min_by) == (98, 26)
