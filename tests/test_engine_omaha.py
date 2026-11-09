"""Omaha (PLO4/5/6) and pot-limit rules in the local simulator."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "stub_sdk"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_agent_sdk import Action, Agent  # noqa: E402

from sim import bots, engine  # noqa: E402
from sim.evaluator import (FLUSH, PAIR, STRAIGHT, cards_int, category, evaluate,  # noqa: E402
                           evaluate_omaha)


def ev(hole, board):
    return evaluate_omaha(cards_int(hole), cards_int(board))


def test_omaha_needs_exactly_two_hole_cards():
    # four hearts in hand + one heart on board is NOT a flush
    v = ev(["Ah", "Kh", "Qh", "Jh"], ["2h", "7c", "8d", "3s", "9c"])
    assert category(v) < FLUSH
    # two hearts in hand + three on board is
    assert category(ev(["Ah", "Kh", "2c", "3d"], ["5h", "9h", "Jh", "4s", "Tc"])) == FLUSH
    # board with four to a straight needs two fitting hole cards
    board = ["5c", "6d", "7h", "8s", "Kd"]
    assert category(ev(["9c", "Ac", "Ad", "2h"], board)) != STRAIGHT  # only one fitting card
    assert category(ev(["9c", "Tc", "Ad", "2h"], board)) == STRAIGHT
    # a board pair plus a pocket pair: hold'em would see two pair, Omaha must use 2 hole cards
    v_omaha = ev(["Ac", "Ad", "2h", "3s"], ["Kc", "Kd", "7h", "8s", "9d"])
    assert category(v_omaha) == 2  # two pair AA + KK (uses AA + K K x) -> still two pair
    # one matching hole card + a paired board: trips, not a full house (no second K in hand)
    v = ev(["7c", "2d", "3h", "4s"], ["7h", "7s", "Kd", "Kh", "9c"])
    assert category(v) == 3
    assert v == max(evaluate(cards_int([a, b] + list(t)))
                    for i, a in enumerate(["7c", "2d", "3h", "4s"]) for b in ["7c", "2d", "3h", "4s"][i + 1:]
                    for t in __import__("itertools").combinations(["7h", "7s", "Kd", "Kh", "9c"], 3))


class _Spy(Agent):
    """Records every state; calls when facing a bet, raises max preflop once."""

    def __init__(self, raise_max=False):
        self.states = []
        self.raise_max = raise_max

    def act(self, state, memory):
        self.states.append(state)
        la = state.can("raise")
        if self.raise_max and la is not None and not state.board:
            return Action.raise_to(la.max_amount), memory
        if state.can("check"):
            return Action.check(), memory
        return Action.call(), memory


def test_plo4_deals_four_cards_and_shows_four():
    a, b = _Spy(), _Spy()
    res = engine.play_table([a, b], seed=3, variant="plo4", hand_cap=3, timing=False, record_hands=True)
    assert res.hands_played >= 1
    st = a.states[0]
    assert len(st.cards) == 4 and st.config.variant == "plo4"
    for h in res.hands:
        for cs in h.hole_cards.values():
            assert len(cs) == 4
        shows = [e for e in h.events if e.kind == "show"]
        for e in shows:
            assert len(e.cards) == 4
    # all 4+4+5 cards distinct in each hand
    for h in res.hands:
        allc = [c for cs in h.hole_cards.values() for c in cs] + list(h.board)
        assert len(set(allc)) == len(allc)


def test_plo6_and_deck_limit():
    res = engine.play_table([_Spy(), _Spy(), _Spy()], seed=5, variant="plo6", hand_cap=2, timing=False)
    assert res.hands_played >= 1
    try:
        engine.Table([_Spy() for _ in range(8)], variant="plo6")
    except ValueError:
        pass
    else:
        raise AssertionError("8 players x 6 cards should not fit a 52-card deck")


def test_pot_limit_max_raise():
    # heads-up, blinds 10/20: SB (button) to act preflop with 10 to call.
    # pot = 30, call 10 -> pot after call 40 -> max raise-to = 20 + 30 + 10 = 60
    a, b = _Spy(raise_max=True), _Spy()
    engine.play_table([a, b], seed=1, variant="plo4", hand_cap=1, timing=False,
                      button=0, blinds=[(10, 20)])
    st = a.states[0]
    la = st.can("raise")
    assert la is not None and la.max_amount == 60, la
    assert la.min_amount == 40
    # BB then faces a raise to 60: pot = 60 + 20 = 80, call 40 -> max = 60 + 80 + 40 = 180
    sb_state = b.states[0]
    la2 = sb_state.can("raise")
    assert la2 is not None and la2.max_amount == 180, la2


def test_pot_limit_holdem_option_and_no_limit_omaha_option():
    a, b = _Spy(raise_max=True), _Spy()
    engine.play_table([a, b], seed=1, hand_cap=1, timing=False, button=0, blinds=[(10, 20)],
                      betting="pot_limit")
    assert a.states[0].can("raise").max_amount == 60
    a, b = _Spy(raise_max=True), _Spy()
    engine.play_table([a, b], seed=1, variant="plo4", betting="no_limit", hand_cap=1, timing=False,
                      button=0, blinds=[(10, 20)])
    assert a.states[0].can("raise").max_amount == 2000


def test_builtin_bots_legal_in_omaha():
    for var in ("plo4", "plo5"):
        lineup = [bots.CheckOrFold(), bots.MinRaiser(seed=1), bots.CallingStation(), bots.Maniac(seed=2),
                  bots.Nit(), bots.Jammer(), bots.RandomLegal(seed=3)]
        res = engine.play_table(lineup, seed=11, variant=var, timing=False)
        for st in res.stats:
            assert st.failed == 0, (var, st.name, st.failures, st.tracebacks[:1], st.illegal_examples[:1])


def test_holdem_unchanged_by_default():
    def run():
        lineup = [bots.Nit(), bots.Maniac(seed=4), bots.CallingStation(), bots.Jammer(), bots.RandomLegal(seed=9)]
        r = engine.play_table(lineup, seed=77, timing=False)
        return r.points, r.final_stacks, r.hands_played
    assert run() == run()
    a, b = run(), engine.play_table([bots.Nit(), bots.Maniac(seed=4), bots.CallingStation(), bots.Jammer(),
                                     bots.RandomLegal(seed=9)], seed=77, timing=False, variant="nlhe",
                                    betting="no_limit")
    assert a == (b.points, b.final_stacks, b.hands_played)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("omaha engine tests ok")
