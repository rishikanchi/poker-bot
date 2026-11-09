"""ICM outcome accounting: side pots, overbets, premiums."""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stub_sdk"))
sys.path.insert(0, os.path.join(ROOT, "bot"))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from poker_agent_sdk import BlindLevel, Event, LegalAction, SeatState, State, TableConfig  # noqa: E402

from pb import icmtools as I  # noqa: E402
from pb import kernels as K  # noqa: E402
from pb import view as V  # noqa: E402

CFG = TableConfig("t", 0, "nlhe", 6, 2000, (BlindLevel(10, 20),), 25, 200, (50, 30, 20))


def mk(seat, seats, history, to_call, button):
    return State(config=CFG, seat=seat, cards=("Ah", "Kd"), board=(), pot=0, to_call=to_call,
                 legal_actions=(LegalAction("fold"), LegalAction("call", to_call, to_call)),
                 history=tuple(history), phase="betting", seats=tuple(seats), button=button,
                 blind_level=0, blinds=BlindLevel(10, 20), hands_until_next_level=5, hand_number=9,
                 players_remaining=len(seats), bust_order=(), events=tuple(history))


def test_overbet_allin_only_our_stack_is_at_stake():
    # 3 left. Seat 4 shoves 12000; hero (seat 1) has 1220 in total; seat 0 posted the big blind.
    hist = [Event("hand", None, 9), Event("blind", 1, 10), Event("blind", 0, 20), Event("raise", 4, 12000)]
    seats = [SeatState(0, 1930, 20, "active"), SeatState(1, 1210, 10, "active"), SeatState(2, 0, 0, "busted"),
             SeatState(3, 0, 0, "busted"), SeatState(4, 0, 12000, "all_in"), SeatState(5, 0, 0, "busted")]
    v = V.build_view(mk(1, seats, hist, 1210, 4))
    assert v.pot == 10 + 20 + 1220, "we can win our own stack from the shover, not 12000"
    win, lose, fold = I.allin_values(v, 4)
    pay = [50.0, 30.0, 20.0]
    # win: hero 1220 + 1220 + 20 = 2460, shover 12000 - 1220, big blind 1930
    assert abs(win - K.icm_values([1930, 2460, 10780], pay)[1]) < 1e-9
    assert lose == 20.0, "busting third of three still pays third place"
    assert abs(fold - K.icm_values([1930, 1210, 12030], pay)[1]) < 1e-9
    ctx = I.IcmCtx(v)
    prem = ctx.premiums(4, [ctx.allin_pair(4)])[0]
    chip_be = 1210 / 2460
    assert 0.0 < prem < 1.0 - chip_be


def test_chips_are_conserved_and_match_reference():
    from pb_ref import icmtools as I0, view as V0   # frozen reference implementation
    rng = random.Random(5)
    for _ in range(300):
        n = rng.choice([3, 4, 6])
        stacks = [rng.choice([300, 800, 1500, 4000, 9000]) for _ in range(6)]
        alive = sorted(rng.sample(range(6), n))
        hero, vil = rng.sample(alive, 2)
        others = [s for s in alive if s not in (hero, vil)]
        sb, bb = others[0], (others[1] if len(others) > 1 else hero)
        vbet = min(rng.choice([60, 200, stacks[vil]]), stacks[vil])
        hist = [Event("hand", None, 9), Event("blind", sb, 10), Event("blind", bb, 20), Event("raise", vil, vbet)]
        seats = []
        for s in range(6):
            if s not in alive:
                seats.append(SeatState(s, 0, 0, "busted"))
                continue
            bet = min(vbet if s == vil else (10 if s == sb else (20 if s == bb else 0)), stacks[s])
            seats.append(SeatState(s, stacks[s] - bet, bet, "all_in" if stacks[s] == bet else "active"))
        hb = [x for x in seats if x.seat == hero][0]
        call = min(vbet - hb.bet, hb.stack)
        st = mk(hero, seats, hist, call, alive[-1])
        v, v0 = V.build_view(st), V0.build_view(st)
        a, b = I.allin_values(v, vil), I0.allin_values(v0, vil)
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9
        ctx = I.IcmCtx(v)
        total = ctx.start.sum()
        vi = ctx.index[vil]
        for risk, v_add in (ctx.allin_pair(vil), (call, 0)):
            w, l, _, _ = ctx._row_win_lose(vi, risk, v_add)
            assert abs(w.sum() - total) < 1e-6 and abs(l.sum() - total) < 1e-6
            assert (w >= -1e-9).all() and (l >= -1e-9).all()


def test_unmatched_chips_come_back():
    """Hero covers the villain and shoves: only the villain's stack is at stake."""
    seats = [SeatState(0, 5800, 200, "active"), SeatState(1, 1300, 200, "active"), SeatState(2, 4500, 0, "folded")]
    hist = [Event("hand", None, 3), Event("blind", 0, 100), Event("blind", 1, 200), Event("call", 0, 100)]
    st = mk(0, seats, hist, 0, 2)
    ctx = I.IcmCtx(V.build_view(st))
    w, l, pot, at_risk = ctx._row_win_lose(ctx.index[1], 5800, 1300)
    assert list(l) == [4500, 3000, 4500] and list(w) == [7500, 0, 4500]
    assert pot == 3000 and at_risk == 1300
    # the premium is measured on the 1300 really at stake, not the 5800 pushed forward
    assert ctx.premiums(1, [(5800, 1300)])[0] == ctx.premiums(1, [(1300, 1300)])[0]


if __name__ == "__main__":
    test_overbet_allin_only_our_stack_is_at_stake(); test_chips_are_conserved_and_match_reference()
    print("icm tests ok")
