"""The self-play abstraction, the auction rules, the trainer and the table hooks."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "stub_sdk"))
sys.path.insert(0, os.path.join(ROOT, "bot"))
sys.path.insert(0, ROOT)

from pb import absx as X  # noqa: E402
from pb import kernels as K  # noqa: E402

R = "23456789TJQKA"
S = "cdhs"


def card(s):
    return R.index(s[0]) * 4 + S.index(s[1])


def eclass(hole, board):
    b = np.zeros(5, np.int64)
    for i, c in enumerate(board):
        b[i] = card(c)
    u = np.array([[card(hole[0]), card(hole[1])]], np.int64)
    return int(X.eqr_classes(u, b, len(board), np.int64(0), K.POP, K.STRAIGHT, 9)[0])


def test_eqr_classes():
    assert eclass(["Ah", "Jd"], ["As", "7c", "2d"]) == 1      # top pair, jack kicker
    assert eclass(["Ah", "Td"], ["As", "7c", "2d"]) == 0      # top pair, ten kicker
    assert eclass(["Kh", "Kd"], ["Qs", "7c", "2d"]) == 1      # overpair
    assert eclass(["9h", "9d"], ["Qs", "7c", "2d"]) == 0      # underpair
    assert eclass(["7h", "7d"], ["Qs", "7c", "2d"]) == 1      # set
    assert eclass(["Ah", "3h"], ["Kh", "7h", "2d"]) == 2      # nut flush draw
    assert eclass(["Qh", "3h"], ["Kh", "7h", "2d"]) == 0      # non-nut flush draw
    assert eclass(["Jh", "Td"], ["9s", "8c", "2d"]) == 2      # open-ended to the nuts
    assert eclass(["7h", "6d"], ["9s", "8c", "2d"]) == 0      # the low end
    assert eclass(["Ah", "3h"], ["Kh", "7h", "2d", "5s", "9c"]) == 0   # river: a missed draw is nothing


def test_flop_selection_follows_the_rules():
    assert X.flop_of(np.zeros(5)) == X.TIDX[0, 1, 2]                      # no bids: cards 0, 1, 2
    assert X.flop_of(np.array([0.0, 0, 0, 0, 1])) == X.TIDX[0, 1, 4]      # one chip puts a card on
    assert X.flop_of(np.array([5.0, 5, 5, 5, 5])) == X.TIDX[0, 1, 2]      # ties go to the earlier card
    assert X.flop_of(np.array([1.0, 9, 2, 9, 2])) == X.TIDX[1, 2, 3]      # tie for third: card 2 beats card 4


def test_pay_buckets():
    assert X.pay_bucket(0, 0) == 0          # could not bid
    assert X.pay_bucket(0, 500) == 1        # nothing
    assert X.pay_bucket(1, 3) == 2          # a token stays a token on a tiny balance
    assert X.pay_bucket(20, 500) == 3
    assert X.pay_bucket(60, 500) == 4
    assert X.pay_bucket(200, 500) == 5


def test_legal_bet_sizes():
    amt = np.zeros(X.NA)
    leg = np.zeros(X.NA, np.bool_)
    X.legal_post(10.0, 0.0, 100.0, 0, 0.0, 1.0, amt, leg)      # SPR 10: four sizes, no all-in
    assert list(leg) == [False, True, True, True, True, True, False]
    assert np.allclose(amt[2:6], [1.0, 3.3, 6.6, 12.5])
    X.legal_post(10.0, 0.0, 30.0, 0, 0.0, 1.0, amt, leg)       # SPR 3: all-in offered
    assert leg[6] and amt[6] == 30.0
    X.legal_post(15.0, 5.0, 100.0, 1, 5.0, 1.0, amt, leg)      # facing a bet: 10% is under the minimum raise
    assert leg[0] and not leg[2] and leg[3]


def test_model_bid_respects_the_balance():
    v = np.array([0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    out = np.zeros(5, np.int64)
    X.model_bid(v, 10.0, 1.0, 7.0, 0.001, np.zeros(5), out)
    assert out.sum() == 7 and out[3] == 0 and out[4] == 0 and out.min() >= 0


def test_trainer_smoke():
    from train import run as TR
    from train import sim as SM
    cfg = dict(TR.CFG)
    tabs = np.load(os.path.join(ROOT, "bot", "data", "holdem_tables.npz"))
    pct = ((tabs["pct_lo"] + tabs["pct_hi"]) / 2).astype(np.float64)
    q0 = np.zeros((X.NROW, X.NA))
    TR.build_q0(q0, 0.15, 0.15, 0.15)
    TR.G.update(E169=tabs["E169"].astype(np.float64),
                coef=np.load(os.path.join(ROOT, "bot", "data", "scorefit_holdem.npz"))["coef"],
                QH=q0.astype(np.float32), QP=TR.build_qp0(cfg, pct).astype(np.float32),
                EQP=TR.build_eqp0(cfg), VFH=TR.build_vf0(cfg).astype(np.float32),
                PAYL=np.ones(X.NBAND * 2 * X.NPAY * X.NB, np.float32), P=TR.params(cfg, 0.004, 0.5, False))
    # generation 0 opens the stated ranges: 15% first to act, 40% on the button (40+ bb)
    summ = TR.preflop_summary(TR.G["QP"], cfg)["40+bb"]
    assert abs(summ[0]["raise"] - 0.15) < 0.02 and abs(summ[5]["raise"] - 0.40) < 0.02
    QS, QN, VFS, VFN, EQS, EQE, EQN, PAYC, AST = TR.run_post(None, 1, 60, 5, 0, 2)
    assert QN.sum() > 0 and VFN.sum() > 0 and np.isfinite(QS).all()
    assert abs(VFS.sum() - VFN.sum() / 2) < 1e-6            # the two players' shares add up to the pot
    assert AST[1] <= AST[2]                                 # nobody pays more than they bid
    qs, qn, kinds = TR.run_pre(None, 1, 3000, 7)
    assert kinds.sum() == 3000 and np.isfinite(qs).all()
    q = TR.update_post(TR.G["QH"], QS, QN, (0, 1, 2), cfg)
    assert q.shape == (X.NROW, X.NA) and np.isfinite(q).all()


TABLES = os.path.join(ROOT, "bot", "data", "trained_holdem.npz")


@pytest.mark.skipif(not os.path.exists(TABLES), reason="tables not trained yet")
def test_hooks_return_legal_actions():
    import test_bot_smoke as B
    from pb import brain, trained
    from pb.settings import P
    from poker_agent_sdk import Event, LegalAction
    assert trained.load(TABLES)
    on = dict(P, use_trained_preflop=True, use_trained_postflop=True, use_trained_auction=True)
    seats = B.seat_states([1990, 1980, 2000, 2000, 2000, 2000], [10, 20, 0, 0, 0, 0])
    kinds = set()
    for cards in (["Ah", "As"], ["7h", "2c"], ["Jh", "Ts"], ["Kd", "Qd"]):
        st = B.mk(2, cards, [], seats, B.H,
                  [LegalAction("fold"), LegalAction("call", 20, 20), LegalAction("raise", 40, 2000)], 20)
        act, _ = brain.decide(st, b"", B.sdk, params=on)
        B.check_legal(st, act)
        kinds.add((cards[0], act.kind))
    assert ("7h", "fold") in kinds
    # heads-up on the flop, checked to us
    hist = B.H + [Event("fold", 2), Event("fold", 3), Event("fold", 4), Event("raise", 5, 50),
                  Event("fold", 0), Event("call", 1, 30), Event("board", None, 0, ("Kh", "7c", "2d")),
                  Event("check", 1)]
    st = B.mk(5, ["Kd", "Qd"], ["Kh", "7c", "2d"],
              B.seat_states([1990, 1950, 2000, 2000, 2000, 1950], [0] * 6,
                            ["folded", "active", "folded", "folded", "folded", "active"]),
              hist, [LegalAction("check"), LegalAction("raise", 20, 1950)], 0)
    act, _ = brain.decide(st, b"", B.sdk, params=on)
    B.check_legal(st, act)
