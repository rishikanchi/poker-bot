"""Kernel checks: the Numba evaluator against a slow reference, ICM sanity."""
import itertools
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
from pb import kernels as K  # noqa: E402


def ref5(cards):
    """Slow, obviously-correct 5-card value comparable across hands."""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = [c & 3 for c in cards]
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight = None
    if len(uniq) == 5 and uniq[0] - uniq[4] == 4:
        straight = uniq[0]
    if uniq == [12, 3, 2, 1, 0]:
        straight = 3
    counts = sorted(((ranks.count(r), r) for r in uniq), reverse=True)
    shape = [c for c, _ in counts]
    order = [r for _, r in counts]
    if straight is not None and flush:
        return (8, [straight])
    if shape == [4, 1]:
        return (7, order)
    if shape == [3, 2]:
        return (6, order)
    if flush:
        return (5, ranks)
    if straight is not None:
        return (4, [straight])
    if shape == [3, 1, 1]:
        return (3, order)
    if shape == [2, 2, 1]:
        return (2, order)
    if shape == [2, 1, 1, 1]:
        return (1, order)
    return (0, ranks)


def ref_best(cards):
    return max(ref5(list(c)) for c in itertools.combinations(cards, 5))


def test_evaluator_matches_reference():
    rng = random.Random(3)
    hands = [rng.sample(range(52), 7) for _ in range(3000)]
    for n in (5, 6, 7):
        vals = [K.eval_hand(h[:n]) for h in hands]
        refs = [ref_best(h[:n]) for h in hands]
        for i in range(0, len(hands) - 1):
            a, b = vals[i], vals[i + 1]
            ra, rb = refs[i], refs[i + 1]
            assert (a > b) == (ra > rb) and (a == b) == (ra == rb), (hands[i][:n], hands[i + 1][:n])


def test_known_hands():
    def c(s):
        return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])
    def v(*cs):
        return K.eval_hand([c(x) for x in cs])
    wheel = v("Ah", "2c", "3d", "4s", "5h", "Kc", "Kd")
    six_high = v("6h", "2c", "3d", "4s", "5h", "Kc", "Kd")
    assert six_high > wheel > v("Kh", "Kc", "Kd", "2s", "3h", "9c", "8d") - 1
    assert v("Ah", "Kh", "Qh", "Jh", "9h", "2c", "2d") > v("Ac", "Kd", "Qh", "Js", "Tc", "2c", "2d")
    assert v("9c", "9d", "9h", "2s", "2c", "Ac", "Kd") > v("Ah", "Kh", "Qh", "Jh", "9h", "2c", "2d")
    assert v("Ac", "Ad", "Kh", "Ks", "Qc", "Qd", "2h") > v("Ac", "Ad", "Kh", "Ks", "Jc", "Jd", "Th")
    assert v("Ac", "Ad", "7h", "6s", "5c", "3d", "2h") < v("Ac", "Ad", "8h", "6s", "5c", "3d", "2h")


def test_icm():
    vals = K.icm_values([1000, 1000], [50, 30, 20])
    assert abs(vals[0] - 40) < 1e-9 and abs(vals[1] - 40) < 1e-9
    vals = K.icm_values([3000, 2000, 1000, 0], [50, 30, 20])
    assert abs(vals.sum() - 100) < 1e-9 and vals[3] == 0
    assert vals[0] > vals[1] > vals[2]
    # chip leader's equity is less than chip-proportional share (ICM compresses)
    assert vals[0] < 100 * 3000 / 6000
    # 8 equal stacks: everyone gets 100/8
    vals = K.icm_values([1500] * 8, [50, 30, 20])
    assert np.allclose(vals, 100 / 8)


def test_equity_sampling():
    from pb.game import Game
    g = Game("holdem", 2, "any", range(13))
    def c(s):
        return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])
    hero = [c("Ah"), c("As")]
    w = np.zeros(g.M)
    for k in range(g.M):
        a, b = g.U[k]
        if a >> 2 == 11 and b >> 2 == 11:
            w[k] = 1.0
    w *= g.live(g.dead_mask(hero))
    _, hu, _ = g.equity(hero, [], [w], 20000, 5)
    assert abs(hu.mean() - 0.82) < 0.015, hu.mean()
    assert abs(g.pre_equity_hu(hero, w) - 0.82) < 0.02


def test_omaha_rule():
    """Omaha: four hearts in hand + one on board is NOT a flush."""
    from pb.game import Game
    g = Game("omaha", 4, "omaha", range(13), universe_size=200)
    def c(s):
        return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])
    hole = [c("Ah"), c("Kh"), c("Qh"), c("Jh")]
    board = [c("2h"), c("7c"), c("8d"), c("3s"), c("9c")]
    v = g.hand_value(hole, board)
    assert v >> 20 < 5, "omaha must use exactly 2 hole cards"
    board2 = [c("2h"), c("7h"), c("8h"), c("3s"), c("9c")]
    assert g.hand_value(hole, board2) >> 20 == 5


if __name__ == "__main__":
    test_known_hands(); test_icm(); test_evaluator_matches_reference(); test_equity_sampling(); test_omaha_rule()
    print("kernel tests ok")


def test_omaha_draws_need_two_hole_cards():
    import numpy as np
    from pb import kernels as K
    from pb.postflop import hero_draw_flags

    def ci(s):
        return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])
    POP, HIGH, STRAIGHT, CATR = K.build_mask_tables(False)
    board = [ci("Ks"), ci("7s"), ci("2s")]
    one_spade = [ci("As"), ci("Kd"), ci("Qh"), ci("Jc")]      # the bare ace of spades: no flush possible
    two_spades = [ci("As"), ci("Qs"), ci("8d"), ci("3c")]     # already a flush, not a draw
    fd_board = [ci("Ks"), ci("7s"), ci("2d")]
    U = np.array([one_spade, two_spades], np.int64)
    X, _ = K.hand_features(U, 4, np.array(board + [0, 0], np.int64), 3, 0, K.RULE_OMAHA, POP, HIGH, STRAIGHT, CATR)
    assert X[0, 3] == 0 and X[1, 3] == 0
    X, _ = K.hand_features(U, 4, np.array(fd_board + [0, 0], np.int64), 3, 0, K.RULE_OMAHA, POP, HIGH, STRAIGHT, CATR)
    assert X[0, 3] == 0 and X[1, 3] == 1                      # one spade: no draw; two spades: flush draw
    # straights: board T 9 2 - J8xx is open-ended (Q or 7), a lone J with 3-2 is not
    hm = lambda cards: sum({1 << (ci(c) >> 2) for c in cards})   # noqa: E731
    bm = hm(["Td", "9c", "2h"])
    assert not K.omaha_straight(hm(["Js", "8d", "3c", "3d"]), bm, POP)
    assert K.omaha_straight(hm(["Js", "8d", "3c", "3d"]), bm | hm(["Qh"]), POP)
    assert K.omaha_straight(hm(["Js", "8d", "3c", "3d"]), bm | hm(["7h"]), POP)
    # four to a straight on board + one hole card is NOT a straight in Omaha
    assert not K.omaha_straight(hm(["Js", "4d", "4c", "2d"]), hm(["Td", "9c", "8h", "7s"]), POP)
    assert not K.omaha_straight(hm(["Js", "6d", "4c", "2d"]), hm(["Td", "9c", "8h", "7s"]), POP)
    assert K.omaha_straight(hm(["Js", "7d", "4c", "2d"]), hm(["Td", "9c", "8h", "7s"]), POP)
    fd, outs, _ = hero_draw_flags(one_spade, board, True)
    assert not fd


def test_nut_draw_flags_and_opening_order():
    from pb.postflop import hero_draw_flags
    from pb.game import get_game
    from pb import preflop as PRE, settings

    def ci(s):
        return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])
    board = [ci("Ks"), ci("7s"), ci("2d")]
    fd, outs, nut = hero_draw_flags([ci("As"), ci("5s")], board)
    assert fd and nut                                   # ace-high flush draw: the nuts
    fd, outs, nut = hero_draw_flags([ci("9s"), ci("5s")], board)
    assert fd and not nut                               # nine-high flush draw: not
    fd, outs, nut = hero_draw_flags([ci("Jh"), ci("Td")], [ci("9c"), ci("8d"), ci("2s")])
    assert outs >= 2 and nut                            # JT on 98x: open-ended to the nut straight
    fd, outs, nut = hero_draw_flags([ci("6h"), ci("5d")], [ci("9c"), ci("8d"), ci("2s")])
    assert outs >= 1 and not nut                        # 65 on 98x: the bottom end of the straight
    # plan: deep opens are ordered by equity x playability - JTs above A2o
    g = get_game()
    g.open_rank_init(lambda hand: PRE._features_eqr(g, hand, settings.P))
    jt, a2 = g.hero_open_pct([ci("Jh"), ci("Th")]), g.hero_open_pct([ci("Ad"), ci("2c")])
    assert jt < a2
    # playability widens the gap between them beyond the raw ranking's
    assert a2 - jt > g.hero_pre_pct([ci("Ad"), ci("2c")]) - g.hero_pre_pct([ci("Jh"), ci("Th")])
    assert g.hero_open_pct([ci("Ad"), ci("Ah")]) < 0.01
