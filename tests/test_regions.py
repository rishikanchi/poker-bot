"""Region model bookkeeping: quantiles, masses with blockers, equities, narrowing."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))

from pb import ranges as Rg  # noqa: E402
from pb.game import get_game  # noqa: E402
from pb.postflop import Regions  # noqa: E402
from pb.stats import Opp, K_STATS, JAM, JAM_EXP, JAM_D, JAM_D_OPP  # noqa: E402
from pb import settings  # noqa: E402

g = get_game()


def c(s):
    return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])


def test_river_regions_partition_and_equity():
    board = [c(x) for x in ("Ks", "7d", "2c", "9s", "3h")]
    hero = [c("Kh"), c("Qh")]
    vals = g.strengths(board, g.dead_mask(board))
    scores = np.where(vals >= 0, vals.astype(float), -1.0)
    order = np.argsort(scores, kind="stable")
    w = g.range_interval(0.0, 0.4, g.dead_mask(board))
    live_h = g.live(g.dead_mask(hero)).astype(float)
    hv = g.hand_value(hero, board)
    beats = np.where(vals < 0, 0.0, np.where(vals < hv, 1.0, np.where(vals == hv, 0.5, 0.0)))
    reg = Regions(order, w, live_h, 6.0, beats=beats)
    cuts = [0.0, 0.13, 0.4, 0.77, 1.0]
    masses = [reg.mass(a, b) for a, b in zip(cuts[:-1], cuts[1:])]
    assert abs(sum(masses) - 1.0) < 1e-9
    direct = float((beats * w * live_h).sum() / (w * live_h).sum())
    assert abs(reg.eq(0.0, 1.0) - direct) < 1e-9 and abs(reg.eq_hu - direct) < 1e-9
    recomposed = sum(m * reg.eq(a, b) for m, a, b in zip(masses, cuts[:-1], cuts[1:]))
    assert abs(recomposed - direct) < 1e-9
    # we beat the weak part of their range more often than the strong part
    assert reg.eq(0.0, 0.3) > reg.eq(0.7, 1.0)


def test_blockers_change_region_probabilities():
    # nut-flush blocker: holding the ace of spades on a three-spade river makes
    # their strongest region (flushes) less likely than a hand without a spade
    board = [c(x) for x in ("Ks", "7s", "2s", "9d", "3h")]
    vals = g.strengths(board, g.dead_mask(board))
    order = np.argsort(np.where(vals >= 0, vals.astype(float), -1.0), kind="stable")
    w = g.range_interval(0.0, 1.0, g.dead_mask(board))
    top = {}
    for name, hero in (("blocker", [c("As"), c("8d")]), ("none", [c("Ad"), c("8d")])):
        live_h = g.live(g.dead_mask(hero)).astype(float)
        beats = np.zeros(g.M)
        reg = Regions(order, w, live_h, 6.0, beats=beats)
        top[name] = reg.mass(0.9, 1.0)
    assert top["blocker"] < top["none"]


def test_sampled_regions_match_samples():
    board = [c(x) for x in ("Ks", "7d", "2c")]
    hero = [c("Kh"), c("Qh")]
    scores, draws = g.scores(board, g.dead_mask(board))
    order = np.argsort(scores, kind="stable")
    w = g.range_interval(0.0, 0.5, g.dead_mask(board)) * (scores >= 0)
    live_h = g.live(g.dead_mask(hero)).astype(float)
    vidx, hu, al = g.equity(hero, board, [w * live_h], 4000, 7)
    reg = Regions(order, w, live_h, 0.0, vidx=vidx, res_hu=hu, res_all=al)
    assert abs(reg.eq(0.0, 1.0) - hu.mean()) < 1e-9
    cuts = [0.0, 0.25, 0.6, 1.0]
    masses = [reg.mass(a, b) for a, b in zip(cuts[:-1], cuts[1:])]
    assert abs(sum(masses) - 1.0) < 1e-9
    assert reg.eq(0.0, 0.25) > reg.eq(0.6, 1.0)
    # sampled share of each region matches its probability (same conditional weights)
    q = np.empty(g.M)
    q[order] = reg.cum
    share = np.mean((q[vidx] > 0.6))
    assert abs(share - masses[2]) < 0.03


def test_narrowing_keeps_the_plan_regions():
    board = [c(x) for x in ("Js", "8s", "3d")]
    scores, draws = g.scores(board, g.dead_mask(board))
    order = np.argsort(scores, kind="stable")
    w = g.range_interval(0.0, 0.5, g.dead_mask(board)) * (scores >= 0)
    tot = w.sum()
    called = Rg.narrow_call(w, order, 0.4, 0.1, 0.0)
    assert abs(called.sum() / tot - 0.5) < 0.02            # middle 50% stays
    q = Rg._quantile(w, order)
    assert q[called > 0].min() > 0.39 and q[called > 0].max() <= 0.9 + 1e-9
    bet_river = Rg.narrow_bet(w, order, 0.5, 0.25, 0.0, None)
    kept = q[bet_river > 0]
    assert ((kept > 0.625 - 1e-9) | (kept <= 0.125 + 1e-9)).all()   # top value + bottom bluffs
    bet_flop = Rg.narrow_bet(w, order, 0.5, 0.25, 0.0, draws)
    extra = (bet_flop > 0) & (q <= 0.625)
    assert extra.any() and (draws[extra] > 0).all()                 # before the river the bluffs are draws
    checked = Rg.narrow_check(w, order, 0.5, 0.25, 0.15, draws)
    assert abs((checked * (bet_flop > 0)).sum() - 0.15 * (w * (bet_flop > 0)).sum()) < 1e-9


def test_band_probs_and_bands():
    dead = g.dead_mask([c("As"), c("Ah")])
    parts = g.band_probs([0.0, 0.05, 0.3, 1.0], dead)
    assert abs(sum(parts) - 1.0) < 1e-9
    # holding two aces makes the very top of their range rarer
    assert parts[0] < g.band_probs([0.0, 0.05], g.dead_mask([c("7d"), c("6d")]))[0]


def test_jam_width_uses_deep_frequency():
    P = settings.P
    row = np.zeros(K_STATS, np.int64)
    assert abs(Opp(row, P).jam_width(0.30) - 0.30) < 1e-9           # no data: the chart
    row[JAM_D_OPP], row[JAM_D] = 40, 12                             # open-shoves 30% of deep hands
    wide = Opp(row, P).jam_width(0.04)
    assert 0.2 < wide < 0.31
    row2 = np.zeros(K_STATS, np.int64)
    row2[JAM], row2[JAM_EXP] = 0, 600                               # never shoves where the chart shoves 6 times
    assert Opp(row2, P).jam_width(0.30) < 0.30


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("region tests ok")
