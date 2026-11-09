"""Memory codec round-trips and stat counting from a known hand."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
from pb import stats as St  # noqa: E402
from pb.game import get_game  # noqa: E402
from pb.memory import Memory  # noqa: E402


def c(s):
    return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])


def test_roundtrip_and_garbage():
    m = Memory()
    m.ensure_seats([0, 1, 2, 3, 4, 5], 2000)
    m.stats[2, St.OPEN] = 7
    m.pending = [("hand", None, 3, ()), ("blind", 0, 10, ()), ("show", 1, 0, (c("Ah"), c("Kd")))]
    m.cache_hand = 3
    m.cache = {1: np.linspace(0, 1, 1326), 2: None}
    b = m.encode()
    m2 = Memory.decode(b)
    assert m2.seats == [0, 1, 2, 3, 4, 5] and m2.stats[2, St.OPEN] == 7
    assert m2.pending[2] == ("show", 1, 0, (c("Ah"), c("Kd")))
    assert m2.cache_hand == 3 and abs(m2.cache[1][-1] - 1.0) < 1e-6
    for bad in (b"", b"PB", b"PB\x01\xff", b[:-7], b[:20], os.urandom(300), b"PB" + os.urandom(100)):
        Memory.decode(bad)  # must not raise
    assert len(m.encode(limit=3000)) <= 3000


def test_replay_counts():
    g = get_game()
    m = Memory()
    m.ensure_seats([0, 1, 2, 3], 2000)
    # seat 2 opens, seat 3 3-bets, seat 2 folds; next hand: flop bet and fold
    hand1 = [("hand", None, 1, ()), ("blind", 0, 10, ()), ("blind", 1, 20, ()),
             ("raise", 2, 50, ()), ("raise", 3, 150, ()), ("fold", 0, 0, ()), ("fold", 1, 0, ()),
             ("fold", 2, 0, ()), ("win", 3, 230, ()),
             ("stack", 0, 1990, ()), ("stack", 1, 1980, ()), ("stack", 2, 1950, ()), ("stack", 3, 2080, ())]
    St.replay_hand(hand1, m, hero_seat=0, game=g, starting_stack=2000)
    S = m.stats
    assert S[m.slot(2), St.OPEN_OPP] == 1 and S[m.slot(2), St.OPEN] == 1
    assert S[m.slot(3), St.TB_OPP] == 1 and S[m.slot(3), St.TB] == 1
    assert S[m.slot(2), St.F3B_OPP] == 1 and S[m.slot(2), St.F3B] == 1
    assert S[m.slot(1), St.BVS_OPP] == 0  # a 3-bet happened before the BB acted
    assert m.last_stacks[m.slot(3)] == 2080
    hand2 = [("hand", None, 2, ()), ("blind", 1, 10, ()), ("blind", 2, 20, ()),
             ("call", 3, 20, ()), ("fold", 0, 0, ()), ("call", 1, 10, ()), ("check", 2, 0, ()),
             ("board", None, 0, (c("Ks"), c("7d"), c("2c"))),
             ("check", 1, 0, ()), ("raise", 2, 40, ()), ("fold", 3, 0, ()), ("call", 1, 40, ()),
             ("board", None, 0, (c("9s"),)), ("check", 1, 0, ()), ("check", 2, 0, ()),
             ("board", None, 0, (c("3h"),)), ("raise", 1, 100, ()), ("call", 2, 100, ()),
             ("show", 1, 0, (c("5h"), c("4h"))), ("show", 2, 0, (c("Kd"), c("Qc"))),
             ("win", 2, 340, ())]
    St.replay_hand(hand2, m, hero_seat=0, game=g, starting_stack=2000)
    i3 = m.slot(3)
    assert S[i3, St.LIMP] == 1
    assert S[i3, St.FTB_OPP0 + 0 + 1] == 1 and S[i3, St.FTB0 + 0 + 1] == 1  # 40 into 60 = big bucket
    i1 = m.slot(1)
    assert S[i1, St.RB_SHOWN] == 1 and S[i1, St.RB_BLUFF] == 1  # river bet with 5-high
    assert S[i1, St.WTSD] == 1 and S[i1, St.SAWF] == 1
    o = St.Opp(S[i3], __import__("pb.settings", fromlist=["P"]).P)
    assert 0 < o.fold_to_bet(1, 0.66) < 1


if __name__ == "__main__":
    test_roundtrip_and_garbage(); test_replay_counts()
    print("memory/stats ok")


def test_busted_seat_leaves_the_player_count():
    """The engine reports end-of-hand stacks only for players who still have
    chips; a seat without one has busted and must be zeroed."""
    g = get_game()
    m = Memory()
    m.ensure_seats([0, 1, 2, 3], 2000)
    hand = [("hand", None, 1, ()), ("blind", 0, 10, ()), ("blind", 1, 20, ()),
            ("raise", 2, 2000, ()), ("call", 3, 2000, ()), ("fold", 0, 0, ()), ("fold", 1, 0, ()),
            ("board", None, 0, (c("Ks"), c("7d"), c("2c"), c("9s"), c("3h"))),
            ("show", 2, 0, (c("5h"), c("4h"))), ("show", 3, 0, (c("Kd"), c("Qc"))), ("win", 3, 4030, ()),
            ("stack", 0, 1990, ()), ("stack", 1, 1980, ()), ("stack", 3, 4030, ())]
    St.replay_hand(hand, m, hero_seat=0, game=g, starting_stack=2000)
    assert m.last_stacks[m.slot(2)] == 0 and m.last_stacks[m.slot(3)] == 4030


def test_min_size_showdowns_and_reraise_response():
    """Plan: watch whether an opponent only min-raises strong hands; learn what
    they do when re-raised, at every level."""
    from pb import settings
    g = get_game()
    m = Memory()
    m.ensure_seats([0, 1, 2, 3], 2000)
    # seat 2 opens, seat 3 3-bets, seat 2 min-4-bets (re-raises a re-raise), seat 3 calls;
    # flop: seat 2 bets the minimum, seat 3 calls; checked down; seat 2 shows seven-deuce
    hand = [("hand", None, 1, ()), ("blind", 0, 10, ()), ("blind", 1, 20, ()),
            ("raise", 2, 50, ()), ("raise", 3, 150, ()), ("fold", 0, 0, ()), ("fold", 1, 0, ()),
            ("raise", 2, 250, ()), ("call", 3, 100, ()),
            ("board", None, 0, (c("Ks"), c("9d"), c("2c"))), ("raise", 2, 20, ()), ("call", 3, 20, ()),
            ("board", None, 0, (c("4s"),)), ("check", 2, 0, ()), ("check", 3, 0, ()),
            ("board", None, 0, (c("3h"),)), ("check", 2, 0, ()), ("check", 3, 0, ()),
            ("show", 2, 0, (c("7h"), c("2d"))), ("show", 3, 0, (c("Kd"), c("Qc"))), ("win", 3, 570, ()),
            ("stack", 0, 1990, ()), ("stack", 1, 1980, ()), ("stack", 2, 1730, ()), ("stack", 3, 2300, ())]
    St.replay_hand(hand, m, hero_seat=0, game=g, starting_stack=2000)
    S = m.stats
    i2 = m.slot(2)
    assert S[i2, St.MZ_SHOWN] == 1 and S[i2, St.MZ_WEAK] == 1      # the min-4-bet was seven-deuce: weak
    assert S[i2, St.F3B_OPP] == 1 and S[i2, St.R4B] == 1
    o = St.Opp(S[i2], settings.P)
    assert o.min_size_wide() == 1.0                                # nothing says their min sizes are strong
    f1, c1, r1 = o.response_to_3bet()
    f2, c2, r2 = o.response_to_3bet(level=2)
    fresh = St.Opp(None, settings.P).response_to_3bet(level=2)
    assert r2 > fresh[2] and f2 < fresh[0]                         # one observed re-raise moves the 4-bet level too
    # an opponent who shows only strong hands after minimum-size bets loses the benefit of the doubt
    S[i2, St.MZ_SHOWN] = 30
    S[i2, St.MZ_WEAK] = 0
    assert St.Opp(S[i2], settings.P).min_size_wide() < 0.35


def test_size_points_pooled_fold_rate_and_usual_size():
    from pb import settings
    assert St.size_bucket(0.5, allin=True) == St.ALLIN_BUCKET
    assert St.size_bucket(1.0) == 2 and St.size_bucket(0.33) == 0 and St.size_bucket(1.25) == 3
    S = np.zeros(St.K_STATS, np.int64)
    o = St.Opp(S, settings.P)
    fresh = o.fold_to_bet(3, 0.66)
    # 20 flop folds out of 20 pull the (unseen) river cell toward folding too: pooled across streets
    S[St.FTB_OPP0 + 1] = 20
    S[St.FTB0 + 1] = 20
    o = St.Opp(S, settings.P)
    assert o.fold_to_bet(1, 0.66) > 0.8 and o.fold_to_bet(3, 0.66) > fresh
    # one-size players: size says nothing once one size dominates
    S[St.BSZ0 + 2] = 9
    assert St.Opp(S, settings.P).same_size_share() == 1.0
    S[St.BSZ0 + 2] = 5
    assert St.Opp(S, settings.P).same_size_share() == 0.0       # too few bets seen
    # usual size: prior 0.75 pot, moving toward what they bet
    S[St.BETF_N] = 24
    S[St.BETF_SUM] = 24 * 33
    assert 0.4 < St.Opp(S, settings.P).usual_bet_frac() < 0.6
