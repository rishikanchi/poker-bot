"""EQR measurement helpers and loader."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "stub_sdk"), os.path.join(ROOT, "bot"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from pb import eqr as E  # noqa: E402
from pb.game import get_game  # noqa: E402


def c(s):
    return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])


def test_buckets():
    assert [E.made_bucket(x) for x in (0.0, 0.34, 0.35, 0.7, 0.9)] == [0, 0, 1, 2, 3]
    assert [E.spr_bucket(x) for x in (0.5, 1.5, 3.9, 4.0, 12)] == [0, 1, 1, 2, 2]
    assert E.pre_key([c("Ah"), c("Ad")]) == "pair"
    assert E.pre_key([c("Jh"), c("Th")]) == "suited_conn"
    assert E.pre_key([c("Kh"), c("4h")]) == "suited"
    assert E.pre_key([c("9h"), c("8d")]) == "conn"
    assert E.pre_key([c("9h"), c("3d")]) == "junk"
    assert E.pre_key([c("Ah"), c("3d")]) == "offsuit"
    assert E.has_draw([c("Ah"), c("2h")], [c("Kh"), c("7h"), c("9c")])
    assert E.has_draw([c("9h"), c("8d")], [c("7c"), c("6s"), c("2d")])
    assert not E.has_draw([c("9h"), c("2d")], [c("7c"), c("6s"), c("Kd")])


def test_made_pct():
    g = get_game()
    nuts = E.hero_made_pct(g, [c("Ah"), c("Ad")], [c("As"), c("Kd"), c("7c")])
    air = E.hero_made_pct(g, [c("3h"), c("2d")], [c("As"), c("Kd"), c("9c")])
    assert nuts > 0.95 and air < 0.1


def test_exact_equity():
    from tools.measure_eqr import exact_equity
    g = get_game()
    h = np.array([c("Ah"), c("Ad")], np.int64)
    v = np.array([c("Kh"), c("Kd")], np.int64)
    b = np.array([c("2c"), c("7s"), c("9d")], np.int64)
    e = exact_equity(h, v, 2, b, 3, g.deck, 5, g.rule, g.POP, g.HIGH, g.STRAIGHT, g.CATR)
    assert 0.88 < e < 0.93, e  # AA vs KK on a dry flop ~ 0.91
    b4 = np.array([c("2c"), c("7s"), c("9d"), c("Kc")], np.int64)
    e4 = exact_equity(h, v, 2, b4, 4, g.deck, 5, g.rule, g.POP, g.HIGH, g.STRAIGHT, g.CATR)
    assert e4 < 0.1  # KK hit a set; AA has 2 outs


def test_measure_hand_accounting():
    """X / pot / position from a hand-built record."""
    from poker_agent_sdk import BlindLevel, Event
    from sim.engine import HandRecord
    from tools.measure_eqr import measure_hand
    g = get_game()
    ev = [Event("hand", None, 1), Event("blind", 0, 10), Event("blind", 1, 20),
          Event("raise", 2, 50), Event("fold", 0), Event("call", 1, 30),
          Event("board", None, 0, ("2c", "7s", "9d")), Event("check", 1), Event("raise", 2, 60),
          Event("fold", 1), Event("win", 2, 230)]
    rec = HandRecord(hand_number=1, button=0, sb=0, bb=1, blinds=BlindLevel(10, 20),
                     stacks_start=[1000, 1000, 1000], hole_cards={0: ("3h", "4h"), 1: ("Kh", "Kd"), 2: ("Ah", "Ad")},
                     board=("2c", "7s", "9d"), events=ev, stacks_end=[990, 950, 1060], busted=[])
    post, pre = measure_hand(rec, {2}, g, np.random.default_rng(0))
    assert len(post) == 1 and len(pre) == 1
    st, ip, mb, dr, sb, X, pot, eq, s = post[0]
    assert st == 1 and pot == 110 and X == 1060 - (1000 - 50)  # +110: won the pot
    assert ip == 1  # seat 2 acts after seat 1 postflop (button is seat 0)
    assert 0.88 < eq < 0.93 and mb == 3


def test_loader(tmp_path=None):
    import tempfile
    d = tempfile.mkdtemp()
    path = os.path.join(d, "eqr_test.json")
    key = E.post_key(1, True, 2, False, 2)
    with open(path, "w") as fh:
        json.dump({"post": {key: {"R": 1.07, "n": 500}, E.post_key(1, False, 0, False, 2): {"R": 0.5, "n": 3}},
                   "pre": {E.pre_bucket_key(True, "pair"): {"R": 1.12, "n": 200}}}, fh)
    E.load(path)
    assert E.lookup_post(1, True, 0.7, False, 6.0) == 1.07
    assert E.lookup_post(1, False, 0.1, False, 6.0) is None  # too few samples
    assert E.lookup_post(3, False, 0.1, False, 6.0) == 1.0
    assert E.lookup_pre(True, [c("9h"), c("9d")]) == 1.12
    assert E.lookup_pre(False, [c("9h"), c("9d")]) is None
    E.load(os.path.join(d, "missing.json"))
    assert E.lookup_post(1, True, 0.7, False, 6.0) is None
    E.load()  # restore the default table


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
    print("eqr tests ok")
