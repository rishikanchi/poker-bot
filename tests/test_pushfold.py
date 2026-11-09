"""Push/fold charts: solver sanity vs known heads-up Nash, loader behaviour."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_pushfold as BP  # noqa: E402
from pb import pushfold  # noqa: E402


def test_heads_up_close_to_nash():
    m = BP.holdem_model()
    j, cb, co, _ = BP.solve_spot(m, 10.0, 1)
    # heads-up Nash at 10bb (no ante): SB jams ~58%, BB calls ~37%
    assert 0.52 <= j <= 0.65, j
    assert 0.32 <= cb <= 0.45, cb
    j2, cb2, _, _ = BP.solve_spot(m, 20.0, 1)
    assert j2 < j and cb2 < cb  # deeper stacks -> tighter
    assert BP.residual(m, 10.0, 1, j, cb, co) < 0.02


def test_more_players_behind_jam_tighter():
    m = BP.holdem_model()
    widths = [BP.solve_spot(m, 10.0, k)[0] for k in (1, 2, 4, 7)]
    assert widths == sorted(widths, reverse=True), widths


def test_ante_widens_jams():
    m = BP.holdem_model()
    assert BP.solve_spot(m, 10.0, 4, ante=0.125)[0] > BP.solve_spot(m, 10.0, 4)[0]


def test_loader_interpolates_and_clamps():
    t = pushfold.table()
    assert t, "pushfold_holdem.npz missing: run tools/build_pushfold.py"
    a = pushfold.jam_width(10, 1)
    b = pushfold.jam_width(12, 1)
    mid = pushfold.jam_width(11, 1)
    assert min(a, b) <= mid <= max(a, b)
    assert 0 < pushfold.jam_width(1, 9) <= 1  # clamped to the grid
    assert pushfold.jam_width(60, 3) <= pushfold.jam_width(25, 3)
    assert pushfold.call_width(10, True, 1) > pushfold.call_width(10, False, 1)


def test_fallback_without_file(tmp_path=None):
    saved = pushfold._T
    try:
        pushfold._T = False
        assert abs(pushfold.jam_width(10, 2) - 0.32) < 1e-9
        assert abs(pushfold.call_width(10, True) - 0.22 * 1.3) < 1e-9
    finally:
        pushfold._T = saved


if __name__ == "__main__":
    test_heads_up_close_to_nash(); test_more_players_behind_jam_tighter(); test_ante_widens_jams()
    test_loader_interpolates_and_clamps(); test_fallback_without_file()
    print("pushfold ok")
