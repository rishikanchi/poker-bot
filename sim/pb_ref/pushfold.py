"""Solved push/fold widths (from tools/build_pushfold.py), looked up at runtime.

  jam_width(eff_bb, n_behind)          share of hands a first-in shover jams with
                                       n_behind players still to act (1 = SB vs BB)
  call_width(eff_bb, is_bb, k_shover)  share of hands a player calls a jam with;
                                       k_shover = players that were behind the
                                       shover when it jammed (sets how wide it jams)

Widths are fractions of all hands (0..1), interpolated linearly in the stack
and clamped to the solved grid. If the data file for the configured variant
is missing, the old hand-written formulas are used (copied here on purpose:
ranges.py may call into this module, so we must not import it).
"""
from __future__ import annotations

import os

import numpy as np

from . import settings

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_T = None  # loaded table, or False when there is no file

_PUSH_W = ((5, 0.55), (8, 0.40), (10, 0.32), (13, 0.25), (16, 0.20), (20, 0.15), (30, 0.10),
           (50, 0.06), (100, 0.04))


def _fallback_jam(eff_bb):
    return float(np.interp(eff_bb, [a for a, _ in _PUSH_W], [b for _, b in _PUSH_W]))


def _fallback_call(eff_bb, is_bb):
    w = float(np.interp(eff_bb, (5, 8, 10, 15, 20, 30, 50), (0.45, 0.30, 0.22, 0.14, 0.11, 0.08, 0.05)))
    return w * (1.3 if is_bb else 1.0)


def _load():
    global _T
    name = str(settings.VARIANT.get("name", "holdem"))
    path = os.path.join(_DATA, f"pushfold_{name}.npz")
    try:
        t = np.load(path)
        _T = {k: np.asarray(t[k], np.float64) for k in ("S", "K", "jam", "call_bb", "call_other")}
    except (OSError, KeyError, ValueError):
        _T = False
    return _T


def table():
    return _T if _T is not None else _load()


def _lookup(arr, eff_bb, k):
    T = table()
    S = T["S"]
    ks = T["K"]
    ki = int(min(max(round(k), ks[0]), ks[-1]) - ks[0])
    return float(np.interp(eff_bb, S, arr[ki]))


def jam_width(eff_bb, n_behind, P=None):
    """Equilibrium-style first-in jam width for this stack and seat."""
    T = table()
    if not T or (P is not None and not P.get("use_pushfold_charts", True)):
        return _fallback_jam(eff_bb)
    if eff_bb > T["S"][-1]:  # beyond the chart: keep shrinking like the old prior
        return min(_lookup(T["jam"], T["S"][-1], n_behind), _fallback_jam(eff_bb))
    return _lookup(T["jam"], eff_bb, n_behind)


def call_width(eff_bb, is_bb, k_shover=1, P=None):
    """Equilibrium-style width for calling an all-in of eff_bb big blinds."""
    T = table()
    if not T or (P is not None and not P.get("use_pushfold_charts", True)):
        return _fallback_call(eff_bb, is_bb)
    arr = T["call_bb"] if is_bb else T["call_other"]
    if eff_bb > T["S"][-1]:
        return min(_lookup(arr, T["S"][-1], k_shover), _fallback_call(eff_bb, is_bb))
    return _lookup(arr, eff_bb, k_shover)
