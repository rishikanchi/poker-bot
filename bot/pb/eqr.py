"""Measured equity realization (EQR) tables, loaded once at import.

EQR = how much of its raw all-in equity a hand actually wins once there is
betting left:  R = E[chips won from this street on] / (equity * pot).
R > 1: the hand wins more than its equity (gets paid when it hits, bluffs).
R < 1: it gets bluffed off or pays off with second-best hands.

The tables come from ``tools/measure_eqr.py`` (simulation, control variate)
and live in ``bot/data/eqr_<variant name>.json``. Every lookup returns None
when there is no table or the bucket has too few samples, so callers keep
their prior. No file I/O happens after import.

API
  lookup_post(street, ip, made_pct, has_draw, spr) -> float | None
      street: 1 = flop, 2 = turn (river is always 1.0, returns 1.0)
      ip: hero acts last postflop (heads-up)
      made_pct: hero's made-hand percentile among live combos on this board
                (0 = weakest .. 1 = strongest; see hero_made_pct)
      has_draw: flush draw or open-ender (see has_draw)
      spr: effective stack / pot at the start of the street
  lookup_pre(ip, hole) -> float | None
      preflop EQR for a heads-up flop after a raise and a call, by position
      and simple hand features (see pre_key)

Bucket helpers (shared by the measurement tool so definitions never drift):
  made_bucket(pct), spr_bucket(spr), pre_key(hole), has_draw(hole, board, omaha),
  hero_made_pct(game, hole, board)
"""
from __future__ import annotations

import json
import os

MIN_N = 40  # buckets with fewer samples are ignored (caller keeps its prior)

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MADE_EDGES = (0.35, 0.65, 0.85)
SPR_EDGES = (1.5, 4.0)


def made_bucket(pct):
    for i, e in enumerate(MADE_EDGES):
        if pct < e:
            return i
    return len(MADE_EDGES)


def spr_bucket(spr):
    for i, e in enumerate(SPR_EDGES):
        if spr < e:
            return i
    return len(SPR_EDGES)


def pre_key(hole):
    """Preflop feature class, same flags as preflop._features_eqr."""
    r = sorted((c >> 2 for c in hole), reverse=True)
    s = [c & 3 for c in hole]
    gaps = [r[i] - r[i + 1] for i in range(len(r) - 1)]
    pair = any(g == 0 for g in gaps)
    suited = len(set(s)) < len(s)
    conn = any(0 < g <= 2 for g in gaps)
    junk = (not suited) and (not pair) and min(gaps) > 3 and r[0] < 12
    if pair:
        return "pair"
    if suited and conn:
        return "suited_conn"
    if suited:
        return "suited"
    if conn:
        return "conn"
    if junk:
        return "junk"
    return "offsuit"


def _straight(m):
    for t in range(12, 3, -1):
        need = 0b11111 << (t - 4)
        if m & need == need:
            return True
    return m & ((1 << 12) | 0b1111) == ((1 << 12) | 0b1111)


def has_draw(hole, board, omaha=False):
    """Flush draw or open-ended straight draw (same rule as postflop.hero_draw_flags)."""
    if len(board) >= 5 or len(board) < 3:
        return False
    hs = [0, 0, 0, 0]
    bs = [0, 0, 0, 0]
    for c in hole:
        hs[c & 3] += 1
    for c in board:
        bs[c & 3] += 1
    for s in range(4):
        cnt = (min(hs[s], 2) + min(bs[s], 3)) if omaha else hs[s] + bs[s]
        if hs[s] > 0 and cnt == 4:
            return True
    bm = 0
    for c in board:
        bm |= 1 << (c >> 2)
    hm = bm
    for c in hole:
        hm |= 1 << (c >> 2)
    if _straight(hm):
        return False
    outs = 0
    for r in range(13):
        b = 1 << r
        if not hm & b and _straight(hm | b) and not _straight(bm | b):
            outs += 1
    return outs >= 2


def hero_made_pct(game, hole, board):
    """Percentile of hero's current made hand among live opponent combos."""
    vals = game.strengths(board, game.dead_mask(list(hole) + list(board)))
    hv = game.hand_value(hole, board)
    live = vals[vals >= 0]
    if live.size == 0:
        return 0.5
    return float(((live < hv).sum() + 0.5 * (live == hv).sum()) / live.size)


def post_key(street, ip, made_b, draw, spr_b):
    return f"{int(street)}|{int(bool(ip))}|{int(made_b)}|{int(bool(draw))}|{int(spr_b)}"


def pre_bucket_key(ip, key):
    return f"{int(bool(ip))}|{key}"


_TABLE = {"post": {}, "pre": {}}


def load(path=None):
    """(Re)load a table. Default: data/eqr_<settings.VARIANT name>.json."""
    global _TABLE
    if path is None:
        try:
            from . import settings
            name = settings.VARIANT.get("name", "holdem")
        except Exception:  # noqa: BLE001
            name = "holdem"
        path = os.path.join(_DATA, f"eqr_{name}.json")
    t = {"post": {}, "pre": {}}
    try:
        with open(path) as fh:
            raw = json.load(fh)
        for sec in ("post", "pre"):
            for k, rec in raw.get(sec, {}).items():
                if rec.get("n", 0) >= MIN_N and rec.get("R") is not None:
                    t[sec][k] = float(rec["R"])
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    _TABLE = t
    return t


def lookup_post(street, ip, made_pct, has_draw, spr):
    if street >= 3:
        return 1.0
    return _TABLE["post"].get(post_key(street, ip, made_bucket(made_pct), has_draw, spr_bucket(spr)))


def lookup_pre(ip, hole):
    return _TABLE["pre"].get(pre_bucket_key(ip, pre_key(hole)))


load()
