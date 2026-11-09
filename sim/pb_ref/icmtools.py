"""ICM helpers: turn chip outcomes into tournament points.

ICM (Independent Chip Model) estimates each player's expected finishing
points from the chip stacks, using the table's actual ``payouts``. Losing
chips hurts more than winning the same chips helps, so all-in decisions are
compared in ICM points instead of chips.
"""
from __future__ import annotations

import numpy as np

from .kernels import icm


def _payouts(v):
    n = len(v.alive)
    pay = list(v.payouts[:n]) + [0.0] * max(0, n - len(v.payouts))
    return np.array(pay, np.float64)


def base_stacks(v):
    """Chips each alive player would have if the hand ended now with nobody
    winning the pot yet: start - committed."""
    return {s: v.players[s].start - v.players[s].committed for s in v.alive}


def hero_value(v, finals):
    """Hero's expected points if the hand ends with these chip counts.

    Players who end the hand with 0 chips bust now and take the bottom places
    among the players alive at the start of the hand (busting 2nd of 2 still
    pays 2nd prize). Everyone with chips splits the top places by ICM.
    """
    seats = v.alive
    st = np.array([max(finals.get(s, 0), 0) for s in seats], np.float64)
    pay = _payouts(v)
    hero_i = seats.index(v.seat)
    k = int((st > 0).sum())
    if st[hero_i] <= 0:
        n_bust = len(seats) - k
        return float(pay[k:k + n_bust].mean()) if n_bust > 0 else 0.0
    vals = icm(st, pay[:max(k, 1)])
    return float(vals[hero_i])


def allin_values(v, villain, pot_winner_if_fold=None):
    """ICM value for hero: (win vs villain, lose vs villain, hero folds now).

    Contest: hero and villain both get their whole remaining stacks in;
    everyone else's committed chips are dead money for the winner.
    """
    hero = v.players[v.seat]
    vil = v.players[villain]
    base = base_stacks(v)
    pot = sum(v.players[s].committed for s in v.alive)
    dead = pot - hero.committed - vil.committed
    x = min(hero.start, vil.start)  # chips each side can lose to the other
    win = dict(base)
    win[v.seat] = hero.start + x + dead
    win[villain] = vil.start - x
    lose = dict(base)
    lose[v.seat] = hero.start - x
    lose[villain] = vil.start + x + dead
    fold = dict(base)
    w = pot_winner_if_fold if pot_winner_if_fold is not None else villain
    fold[w] = fold.get(w, 0) + pot
    return hero_value(v, win), hero_value(v, lose), hero_value(v, fold)


def value_all_fold(v):
    """Hero wins the pot uncontested."""
    base = base_stacks(v)
    pot = sum(v.players[s].committed for s in v.alive)
    base[v.seat] += pot
    return hero_value(v, base)


def value_fold(v, winner):
    base = base_stacks(v)
    pot = sum(v.players[s].committed for s in v.alive)
    if winner in base:
        base[winner] += pot
    return hero_value(v, base)


def breakeven_equity(v, villain):
    """Equity needed to get it all in vs villain: (ICM, chips)."""
    w, l, f = allin_values(v, villain)
    icm_be = (f - l) / (w - l) if w > l else 1.0
    hero = v.players[v.seat]
    vil = v.players[villain]
    x = min(hero.start, vil.start)
    pot = sum(v.players[s].committed for s in v.alive)
    risk = x - hero.committed
    final = pot + (x - hero.committed) + (x - vil.committed)
    chip_be = risk / final if final > 0 else 0.5
    return float(np.clip(icm_be, 0, 1)), float(np.clip(chip_be, 0, 1))


def risk_premium(v, villain, scale=1.0):
    """Extra equity ICM demands for stacking off vs ``villain`` (>= 0)."""
    try:
        icm_be, chip_be = breakeven_equity(v, villain)
    except Exception:  # noqa: BLE001
        return 0.0
    return max(0.0, icm_be - chip_be) * scale
