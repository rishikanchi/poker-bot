"""ICM helpers: turn chip outcomes into tournament points.

ICM (Independent Chip Model) estimates each player's expected finishing
points from the chip stacks, using the table's actual ``payouts``. Losing
chips hurts more than winning the same chips helps.

Plan rules implemented here:
  * every all-in decision is compared in exact ICM points at the real stacks;
  * for other bets and calls, ICM gives a *risk premium*: the extra equity
    needed because the chips we can lose are worth more than the chips we
    can win. It is computed for the chips actually at stake in that option.
"""
from __future__ import annotations

import numpy as np

from .kernels import hero_values


class IcmCtx:
    """ICM values for the outcomes of the current hand (one per decision).

    Chips are accounted the way side pots work: if hero's total commitment in
    the hand is H, hero can win at most H from every other player; anything a
    player has put in beyond H is never hero's (it goes back to, or stays
    with, the opponent hero is up against).
    """

    def __init__(self, v):
        self.seats = list(v.alive)
        self.index = {s: i for i, s in enumerate(self.seats)}
        self.hi = self.index[v.seat]
        n = len(self.seats)
        pay = list(v.payouts[:n]) + [0.0] * max(0, n - len(v.payouts))
        self.pay = np.array(pay, np.float64)
        self.start = np.array([v.players[s].start for s in self.seats], np.float64)
        self.com = np.array([v.players[s].committed for s in self.seats], np.float64)
        self.base = self.start - self.com         # chips each player has behind right now
        self.pot = float(self.com.sum())
        self.hero_seat = v.seat

    def _row_win_lose(self, vi, risk, v_add):
        """End-of-hand stacks if hero puts ``risk`` more in, the villain
        ``v_add`` more, everyone else is done, and hero wins / loses."""
        hi = self.hi
        C = self.com.copy()
        if vi >= 0:
            C[vi] += v_add
        H = self.com[hi] + risk
        C[hi] = 0.0
        H = min(H, max(float(C.max()), self.com[hi]))  # chips nobody can match come straight back
        C[hi] = H
        main = np.minimum(C, H)                    # what each player has in hero's pot
        pot_main = float(main.sum())
        side = float((C - main).sum())             # beyond hero's reach: the villain's
        win = self.start - C
        lose = self.start - C
        win[hi] = self.start[hi] - H + pot_main
        lose[hi] = self.start[hi] - H
        if vi >= 0:
            win[vi] += side
            lose[vi] += side + pot_main
        return win, lose, pot_main, H - self.com[hi]

    def outcomes(self, villain, pairs):
        """ICM points for hero in each outcome against ``villain``.

        pairs: list of (risk, v_add): hero puts ``risk`` more chips in and the
        villain ``v_add`` more. Returns (fold, all_fold, win[], lose[], pots[]):
          fold      hero gives up now, the villain takes the pot
          all_fold  everyone folds to hero, who takes what he can win of the pot
          win[j]    both put the chips in and hero wins
          lose[j]   both put the chips in and hero loses
          pots[j]   the pot hero is playing for in option j (chips)
        """
        hi = self.hi
        vi = self.index.get(villain, -1)
        m = len(pairs)
        rows = np.empty((2 * m + 2, len(self.seats)))
        rows[0] = self.base
        if vi >= 0:
            rows[0, vi] += self.pot
        cap = np.minimum(self.com, self.start[hi])      # hero wins at most his own stack from each
        cap[hi] = self.com[hi]
        rows[1] = self.start - self.com + (self.com - cap)
        rows[1, hi] = self.base[hi] + float(cap.sum())
        pots = []
        self.risks = []          # chips hero really has at stake in each option
        for j, (risk, v_add) in enumerate(pairs):
            risk = min(max(float(risk), 0.0), self.base[hi])
            v_add = min(max(float(v_add), 0.0), self.base[vi]) if vi >= 0 else 0.0
            rows[2 + 2 * j], rows[3 + 2 * j], pm, at_risk = self._row_win_lose(vi, risk, v_add)
            pots.append(pm)
            self.risks.append(at_risk)
        vals = hero_values(rows, self.pay, hi)
        return float(vals[0]), float(vals[1]), vals[2::2], vals[3::2], pots

    def allin_pair(self, villain):
        """(risk, v_add) when hero and the villain put in everything they can lose to each other."""
        vi = self.index.get(villain, -1)
        if vi < 0:
            return self.base[self.hi], 0.0
        x = min(self.start[self.hi], self.start[vi])
        return max(x - self.com[self.hi], 0.0), max(x - self.com[vi], 0.0)

    def allin(self, villain):
        """(win, lose, fold, all_fold) for getting all-in against the villain."""
        f, af, w, l, _ = self.outcomes(villain, [self.allin_pair(villain)])
        return float(w[0]), float(l[0]), f, af

    def premiums(self, villain, pairs, scale=1.0):
        """Risk premium (extra equity needed) for each (risk, v_add) option.

        chips: break-even equity = risk / the pot hero plays for.
        ICM:   break-even equity = (V(fold) - V(lose)) / (V(win) - V(lose)).
        premium = max(0, ICM break-even - chip break-even) * scale.
        """
        f, _, w, l, pots = self.outcomes(villain, pairs)
        out = []
        for j in range(len(pairs)):
            chip_be = self.risks[j] / pots[j] if pots[j] > 0 else 0.0
            icm_be = (f - l[j]) / (w[j] - l[j]) if w[j] > l[j] else 1.0
            out.append(max(0.0, min(1.0, icm_be) - chip_be) * scale)
        return out


# ---- thin wrappers kept for callers that need a single number -----------------
def allin_values(v, villain):
    """(win, lose, fold) ICM points for an all-in against the villain."""
    w, l, f, _ = IcmCtx(v).allin(villain)
    return w, l, f


def value_all_fold(v):
    return IcmCtx(v).outcomes(None, [])[1]


def risk_premium(v, villain, scale=1.0):
    """All-in risk premium against the villain (>= 0)."""
    try:
        ctx = IcmCtx(v)
        return ctx.premiums(villain, [ctx.allin_pair(villain)], scale)[0]
    except Exception:  # noqa: BLE001
        return 0.0
