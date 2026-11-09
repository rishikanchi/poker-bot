"""The variant PLUG-IN layer: everything that depends on which game is played.

Architecture: one variant-agnostic CORE (view, memory, stats, ranges,
preflop, postflop, icm, actions, brain) + one PLUG-IN per variant. The core
never looks at cards; it only calls the methods of ``VariantPlugin`` below.

``CommunityCardGame`` is the plug-in for the whole community-card family
(hold'em, short deck, Omaha 4/5/6, 3-card games). It is configured from
``settings.VARIANT``, so most build-hour variants need no new code.
A game with a different structure (draw, stud, hi-lo split) needs a new
class implementing the same interface; until then the core runs it in
safe mode (see brain.py).
"""
from __future__ import annotations

import os

import numpy as np

from . import kernels as K
from . import settings

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class VariantPlugin:
    """What the core needs from a variant. Implement these for a new game.

    A *range* is a weight vector over ``self.U`` (the universe of hands,
    shape [M, H]). Percentiles are 0 = strongest, 1 = weakest.
    """
    H: int            # hole cards per player
    M: int            # universe size
    U: np.ndarray     # universe hands [M, H]

    def hero_pre_pct(self, hole):                      # preflop strength percentile
        raise NotImplementedError

    def dead_mask(self, cards):                        # 52-bit mask of cards
        raise NotImplementedError

    def range_interval(self, lo, hi, dead):            # range from a percentile band
        raise NotImplementedError

    def scores(self, board, dead):                     # postflop strength order in [0, 1]
        raise NotImplementedError

    def equity(self, hole, board, weights, n, main_scores, seed):
        """Sampled equity vs ranges -> (vscore, res_hu, res_all) per sample."""
        raise NotImplementedError

    def pre_equity_hu(self, hole, w):                  # heads-up preflop equity
        raise NotImplementedError

    def hand_value(self, hole, board):                 # showdown value (bigger wins)
        raise NotImplementedError


class CommunityCardGame(VariantPlugin):
    def __init__(self, name, H, rule, deck_ranks, board_total=5, short_deck=False,
                 universe_size=1500, seed=11):
        self.name = name
        self.H = H
        self.rule = K.RULE_OMAHA if rule == "omaha" else K.RULE_ANY
        self.board_total = board_total
        self.deck = np.array(sorted(r * 4 + s for r in deck_ranks for s in range(4)), np.int64)
        self.POP, self.HIGH, self.STRAIGHT, self.CATR = K.build_mask_tables(short_deck)
        self.exact = H == 2
        rng = np.random.default_rng(seed)
        if self.exact:
            self.U = K.build_combos(self.deck.tolist())
        else:
            hands = np.array([np.sort(rng.choice(self.deck, H, replace=False))[::-1]
                              for _ in range(universe_size)], np.int64)
            self.U = hands
        self.M = self.U.shape[0]
        self.U_mask = np.zeros(self.M, np.int64)
        for i in range(H):
            self.U_mask |= np.left_shift(np.int64(1), self.U[:, i])
        self.E169 = None
        self.cidx = None
        if self.exact:
            self.cidx = {}
            for k, (a, b) in enumerate(self.U):
                self.cidx[(int(a), int(b))] = k
                self.cidx[(int(b), int(a))] = k
        self._rank_universe()

    # -- preflop ranking ----------------------------------------------------
    def _rank_universe(self):
        holdem = self.exact and self.M == 1326 and self.CATR[5] == 5
        path = os.path.join(_DATA, "holdem_tables.npz")
        if holdem and os.path.exists(path):
            t = np.load(path)
            cls = K.COMBO_CLASS
            self.E169 = t["E169"].astype(np.float64)
            self.combo_class = cls
            self.p_lo = t["pct_lo"].astype(np.float64)[cls]
            self.p_hi = t["pct_hi"].astype(np.float64)[cls]
            self.pre_pct = 0.5 * (self.p_lo + self.p_hi)
            return
        # generic: equity of each universe hand vs a random hand, then rank
        n = 300 if self.exact else 60
        eq = K.preflop_equity_vs_random(self.U, self.H, self.board_total, self.deck, n, 5,
                                        self.rule, self.POP, self.HIGH, self.STRAIGHT, self.CATR)
        if self.exact:
            score = eq
        else:
            # fit a cheap feature score to the noisy equities so hero's own hand
            # can be scored in microseconds at decision time
            X = self.features(self.U)
            beta, *_ = np.linalg.lstsq(X, eq, rcond=None)
            self.beta = beta
            score = X @ beta
        order = np.argsort(-score, kind="stable")
        self.score_sorted = -np.sort(-score)
        ranks = np.empty(self.M)
        ranks[order] = np.arange(self.M)
        self.p_lo = ranks / self.M
        self.p_hi = (ranks + 1) / self.M
        self.pre_pct = 0.5 * (self.p_lo + self.p_hi)
        self.pre_score = score
        self._build_pre_table()

    def features(self, hands):
        """Variant-agnostic preflop features for hands with H cards."""
        hands = np.atleast_2d(hands)
        r = hands >> 2
        s = hands & 3
        n = hands.shape[0]
        H = hands.shape[1]
        rank_cnt = np.zeros((n, 13))
        suit_cnt = np.zeros((n, 4))
        for i in range(H):
            rank_cnt[np.arange(n), r[:, i]] += 1
            suit_cnt[np.arange(n), s[:, i]] += 1
        pairs = (rank_cnt >= 2).sum(1)
        trips = (rank_cnt >= 3).sum(1)
        suited_groups = (suit_cnt >= 2).sum(1)
        flush_heavy = (suit_cnt >= 3).sum(1)
        ace = (r == 12).any(1).astype(float)
        hi_sum = np.sort(r, 1)[:, ::-1][:, : min(H, 3)].sum(1) / 36.0
        pair_rank = (rank_cnt >= 2) @ np.arange(13) / 12.0
        uniq = rank_cnt > 0
        conn = (uniq[:, :12] & uniq[:, 1:]).sum(1).astype(float)
        nut_suit = np.zeros(n)
        for i in range(H):
            is_ace = r[:, i] == 12
            nut_suit += is_ace * (suit_cnt[np.arange(n), s[:, i]] >= 2)
        return np.column_stack([np.ones(n), hi_sum, ace, pairs, pair_rank, trips, suited_groups,
                                flush_heavy, conn, nut_suit])

    def hero_pre_pct(self, hole):
        if self.exact:
            k = self.cidx.get((hole[0], hole[1]))
            return float(self.pre_pct[k]) if k is not None else 0.5
        sc = float((self.features(np.array([hole], np.int64)) @ self.beta)[0])
        # position of this score within the universe's sorted scores
        idx = np.searchsorted(-self.score_sorted, -sc)
        return float(min(max(idx / self.M, 0.0), 1.0))

    # -- ranges ---------------------------------------------------------------
    def dead_mask(self, cards):
        m = 0
        for c in cards:
            m |= 1 << int(c)
        return m

    def live(self, dead):
        return (self.U_mask & np.int64(dead)) == 0

    def range_interval(self, lo, hi, dead):
        """Weights of universe hands whose preflop percentile is in [lo, hi)."""
        lo = max(0.0, lo)
        hi = min(1.0, hi)
        w = np.clip((np.minimum(hi, self.p_hi) - np.maximum(lo, self.p_lo))
                    / np.maximum(self.p_hi - self.p_lo, 1e-9), 0.0, 1.0)
        return w * self.live(dead)

    # -- postflop -----------------------------------------------------------------
    def scores(self, board, dead):
        b = np.asarray(board, np.int64)
        return K.combo_scores(self.U, self.H, b, b.shape[0], np.int64(dead), self.rule,
                              self.POP, self.HIGH, self.STRAIGHT, self.CATR)

    def strengths(self, board, dead):
        b = np.asarray(board, np.int64)
        return K.strengths_all(self.U, self.H, b, b.shape[0], np.int64(dead), self.rule,
                               self.POP, self.HIGH, self.STRAIGHT, self.CATR)

    def hand_value(self, hole, board):
        h = np.asarray(hole, np.int64)
        b = np.asarray(board, np.int64)
        buf = np.zeros(K.MAXC, np.int64)
        return int(K.eval_rule(h, self.H, b, b.shape[0], self.rule, buf,
                               self.POP, self.HIGH, self.STRAIGHT, self.CATR))

    def equity(self, hole, board, weights, n, main_scores, seed):
        """Sampled equity vs the ranges in ``weights`` (list of [M] arrays).

        Returns (vscore, res_hu, res_all) per sample; see kernels.mc_equity.
        """
        cdfs = np.cumsum(np.vstack(weights), axis=1)
        h = np.asarray(hole, np.int64)
        b = np.asarray(board, np.int64)
        return K.mc_equity(h, self.H, b, b.shape[0], self.board_total, self.deck, self.U, cdfs,
                           cdfs.shape[0], int(n), main_scores, int(seed) & 0x7FFFFFFF, self.rule,
                           self.POP, self.HIGH, self.STRAIGHT, self.CATR)

    # -- preflop equity vs a percentile band ------------------------------------
    PRE_XS = np.array([0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0])
    PRE_BINS = 20

    def _build_pre_table(self):
        self.pre_T = K.preflop_table(self.U, self.H, self.pre_pct, self.PRE_XS, self.PRE_BINS,
                                     150, self.board_total, self.deck, 9, self.rule,
                                     self.POP, self.HIGH, self.STRAIGHT, self.CATR)

    def _top_eq(self, row, x):
        xs = self.PRE_XS
        if x <= xs[0]:
            return row[0]
        return float(np.interp(x, xs, row))

    def pre_equity(self, hole, lo, hi):
        """Heads-up all-in preflop equity of ``hole`` vs the band [lo, hi)."""
        hi = min(max(hi, lo + 1e-3), 1.0)
        if self.E169 is not None:
            w = self.range_interval(lo, hi, self.dead_mask(hole))
            return self.pre_equity_hu(hole, w)
        if getattr(self, "pre_T", None) is None:
            self._build_pre_table()
        p = self.hero_pre_pct(hole) * self.PRE_BINS - 0.5
        b0 = int(min(max(np.floor(p), 0), self.PRE_BINS - 1))
        b1 = min(b0 + 1, self.PRE_BINS - 1)
        t = min(max(p - b0, 0.0), 1.0)
        row = (1 - t) * self.pre_T[b0] + t * self.pre_T[b1]
        fh, fl = self._top_eq(row, hi), self._top_eq(row, lo) if lo > 0 else 0.0
        eq = (hi * fh - lo * fl) / (hi - lo) if lo > 0 else fh
        return float(min(max(eq, 0.0), 1.0))

    def pre_equity_hu(self, hole, w, n=300, seed=1):
        """Heads-up preflop equity vs range weights w (card removal included)."""
        tot = w.sum()
        if tot <= 0:
            return 0.5
        if self.E169 is not None:
            k = self.cidx.get((hole[0], hole[1]))
            if k is not None:
                row = self.E169[self.combo_class[k]][self.combo_class]
                return float((row * w).sum() / tot)
        _, hu, _ = self.equity(hole, [], [w], n, np.zeros(self.M), seed)
        return float(hu.mean())


Game = CommunityCardGame  # short alias

_GAMES = {}


def get_game(H=None):
    """Return the configured game (built once per process, at import)."""
    cfg = settings.VARIANT
    if H is None:
        H = cfg["hole_cards"]
    key = (H, cfg["rule"], tuple(cfg["deck_ranks"]), cfg["short_deck"])
    g = _GAMES.get(key)
    if g is None:
        g = Game(cfg["name"], H, cfg["rule"], cfg["deck_ranks"], cfg["board_total"],
                 cfg["short_deck"], cfg.get("universe_size", 1500))
        _GAMES[key] = g
    return g


def warmup():
    """Build the configured game and compile every kernel it uses."""
    g = get_game()
    rng = np.random.default_rng(0)
    cards = rng.choice(g.deck, g.H + 5, replace=False).astype(np.int64)
    hole, board = list(cards[: g.H]), list(cards[g.H:])
    dead = g.dead_mask(hole + board[:3])
    s = g.scores(board[:3], dead)
    g.scores(board, g.dead_mask(hole + board))
    w = g.range_interval(0.0, 0.5, dead)
    g.equity(hole, board[:3], [w], 16, s, 1)
    g.equity(hole, board, [w, w], 16, s, 1)
    g.pre_equity(hole, 0.0, 0.3)
    g.pre_equity(hole, 0.1, 0.4)
    g.hero_pre_pct(hole)
    K.icm_values([100.0, 50.0, 25.0], [50.0, 30.0, 20.0])
    return g
