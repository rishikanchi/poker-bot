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

    def band_probs(self, edges, dead):
        """P(hand in [edges[i], edges[i+1])) for a hand that avoids the dead
        cards. This is how blockers enter: holding an ace makes the top of
        an opponent's range less likely."""
        live = self.live(dead)
        tot = float(live.sum())
        if tot <= 0:
            return [max(b - a, 0.0) for a, b in zip(edges[:-1], edges[1:])]
        span = np.maximum(self.p_hi - self.p_lo, 1e-9)
        below = [float((np.clip((min(max(e, 0.0), 1.0) - self.p_lo) / span, 0.0, 1.0) * live).sum()) / tot
                 for e in edges]
        return [max(b - a, 0.0) for a, b in zip(below[:-1], below[1:])]

    def pre_equity_multi(self, hole, bands, seed=1, n=200):
        """Preflop equity against several percentile bands at once.

        Opponents are sampled jointly (no shared cards) and the board is dealt,
        so we count only the runouts where we beat everyone (plan: "do it per
        runout", never the product of heads-up equities). Sampled games get fewer
        runouts because each costs more; the product is only the fallback for an
        empty band."""
        if len(bands) == 1:
            return self.pre_equity(hole, bands[0][0], bands[0][1])
        dead = self.dead_mask(hole)
        ws = [self.range_interval(lo, hi, dead) for lo, hi in bands]
        if all(w.sum() > 0 for w in ws):
            if not self.exact:
                n = min(n, self.PRE_MULTI_SAMPLES)
            _, _, al = self.equity(hole, [], ws, n, seed)
            return float(al.mean())
        eq = 1.0
        for lo, hi in bands:
            eq *= self.pre_equity(hole, lo, hi)
        return eq

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
        self.score_coef = np.zeros((2, K.NF))   # fitted strength weights (flop, turn); zeros = none
        self._load_score_fit()
        self.E169 = None
        self.cidx = None
        if self.exact:
            self.cidx = {}
            for k, (a, b) in enumerate(self.U):
                self.cidx[(int(a), int(b))] = k
                self.cidx[(int(b), int(a))] = k
        self._rank_universe()

    def _load_score_fit(self):
        """Fitted weights for the flop/turn strength score (tools/fit_scores.py)."""
        path = os.path.join(_DATA, f"scorefit_{self.name}.npz")
        try:
            t = np.load(path)
            c = np.asarray(t["coef"], np.float64)
            if c.shape == (2, K.NF):
                self.score_coef = c
        except (OSError, KeyError, ValueError):
            pass

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
        # generic (plan): rank hands by equity against a mid-strength (~25%)
        # range. First pass vs a random hand gives a ranking to define "top
        # 25%"; then two passes vs the top quarter of the current ranking.
        n = 300 if self.exact else 60
        eq = K.preflop_equity_vs_random(self.U, self.H, self.board_total, self.deck, n, 5,
                                        self.rule, self.POP, self.HIGH, self.STRAIGHT, self.CATR)
        X = None if self.exact else self.pre_features(self.U)
        score = eq
        for it in range(3):
            if not self.exact:
                # fit a cheap feature score to the noisy equities so hero's own
                # hand can be scored in microseconds at decision time (plan:
                # "a scoring function ... fit its weights to sampled equities")
                beta, *_ = np.linalg.lstsq(X, eq, rcond=None)
                self.beta = beta
                score = X @ beta
            else:
                score = eq
            if it == 2:
                break
            order = np.argsort(-score, kind="stable").astype(np.int64)
            top = max(int(0.25 * self.M), 2)
            eq = K.preflop_equity_vs_top(self.U, self.H, order, top, self.board_total, self.deck, n, 7 + it,
                                         self.rule, self.POP, self.HIGH, self.STRAIGHT, self.CATR)
        order = np.argsort(-score, kind="stable")
        self.score_sorted = -np.sort(-score)
        ranks = np.empty(self.M)
        ranks[order] = np.arange(self.M)
        self.p_lo = ranks / self.M
        self.p_hi = (ranks + 1) / self.M
        self.pre_pct = 0.5 * (self.p_lo + self.p_hi)
        self.pre_score = score
        self._build_pre_table()

    def pre_features(self, hands):
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

    def open_rank_init(self, adj_fn):
        """Order the universe for OPENING decisions by equity (vs the top-25%
        range) x playability (plan: "deep opens by realized EV, JTs ahead of
        A2o"; push/fold keeps the raw ranking). ``adj_fn(hand) -> R adjustment``."""
        if self.exact and self.E169 is not None:
            top = (self.pre_pct < 0.25).astype(np.float64)
            top_cls = np.bincount(self.combo_class, weights=top, minlength=self.E169.shape[0])
            eq_cls = self.E169 @ top_cls / max(top_cls.sum(), 1e-9)
            eq = eq_cls[self.combo_class]
        elif getattr(self, "pre_score", None) is not None:
            eq = np.asarray(self.pre_score, np.float64)
        else:
            eq = 1.0 - self.pre_pct
        adj = np.array([adj_fn([int(x) for x in h]) for h in self.U], np.float64)
        score = eq * (1.0 + adj)
        order = np.argsort(-score, kind="stable")
        ranks = np.empty(self.M)
        ranks[order] = np.arange(self.M)
        self.open_pct = (ranks + 0.5) / self.M
        self.open_score_sorted = -np.sort(-score)
        self._open_adj_fn = adj_fn

    def hero_open_pct(self, hole):
        """Percentile of ``hole`` in the opening order (0 = best)."""
        if getattr(self, "open_pct", None) is None:
            return self.hero_pre_pct(hole)
        if self.exact:
            k = self.cidx.get((hole[0], hole[1]))
            return float(self.open_pct[k]) if k is not None else 0.5
        key = ("open",) + tuple(hole)
        if getattr(self, "_op_key", None) == key:
            return self._op_val
        sc = float((self.pre_features(np.array([hole], np.int64)) @ self.beta)[0])
        sc *= 1.0 + self._open_adj_fn([int(x) for x in hole])
        idx = np.searchsorted(-self.open_score_sorted, -sc)
        self._op_key, self._op_val = key, float(min(max(idx / self.M, 0.0), 1.0))
        return self._op_val

    def hero_pre_pct(self, hole):
        if self.exact:
            k = self.cidx.get((hole[0], hole[1]))
            return float(self.pre_pct[k]) if k is not None else 0.5
        key = tuple(hole)
        if getattr(self, "_pp_key", None) == key:      # same hand as the last call: reuse
            return self._pp_val
        sc = float((self.pre_features(np.array([hole], np.int64)) @ self.beta)[0])
        # position of this score within the universe's sorted scores
        idx = np.searchsorted(-self.score_sorted, -sc)
        self._pp_key, self._pp_val = key, float(min(max(idx / self.M, 0.0), 1.0))
        return self._pp_val

    # -- ranges ---------------------------------------------------------------
    extra_dead = 0  # set per decision by brain (e.g. dead auction cards); never carried over

    def dead_mask(self, cards):
        m = int(self.extra_dead)
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
        """(strength score per universe hand, strong-draw flag per hand)."""
        b = np.asarray(board, np.int64)
        return K.combo_scores(self.U, self.H, b, b.shape[0], np.int64(dead), self.rule, self.score_coef,
                              self.POP, self.HIGH, self.STRAIGHT, self.CATR)

    def board_features(self, board, dead):
        """(feature matrix, draw flags) - used by tools/fit_scores.py."""
        b = np.asarray(board, np.int64)
        return K.hand_features(self.U, self.H, b, b.shape[0], np.int64(dead), self.rule,
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

    def auction_equities(self, hole, auction, weights, n, seed):
        """Pot share on each of the 10 flops of the 5 auction cards (kernels.TRIPLES order)."""
        cdfs = np.cumsum(np.vstack(weights), axis=1)
        return K.auction_equities(np.asarray(hole, np.int64), self.H, np.asarray(auction, np.int64),
                                  self.deck, self.U, cdfs, cdfs.shape[0], int(n), int(seed) & 0x7FFFFFFF,
                                  self.rule, K.TRIPLES, self.POP, self.HIGH, self.STRAIGHT, self.CATR)

    def equity(self, hole, board, weights, n, seed):
        """Sampled equity vs the ranges in ``weights`` (list of [M] arrays).

        Returns (vidx, res_hu, res_all) per sample; see kernels.mc_equity.
        """
        cdfs = np.cumsum(np.vstack(weights), axis=1)
        h = np.asarray(hole, np.int64)
        b = np.asarray(board, np.int64)
        return K.mc_equity(h, self.H, b, b.shape[0], self.board_total, self.deck, self.U, cdfs,
                           cdfs.shape[0], int(n), int(seed) & 0x7FFFFFFF, self.rule,
                           self.POP, self.HIGH, self.STRAIGHT, self.CATR, np.int64(self.extra_dead))

    # -- preflop equity vs a percentile band ------------------------------------
    PRE_XS = np.array([0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0])
    PRE_BINS = 20

    PRE_SAMPLES = 300
    PRE_MULTI_SAMPLES = 40      # runouts for a multiway preflop equity in a sampled game (PRIOR: fits the time budget)

    def _build_pre_table(self):
        """pre_Tb[bin, cell]: equity vs a villain in the band (xs[cell-1], xs[cell]];
        pre_T[bin, cell]: equity vs the top xs[cell] (the running average of pre_Tb)."""
        self.pre_Tb = K.preflop_table(self.U, self.H, self.pre_pct, self.PRE_XS, self.PRE_BINS,
                                      self.PRE_SAMPLES, self.board_total, self.deck, 9, self.rule,
                                      self.POP, self.HIGH, self.STRAIGHT, self.CATR)
        self._pre_lo = np.concatenate(([0.0], self.PRE_XS[:-1]))     # lower edge of each cell
        widths = self.PRE_XS - self._pre_lo
        self.pre_T = np.cumsum(self.pre_Tb * widths, axis=1) / self.PRE_XS

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
        row = (1 - t) * self.pre_Tb[b0] + t * self.pre_Tb[b1]
        # average of the band cells, weighted by how much of [lo, hi) lies in each
        ov = np.clip(np.minimum(self.PRE_XS, hi) - np.maximum(self._pre_lo, lo), 0.0, None)
        tot = float(ov.sum())
        if tot <= 0:
            return float(row[min(int(np.searchsorted(self.PRE_XS, lo)), row.size - 1)])
        return float(min(max(float((row * ov).sum()) / tot, 0.0), 1.0))

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
        _, hu, _ = self.equity(hole, [], [w], n, seed)
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
    hole, board = [int(x) for x in cards[: g.H]], [int(x) for x in cards[g.H:]]
    dead = g.dead_mask(hole + board[:3])
    g.scores(board[:3], g.dead_mask(board[:3]))
    g.scores(board, g.dead_mask(board))
    g.strengths(board, g.dead_mask(board))
    w = g.range_interval(0.0, 0.5, dead)
    g.equity(hole, board[:3], [w], 16, 1)
    g.equity(hole, board, [w, w], 16, 1)
    g.pre_equity(hole, 0.0, 0.3)
    g.pre_equity(hole, 0.1, 0.4)
    g.pre_equity_multi(hole, [(0.0, 0.3), (0.1, 0.5)], 1, 16)
    g.band_probs([0.0, 0.1, 0.5], g.dead_mask(hole))
    g.hero_pre_pct(hole)
    g.hand_value(hole, board)
    K.icm_values([100.0, 50.0, 25.0], [50.0, 30.0, 20.0])
    K.hero_values(np.array([[100.0, 50.0, 25.0], [0.0, 150.0, 25.0]]), np.array([50.0, 30.0, 20.0]), 0)
    g.auction_equities(hole, board[:5], [w], 8, 1)
    return g
