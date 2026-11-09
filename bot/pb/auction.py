"""Sealed flop auction (Boston variant): five bids, one per face-up auction card.

Pipeline (after the team's my_bot/auction.py on the build box):
  1. value the 10 possible flops: our pot share on each, vs the opponents'
     estimated preflop ranges (Monte Carlo, kernels.auction_equities; same
     samples for all ten flops so their differences are low-noise)
  2. card marginals: mean value of the flops holding a card minus those not
  3. budget: pace the flop-chip balance over the auctions expected to remain,
     scaled by how much this auction is worth (pot x equity edge, in stacks)
  4. bid on the best flop's three cards: 1 chip each beats silent bidders,
     the rest of the budget split by the cards' marginal values

Rules that matter: we pay only our bids on cards that make the flop; ties go
to the card dealt first; no bids = cards 0, 1, 2; flop-chips never refill and
are worth no points, so unspent chips at the end of a table are wasted.
"""
from __future__ import annotations

import numpy as np

from . import kernels as K
from . import ranges as Rg
from . import view as V

TRIPLES = [tuple(int(x) for x in t) for t in K.TRIPLES]
DEFAULT_TRIPLE = (0, 1, 2)
NO_BID = (0, 0, 0, 0, 0)


def card_marginals(values):
    total = float(np.sum(values))
    out = []
    for card in range(5):
        inside = sum(val for t, val in zip(TRIPLES, values) if card in t)
        out.append(inside / 6.0 - (total - inside) / 4.0)  # 6 flops hold a card, 4 do not
    return out


def budget(P, balance, stake, hand_number, hand_cap, players_remaining):
    """Flop-chips this auction may spend."""
    table_hands = min(P["auction_table_hands"], hand_cap)
    hands_left = max(table_hands - hand_number, 0)
    # fewer players left -> we are in more of the auctions
    rate = min(0.9, P["auction_flop_rate"] * 8.0 / max(players_remaining, 2))
    auctions_left = max(hands_left * rate, P["auction_min_left"])
    paced = balance / auctions_left * (stake / P["auction_ref_stake"])
    return int(min(paced, balance * P["auction_max_fraction"]))


def plan(c, P):
    """Return the five bids for this auction."""
    v, g = c.v, c.game
    auc = v.auction_cards
    lo, hi = v.legal.get("bid", (0, 0))
    balance = int(hi or 0)
    if balance <= 0 or len(auc) != 5 or len(v.hole) != g.H:
        return NO_BID
    dead = g.dead_mask(list(auc))
    ranges, _ = Rg.build_ranges(v, g, c.opps, P, {}, {}, dead)
    opp = [s for s in ranges if v.players[s].in_hand]
    if not opp:
        return NO_BID
    live = g.live(g.dead_mask(list(v.hole) + list(auc))).astype(np.float64)
    ws = []
    for s in opp[: P["auction_max_opps"]]:
        w = ranges[s] * live
        ws.append(w if w.sum() > 0 else live)
    seed = V.stable_seed(v.table_id, v.hand_number, "auction")
    values = g.auction_equities(v.hole, auc, ws, P["auction_samples"], seed)
    best = int(np.argmax(values))
    mean = float(np.mean(values))
    edge = float(values[best]) - mean
    if edge < P["auction_min_edge"]:
        return NO_BID
    target = TRIPLES[best]
    marg = card_marginals(values)
    chips = v.pot + sum(p.stack for p in v.players.values() if not p.busted)
    avg_stack = chips / max(v.players_remaining or len(v.alive), 1)
    stake = v.pot * edge / max(avg_stack, 1.0)
    spend = min(budget(P, balance, stake, v.hand_number, v.hand_cap,
                       v.players_remaining or len(v.alive)), balance)
    bids = [0] * 5
    floor = 0 if target == DEFAULT_TRIPLE and spend < 3 else 1
    left = balance
    for card in reversed(target):   # later cards lose ties: they need it most
        if left >= floor:
            bids[card] = floor
            left -= floor
    extra = min(max(spend - sum(bids), 0), left)
    weights = [max(marg[card], 0.0) for card in target]
    wsum = sum(weights)
    if extra and wsum > 0:
        for card, wt in zip(target, weights):
            bids[card] += int(extra * wt / wsum)
    return tuple(int(b) for b in bids)
