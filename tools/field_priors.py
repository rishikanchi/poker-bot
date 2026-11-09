"""Population statistics from hand histories, next to the bot's priors (plan item 10:
"a script that analyzes hand histories between rounds to update priors").

Input: JSON lines, one hand per line, in the local simulator's format (sim/run.py
--hands-out): {"seats": [...labels...], "blinds": [sb, bb, ante], "stacks_start": [...],
"events": [[kind, seat, amount, cards], ...], ...}. For the kit's own hand-history
files, write a small loader that yields the same dict per hand and pass --loader
module:function (see load_sim below for the shape).

Every hand is replayed through pb.stats.replay_hand with EVERY seat tracked (no
hero), so the counters are exactly the ones the bot keeps at the table. The report
prints, per statistic, the population rate beside the prior in settings.P and the
number of observations, plus a bet-size histogram. Edit settings.P by hand from it:
the plan wants priors near equilibrium and moved only where the field clearly differs.

Run:  python3 tools/field_priors.py hands.jsonl [--variant omaha4] [--json out.json]
"""
from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bot"))


def load_sim(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def card_ints(cards):
    from pb.view import cards_int
    return tuple(cards_int(cards))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--variant", default=None, help="holdem (default) / omaha4 / ... (sets PB_VARIANT)")
    ap.add_argument("--loader", default=None, help="module:function yielding hand dicts (default: simulator format)")
    ap.add_argument("--json", default=None, help="also write the rates here")
    a = ap.parse_args(argv)
    if a.variant:
        os.environ["PB_VARIANT"] = a.variant
    from pb import settings, stats as St
    from pb.game import get_game
    from pb.memory import Memory
    P = settings.P
    game = get_game()
    loader = load_sim
    if a.loader:
        mod, fn = a.loader.split(":")
        loader = getattr(importlib.import_module(mod), fn)

    # one Memory per table so the counters start fresh per table, as the bot's do
    mems = {}
    sizes = collections.Counter()
    n_hands = 0
    for h in loader(a.path):
        key = h.get("table", 0)
        m = mems.get(key)
        seats = list(range(len(h["seats"])))
        if m is None:
            m = Memory()
            m.ensure_seats(seats, int(max(h["stacks_start"])))
            mems[key] = m
        events = [(k, s, int(amt or 0), card_ints(c) if c else ()) for k, s, amt, c in h["events"]]
        # bet sizes as a fraction of the pot (for the histogram)
        pot = 0
        high = 0
        street = 0
        bets = {}
        for k, s, amt, _ in events:
            if k in ("blind", "ante"):
                pot += amt
                if k == "blind":
                    bets[s] = bets.get(s, 0) + amt
                    high = max(high, bets[s])
            elif k == "board":
                street += 1
                bets = {}
                high = 0
            elif k == "raise" and street >= 1:
                sizes[round(min((amt - high) / max(pot + high - bets.get(s, 0), 1), 3.0), 1)] += 1
            if k in ("call", "raise"):
                inc = amt if k == "call" else amt - bets.get(s, 0)
                pot += inc
                bets[s] = bets.get(s, 0) + inc
                high = max(high, bets[s])
        St.replay_hand(events, m, hero_seat=-1, game=game, starting_stack=int(max(h["stacks_start"])), P=P)
        n_hands += 1
    S = np.zeros((1, St.K_STATS))
    for m in mems.values():
        S[0] += m.stats.sum(axis=0)
    tot = S[0]

    def rate(num, den):
        return (tot[num] / tot[den] if tot[den] > 0 else float("nan")), int(tot[den])

    rows = [
        ("open rate / baseline (x1.0 = plan widths)", (tot[St.OPEN] / max(tot[St.OPEN_EXP] / 100.0, 1e-9), int(tot[St.OPEN_OPP])), 1.0),
        ("limp first in", rate(St.LIMP, St.LIMP_OPP), P["prior_limp"]),
        ("3-bet vs a lone open", rate(St.TB, St.TB_OPP), P["prior_3bet"]),
        ("cold call (not BB)", rate(St.CCN, St.CCN_OPP), P["prior_coldcall"]),
        ("blind folds to a steal", rate(St.BVS_FOLD, St.BVS_OPP), P["prior_blind_fold"]),
        ("fold to 3-bet", rate(St.F3B, St.F3B_OPP), 1 - P["prior_4bet"] - P["prior_call_3bet"]),
        ("call 3-bet", rate(St.C3B, St.F3B_OPP), P["prior_call_3bet"]),
        ("4-bet", rate(St.R4B, St.F3B_OPP), P["prior_4bet"]),
        ("calls of all-ins / chart", (tot[St.CJ] / max(tot[St.CJ_EXP] / 100.0, 1e-9), int(tot[St.CJ_OPP])), 1.0),
        ("c-bet", rate(St.CB, St.CB_OPP), P["prior_cbet"]),
        ("bet when checked to", rate(St.BU, St.BU_OPP), P["prior_bet_when_checked"]),
        ("raise vs bet", rate(St.RVB, St.RVB_OPP), P["prior_raise_vs_bet"]),
        ("fold to raise", rate(St.FTR, St.FTR_OPP), P["prior_fold_to_raise"]),
        ("went to showdown (saw flop)", rate(St.WTSD, St.SAWF), None),
        ("river bets shown: weak", rate(St.RB_BLUFF, St.RB_SHOWN), None),
        ("raises shown: weak then", rate(St.RZ_BLUFF, St.RZ_SHOWN), P["prior_raise_bluff_share"]),
        ("min-size bets shown: weak", rate(St.MZ_WEAK, St.MZ_SHOWN), P["bluff_showdown_pct"]),
        ("calls shown: weak", rate(St.CALL_WEAK, St.CALL_SHOWN), None),
        ("usual bet size (pots)", ((tot[St.BETF_SUM] / 100.0 / tot[St.BETF_N]) if tot[St.BETF_N] else float("nan"), int(tot[St.BETF_N])), P["villain_bet_frac"]),
    ]
    names = ["33%", "66%", "100%", "125%", "all-in"]
    for st in (1, 2, 3):
        for b in range(St.N_SIZES):
            j = (st - 1) * St.N_SIZES + b
            rows.append((f"fold to bet street {st} size {names[b]}", rate(St.FTB0 + j, St.FTB_OPP0 + j),
                         St.MENU[b] / (1 + St.MENU[b]) if b < len(St.MENU) else None))
    print(f"{n_hands} hands, {len(mems)} tables")
    print(f"{'statistic':40s} {'field':>8s} {'prior':>8s} {'n':>7s}")
    out = {}
    for name, (val, n), prior in rows:
        ps = "" if prior is None else f"{prior:8.3f}"
        print(f"{name:40s} {val:8.3f} {ps:>8s} {n:7d}")
        out[name] = {"field": None if val != val else round(float(val), 4), "prior": prior, "n": n}
    print("\nbet sizes (fraction of the pot, rounded to 0.1):")
    total = sum(sizes.values()) or 1
    for k in sorted(sizes):
        print(f"  {k:4.1f}  {100 * sizes[k] / total:5.1f}%")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"hands": n_hands, "rates": out, "bet_sizes": {str(k): v for k, v in sorted(sizes.items())}}, fh, indent=2)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
