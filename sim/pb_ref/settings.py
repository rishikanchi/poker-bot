"""Variant settings and tunable strategy parameters.

BUILD HOUR: read RULES.md, then set VARIANT below. Everything else adapts.

  Texas hold'em:      hole_cards=2, rule="any",   deck_ranks=range(13)
  Short deck (6+):    hole_cards=2, rule="any",   deck_ranks=range(4, 13), short_deck=True
                      (check RULES.md: some short-deck rules keep full house > flush)
  Omaha (PLO4/5/6):   hole_cards=4/5/6, rule="omaha" (exactly 2 hole + 3 board)
  Pineapple-style:    hole_cards=3, rule="any" (until the discard; see README)

Anything outside this family (draw games, stud, split pots) runs in safe mode.
"""

import os as _os

VARIANT = {
    "name": "holdem",
    "hole_cards": 2,
    "rule": "any",          # "any" = best 5 of hole+board; "omaha" = exactly 2 hole + 3 board
    "deck_ranks": list(range(13)),  # 0=2 ... 12=A
    "short_deck": False,
    "board_total": 5,
    "universe_size": 800,   # sampled hands per range for games with >2 hole cards
}

# Presets. PB_VARIANT=<name> in the environment picks one (used by the local
# simulator; in the build hour you can also just edit VARIANT above).
PRESETS = {
    "holdem": {"name": "holdem", "hole_cards": 2, "rule": "any", "deck_ranks": list(range(13)), "short_deck": False},
    "shortdeck": {"name": "shortdeck", "hole_cards": 2, "rule": "any", "deck_ranks": list(range(4, 13)), "short_deck": True},
    "omaha4": {"name": "omaha4", "hole_cards": 4, "rule": "omaha", "deck_ranks": list(range(13)), "short_deck": False},
    "omaha5": {"name": "omaha5", "hole_cards": 5, "rule": "omaha", "deck_ranks": list(range(13)), "short_deck": False},
    "omaha6": {"name": "omaha6", "hole_cards": 6, "rule": "omaha", "deck_ranks": list(range(13)), "short_deck": False},
}
if _os.environ.get("PB_VARIANT") in PRESETS:
    VARIANT.update(PRESETS[_os.environ["PB_VARIANT"]])

# Every number the optimizer (or you) may want to tune lives here.
P = {
    # ---- time budget (ms) for one decision, of decision_ms ----
    "budget_frac": 0.55,
    "low_bank_ms": 40.0,             # below this much bank left, halve the budget
    # ---- preflop: open widths (fraction of hands) by players left to act behind ----
    # plan: UTG ~14%, middle ~20%, button ~42%, raise-or-fold. index = players
    # still to act after us (incl. blinds); 1 = SB vs BB
    "open_width": {1: 0.42, 2: 0.42, 3: 0.27, 4: 0.21, 5: 0.18, 6: 0.16, 7: 0.14, 8: 0.13, 9: 0.12},
    "open_width_hu": 0.80,           # heads-up button/SB
    "open_scale": 1.0,               # multiplies OUR open widths (not the opponent model)
    "open_size_bb": 2.2,             # plan: ~2.2-2.5bb (+1bb per limper)
    "open_size_bb_short": 2.0,       # plan: 16-40bb opens ~2-2.2bb
    "iso_size_bb": 2.5,              # plan: isolate limpers to ~2.5bb + 1bb per limper
    "limp_iso_extra_bb": 1.0,
    "iso_width_mult": 1.1,           # plan: isolate with the opening range "plus a bit"
    "steal_ev": False,               # plan: fixed widths (EV steals tested: no gain, +0.14 ± 0.37)
    "use_pushfold_charts": True,     # plan: opponents' push/call ranges from solved charts
    # ---- stack bands (effective stack in big blinds) ----
    "push_fold_bb": 15.0,            # plan: <= ~15bb shove or fold
    "open_jam_bb": 20.0,             # plan: 16-40bb band may open-jam (we allow up to here)
    "threebet_jam_bb": 35.0,         # plan ~20-25bb; tuned: 35 (+1.51 ± 0.38 on fresh tables)
    "fourbet_jam_bb": 35.0,          # plan ~40bb, else ~2.3x; tuned with the 3-bet depth
    "fourbet_size": 2.3,
    # ---- preflop equity realization priors (EQR) ----
    "eqr_pre_ip": 0.92,
    "eqr_pre_oop": 0.78,
    "eqr_suited": 0.06,
    "eqr_connected": 0.04,
    "eqr_pair": 0.04,
    "eqr_offsuit_junk": -0.06,
    # ---- postflop EQR priors (plan: 1.0 IP, 0.85 OOP, +0.1 strong draws/nuts, -0.1 weak made) ----
    "eqr_ip": 1.0,
    "eqr_oop": 0.85,
    "eqr_draw": 0.10,
    "eqr_weak_made": -0.10,
    "eqr_strong": 0.10,
    "eqr_spr_full": 3.0,             # SPR at which the prior fully applies (shrinks to 1 below)
    "use_measured_eqr": False,       # DEVIATION: measured EQR (vs test bots) scored worse twice
                                     # (+1.44±0.40, +0.93±0.48); re-measure from real hand histories
    "eqr_measured_weight": 0.5,      # blend with the prior (measured vs test bots, not the real field)
    "eqr_measured_cap": 1.5,
    # ---- equilibrium priors for the opponent model (shrinkage targets) ----
    "prior_fold_to_3bet": 0.55,
    "prior_4bet": 0.10,
    "prior_call_3bet": 0.35,         # opener calls a 3-bet (fold = 1 - call - 4-bet = 0.55)
    "behind_continue_vs_3bet": 0.04, # each player still to act behind continues vs a 3-bet
    "threebet_eqr_adj": -0.15,       # EQR change in called 3-bet pots (plan: EQR differs by spot); tuned
    "threebet_conf_z": 1.5,          # pessimistic fold estimate for non-value 3-bets; tuned
    "threebet_value_pct": 0.06,      # 3-bets with hands better than this are value, not bluffs
    "prior_fold_to_4bet": 0.50,
    "prior_raise_vs_bet": 0.08,      # plan: ~10-15% of the continuing range raises
    "prior_bet_when_checked": 0.40,  # plan: bet ~25-35% for value + s/(1+2s) bluffs
    "prior_cbet": 0.55,
    "prior_raise_bluff_share": 0.10, # plan: start low (~10%), update from showdowns
    "prior_fold_to_raise": 0.45,     # used only when the price is unknown
    "raise_fold_price_prior": True,  # plan: fold-to-raise prior = s/(1+s) of the price they face
    "size_strength": 0.15,           # plan: bigger bets -> tighter assumed range
    "k_common": 12.0,                # observations needed to move halfway from the prior
    "k_rare": 24.0,
    # ---- postflop sizing (plan + Rishi: 33 / 66 / 125 % of the pot, all-in) ----
    "bet_sizes": {1: (0.33, 0.66, 1.25), 2: (0.33, 0.66, 1.25), 3: (0.33, 0.66, 1.25)},
    "bet_sizes_no_initiative": (0.33, 0.66),  # plan: smaller menu for probes / leads
    "raise_pot_frac": 0.75,          # plan: raise to call + 0.75 * (pot after calling)
    "jam_left_frac": 0.5,            # plan: a bet leaving < ~half a pot behind becomes all-in
    "raise_jam_left_frac": 1.0,      # plan: a raise leaving < ~1 pot behind becomes all-in
    "allin_spr": 3.0,                # plan: offer all-in explicitly when SPR < 3
    # ---- river bluffs (plan) ----
    "river_bluff_cap": True,         # bluff only if P(fold | our cards) > s/(1+s), capped at s/(1+2s)
    "river_sdv_eq": 0.30,            # below this equity vs their range = no showdown value
    # ---- checking-range protection (plan: target share, tunable, probably low) ----
    "trap_eq": 0.80,
    "trap_share": 0.10,
    # ---- fold equity on the next street (plan; discounted) ----
    "multistreet_lo": 0.20,
    "multistreet_hi": 0.55,
    "multistreet_disc": 0.70,
    # ---- ICM ----
    "icm_scale": 1.0,                # multiplies the ICM risk premium
    "cap_rp_boost": 0.5,             # plan: risk premium grows as the hand cap nears
    "cap_rp_hands": 40.0,            # ... over the last this-many hands
    "postflop_icm_allin": True,      # plan: exact ICM check for postflop all-ins
    # ---- postflop decision details ----
    "mc_samples_min": 80,
    "mc_samples_min_sampled": 30,    # games with >2 hole cards (each sample is costlier)
    "mc_samples_max": 700,
    "mc_fixed_samples": 0,           # >0: fixed sample count (reproducible tuning runs)
    "tie_margin": 0.02,              # EVs within this fraction of the pot are mixed
    "multiway_call_margin": 0.03,    # extra equity needed per player still to act behind
    "bluff_conf_z": 0.5,             # pessimistic fold estimate for bluffs (plan: lower CI bound)
}
