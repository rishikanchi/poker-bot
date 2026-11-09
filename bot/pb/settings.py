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

# Every number the bot uses lives here. Three kinds:
#   PLAN   a number or rule stated in Solon's plan
#   PRIOR  a starting value the plan asks for without fixing the number
#          ("start from equilibrium-style frequencies"); observation replaces it
#   TUNED  chosen by paired simulation tests (see README "Plan compliance")
P = {
    # ---- time budget for one decision, as a share of decision_ms ----
    "budget_frac": 0.55,
    "low_bank_ms": 40.0,             # below this much bank left, halve the budget
    "lean_at": 0.80,                 # PLAN "fallback to a simpler policy if the time budget is running out":
                                     # past this share of decision_ms, price only check / call / fold / one size
    # ---- preflop opening (PLAN: UTG ~14%, middle ~20%, button ~42%, raise or fold) ----
    # index = players still to act after us (incl. blinds); 1 = small blind vs big blind
    "open_width": {1: 0.38, 2: 0.42, 3: 0.27, 4: 0.21, 5: 0.18, 6: 0.16, 7: 0.14, 8: 0.13, 9: 0.12},  # PLAN: SB ~38%, BTN ~42% ...
    "open_width_hu": 0.80,           # PRIOR: heads-up button
    "open_scale": 1.0,               # TUNED multiplier on OUR open widths
    "open_jam_bb": 25.0,             # PLAN 16-40bb band: fold / small open / open-jam; the chart covers <= 25bb
    "open_size_bb": 2.2,             # PLAN: ~2.2-2.5bb
    "open_size_bb_short": 2.0,       # PLAN: ~2-2.2bb at 16-40bb
    "iso_size_bb": 2.5,              # PLAN: isolate limpers to ~2.5bb + 1bb per limper
    "limp_iso_extra_bb": 1.0,
    "iso_width_mult": 1.1,           # PLAN: isolate with the opening range "plus a bit"
    "use_pushfold_charts": True,     # PLAN: push/call ranges solved by iterated best response
    # ---- stack bands, effective stack in big blinds ----
    "push_fold_bb": 15.0,            # PLAN: <= ~15bb shove or fold
    "threebet_jam_bb": 35.0,         # PLAN ~20-25bb; TUNED to 35 (+1.51 +- 0.38)
    "fourbet_jam_bb": 35.0,          # PLAN ~40bb; TUNED with the 3-bet depth
    "fourbet_size": 2.3,             # PLAN: 4-bet ~2.2-2.5x the 3-bet
    "commit_left_frac": 1.0,         # PLAN: a raise that leaves < ~1 pot behind is an all-in
    # ---- preflop equity realization (EQR). PLAN gives the features and signs, not the sizes ----
    "eqr_pre_ip": 0.92,              # PRIOR / TUNED
    "eqr_pre_oop": 0.78,             # PRIOR / TUNED
    "eqr_suited": 0.06,              # PRIOR: suited (+)
    "eqr_connected": 0.04,           # PRIOR: connected (+)
    "eqr_pair": 0.04,                # PRIOR: pair (+)
    "eqr_offsuit_junk": -0.06,       # PRIOR: offsuit weak kicker (-)
    "threebet_eqr_adj": -0.15,       # TUNED: EQR in 3-bet pots (PLAN: EQR differs by spot)
    "eqr_multiway": 0.05,            # PLAN: +-0.05 per extra opponent (nut-type hands +, others -)
    # ---- postflop EQR (PLAN: 1.0 IP, 0.85 OOP, +0.1 strong draws / near-nuts, -0.1 weak made hands) ----
    "eqr_ip": 1.0,
    "eqr_oop": 0.85,                 # flop
    "eqr_oop_turn": 0.90,            # PLAN: the turn solves used river EQR 1.0 in position / 0.9 out of position
    "eqr_draw": 0.10,                # nut draws
    "eqr_nonnut_draw": -0.10,        # PLAN (PLO discussion): non-nut draws realize less than their equity
    "eqr_weak_made": -0.10,
    "eqr_strong": 0.10,
    "eqr_strong_eq": 0.75,           # PRIOR: equity that counts as "strong / near-nut"
    "eqr_weak_lo": 0.35,             # PRIOR: equity band that counts as a "weak made hand"
    "eqr_weak_hi": 0.62,
    "eqr_spr_full": 3.0,             # PLAN: EQR shrinks toward 1 at low SPR; full effect from this SPR
    "eqr_multiway_pos": 0.05,        # PLAN multiway: first to act -0.05, last to act +0.05
    "use_measured_eqr": False,       # DEVIATION: measured EQR (vs test bots) scored worse twice
    "eqr_measured_weight": 0.5,
    "eqr_measured_cap": 1.5,
    # ---- opponent-model starting values (shrinkage targets) ----
    "prior_open": 0.22,              # PRIOR: average of our own opening widths
    "prior_3bet": 0.04,              # PLAN: 3-bet the top ~3-4%
    "prior_4bet_width": 0.025,       # PLAN: 4-bet the top ~2-3%
    "prior_coldcall": 0.07,          # PLAN sample ranges: in-position caller ~7%
    "bb_defend": (0.25, 0.45),       # PLAN sample ranges: BB calls ~25% vs a 15% open ... ~45% vs a 45% open
    "prior_limp": 0.03,              # PRIOR: equilibrium play almost never limps
    "prior_vpip": 0.25,              # PRIOR
    "prior_pfr": 0.15,               # PRIOR
    "prior_blind_fold": 0.62,        # PRIOR: PLAN has the BB defending 25-45% and the SB less
    "prior_fold_to_3bet": 0.55,      # PRIOR
    "prior_4bet": 0.10,              # PRIOR: share of an opening range that 4-bets a 3-bet
    "prior_call_3bet": 0.35,         # PRIOR: share that calls it (PLAN samples: ~7% of 15%, ~16% of 45%)
    "prior_fold_to_4bet": 0.50,      # PRIOR
    "prior_5bet": 0.15,              # PRIOR
    "behind_continue_vs_3bet": 0.03, # PLAN: a player behind continues with about a 4-bet range (2-3%)
    "threebet_conf_z": 1.5,          # TUNED: PLAN "deviate only as far as the evidence supports"
    "cover_call_mult": 0.85,         # PRIOR: PLAN "a player you would knock out calls tighter"
    "k_ratio": 3.0,                  # PRIOR: pseudo-count (in events) for observed/expected ratios
    "prior_raise_vs_bet": 0.08,      # PLAN: ~10-15% of the continuing range raises
    "prior_raise_vs_bet_river": 0.05,  # PLAN river rule: "raise all-in with the top ~5%"
    "prior_bet_when_checked": 0.40,  # PLAN: bets ~25-35% for value + s/(1+2s) bluffs when checked to
    "prior_cbet": 0.55,              # PRIOR
    "prior_raise_bluff_share": 0.10, # PLAN: start low (~10%), update from showdowns
    "prior_fold_to_raise": 0.45,     # used only when the price is unknown
    "raise_fold_price_prior": True,  # PLAN: fold-to-raise prior = s/(1+s) of the price they face
    "range_leak": 0.05,              # PRIOR: weight left on hands outside the region an action implies
    "range_leak_check": 0.15,        # PLAN: players check 10-20% of their strong hands (traps)
    "bluff_showdown_pct": 0.45,      # PRIOR: a shown hand below this percentile counts as a bluff / weak
    "same_size_share": 0.85,         # PLAN "repeated identical sizes": size says nothing once one size is this common
    "same_size_min_bets": 8,         # ... after this many bets seen
    "k_common": 12.0,                # PLAN k: observations to move halfway from the prior (common stats)
    "k_rare": 24.0,                  # ... rare stats
    # ---- postflop sizing (PLAN + Rishi: 33 / 66 / 125 % of the pot, all-in) ----
    "bet_sizes": {1: (0.33, 0.66, 1.25), 2: (0.33, 0.66, 1.25), 3: (0.33, 0.66, 1.25)},
    "bet_sizes_no_initiative": (0.33, 0.66),  # PLAN: smaller menu for probes / leads
    "raise_pot_frac": 0.75,          # PLAN: raise to call + 0.75 * (pot after calling)
    "jam_left_frac": 0.5,            # PLAN: a bet leaving < ~half a pot behind becomes all-in
    "raise_jam_left_frac": 1.0,      # PLAN: a raise leaving < ~1 pot behind becomes all-in
    "allin_spr": 3.0,                # PLAN: offer all-in explicitly when SPR < 3
    "villain_bet_frac": 0.75,        # PLAN: "their usual size, or the 75% translation point" (prior for the usual size)
    # PLAN: opponents' bets are translated onto min / 33% / 75% / 125% / all-in. A minimum-size bet
    # (or a min-raise) is cheap and "often weak": it does not narrow their range. This is the next
    # translation point above the minimum; sizes in between are split with the pseudo-harmonic rule.
    "min_translate_bet": 0.33,
    # ---- river bluffs (PLAN: bluff when P(fold | our cards) > s/(1+s), with the best blockers,
    #      capped at s/(1+2s) of our betting range) ----
    "river_bluff_cap": True,
    "river_sdv_eq": 0.30,            # PRIOR: below this equity vs their range a hand has no showdown value
    # ---- checking-range protection (PLAN: tunable share of very strong hands that check) ----
    "trap_eq": 0.80,
    "trap_share": 0.10,
    # ---- fold equity on the next street (PLAN, discounted ~0.7) ----
    "multistreet_lo": 0.20,
    "multistreet_hi": 0.55,
    "multistreet_disc": 0.70,
    # ---- ICM ----
    "icm_scale": 1.0,                # TUNED multiplier on the ICM risk premium
    "cap_rp_boost": 0.5,             # PRIOR: PLAN "the risk premium grows as the hand cap nears"
    "cap_rp_hands": 40.0,            # ... over the last this-many hands
    "postflop_icm_allin": True,      # PLAN: every all-in is checked in exact ICM points
    # ---- decision details ----
    "mc_samples_min": 80,
    "mc_samples_min_sampled": 30,    # games with >2 hole cards (each sample is costlier)
    "mc_samples_max": 700,
    "mc_fixed_samples": 0,           # >0: fixed sample count (reproducible test runs)
    "region_prior_n": 6.0,           # PRIOR: pseudo-samples pulling a thin region's equity to the fitted line
    "tie_margin": 0.02,              # PLAN: mix near-ties; EVs within this share of the pot ...
    "tie_mix_prob": 0.35,            # ... pick the runner-up this often
    "multiway_call_margin": 0.03,    # PLAN: extra equity needed per player still to act behind
    "bluff_conf_z": 0.5,             # PLAN: pessimistic fold estimate for bluffs
}
