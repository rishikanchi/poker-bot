# poker-bot — No Limit '26

A competition bot built as **one variant-agnostic core + one plug-in per variant**,
following Solon's plan (EV per action from equity, equity realization and
fold equity; equilibrium-style priors shrunk toward what each opponent shows;
exact ICM for all-ins).

```
bot/                       <- the folder you upload (agent.py at the zip root)
  agent.py                 entry point: thread limits, warm-up at import, safe act()
  data/
    holdem_tables.npz      hold'em preflop equities (169x169) + ranking     tools/build_tables.py
    pushfold_<variant>.npz solved jam / call / re-jam charts                tools/build_pushfold.py
    scorefit_<variant>.npz fitted flop/turn hand-strength weights           tools/fit_scores.py
                           (both shipped for holdem, omaha4, omaha5, omaha6, shortdeck)
    eqr_holdem.json        measured EQR (kept, switched off - see below)    tools/measure_eqr.py
  pb/
    settings.py            VARIANT switch + every number the bot uses, each labelled
                           PLAN / PRIOR / TUNED
    game.py                PLUG-IN layer: VariantPlugin interface, CommunityCardGame
    kernels.py             Numba: evaluator (any hole count, hold'em / Omaha rules,
                           short deck), equity sampling, hand features, ICM
    view.py                ADAPTER: SDK state -> plain View (the only SDK-aware file)
    memory.py              compact bytes carried between turns (never raises)
    stats.py               opponent counters from finished hands + shrunk estimates
    ranges.py              what each opponent likely holds, given their actions
    preflop.py             open / steal / defend / 3-bet / push-fold by EV and ICM
    postflop.py            check / bet / call / raise by EV with the region model
    icmtools.py            chips -> tournament points (ICM), side pots included
    pushfold.py            lookups into the solved charts
    actions.py             decision -> always-legal SDK Action
    brain.py               one act(): memory, events, view, decide, encode, timing
stub_sdk/                  local stand-in for poker_agent_sdk (NOT uploaded)
sim/                       local sit-and-go simulator, simple bots, seven strong bots
  pb_ref/                  frozen copy of an older calculator the strong bots use,
                           so editing bot/pb never changes the opponents
tools/                     build_tables, build_pushfold, fit_scores, measure_eqr,
                           tune (parameter search), ab_code (version vs version),
                           field_priors (hand histories -> field rates vs priors), pack
tests/                     84 tests: evaluator, ICM, memory, stats, view, regions,
                           push/fold, engine rules, whole-bot spots
```

## Core vs plug-in

The core never looks at cards. It asks the plug-in (`pb/game.py`) for: the
preflop strength percentile of a hand, a range from a percentile band, the
probability of a band given our cards (blockers), a strength order of every
hand on a board, equity against ranges, and a showdown value. Ranges are
weight vectors over a *universe* of hands: all 1326 combos for 2-card games
(exact), a fixed sample for 3-6 card games (800 hands for Omaha; same code).

| Variant | What to do in the build hour |
|---|---|
| Hold'em (any blinds / antes / payouts) | nothing — data files are built |
| Omaha 4 | `PB_VARIANT=omaha4` or edit `settings.VARIANT` — data files are built |
| Short deck, Omaha 5 / 6 | `PB_VARIANT=shortdeck` / `omaha5` / `omaha6` — data files are built |
| another community-card variant (3 hole cards, other deck) | add a preset in `settings.py`, then run `tools/fit_scores.py` and `tools/build_pushfold.py` (about 15 s each) |
| Draw, stud, hi-lo split, discard phases | needs a new `VariantPlugin` class; until then the bot runs **safe mode** (legal check / call-small / fold) |

Non-hold'em games rank hands at import (equity against a top-25% range,
fitted to cheap features) and build a preflop equity table (about 7 s in total
for Omaha, including compiling).

## How decisions are made (Solon's plan)

* **Preflop.** Hands are ranked by equity against a top-25% range. Unopened:
  raise or fold by position width (UTG ~14% … button ~42%), wider against
  blinds that fold a lot; isolate limpers with the opening range plus a bit at
  2.5 bb + 1 bb per limper; over-limp only in position with hands that play
  well multiway. Facing raises: fold / call / re-raise (3x in position, 4x
  out, +1x per caller; 4-bet 2.3x) by EV = equity x EQR x pot − cost, using
  the opponent's observed fold / call / re-raise response and our blockers.
  At 15 bb or less: shove or fold. Every all-in decision is compared in
  **exact ICM points** at the real stacks and payouts; opponents' jam, call
  and re-jam ranges come from charts **solved by iterated best response**,
  scaled by what each opponent has shown.
* **Postflop.** Each opponent's range is narrowed by their actions with the
  region model: hands are ordered by strength (fitted equity on flop and
  turn, hand rank on the river); a bet keeps the strongest part plus bluffs
  from the bottom (s/(1+2s) of the betting range; on the river learned from
  showdowns), a raise the strongest part plus strong draws, a call the middle,
  a check what would not have bet (with 15% of strong hands kept as traps);
  bigger bets mean tighter ranges. Equity is Monte Carlo on flop and turn
  and exact on the river. Options: check, bet **33% / 66% / 125%** of the pot
  (33 / 66 without the initiative), all-in when SPR < 3, call, raise to call +
  75% of the pot. For each one the opponent's range is split into the part
  that folds, calls and raises, **given our cards**, and the EVs are compared.
  A bet that does not win more than half the time when called is "not for
  value" and may count on folds only at the pessimistic end of the estimate.
  Every option carries the ICM risk premium for the chips it puts at stake;
  every all-in must win in exact ICM points; 10% of very strong hands check.
* **Opponent model.** Opens, limps, 3-bets, cold calls, blind defence,
  response to 3-bets, jams and calls of jams (each against the frequency a
  baseline player would have in the same spots), c-bets, fold to bet by
  street and size (sizes mapped onto 33 / 66 / 125 with the pseudo-harmonic
  rule), raises, bets when checked to, fold to raise by the price faced, and
  showdown-verified bluff shares. Every estimate is
  `(observed + k x prior) / (n + k)` with equilibrium-style priors.
* **Safety.** Memory decode never raises; every step is wrapped; the action
  is always one of the offered kinds with an int amount inside the range; a
  time budget sizes the Monte Carlo and halves when the table's bank runs
  low; practice decisions run at import so the first real one is warm.
  `PB_DEBUG=1` makes strategy errors crash instead of falling back (local only).

## Plan compliance

`settings.py` labels every number: **PLAN** (stated in the plan), **PRIOR**
(a starting value the plan asks for without fixing the number; observation
replaces it), **TUNED** (chosen by paired simulation). The stack-loss fixes
and the 2026-11-09 review went through every decision path and replaced what
was not in the plan:

| Was in the code | Now |
|---|---|
| hand-picked draw bonuses to order hands on flop / turn | a scoring function fitted to sampled equities (the plan's wording); rank correlation with true equity 0.99 / 0.99 hold'em, 0.96 / 0.98 Omaha |
| fixed 45% fold-to-raise, fixed 3-bet responses, invented +0.05 EQR in 3-bet pots | observed responses shrunk to priors; fold-to-raise prior s/(1+s) of the price faced |
| caps on how far an observed rate could move a range (3-bet ≤ 4x prior, jam ≤ 6x chart) | 3-bet, open and cold-call rates: plain shrinkage, no upper cap; jam range: chart x observed ratio, never tighter than their observed deep open-jam rate |
| river bluff gate (blocker ranking inside a capped share of our range) | the EV comparison alone, with the pessimistic fold estimate |
| an opponent's check-raise was read from the range left after their check (strong hands already removed) | read from the range they had before checking; our own check-raise EV uses the region model on their betting range |
| "shover = tight range" side effect of a wrong all-in flag | all-in flag fixed; jam ranges from the solved charts x observed jam ratio |
| every bet narrowed a range, however small (a 1 bb bet into 90 bb was read as the top few percent) | the plan's "min" translation point: a minimum-size bet or min-raise is cheap and "often weak", so it leaves their range as it was, unless their showdowns show they only do it with strong hands |
| fixed 50% fold chance whenever we 4-bet or 5-bet | what the player has done when re-raised, shrunk to that level's starting values; the branch where we get it all in carries the all-in ICM risk premium |

**Second pass (2026-11-09, evening): line-by-line audit of the whole plan.**
Four reviewers went through the 290 KB chat one quarter each and checked every
directive against the code (about 400 items). Items the first pass had missed
or implemented loosely, now done the way the plan states them:

* **Our own river bluffs** are capped at s/(1+2s) of our betting range, best
  blockers first, after the alpha test with card removal; the same rule for
  river bluff-raises. (The first pass had dropped this as "optional"; the plan
  calls only the checking-range protection optional.)
* **Opponents' bet sizes translate onto min / 33 / 66 / 100 / 125 / all-in**
  (fold-to-bet is kept per point; the earlier code had 33 / 66 / 125 only).
* **Fold-to-bet is pooled across streets** ("one overall fold-to-bet plus a
  per-street offset") with the larger k for the rare later-street cells.
* **Showdowns calibrate the calling range** ("how strong their calling range
  is") and the flop / turn bluff shares of their bets, and the river bluff
  share is kept per bet size ("record the bet size next to the hand they
  showed").
* **A player who always uses one size gets no size inference** ("rely on its
  frequencies instead"); **their usual bet size** (prior: the 75% point) is
  what we expect when we check.
* **Deep opens are ordered by equity x playability** (JTs above A2o; push/fold
  keeps the raw ranking), hands at the edge of the range are mixed, and the
  opening range tightens against players who 3-bet a lot.
* **16-25 bb: fold / small open / open-jam** - the jam is taken when it is worth
  more in ICM points than the small open (the plan's middle band).
* **Re-jamming over a shove with players behind** prices a bigger stack
  behind calling for our whole stack (first-caller chain).
* **The hand-cap ICM boost applies preflop too** (it was postflop only).
* **Nut draws vs the rest**: +0.1 EQR only for nut draws and near-nut hands,
  -0.1 for non-nut draws (the plan's PLO discussion); nut flush and nut
  straight draws are detected.
* **Multiway**: 33-66% sizes only, raises only with very strong hands, bluffs
  only as nut-draw semi-bluffs; the pot when called counts the callers we
  expect, and players after a caller are priced at the better odds they get.
* **Geometric sizing with very strong hands**: one extra candidate size on the
  flop and turn (the pot fraction that gets stacks in by the river), chosen by
  the same EV comparison - the 33 / 66 / 125 menu is unchanged.
* **Their re-raise of our check-raise** and **their raise of our bet** carry
  EQR and the ICM risk premium like every other branch; on the river their
  raise is read as all-in with the plan's ~5% prior.
* **Turn EQR out of position 0.90** (the plan's turn solves), stratified
  runouts in the Monte Carlo, joint sampling for Omaha multiway preflop equity.
* **A simpler policy when time runs out** (past 80% of the decision budget:
  check / call / fold / one size, no check-raise model).
* **Between rounds**: `tools/field_priors.py` replays hand histories through the
  bot's own counters and prints the field's rates beside every prior; the
  tuner's space now includes the k values, the mixing band and the all-in
  threshold; data files for short deck and Omaha 5 / 6 are built.

Still not as the plan states it, and why:

| Plan item | State | Why |
|---|---|---|
| flop texture buckets, range statistics, presolved flop / turn / river games, EQR tables by bucket, per-bucket default sizes and frequencies | not built | the plan's own later ranking puts them after the core ("skip the full Hold'em solve"); the live EV with the region model is what the plan falls back to |
| measured EQR replacing the priors; the solve -> measure -> solve loop | built, switched off | scored worse twice in paired tests (+1.44 ± 0.40, +0.93 ± 0.48 for switching it off) |
| push/fold charts solved in ICM | chip-EV charts, exact ICM for our own decision | the plan revised this itself ("compute EV at the exact stacks ... better than interpolating"); opponents' modelled call widths near the bubble are a little wide |
| EQR depending on the line (check-through vs bet-and-call) | position, hand type, SPR, street only | the plan gives no values; nothing invented |
| implied / reverse implied odds for draws when calling | nut vs non-nut EQR only | the plan gives direction, not amounts |
| 3-bet / 4-bet all-in depth ~25 / ~40 bb | 35 / 35 | paired test +1.51 ± 0.38 |
| own bet sizes 33 / 125 / all-in, sizes in the tuner | 33 / 66 / 125, fixed | the team's decision |
| Omaha ranges as strength histograms | sampled 800-hand universe | one code path for every variant inside 1 ms |
| plug-ins for draw / stud / hi-lo / discard phases | safe mode | not built |
| a pure-numpy fallback if the sandbox lacks Numba | none | check in the first minutes of the build hour |

Deviations from the plan that remain, with the reason:

| Item | Plan | Bot | Why |
|---|---|---|---|
| Measured EQR | replace priors with measured values | priors only (`use_measured_eqr: False`; tool kept) | measured values scored worse twice in paired tests (+1.44 ± 0.40 and +0.93 ± 0.48 for switching them off) |
| 3-bet / 4-bet all-in depth | ~25 bb / ~40 bb | 35 bb / 35 bb | paired test +1.51 ± 0.38 |
| Omaha ranges | strength histograms + fitted model | sampled 800-hand universe, same code as hold'em | one code path for every variant inside 1 ms |

Not built (the plan put them after the core): the preflop solve, flop buckets
and presolved flop / turn / river games, the measure-and-iterate EQR loop
against the real field, plug-ins for draw / stud / hi-lo and discard phases.

Guard rails that are not plan numbers (they bound how far a small sample can
move an estimate): steal-width multiplier 0.6-1.8, jam and call-of-jam ratios
0.5/0.6-3, open / cold-call / blind-defence ratios at least 0.4-0.5 of the
baseline, EQR kept inside 0.6-1.3.

## Review of 2026-11-09: bugs fixed

Found by an independent review (about 100,000 debug decisions, no crashes or
illegal actions before or after) and fixed, each with a regression test:

* **Busted players were still counted** by the opponent statistics, so after
  the first bust every opponent looked looser than they were.
* **Chips nobody could call were priced as at risk / winnable** when we
  covered the opponent (preflop re-raises, postflop raises, check-raises, bets
  into a shorter stack, and the ICM rows). Now only the matched chips count.
* **Call-or-fold spots with no raise** (small blind against a big blind who is
  all-in from posting) always folded. Now decided in exact ICM points.
* **Pot-limit games**: a raise capped by the pot was priced as an all-in. Now
  a shove is assumed only when one is really on offer.
* **Omaha draws** ignored the two-hole-card rule (a single suited card counted
  as a flush draw, one-card straights too). Fixed and the strength weights
  refitted: rank correlation with true equity 0.90 -> 0.96 (flop), 0.97 -> 0.98 (turn).
* **Non-hold'em preflop equity table**: equity against a narrow band was a
  difference of two noisy numbers (errors up to 0.22). Now each band is
  sampled on its own (RMS error 0.045 against direct sampling).
* Smaller ones: big-blind defends were mixed into the cold-call rate; a lone
  short blind post could be read as the wrong blind; heads-up, the big blind
  raised a limp with the button's 80% range; a pessimistic estimate could go
  out of range; with an all-in main opponent the model now uses one who can act.

**Why it earned less from always-raising bots** (follow-up on the same day).
Against a bot that min-raises every time, the bot folded to 1 bb bets with
enormous pot odds (king-high folded for 1 bb into a 92 bb pot) because every
bet, however small, narrowed the opponent's range and three streets of that
compounded to "top few percent". The plan covers this (the "min" translation
point for opponents' sizes, "min-bets are cheap and often weak: call or raise
wide", "watch whether a particular opponent only min-raises strong hands") and
the code had not implemented it. Now a minimum-size bet or a min-raise leaves
the range as it was, in the share the pseudo-harmonic rule gives the "min"
point, unless that player's showdowns after min-size bets say otherwise. The
second cause: when we 4-bet or 5-bet, the opponent's fold chance was a fixed
50% that never learned, so against a bot that never folds the bot kept
re-raising marginal hands and busted before the money (23 vs 19 pre-money
busts per 100 tables). Now the response uses what the player has done when
re-raised, shrunk to that level's starting values, and the stack-off branch
carries the all-in ICM risk premium. Paired effect of the two changes (900
tables): simple field +0.6 ± 0.2 in hold'em and +3.2 ± 0.9 in Omaha, nothing
measurable elsewhere. Chips won from the min-raiser doubled; most of that is
variance rather than points.

## Results (local simulator, M-series Mac, 2026-11-09)

**Paired tests** (`tools/ab_code.py`: both versions play the same tables,
seats and payouts; points per table, new minus old; two seeds pooled).
strong = 1 of ours + the 7 strong bots, simple = + the 7 simple bots,
mixed = 2 + 2 + strong bots, head-to-head = 4 copies vs 4 copies.

| Final code against | Variant | strong | simple | overall |
|---|---|---|---|---|
| the code before this review (fixes only, before the min-size work) | hold'em | +1.4 ± 1.0 | +0.2 ± 0.8 | **+0.6 ± 0.35** |
| | Omaha | +2.3 ± 1.1 | +0.1 ± 1.0 | **+0.45 ± 0.4** |
| the same code with the *old* (pre-review) postflop, final code | hold'em | +0.5 ± 1.0 | −0.8 ± 0.75 | **−0.1 ± 0.35** |
| | Omaha | +0.8 ± 1.2 | +0.5 ± 1.1 | **+0.1 ± 0.5** |

Reading: the plan-faithful postflop is about equal overall to the old one,
a little better against competent opponents, and in hold'em still a little
worse against the simple bots (the remaining difference is a few more
deep-stacked stack-offs before the money). Head-to-head the older versions
score 0 to 0.6 more than the final one; that scenario rewards modelling copies
of yourself. The bug fixes alone: hold'em +0.63 ± 0.38, Omaha +0.21 ± 0.52.

**Final code vs the seven strong bots**, 8-max, 600 tables per variant, seats
rotated, payouts 50/30/20 (`tools/final_nlhe.json`, `tools/final_plo4.json`):

| Hold'em (no-limit) | pts ± se | avg place | win % | in the money % |
|---|---|---|---|---|
| **our bot** | **20.22 ± 0.83** | 3.58 | 25 | 56 |
| ValueReg | 13.90 ± 0.70 | 4.13 | 10 | 44 |
| NashICM | 13.12 ± 0.71 | 4.29 | 11 | 41 |
| TAG | 11.90 ± 0.71 | 4.49 | 11 | 37 |
| BayesReader | 11.63 ± 0.75 | 4.77 | 14 | 32 |
| LAG | 11.30 ± 0.74 | 4.85 | 14 | 32 |
| Adaptive | 10.44 ± 0.69 | 4.64 | 11 | 32 |
| Balanced | 7.48 ± 0.59 | 5.25 | 6 | 24 |

| Omaha-4 (pot-limit) | pts ± se | avg place | win % | in the money % |
|---|---|---|---|---|
| **our bot** | **17.80 ± 0.88** | 4.11 | 27 | 45 |
| NashICM | 14.26 ± 0.71 | 4.16 | 12 | 46 |
| ValueReg | 13.80 ± 0.66 | 3.97 | 8 | 47 |
| TAG | 12.57 ± 0.71 | 4.52 | 11 | 39 |
| Adaptive | 11.99 ± 0.72 | 4.45 | 11 | 36 |
| LAG | 11.42 ± 0.73 | 4.80 | 13 | 33 |
| BayesReader | 10.97 ± 0.73 | 4.88 | 13 | 31 |
| Balanced | 7.20 ± 0.58 | 5.12 | 6 | 24 |

(Fair share is 12.5. The pre-review code scored 18.11 and 15.28 in the same
set-up; the paired tests above are the cleaner comparison.)
Against the seven simple bots, 600 tables: 30.2 ± 0.8 (hold'em), 27.7 ± 0.8 (Omaha).

**Checks.** 82 tests pass. Debug runs (`PB_DEBUG=1`, strategy errors crash):
about 95,000 decisions in hold'em and Omaha at 2, 3, 6 and 8 players, 0 errors,
0 illegal actions. Upload check (heads-up vs check/fold, no time bank, 60 runs
each): 0 failed decisions; slowest decision 0.15 ms (hold'em), 0.39 ms (Omaha).

**Timing** (real rules: 1 ms + 100 ms bank per table; one process, idle
machine, 8-max vs the strong bots, 30 tables):

| | p50 | p90 | p99 | max | least bank left |
|---|---|---|---|---|---|
| Hold'em | 0.42 ms | 0.79 ms | 1.00 ms | 2.7 ms | 94.0 of 100 ms |
| Omaha-4 | 0.45 ms | 0.82 ms | 1.03 ms | 3.2 ms | 96.4 of 100 ms |

(One earlier run on a busy machine had a single 9 ms decision and one table
that used 34 ms of its bank; the tables above are from an idle machine.)

The plan-faithful postflop costs more than the old one (about 0.75 ms for a
postflop decision; three bet sizes, the region model and ICM for each option).
No timeouts here, but a build box 2x slower would lean on the bank for most
postflop decisions. **Benchmark there first.** If p90 is over 1 ms, lower
`budget_frac` (0.55) and `mc_samples_min` (80), in that order.

## Test opponents

`sim/bots.py`: simple bots (check/fold, min-raiser, calling station, maniac,
nit, jammer, random). `sim/strong_bots.py`: seven competent bots that use
real equity, ranges, pot odds, solved push/fold and ICM, in hold'em and
Omaha: **TAG**, **LAG**, **NashICM**, **Adaptive**, **Balanced**,
**ValueReg**, **BayesReader** (`tag, lag, nashicm, adaptive, balanced,
valuereg, bayes` in `sim/run.py`). They use `sim/pb_ref`, a frozen copy of an
older calculator, so they do not change when the bot does.

The simulator plays `--variant nlhe` (no-limit) and `--variant plo4|plo5|plo6`
(pot-limit by default) and sets `PB_VARIANT` so the bot loads the right plug-in.

## Run it

```bash
python3 -m pytest -q tests/
python3 sim/run.py --bot bot --opponents tag,lag,nashicm,valuereg,adaptive,balanced,bayes --players 8 --tables 400 --procs 8 --no-timing
python3 sim/run.py --bot bot --opponents tag,lag,nashicm,valuereg,adaptive,balanced,bayes --players 8 --tables 400 --procs 8 --no-timing --variant plo4
python3 sim/run.py --bot bot --opponents nit,station,maniac,jammer,minraiser,random,checkfold --players 8 --tables 400 --procs 8 --no-timing
python3 sim/run.py --bot bot --heads-up-test --tables 50                 # the upload check
python3 sim/run.py --bot bot --opponents tag,lag,nashicm --players 8 --tables 40 --procs 1   # real timing
python3 tools/pack.py
```

Compare two versions of the bot on identical tables (seeds, seats, payouts):

```bash
tools/snapshot_bot.sh /tmp/old_bot pb_old            # before you change something
python3 tools/ab_code.py --old /tmp/old_bot --old-pkg pb_old --tables 300 [--variant plo4]
```

`tools/tune.py` searches the TUNED knobs by simulation (every candidate plays
the same tables). Only adopt a change that wins a paired validation by about
two standard errors.

## Build-hour checklist

1. Download the kit. Read `RULES.md` (variant, betting limit, any new phases or action kinds) and the Format page (payouts, round weights).
2. `python3.12 -m venv .venv`, install the kit requirements, run our tests **against the real SDK** (drop `stub_sdk` from the path).
3. Check the real shapes against `pb/view.py` (blinds object, `pot`, event `cards`, legal action fields). Fix there only.
4. Set `settings.VARIANT`. Hold'em and Omaha-4 need nothing else. Any other community-card variant: `python3 tools/fit_scores.py` and `python3 tools/build_pushfold.py` (they write `bot/data/scorefit_<name>.npz` and `pushfold_<name>.npz`).
5. `poker-agent check bot/agent.py`, then `poker-agent play bot examples/check_or_fold examples/min_raiser --tables 200`, and an 8-seat run.
6. **Benchmark on the build box** (p99 and max of act). If p99 is near 1 ms, lower `budget_frac` / `mc_samples_max`.
7. `python3 tools/pack.py`, upload, confirm the test match passed. Submit early.
8. Between rounds: download hand histories, compare our priors with the field, resubmit before each freeze.
