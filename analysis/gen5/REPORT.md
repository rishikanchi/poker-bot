# Guppy gen 5 — analysis of 1,001 tables (flop-auction hold'em)

Source: `hands-690def26-…-first-1000.jsonl.gz` (1,001 eight-handed sit-and-gos, 139,158 hands,
payouts 50/30/20, 192 distinct opponents, no filler seats) and the uploaded `guppy_gen5.zip`.
All "pts" below are tournament payout points per table; the field average is 12.5.

## 1. Where we stand

| | Guppy gen 5 | Break-even |
|---|---|---|
| Average payout | **14.85** (± 0.60 s.e.) | 12.5 |
| 1st / top 3 | 15.9% / 43.6% | 12.5% / 37.5% |
| 8th place | 13.2% | 12.5% |
| Rank among the 192 opponents seen | about 59th | |

The best opponents in this sample average 23–27 pts (each over only 30–55 tables, so those
numbers are inflated by luck, but the gap is real). Against the weakest quarter of tables we
score 16.8; against the strongest quarter we score **11.0** — we beat weak bots and lose to
good ones.

Where the points come from, by stage (sum of our ICM change per table):

| Players left | 8 | 7 | 6 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|---|---|
| pts / table | +1.08 | +0.53 | +0.35 | +0.33 | **−0.61** | **−0.24** | +0.07 |

By stack depth: 40bb+ **+2.07**, 20–40bb +0.35, 10–20bb 0.00, under 10bb **−0.91**.
We build an edge deep-stacked and early, then give a third of it back short-stacked and on
the bubble.

"TOP30" below means the 30 opponents with the highest average payout in this sample.

## 2. Finding 1 — the auction is our largest leak

Rules as confirmed from the data: five cards are shown, the three with the highest total
bids become the flop (ties go to the earlier card; with no bids the flop is the first three,
which happened in 19% of auctions), and each player pays only their bids on the cards that
were selected. Checked against all 10,238 of our own bids: no exceptions.

**How much the flop choice matters.** Using 18,136 heads-up auctions where both hands were
later shown, I computed each player's exact equity on all 10 possible flops:

* The gap between a player's best and worst flop averages **42% of the pot** (all-in pots)
  and 46% (pots with later betting).
* The player who pays more ends up with about **+7% to +11% of the pot** in equity
  relative to the average flop; the one who pays nothing while the opponent pays gets −11%.

**What we do.**

| | Guppy | TOP30 | Middle | Bottom 60 |
|---|---|---|---|---|
| Chips paid per auction | **17** | 51 | 47 | 20 |
| Flop chips unspent at table end (of 1,000) | **782** | 351 | 499 | 808 |
| Chips left at hand 100, if alive | 790 | 435 | | |
| Equity gained in all-in pots vs average flop | **−1.3%** | +2.1% | +0.8% | −3.6% |

Unspent chips are worth nothing, and across opponents unspent chips correlate −0.61 with
average payout. We bid a median 1.6 chips per big blind of pot; opponents pay a median 2.9,
upper quartile 7.7, top decile 14.6. We bid **zero in 19% of auctions**, including 20% of
all-in pots averaging 33bb, where we then run 4.9% below the average flop.

**What it costs.** We have 2.6 heads-up preflop all-ins per table, and winning versus losing
one is worth 11.1 pts on average. So:

| Our equity vs average flop in all-in pots | pts / table vs now |
|---|---|
| 0% (just stop losing auctions) | +0.4 |
| +5% | **+1.8** |
| +10% (what consistent out-bidders get) | **+3.3** |

This is for all-in pots only; ordinary pots add more. Our own data shows the response: when
we bid 36–110 chips we gained +5.4%, and at 111+ we gained +9.5%.

**Why gen 5 under-bids (from `agent.py` / `core.py`).**

1. Flop values come from the 10×4 strength/EQR buckets. When all ten flops land in the same
   bucket (for example ace-high that misses every flop), every candidate bid has zero edge
   and `choose_bid` returns nothing — even though real equity against the opponent's range
   differs a lot by flop.
2. All-in pots use the same bucket table (`EQUITY[0][strength][eqr]`), not equity against a
   calling or jamming range.
3. The opponent model `OPP` came from self-play, where bids are far smaller than the real
   field's.
4. Pacing can only cut the chip price to 0.25×, and assumes 150 hands. The median table is
   139 hands and we are eliminated at a median of hand 99, so the budget is never reached.

**Recommended changes.**

* Price chips by budget, not by a fixed constant: we see about 12.6 auctions per table, so
  roughly 80 chips per auction are available. Target spending the stack down by about hand
  100–120, and weight by pot size in big blinds and by whether our tournament life is at
  stake.
* In all-in pots bid on real equity: for each of the 10 flops, equity of our two cards
  against the opponent's likely range. To beat the field three times in four in a big pot
  takes roughly 100–150 chips; the top decile of opponents pays 200+.
* Never bid zero when we hold chips and the pot is contested; replace the bucket values
  with something that distinguishes flops (even a cheap lookup: pair/overcard/suit/
  connectedness of each candidate card with our hand and with a typical opponent range).
* Track each opponent's chips paid per auction (visible every hand via the flop-chip
  balances) and bid just above what that opponent usually pays. Fifteen opponents never bid
  at all; against them a token bid wins the flop.

## 3. Finding 2 — short-stack and bubble play is too tight

First-in (folded to us) action:

| Spot | Guppy raise + jam | TOP30 | Middle |
|---|---|---|---|
| Button, all depths | **26%** | 54% | 46% |
| Small blind, all depths | **43%** | 74% | 57% |
| Cutoff, all depths | 20% | 35% | 33% |
| Any seat, under 10bb (jam) | **29%** | 59% | 44% |
| Button, under 10bb | 19% | 55% | 41% |
| Any seat, 10–20bb, open-jam | 3% | 19% | 13% |
| PFR with 4 players left | **18%** | 34% | 26% |
| PFR with 3 players left | **23%** | 45% | 31% |

Across opponents, steal frequency and under-10bb jam frequency are the two statistics most
correlated with payout (both +0.59).

**Counterfactual.** For every first-in hand we folded with 20bb or less (6,833 hands, our
cards known), I computed the exact ICM value of jamming instead, using the field's measured
call frequencies by seat and stack and the hands callers actually showed. The model
predicts +0.82 pts per jam on the jams we really made, against +0.61 realized, so it runs
about 0.2 optimistic; the conservative column applies that haircut and counts only hands
ahead by more than 0.5 pts.

| Hands we folded | Jam was better | Gain, all +EV jams | Gain, conservative |
|---|---|---|---|
| All, ≤ 20bb | 59% | +2.15 | **+1.26 pts/table** |
| Small blind | 92% | +0.68 | +0.46 |
| Button | 80% | +0.85 | +0.49 |
| Cutoff | 48% | +0.42 | +0.25 |
| 4 players left | 51% | +0.80 | +0.59 |
| 3 players left | 89% | +0.54 | +0.30 |
| Under 4bb | 87% | +0.59 | +0.46 |

Open-jams get everyone to fold 73% of the time on average (85% from the small blind).

Two related leaks:

* **Limping.** We open-limp 5% of first-in hands (20% from the small blind at 10–20bb),
  then fold to a raise 76% of the time. Realized cost −0.23 pts/table; jamming those same
  hands at ≤ 20bb would have been worth about +0.4. `pre_legal` allows CALL first-in in
  every regime, including under 10bb.
* **Non-all-in steals deep.** A 2.4bb small-blind open folds out the big blind 62% of the
  time against a break-even of 56% — profitable with any two cards — and we fold 42–50% of
  our small blinds.

**Calling jams.** Our calls are sound (81% of them correct, +1.5 pts/table earned). Of
3,113 folds facing a single open-jam, 540 were clear calls by the same method, worth about
**+1.5 pts/table**, mostly in the big blind and with 2–4 players left; at under 4bb jam
size we call 34% in the big blind where TOP30 call 50%. This estimate relies on
field-average jam ranges, so treat it as an upper bound unless we track each opponent.

**Likely cause.** `icm_means` converts a chip EV into a two-outcome bet (win `gain` or lose
`risk`) priced against the biggest stack. A jam is a three-outcome bet — usually everyone
folds, sometimes we are called and win, sometimes called and lose — so this conversion
understates jams whose value is mostly fold equity. Computing the three ICM outcomes
directly (as the counterfactual does) is cheap with `hero_icm`.

## 4. Finding 3 — postflop: calls the turn too much, small bets, gives up rivers

| Statistic | Guppy | TOP30 | Middle |
|---|---|---|---|
| Fold to turn bet | **27%** | 47% | 38% |
| Fold to river bet | 55% | 57% | 45% |
| Fold to a raise of our flop bet | **60%** | 39% | 28% |
| River bet when we bet the turn | **24%** | 72% | 75% |
| Median bet size, flop / turn (pot) | **0.40 / 0.33** | 0.60 / 0.65 | 0.60 / 0.60 |
| Went to showdown / won at showdown | 34% / 58% | 29% / 53% | 35% / 51% |

* Calling a big turn bet (737 hands) loses 2.8bb per hand; 30% of those hands end with us
  folding the river for −16.7bb. `postflop()` uses the flop tables on the turn ("draws are
  still live"), which over-values draws and marginal pairs with one card to come.
* After betting the turn we check the river 76% of the time and lose 1.5bb per hand on
  those; when we do bet we win the pot 77% of the time. The pattern is a small turn bet
  with a weak hand followed by giving up.
* Betting the flop then folding to a raise 60% of the time is exploitable; good opponents
  fold 39%.
* Sets, straights and better are checked on the flop about 60% of the time. Betting them
  earned more than checking (sets 17.8bb vs 11.9bb; nut hands 30bb vs 13bb), though
  position partly explains the difference.
* By first flop action, calling a flop bet is our worst line: −1.0 pts per hand as the
  preflop raiser, −1.9 with weak top pair, −2.2 with an overpair.

Overall, hands that see a flop win +1.18bb each yet net −0.08 pts/table: the chips we win
postflop are offset by the ICM cost of the pots we lose.

## 5. Finding 4 — no opponent memory in a very uneven field

`act()` passes `memory` through untouched. The field differs enormously per opponent
(10th to 90th percentile across entries):

| Statistic | 10th | Median | 90th |
|---|---|---|---|
| Big blind folds to a steal | 27% | 74% | 91% |
| Calls an open-jam | 3% | 12% | 24% |
| Steals from CO/BTN/SB | 10% | 42% | 66% |
| Folds to a 3-bet | 27% | 68% | 86% |
| Folds to a postflop bet | 25% | 46% | 73% |
| Chips paid per auction | 3 | 42 | 74 |

There are 41 opponents whose big blind folds to steals 85%+ of the time, 15 that never call
a jam, 15 that never raise, 15 that never bid, and 28 that fold to 65%+ of postflop bets.
A table lasts about 139 hands, so a handful of per-seat counters in the memory bytes
(fold-to-steal, call-jam, chips paid per auction, fold-to-bet) would converge quickly and
feed directly into the jam, steal and bid decisions above.

## 6. Smaller observations

* **How we bust** (842 eliminations): 30% jamming first-in (median stack 3.7bb — we wait
  too long), 25% calling an all-in, 25% stacking off postflop (median 28.7bb), 14%
  re-jamming, 6% blinded out.
* **Big blind under 10bb** costs −2.1 pts/table on its own; it is where both the tight
  jam-calling and the late jamming show up.
* Our fold-to-3-bet is 53% against 70% for good opponents, and calling 3-bets after opening
  loses −0.59 pts per hand.
* Five automatic (timed-out or defaulted) actions in three tables — negligible.
* Payout rose from 12.8 (first 200 tables) to 16.6 (tables 800–999) with opponent strength
  flat; each block has a standard error of about 1.3, so this is probably noise.

## 7. Priorities for generation 6

| # | Change | Evidence | Rough upside |
|---|---|---|---|
| 1 | Spend the flop-chip budget; bid on real equity, hardest in all-in pots | §2 | +2 to +3 pts |
| 2 | Three-outcome ICM for jams; much wider first-in jams ≤ 20bb, especially BTN/SB and 3–5 handed | §3 | +1.3 to +2 |
| 3 | Remove first-in limps; open any reasonable hand from the small blind and button | §3 | +0.4 to +0.8 |
| 4 | Per-opponent counters in memory (steal, call-jam, bid size, fold-to-bet) | §5 | multiplies 1–3 |
| 5 | Separate turn table; fold more to big turn bets; bigger bets; stop bet-turn / check-river with air | §4 | uncertain, likely +0.5 to +1 |
| 6 | Call jams wider in the big blind when short or short-handed | §3 | up to +1.5, opponent-dependent |

These are not additive (a wider jam range changes how often the auction matters, and so
on), but items 1 and 2 are each comparable to our entire current edge of +2.35 pts.

## 8. Caveats

* Opponents' bids are private; "paid" is what they were charged on selected cards, so it
  understates their bids when they lose the auction.
* The auction equity study uses hands shown at showdown. The all-in subset is unbiased;
  the subset with later betting is not.
* Push/fold counterfactuals use field-average call and jam ranges, ordinary hold'em
  all-in equity (the auction's effect on all-in equity is ignored), and at most one caller.
* Group comparisons (TOP30 vs us) are correlations, and TOP30 membership is itself partly
  luck at 30–55 tables each.
* The gen 5 docstring mentions a ~1 ms decision budget; the equity-based bidding in §2 has
  to fit that, so it likely needs precomputed tables rather than live simulation.

## Reproducing

`scripts/` holds everything used here. `load.py` reads the `.jsonl` (pass its path as the
first argument on the first run; a pickle cache is written next to the scripts), `hp.py`
parses hands, `ev.py` / `icm.py` are the evaluator and ICM (ICM reproduces the logged
`equity_before` to 5e-5). Then run `a1_results.py` … `a9_misc.py`; `a4` must run before
`a5`, and `a6` needs `GEN5_DIR` pointing at the unpacked gen 5 bot. Requires numpy and numba.
