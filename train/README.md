# Self-play trainer (Boston: hold'em with a sealed flop auction)

Builds the lookup tables in `bot/data/trained_holdem.npz` by playing the game
against itself, one generation at a time. Not uploaded; only the tables and
`bot/pb/absx.py` + `bot/pb/trained.py` go into the bot.

```bash
python3 train/run.py --gens 3                               # generations 1-3 (about 10 min each on 4 cores)
python3 train/run.py --resume train/out/gen3.npz --gens 2   # later: generations 4 and 5
```

`--post` (heads-up flops per postflop pass, default 300,000) and `--pre` (hands
per preflop pass, default 24,000,000) set the sample sizes. Each generation
writes `train/out/gen<N>.npz` and adds to `train/out/report.json`.

## What a hand is reduced to

| Street | Looked up by |
|---|---|
| Preflop | stack band (<10, 10-20, 20-40, 40+ bb), situation (unopened, facing limp, facing open, limped facing raise, raised facing re-raise, cold facing re-raise, facing jam), seat, where the last raiser sits, the 169 starting hands |
| Flop | position, SPR, what we face (size class, bet or raise), who raised last preflop, **range-vs-range equity** and **nut advantage** (three levels each), and the hand's bucket |
| Turn, river | position, SPR, what we face, who bet last on the street before, and the hand's bucket |

A hand's bucket has two factors, and hands in the same bucket are the same hand:

* **strength**: the share of the opponent's range it beats on this board (10 levels,
  finer at the top). The opponent's range is tracked through every action they take.
* **equity realization class**: nut draw / made hand of at least top pair with a jack
  kicker / everything else (non-nut draws and weaker made hands).

Actions: fold, check or call, bet or raise 10 / 33 / 66 / 125% of the pot, all-in
when the stack is under 5 pots. Preflop: under 10 bb jam / limp / fold; 10-20 bb
jam / limp / min-raise; 20-40 bb open 2.2x, one re-raise, then jam; 40+ bb open
2.5x, re-raise, 4-bet, then jam.

## Generations

Generation 0 is the stated starting point: open the top 15% first to act and 5%
more per seat (small blind 45%); equity realization 1.0, minus 0.15 out of
position, plus 0.15 for nut draws, minus 0.15 for non-nut draws and made hands
under top pair with a jack kicker. Postflop EVs come from equity x equity
realization, with a bet called by the share of a range the pot odds require.

Each later generation runs three passes with the previous generation's tables:

1. **Turn and river together.** Hands are played from the deal; the EV of every
   turn and river action is measured and those tables rebuilt.
2. **Flop**, played through the new turn and river tables. Besides the flop EVs
   this pass measures the value of each flop bucket (what the auction bids on),
   the equity realization of each starting hand by position, stack band and
   pot type, and how payments relate to hands.
3. **Preflop.** The EV of every preflop action, where a hand that sees a flop
   is worth `equity x measured equity realization x pot` and an all-in is
   worth its equity.

EVs are measured by letting one player make one uniformly random deviation per
hand and recording what that decision earned; everything else follows the
tables. Each new estimate is shrunk to the previous generation's (and flop cells
split by range-vs-range equity and nut advantage shrink to the unsplit cell).
Actions are chosen by a softmax over the EVs, so hands with clearly different
EVs play one action and close ones mix.

## The auction

After preflop, both players value all 10 possible flops (the measured value of
the bucket their hand lands in on each), then bid:

`EV(bid) = sum over flops P(flop | bid) x value(flop) x pot - price x chips paid`

* `P(flop | bid)` comes from the opponent's bids: 24 hands sampled from their
  range, each bidding a share `kappa` of what its best flop is worth. `kappa`
  is refitted every generation to what the best-response bids actually spent.
* The bid is the best of 13 candidates (nothing, the best two flops at six
  spend levels, one card alone at two levels).
* `price` (big blinds per flop-chip) is retuned each generation so the average
  payment is 8% of the balance, and scales with 1 / balance.
* What a seat paid then narrows its range, using your payment buckets (cannot
  bid, nothing, token 1-3, up to 5%, 5-15%, over 15% of the balance), by stack band.

How bidding enters the rest: the flop values and each starting hand's equity
realization are measured on hands that went through the auction, so preflop and
the flop tables already price it in. Range-vs-range equity and nut advantage are
computed on the flop the auction selected, with the two dead cards removed.

## What is simplified

* Heads-up after the flop. Multiway flops (about 5% of hands in self-play) are
  not trained; preflop values them with the product of pairwise equities.
* Equal stacks at the table and chip EV. No ICM, no antes.
* A preflop all-in is worth its equity; the auction that follows it is ignored.
* The opponent's bids are not used to narrow ranges before the flop is known,
  only the payment afterwards.
* No exploits: the spending profile per seat is runtime-only and not built yet.

## Results so far (three generations, 2026-11-10)

Generation 1 used 300,000 flops per postflop pass and 24M preflop hands;
generations 2 and 3 used 180,000 and 12M to save time. Each table-driven hook
was then played against the current bot on the starter-kit engine
(`tools/ab_kit.py`: four copies of each at 8-player tables, 144 tables, real
1 ms clock, no failed decisions on either side):

| Hook | Points per table vs the current bot | Setting |
|---|---|---|
| postflop tables | -2.71 ± 1.11 | off |
| auction bids | +1.15 ± 1.15 | off (not yet distinguishable from zero) |
| preflop tables | not played | off |

The preflop table is not usable yet: plain best responses swing from one
generation to the next (first-to-act opens at 40+ bb went 11% -> 2% -> 30%),
and generation 3 opens hands like K5o and 87o. `run.py` now moves only half of
the way to each new estimate from generation 2 on (`damp`), which the first
three generations did not have. Generations 4 and 5 resumed from `gen3.npz`
use it.
