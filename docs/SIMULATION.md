# How the score simulation works

The simulator produces a range of possible games rather than one predicted score. It does not
start from a win probability. Win chances are simply counted from the simulated scores.

## The two steps

**1. Expected points.** A regularized linear model predicts each team's points after four quarters,
one team at a time. Inputs are that team's offense, the opponent's defense, pace, turnovers, Elo
ratings and schedule strength, recent form, recruiting, returning production, and home or away.
Offense and defense stats are adjusted for opponent strength and use only earlier games.

**2. Monte Carlo draws.** For each simulated game:

- A real pair of regulation scores is drawn from 2016-20 games where both teams had similar
  expected points. Weights are nudged so the averages match this matchup.
- If that pair is tied, the game goes to overtime. A winner is drawn using a small model of how
  the gap in expected points affected the winner of real overtime games.
- Both teams then add the points from a randomly chosen real overtime. Games never end tied.

## Why real pairs

Two designs were compared by forward testing inside 2016-20:

| Forward test, 2018-20 | Actual | Separate draws | Real pairs |
|---|---:|---:|---:|
| Overtime rate | 3.9% | 2.7% | 4.4% |
| Margin within 3 | 15.8% | 11.2% | 15.2% |
| Margin exactly 3 or 7 | 17.8% | 11.5% | 18.0% |

After accounting for expected points, the two teams' results have almost no overall correlation
(0.02 on 2016-20). A single correlation number therefore can't reproduce close finishes and key
margins. Drawing real pairs does, because it keeps whatever happened together in real games.

The raw correlation between the two scores is negative (about -0.16) mostly because strong teams
score more and allow less. The simulation reproduces this through the expected points (-0.15).

2021-22 first revealed the problem, but this design was chosen from the 2016-20 test above and
checked on 2021-22 only once afterward.

## Assumptions

- **The past resembles the future.** Scoring patterns from 2016-20 are assumed to carry forward.
- **Expected points capture what's knowable before kickoff.** Injuries, weather, suspensions,
  coaching changes, and player opt-outs are not modeled.
- **Similar expectations mean similar game shapes.** A game's pattern is assumed to depend only
  on the two expected scores, and not on styles such as a run-heavy team against a pass-heavy one.
- **Team strength is fixed for a game.** Uncertainty about how good a team really is only appears
  through past forecast errors.
- **Each team's expected points are additive.** The linear model has no interaction terms, so
  an unusual matchup can't produce an effect larger than its parts.
- **Overtime follows 2016-20 patterns.** NCAA overtime rules changed in 2021, so very long overtimes
  now play differently. The effect on win probability should be small because overtime is rare, but
  this hasn't been measured separately.
- **No score can be outside what was seen.** Regulation pairs come only from 2016-20 games, so a
  scoreline that never occurred can't be drawn before overtime is added.

## What 10,000 simulations does and doesn't do

More simulations reduce random noise only. Two independent 10,000-game runs differ by about 0.5
percentage points in win probability on average, and by at most 1.9 across 2021-22 games. If the
expected points are wrong, 10,000 runs simply measure the wrong answer precisely.

## How realism is tested

Full results are in the [Stage 3 report](../reports/stage3_diagnostics.md).

**Matches well on 2021-22**

- Overtime: 4.2% simulated, 4.5% actual.
- Margin of exactly 3 or 7: 18.1% simulated, 19.4% actual.
- Rare team scores (1, 2, 4, 5, 8): 0.2% simulated, 0.1% actual. A bell curve gives 3.6%.

**Known weaknesses on 2021-22**

- **Ranges run too wide.** 80% intervals held the result about 83% of the time, and 90% intervals
  about 93%. Simulated totals have a standard deviation of 18.6 against 17.3 actual. That 18.6 equals
  the 2016-20 actual, so the model appears to be reproducing the wider-scoring seasons it trained on.
- **High-scoring games are overstated.** Totals of 80 or more: 10.1% simulated, 7.9% actual.
- **Close games are still a little low.** Margin within 3: 15.5% simulated, 17.9% actual.

These were left uncorrected, so 2021-22 stays a diagnostic rather than a tuning set.

**Other risks**

- College football changed its clock rules in 2023, which could have caused overpredicted totals. The
  [final evaluation](FINAL_EVALUATION.md) found no such problem overall (average total off by +0.16 points).
- New FBS teams have no returning-production data. All five of the largest disagreements with the
  stats model involve such a team, so forecasts for those teams deserve extra caution.
- Returning production is one of the larger factors, and its exact publication timing isn't
  confirmed. See [data sources](DATA_SOURCES.md).
