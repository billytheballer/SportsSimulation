# Final evaluation: what the results mean

This explains the permanent results in the [final evaluation report](../reports/final_evaluation/final_evaluation.md).
That report and its data file were generated once and are read-only. This page is commentary on them
and changes no numbers.

## How the test was protected

- **Frozen first.** Settings and fingerprints of 13 source files, the Stage 3 report, and cached
  data were saved before the run. The run checks them and refuses if anything changed.
- **Pipeline checked without touching the test.** The same code ran on 2021-22 first and reproduced
  the Stage 3 numbers exactly on all seven checks.
- **Run once.** One attempt, on 2,398 FBS vs FBS games. A second attempt was tried on purpose and refused.
- **No tuning from these results.** The evaluated models were fit on 2016-20 and never changed.
- **No forecast saw its own result.** Ratings and team stats updated game by game, using only earlier games.

**What was seen beforehand.** Early data checks printed how many 2023-25 games existed, per-season
counts of neutral-site games and missing CFBD Elo values, and the largest margin across all seasons.
Earlier development runs also calculated ratings through those seasons without scoring them. No
accuracy or error figures for these seasons were seen before this run. Model settings had already
been chosen by then and were not changed.

## Main results

| Forecast | Accuracy | Log loss | Margin MAE | Total MAE |
|---|---:|---:|---:|---:|
| Listed betting line (benchmark, timing unverified) | not comparable | 0.5213 | 11.99 | 12.68 |
| **Simulation** | **71.6%** | **0.5376** | **12.47** | **12.84** |
| Team-stat model | 71.6% | 0.5404 | 12.45 (regression) | n/a |
| CFBD's Elo (benchmark) | 70.8% | 0.5529 | n/a | n/a |
| Our Elo | 70.2% | 0.5589 | 13.28 (regression) | n/a |
| Home win rate from training | 58.4% | 0.6776 | 15.78 | 13.87 |
| Coin flip | 50.0% | 0.6931 | n/a | n/a |

- **Both stat-based models clearly beat Elo.** The simulation's log-loss edge is +0.0213
  (95% interval +0.0114 to +0.0316), and the stats model's is +0.0186 (+0.0076 to +0.0301).
- **The simulation and stats model are effectively tied.** The simulation's edge is +0.0028, with
  an interval of -0.0015 to +0.0068 that includes zero. Using the simulation as the main number
  is a design choice, as it keeps win chances and score ranges consistent.
- **The listed betting line is better.** Its log-loss advantage over the simulation is 0.0163
  (interval 0.0084 to 0.0242). Markets can react to news this model can't see.

**These scores are better than 2021-22 (0.5551).** That doesn't mean the model improved, since
it's the same model. Elo also improved, from 0.5757 to 0.5589, and CFBD's Elo from 0.5702 to 0.5529,
so these seasons appear to have been easier to predict.

## Where it does well

- **Calibration is good.** From the favorite's side, forecasts of about 75%, 85%, and 95%
  came true 75.6%, 86.2%, and 94.6% of the time. The largest gap is 65.0% forecast against 62.7%
  actual, which is about one standard error. The team-stat model's largest gap is at 74.7% against 70.3%.
- **Mismatches.** Power vs. Group of Five games: 82.4% accuracy, log loss 0.3610.
- **Early season.** Weeks 1-4: 75.2% accuracy, log loss 0.4925, against 0.5433 for Elo. Carrying
  over stats, recruiting, and returning production helps most here. Those weeks also include many mismatches.
- **Strong favorites.** When the favorite is given 80% or more, it won 90.1% of the time.
- **Total scoring level.** Average total points were off by only +0.16. The concern that 2023 clock
  rule changes would cause overpredicted totals didn't show up overall.
- **Realistic shapes.** Overtime was 4.2% simulated against 4.8% actual. Margins of exactly 3 or 7
  were 18.1% against 19.2%.

## Where it struggles

- **Bowl games and the rest of the postseason.** 57.5% accuracy on 134 games, with log loss
  0.7022, worse than a coin flip's 0.6931. The stats model and Elo did no better. Player opt-outs,
  transfers, and coaching changes are likely reasons, but that hasn't been tested.
- **Neutral sites.** 60.4% accuracy on 202 games, log loss 0.6649. This overlaps heavily with bowl games.
- **Teams with no returning-production data.** On 57 games, log loss was 0.6142 against Elo's 0.6136,
  so the extra model complexity added nothing there. The app warns when this applies.
- **Close matchups.** When the favorite is under 60%, it won 54.5% of the time. That matches its
  forecast, but such games remain close to a coin flip by nature.
- **Group of Five vs. Group of Five.** 68.7% accuracy, with the smallest edge over Elo (0.5806
  against 0.5877), and the largest total error of the game types (13.50 points).
- **Score ranges run wide.** 80% ranges held 83-85% of results, and 90% ranges held 92-93%. Simulated
  totals vary more than real ones (standard deviation 18.0 against 16.8). This matches the 2021-22 finding.
- **Lopsided and very low-scoring games are overstated.** Margins of 28 or more: 21.1% simulated against
  18.9% actual. Totals of 30 or less: 9.9% against 8.1%.
- **Season-to-season variation is real.** Log loss ran from 0.5210 in 2023 to 0.5508 in 2024.
  Expect any one season to differ from the average.
- **Margin forecasts may be drifting.** Forecasts ran below actual home margins in 2024 (-0.53) and
  2025 (-1.09). Two seasons aren't enough to call this a trend. It's a good question for future retraining.

## What this result can't tell you

- A single exact score remains very uncertain. The typical team-points miss is about 9 points.
- These figures are averages over thousands of games, not a measure of certainty about any one game.
- Injuries, weather, suspensions, and player availability aren't included, and their effect was not measured.
- The production model uses more training data than the evaluated model, so its exact performance
  isn't what's measured here. Its 2026 scorecard is tracked separately in the app.
