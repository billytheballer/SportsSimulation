# Final evaluation

**This file is permanent. It was generated once by `cfbsim final-eval` and is read-only.**

- Scored on 2,398 FBS vs FBS games from [2023, 2024, 2025].
- Models were fit on 3,616 games from 2016-20 and never refit.
- Team ratings and stats updated game by game, using only games played before each forecast.
- Settings were frozen at 2026-11-03T17:44:32.690652+00:00, before any of these games were scored.
- Completed at 2026-11-03T17:46:18.759354+00:00, using 1 run attempt(s).
- Frozen settings: `{'elo_params': {'k': 40.0, 'home_advantage': 55.0, 'season_carryover': 0.7, 'lower_division_initial': 1000.0}, 'score_feature_params': {'halflife': 8.0, 'season_weight': 0.5}, 'alpha': 300.0, 'bandwidth': 3.0, 'win_feature_params': {'halflife': 8.0, 'season_weight': 0.5}, 'win_C': 0.03}`

Data provided by CollegeFootballData.com. Listed betting lines are benchmarks only, never model inputs, and their timing is unverified, so they are not described as closing lines.

## Win probability

| Forecast | Games | Accuracy | Log loss | Brier |
|---|---:|---:|---:|---:|
| simulation | 2,398 | 71.6% | 0.5376 | 0.1822 |
| stats_model | 2,398 | 71.6% | 0.5404 | 0.1836 |
| elo | 2,398 | 70.2% | 0.5589 | 0.1901 |
| baseline: coin_flip | 2,398 | 50.0% | 0.6931 | 0.2500 |
| baseline: home_rate | 2,398 | 58.4% | 0.6776 | 0.2423 |

Log-loss edges (above zero means the first model is better):

- Simulation over stats model: +0.0028 (95% interval -0.0015 to +0.0068)
- Simulation over Elo: +0.0213 (95% interval +0.0114 to +0.0316)
- Stats model over Elo: +0.0186 (95% interval +0.0076 to +0.0301)

CFBD's own Elo on 2,398 games: log loss 0.5529, accuracy 70.8%. Simulation on those games: 0.5376, 71.6%.

Listed betting line (timing unverified) on 2,398 games: log loss 0.5213. Simulation on those games: 0.5376. Simulation edge: -0.0163 (95% interval -0.0242 to -0.0084).

Simulation and stats model agreement: correlation 0.987, average gap 3.1%, same favorite in 95.7% of games.

## Calibration from the favorite's side

| Forecast for favorite | Model | Games | Mean forecast | Favorite won | Standard error |
|---|---|---:|---:|---:|---:|
| (0.499, 0.6] | simulation | 595 | 55.0% | 54.5% | 2.0% |
| (0.6, 0.7] | simulation | 579 | 65.0% | 62.7% | 2.0% |
| (0.7, 0.8] | simulation | 504 | 74.9% | 75.6% | 1.9% |
| (0.8, 0.9] | simulation | 384 | 84.9% | 86.2% | 1.8% |
| (0.9, 1.0] | simulation | 336 | 95.0% | 94.6% | 1.2% |
| (0.499, 0.6] | stats_model | 514 | 54.6% | 52.9% | 2.2% |
| (0.6, 0.7] | stats_model | 543 | 64.9% | 64.1% | 2.0% |
| (0.7, 0.8] | stats_model | 526 | 74.7% | 70.3% | 1.9% |
| (0.8, 0.9] | stats_model | 457 | 84.8% | 85.1% | 1.7% |
| (0.9, 1.0] | stats_model | 358 | 94.4% | 94.4% | 1.2% |
| (0.499, 0.6] | elo | 605 | 55.0% | 53.7% | 2.0% |
| (0.6, 0.7] | elo | 564 | 64.9% | 64.2% | 2.0% |
| (0.7, 0.8] | elo | 584 | 74.9% | 71.7% | 1.8% |
| (0.8, 0.9] | elo | 423 | 85.0% | 87.5% | 1.7% |
| (0.9, 1.0] | elo | 222 | 93.4% | 93.7% | 1.7% |

## Score forecasts

### Team Points

| Forecast | Games | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| simulation mean | 4,796 | 8.97 | 11.23 | +0.08 |
| simulation median | 4,796 | 8.94 | 11.27 | -0.61 |
| training average | 4,796 | 10.95 | 13.43 | +1.80 |

### Margin

| Forecast | Games | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| simulation mean | 2,398 | 12.47 | 15.75 | -0.54 |
| simulation median | 2,398 | 12.46 | 15.78 | -0.60 |
| stats regression | 2,398 | 12.45 | 15.74 | -0.32 |
| elo regression | 2,398 | 13.28 | 16.70 | -0.50 |
| training average | 2,398 | 15.78 | 20.15 | -0.83 |
| listed spread timing unverified | 2,398 | 11.99 | 15.19 | -0.48 |

### Total

| Forecast | Games | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| simulation mean | 2,398 | 12.84 | 16.03 | +0.16 |
| simulation median | 2,398 | 12.76 | 15.99 | -0.66 |
| training average | 2,398 | 13.87 | 17.15 | +3.60 |
| listed total timing unverified | 2,398 | 12.68 | 15.87 | -0.93 |

## By season

| Season | Model | Games | Accuracy | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| 2023 | simulation | 792 | 73.5% | 0.5210 | 0.1748 |
| 2023 | stats_model | 792 | 73.9% | 0.5235 | 0.1764 |
| 2023 | elo | 792 | 71.0% | 0.5523 | 0.1866 |
| 2024 | simulation | 798 | 69.9% | 0.5508 | 0.1885 |
| 2024 | stats_model | 798 | 69.5% | 0.5514 | 0.1889 |
| 2024 | elo | 798 | 68.5% | 0.5759 | 0.1975 |
| 2025 | simulation | 808 | 71.4% | 0.5408 | 0.1833 |
| 2025 | stats_model | 808 | 71.4% | 0.5461 | 0.1853 |
| 2025 | elo | 808 | 71.2% | 0.5487 | 0.1862 |

| Season | Target | Games | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| 2023 | team points | 1,584 | 9.03 | 11.25 | +0.12 |
| 2023 | margin | 792 | 12.47 | 15.62 | +0.00 |
| 2023 | total | 792 | 13.13 | 16.20 | +0.25 |
| 2024 | team points | 1,596 | 9.01 | 11.38 | -0.18 |
| 2024 | margin | 798 | 12.61 | 15.98 | -0.53 |
| 2024 | total | 798 | 12.77 | 16.20 | -0.37 |
| 2025 | team points | 1,616 | 8.86 | 11.07 | +0.29 |
| 2025 | margin | 808 | 12.32 | 15.64 | -1.09 |
| 2025 | total | 808 | 12.63 | 15.68 | +0.59 |

## Where the model does well and struggles

| Group | Games | Sim accuracy | Sim log loss | Stats log loss | Elo log loss | Margin MAE | Total MAE | Total bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Weeks 1-4 | 605 | 75.2% | 0.4925 | 0.4948 | 0.5433 | 12.66 | 13.12 | +0.63 |
| Week 5 through regular season | 1,659 | 71.4% | 0.5408 | 0.5436 | 0.5529 | 12.33 | 12.69 | -0.05 |
| Postseason | 134 | 57.5% | 0.7022 | 0.7066 | 0.7039 | 13.39 | 13.51 | +0.57 |
| Conference games | 1,633 | 70.7% | 0.5525 | 0.5551 | 0.5668 | 12.32 | 12.73 | -0.16 |
| Non-conference games | 765 | 73.5% | 0.5059 | 0.5089 | 0.5422 | 12.78 | 13.09 | +0.84 |
| True home games | 2,196 | 72.6% | 0.5259 | 0.5285 | 0.5477 | 12.43 | 12.81 | -0.01 |
| Neutral sites | 202 | 60.4% | 0.6649 | 0.6694 | 0.6816 | 12.87 | 13.20 | +1.98 |
| Power vs. power | 1,054 | 70.5% | 0.5598 | 0.5623 | 0.5765 | 12.18 | 12.18 | +0.82 |
| Power vs. Group of Five | 370 | 82.4% | 0.3610 | 0.3647 | 0.4333 | 12.73 | 13.01 | -0.26 |
| Group of Five vs. Group of Five | 974 | 68.7% | 0.5806 | 0.5834 | 0.5877 | 12.68 | 13.50 | -0.40 |
| Team missing returning data | 57 | 64.9% | 0.6142 | 0.6176 | 0.6136 | 11.81 | 14.93 | -1.39 |
| Favorite under 60% | 595 | 54.5% | 0.6872 | 0.7001 | 0.7073 | 12.91 | 13.06 | +0.05 |
| Favorite 60-80% | 1,083 | 68.7% | 0.6109 | 0.6110 | 0.6234 | 12.09 | 13.02 | +0.90 |
| Favorite 80% or more | 720 | 90.1% | 0.3037 | 0.3022 | 0.3394 | 12.67 | 12.40 | -0.86 |

## Do the simulated ranges hold up?

| Interval | Team points | Margin | Total |
|---|---:|---:|---:|
| 50% | 56.9% | 53.8% | 54.2% |
| 80% | 84.7% | 83.5% | 83.1% |
| 90% | 93.2% | 92.6% | 92.1% |
| 95% | 97.0% | 96.2% | 96.4% |

Outside the 90% range on totals: 3.6% below and 4.3% above. On margins: 3.2% below and 4.2% above. About 5% each way is ideal.

## Is the distribution realistic?

| Statistic | Actual | Simulation |
|---|---:|---:|
| Team shut out | 1.5% | 2.0% |
| Team scores 50+ | 4.8% | 6.0% |
| Total 30 or less | 8.1% | 9.9% |
| Total 80+ | 7.0% | 7.9% |
| Margin 28+ | 18.9% | 21.1% |
| Margin within 3 | 16.9% | 15.5% |
| Margin exactly 3 or 7 | 19.2% | 18.1% |
| Rare scores (1, 2, 4, 5, 8) | 0.2% | 0.2% |
| Team points, mean | 26.6 | 26.6 |
| Team points, sd | 13.3 | 14.1 |
| Total, mean | 53.1 | 53.3 |
| Total, sd | 16.8 | 18.0 |
| Overtime rate | 4.8% | 4.2% |

## Factors behind projected margins

| Factor | Average push (points) |
|---|---:|
| team offense | 3.87 |
| opponent defense | 3.31 |
| site | 2.33 |
| team strength | 1.90 |
| returning | 1.83 |
| recruiting | 1.71 |
| recent form | 1.06 |
| pace | 0.67 |
| turnovers | 0.25 |
