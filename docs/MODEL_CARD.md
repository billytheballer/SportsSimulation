# Model card

A one-page summary of what this model is, how it was tested, and when not to rely on it.

## What it does

For a matchup between two FBS college football teams, it estimates each team's win probability and
expected points, then simulates 10,000 games to produce a range of possible scores.

**Intended use:** learning, analysis, and portfolio demonstration.

**Not intended for:** betting or any decision with real stakes. Listed betting lines were more accurate
than this model in testing, and it can't see injuries or other late news.

## How it works

| Component | Method | Inputs |
|---|---|---|
| Team ratings | Walk-forward Elo | Past results, margin of victory, site |
| Team stats | Opponent-adjusted running averages | Scoring, yardage, efficiency, turnovers, pace |
| Win model (comparison) | Regularized logistic regression | Elo, team stats, schedule, form, recruiting, returning production, site |
| Score model | Ridge regression, one team at a time | The same categories, as offense against opposing defense |
| Simulation (main output) | Monte Carlo draws of real score pairs from similar past games, plus an overtime step | Expected points from the score model |

Betting lines are never inputs. No language model produces any prediction.

## Data

All data comes from the [CollegeFootballData.com](https://collegefootballdata.com) free tier. Raw data isn't in
this repo. See [data sources](DATA_SOURCES.md).

| Seasons | Purpose |
|---|---|
| 2014-15 | Warm-up for ratings. Never scored |
| 2016-20 | Training and choosing settings |
| 2021-22 | Development and diagnostics |
| 2023-25 | Final evaluation, run once |
| 2016-25 | Training for the separate production model |
| 2026 | Updates production team data. Never refits the model |

## Performance

The final evaluation covered 2,398 FBS vs FBS games from 2023-25. The evaluated model trained only on 2016-20.

| Measure | Result |
|---|---:|
| Accuracy | 71.6% |
| Log loss | 0.5376 |
| Brier score | 0.1822 |
| Team points error (MAE / RMSE) | 8.97 / 11.23 |
| Margin error (MAE / RMSE) | 12.47 / 15.75 |
| Total points error (MAE / RMSE) | 12.84 / 16.03 |

Log loss comparisons on the same games: listed betting line 0.5213 (timing unverified), team-stat model 0.5404,
CFBD's Elo 0.5529, our Elo 0.5589, coin flip 0.6931.

The simulation is statistically tied with the team-stat model, clearly beats Elo, and clearly trails the
listed betting line. See [what the results mean](FINAL_EVALUATION.md).

**Live check:** on 500 FBS games from the 2026 season through October 31, played after production training ended,
accuracy was 75.0%, log loss 0.536, and margin MAE 12.4. This is a partial season, so it may not hold up.

## Known limitations

- **Bowl games:** 57.5% accuracy on 134 postseason games, with log loss worse than a coin flip.
- **Neutral sites:** 60.4% accuracy on 202 games.
- **New FBS teams:** no returning-production data. On 57 games, it was no better than Elo.
- **Wide ranges:** 90% ranges contained 92-93% of results. Lopsided games are somewhat overstated.
- **Not modeled:** injuries, player availability, suspensions, weather, coaching changes, and team styles.
- **Possible timing issue:** returning production and recruiting data may include later corrections, and
  their exact publication dates aren't confirmed.
- **Change over time:** the model doesn't refit during a season, and college football keeps changing.

## How leakage was prevented

Each forecast uses only data from before kickoff, enforced by the order of the code and checked by tests that
rewrite later results. Settings and code were frozen before the final evaluation, which refuses to run twice.
[Read how the test was protected](FINAL_EVALUATION.md#how-the-test-was-protected).

## Reproducing the results

The saved results can't be regenerated with the one-time command, since a completed evaluation refuses to run
again. To inspect the method, read `src/cfbsim/final_eval.py`, run the tests with `uv run pytest`, and rerun
the development stages with your own free API key.
