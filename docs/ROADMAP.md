# Roadmap

Each stage has to beat the previous one on validation seasons before its additions stay in.

## Stage 1: Foundation and baseline (complete)

- CFBD client with a local cache and monthly call budget
- Clean game table with post-game fields excluded
- Walk-forward Elo with margin of victory, home field, and offseason regression
- Logistic calibration fit on tuning seasons only
- Scoring against baselines: accuracy, log loss, Brier, calibration table, bootstrap interval

**Result:** the model beats the home-rate baseline on validation seasons, with log loss of
0.5757 against 0.6797. It trails CFBD's own Elo at 0.5702, which is the number for Stage 2 to beat.

## Stage 2: Team performance features (complete)

- Per-game box scores and advanced stats, turned into opponent-adjusted rolling averages from
  earlier games only
- Yards per play, passing and rushing efficiency, turnovers, EPA, points scored and allowed
- Strength of schedule, recent form, returning production, recruiting classes
- Regularized logistic regression, measured against Elo
- Listed betting line as a tough outside benchmark (benchmark only, not a feature)

Season-total stats can't be used for past games, because they include games played after
the one being predicted.

**Result:** log loss of 0.5592 on validation seasons, beating Elo (0.5757) with an interval just
above zero. It leads CFBD's Elo (0.5702) inconclusively and trails the betting line (0.5380).

**Known limits to carry into Stage 3:**
- The gain over Elo is concentrated in weeks 1-4 (edge +0.051). After that it is +0.003, with an
  interval that includes zero.
- Favorites were forecast to win 73.4% of validation games and won 70.0%, so probabilities run
  slightly overconfident.
- Several feature groups overlap, so removing any one barely changes results. A smaller model
  may do just as well and be easier to explain.

## Stage 3: Score model and Monte Carlo (complete)

- Separate expected points for each team, based on its offense and the opponent's defense
- 10,000 simulated games per matchup, drawn from real pairs of scores in similar past games
- Regulation and overtime simulated as separate steps, so games never end tied
- Scoring: margin and total error, plus whether the simulated ranges contain actual results as
  often as they claim
- Command-line matchup simulator with a factor breakdown

**Result:** log loss of 0.5551 on 2021-22, effectively tied with the stats model (0.5592) and
clearly ahead of Elo (0.5757). Margin MAE is 12.69 and total MAE is 12.82 points, behind the
listed lines at 12.29 and 12.32. Details and assumptions are in
[how the simulation works](SIMULATION.md).

**Known limits to carry forward:**
- Simulated ranges run somewhat wide and overstate high-scoring games.
- Forecasts are less reliable for new FBS teams, which have no returning-production data.
- The 2023 clock rule changes could have caused overpredicted totals. The final evaluation found no
  such problem overall (average total off by +0.16 points).

## Stage 4: App, final evaluation, and production model (complete)

**4A: Streamlit app.** Team pickers, win probability, projected scores and ranges, simulated outcomes,
team comparison, factor breakdown, and a visible notice about how old the data is.

**4B: Final evaluation.** Run once on 2,398 games from 2023-25, with settings and code frozen beforehand.
The simulation scored 71.6% accuracy and log loss 0.5376, effectively tied with the team-stat model, clearly
ahead of Elo, and behind listed betting lines. See [what the results mean](FINAL_EVALUATION.md).

**4C: Production model.** Trains on 2016-25 with the same frozen settings. Completed 2026 games update
team ratings and stats without refitting anything, and they're scored as a live test in the app.

## Stage 5: Publish (prepared; publishing decisions are the owner's)

- README with real results and screenshots, a [model card](MODEL_CARD.md), and a
  [publishing checklist](PUBLISHING.md)
- Tested from a clean clone, and with a hosting-style install that has no raw data or API key
- A `--public` bundle option that leaves out this season's scores and schedule
- Still to do by the owner: merge to `main`, make the repo public, and optionally deploy

## Stage 6: Advanced experiments (kept only if they win on validation)

- Gradient boosting and Bayesian hierarchical team models, using Colab's free tier if needed
- Weather and game location, if the leakage problem below can be solved
- Possible improvements to investigate: bowl games, slightly wide score ranges. 2023-25 is now used, so
  any change must be judged on new games as they're played
