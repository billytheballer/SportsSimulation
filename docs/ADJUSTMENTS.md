# FPI and player availability

The current-season forecast adds ESPN FPI and official availability reports to the production model.
**This layer hasn't been tested.** The [final evaluation](FINAL_EVALUATION.md) measured the model without it, and
those results don't describe it. The app always shows the tested model's number next to the adjusted one.

## Why it's a separate layer

Training a feature needs values as they stood before past games. Neither input has that history:

- **FPI:** CFBD's historical values look like final, end-of-season numbers. In a 2022 test they beat a listed betting
  line in Weeks 1-4 (log loss 0.42 against 0.50), which suggests they contain later results.
- **Availability reports:** conferences began publishing them between 2023 and 2025, mostly for conference games.
  No public archive was found, and an archive from those years would overlap the final evaluation anyway.

So these are applied to the production model's expected points, and the 2023-25 seasons weren't used in any way.

## What it does

1. **Availability.** Each listed player's expected cost is subtracted from that player's own team:
   `chance of missing the game × points per 100 yards × the player's yards per game ÷ 100`.
2. **FPI.** FPI's margin is `home FPI − away FPI + the model's home-field points`. The expected margin then moves
   25% of the way toward it. The projected total doesn't change.
3. **Simulation.** The adjusted expected points go into the same 10,000-game simulation.

| Value | Setting | Where it comes from |
|---|---:|---|
| QB impact | 0.97 points per 100 lost yards (95% interval 0.48 to 1.44) | Estimated from 2016-22 |
| Rusher or receiver impact | 1.10 points per 100 lost yards | Low end of its 2016-22 estimate |
| Who counts as a valued player | The starting QB, or a rusher or receiver with 70+ yards a game | Chosen to reduce a false signal |
| FPI weight | 25% | **Assumption.** No data exists to fit it |
| Chance of missing the game | Out 100%, Doubtful 80%, Questionable or Game Time Decision 50%, Out 1st Half 50%, Probable 10% | **Assumption** |

## How player impact was estimated

Using 5,157 games from 2016-22, the model's forecast errors were compared with the production of players who didn't
appear, measured from what each player had done earlier that season.

[These estimates](../reports/availability_impact.json) have real weaknesses:

- **Box scores list only players with stats.** A receiver with no catches in a bad game can look absent, so poor play
  can cause an apparent absence. The QB estimate stays steady across thresholds, but the rusher and receiver
  estimate doesn't, which is why it uses the low end of its interval.
- **Absence isn't always injury.** Benchings and suspensions count too.
- **Backup quality is ignored.** A strong backup costs less than this estimates, and a weak one costs more.
- **The gain is small.** Knowing who actually played, which no forecast can know in advance, improved 2021-22 margin
  error only from 12.69 to about 12.6 points.

**Not covered:** defensive players and offensive linemen. They're listed when reported, with no estimated effect,
because free data doesn't give them a comparable value. An out star defender therefore doesn't change the forecast.

## Avoiding double counting

- A player who has already missed three or more straight games gets no added effect, since recent team stats reflect it.
- FPI may already reflect some injuries, so the two adjustments can partly overlap. This hasn't been measured.

## Recording an official report

Reports are entered with their source because official pages differ by conference and school, and their reuse
terms haven't been confirmed. An entry needs an official link, and anything else is refused. Reports count
only for that team's next game.

```bash
uv run cfbsim add-availability --team "Ohio State" --player "Example Player" --status out --game-date 2026-11-07 --reported-at 2026-11-05T20:00:00-05:00 --source-url https://bigten.org/example
```

Accepted sites are the conference domains and school sites listed as `OFFICIAL_DOMAINS` in `src/cfbsim/adjustments.py`,
plus `.edu` addresses. Add a school's athletics site to that list once you've confirmed it's official.

A team with no report on file isn't treated as healthy. The app says so, and many non-conference games have no report.

## Weekly routine

```bash
uv run cfbsim update-season
```

Add any official reports, then:

```bash
uv run cfbsim build-production
```

This saves a dated FPI snapshot in `data/snapshots/fpi/` and logs each coming week's forecasts in `data/forecast_log/`.
Rebuild after new reports, since they take effect only at the next build.

## How this will be tested

The forecast log records both the tested and adjusted numbers before kickoff. Once enough games are played, the
comparison can be run once on those new games. Decide the test before looking at results, and don't use 2023-25.
A small improvement will probably take at least a full season to confirm. Scoring the log hasn't been built yet.

Until then, treat the tested model's number as the one with evidence behind it.
