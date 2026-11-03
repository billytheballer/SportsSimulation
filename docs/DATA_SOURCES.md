# Data sources

Checked November 2026. CFBD can change tiers and terms, so recheck
[its tiers page](https://collegefootballdata.com/api-tiers) and
[its terms](https://collegefootballdata.com/terms) before relying on this.

## CollegeFootballData.com free tier

| | |
|---|---|
| Cost | $0, email signup, no card |
| Limit | 1,000 calls a month |
| Included | Games, teams, team and player stats, history, recruiting, betting lines, EPA/PPA |
| Not included | Opponent-adjusted metrics, weather, live scoreboard, GraphQL |

### Terms that shape this project

- Raw API data can't be committed to a public repo. `data/` is ignored by git.
- Derived outputs such as models, ratings, predictions, and charts may be published.
- API keys must stay server-side and out of the repo. `.env` is ignored by git.
- Attribution isn't required, but the project gives it because CFBD recommends it.

### Call budget

| Data | Calls | Stage |
|---|---:|---|
| Games, 2014-2025 (regular and postseason) | 24 | 1 |
| Per-game advanced stats, box scores, betting lines, recruiting, returning production | 271 | 2 |
| Two requests while exploring the API (one rejected, one for unused talent data) | 2 | 2 |
| Teams and venues | about 2 (estimate) | later |
| 2026 check and first season update | 20 | 4 |
| Weekly updates during a season | about 10 a week (estimate) | 4 |

The full download used 297 calls, which fits comfortably in one month. The client stops at 900
counted calls a month, leaving a margin below the real limit.

## What's available for the 2026 season

Checked on November 3, 2026 with the free tier. Every input the model needs is available:

| Data | Status |
|---|---|
| Games | 622 of 888 regular-season games completed, through week 9 |
| Box scores and per-game advanced stats | Available through week 9 |
| Returning production | Available for 136 teams. North Dakota State and Sacramento State, both new to FBS, have none, and the app says so |
| Recruiting classes | Available for 221 teams |
| ESPN FPI | Current values through CFBD, saved weekly because past pre-game values can't be recovered |
| Official availability reports | Published by conferences and schools, mostly for conference games. Entered with their source |
| Weather | Not used: CFBD weather requires payment |

CFBD doesn't publish when each game's stats appear after it ends, so a recently finished game
could lack stats for a short time. Its Elo rating would still update, and the missing stats stay missing.

## Requested variables

| Variable | Status | Source or reason |
|---|---|---|
| Points, home field, game location | Free | CFBD `/games`, `/venues` |
| Yards per play, passing and rushing efficiency, turnovers | Free | CFBD `/games/teams`, `/stats/game/advanced` |
| Strength of schedule | Calculated | Derived from game results; no external data needed |
| Recent form | Calculated | Derived from earlier games |
| Betting lines | Free | CFBD `/lines`, benchmark only |
| Weather | Unresolved | CFBD charges for it (Tier 1). A free weather archive could work, but see below |
| Injuries and availability | Current reports only | Official reports exist for recent seasons, but no usable history. See [FPI and player availability](ADJUSTMENTS.md) |

### Timing assumptions in the model

These inputs come from CFBD as season-level or game-level values, and I can't confirm exactly
when each one became public:

- **Returning production:** CFBD lists this as a [preseason dataset](https://apinext.collegefootballdata.com/data-availability),
  but also notes that past data can be corrected after publication. Its exact formula isn't
  documented, so a small amount of later knowledge could be included. The model scored about the
  same without it (0.5604 vs. 0.5592), so results don't depend on it.
- **Recruiting:** each season uses that year's class and the three before it. Those classes sign
  before the season, but later rating revisions can't be ruled out.
- **Talent composite:** not used, because its publication timing is unclear and four-year
  recruiting covers similar ground.
- **Betting lines:** CFBD doesn't say when in the week the `spread` and `overUnder` fields were
  captured, so this project calls them listed lines and never closing lines. The consensus line is
  used where offered (about 60% of games), and otherwise the median across providers. Lines are a
  benchmark only and never feed any model.

### Weather needs care

A weather archive records what actually happened. Before a game, people only had a forecast,
so training on actual weather overstates accuracy. It also needs its terms and accuracy
checked first. Until both are settled, weather stays out of the model.

### Injuries are a real limitation

No national rule requires reports, and coverage varies by conference. Official reports began between 2023 and 2025,
mostly for conference games, and no usable archive was found. The tested model therefore leaves injuries out. A
separate, untested layer uses current reports when they're on file. See [FPI and player availability](ADJUSTMENTS.md).
