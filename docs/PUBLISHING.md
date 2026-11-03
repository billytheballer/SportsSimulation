# Publishing checklist

These steps are for the project's owner. Everything here is free. Do them in order.

## 1. Decide about your email address

Git commits show the author's email address, and anyone can read it once a repo is public. Existing commits
use a personal address. Before making the repo public, choose one:

- **Keep it.** Nothing else to do.
- **Hide it on future commits.** In GitHub, go to **Settings > Emails** and turn on **Keep my email addresses
  private**, then set your Git email to the GitHub-provided `noreply` address. Earlier commits keep the old address.
- **Remove it from existing commits.** This rewrites history and needs a force push, so do it before anyone relies
  on the repo. Ask for help with this step, as it can't be undone easily.

## 2. Merge into `main`

In GitHub Desktop, choose **Branch > Create Pull Request**, then merge it on GitHub. The project shows on the repo's
front page only after this step.

## 3. Check what's public, then make the repo public

Confirm these are absent from the repo: `.env`, `data/`, and `bundles/`. Then, in GitHub, go to
**Settings > General > Danger Zone > Change repository visibility**. A public repo can be copied by anyone,
even if you make it private later.

## 4. Optional: host the app

[Streamlit Community Cloud](https://streamlit.io/cloud) hosts apps from public GitHub repos for free.

1. Build a public copy of the model:

   ```bash
   uv run cfbsim update-season
   ```

   ```bash
   uv run cfbsim build-production --public
   ```

2. Remove the `bundles/` line from `.gitignore`. Commit only `bundles/production_2026.joblib`, then push.
3. In Streamlit Community Cloud, create an app that points at `streamlit_app.py`, and choose Python 3.12 under
   **Advanced settings**.

The hosted app needs no API key, so don't add one. Keep your key in `.env` on your own computer.

**To update it:** repeat steps 1 and 2. If you stop updating, the app shows a warning once its data is over 9 days old.

### What a public model file contains

| Included | Why |
|---|---|
| Model coefficients and settings | Derived outputs |
| Each team's running averages, ratings, and win-loss record | Derived outputs |
| Regulation score pairs and expected points from about 7,500 past games, with no teams or dates | Used to draw simulated scores |
| Team names and conferences | Needed for the team pickers |
| This season's team pairings and dates, without scores | Used for rematch warnings |

The `--public` option leaves out this season's scores and upcoming schedule, so a hosted app has no upcoming-games
picker.

**Your decision:** [CFBD's terms](https://collegefootballdata.com/terms) allow publishing models and other derived
outputs, and don't allow redistributing raw data as a dataset. This file appears to fit as derived output, but the
score pairs are the closest call. If you want certainty, ask CFBD before publishing. This isn't legal advice.

## 5. Add a link to the README

If you host the app, add its address near the top of the README, then commit and push.
