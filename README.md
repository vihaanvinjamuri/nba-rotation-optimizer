# NBA Rotation Optimizer

New York Knicks, 2025-26. A data pipeline and Streamlit app that reconstructs real on-court lineups and box scores from raw play-by-play data, then uses a fatigue-adjusted optimization model to check whether the team's real minutes decisions matched what the data says was optimal. It also includes a season simulator where you can build your own starting five and minutes distribution and simulate a full season through the playoffs.

**Live app:** _add your deployed link here once it's up on Streamlit Community Cloud_

## What it does

The season is told chronologically first, opening night through the championship, then the app digs into the numbers underneath it.

**Was the real rotation optimal?** For every game, a CVXPY optimizer re-allocates that game's real total minutes across the roster to maximize predicted value minus a fatigue penalty, and the app compares that model-optimal allocation to what the coach actually did, at both the season and single-game level.

**Build your Knicks and simulate the season.** Pick a starting five, a sixth man, and how many minutes each plays. The app flags unrealistic position assignments (playing a point guard at center, for example) and simulates an 82-game season plus the play-in and playoffs based on your choices, reporting a final result and tips for adjusting your lineup.

## How the data pipeline works

1. **Lineups from play-by-play.** Every substitution event (`SUB: <in> FOR <out>`) is walked in order from a detected starting five to reconstruct exactly which five players were on the court for every second of the game. The starting lineup is re-detected at the top of every period, not just once at tip-off, since a lineup change between quarters isn't logged as an in-game substitution.
2. **Box scores from play-by-play.** Points, rebounds, assists, steals, blocks, turnovers, and fouls are aggregated directly from play-by-play events, independent of the lineup reconstruction, and checked against real final scores and season totals.
3. **Calibration.** Minutes are lightly anchored to verified season totals (basketball-reference) to correct residual lineup-tracking noise.
4. **The optimizer.** Each player's value coefficient blends their trailing (no-lookahead) on/off net rating with their trailing Game Score, both shrunk toward a neutral prior for small samples. A CVXPY program then re-allocates each game's real total minutes to maximize value minus a fatigue penalty based on recent workload, subject to a 48-minute per-player cap.

Full formulas and reasoning are in the app itself under "How this was built."

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py                  Streamlit app
src/                     data pipeline scripts (lineup parsing, box scores, calibration, the optimizer)
data/processed/          reconstructed lineups, box scores, and optimizer results
data/reference/           verified season stats, player positions, and illustrative team strength ratings used by the season simulator
```

## What's real data vs. illustrative

Lineups, box scores, and minutes are reconstructed from real play-by-play and checked against basketball-reference and national game coverage. The season simulator's other 29 teams are represented by a single illustrative net-rating number each, not real opposing players or real 2025-26 results, since the simulator is a what-if tool for the Knicks' own lineup choices, not a league-wide prediction model.
