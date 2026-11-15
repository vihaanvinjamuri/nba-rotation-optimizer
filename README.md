# NBA Rotation Optimizer

New York Knicks, 2024-25 and 2025-26. A data pipeline and Streamlit app that reconstructs real on-court lineups and box scores from raw play-by-play data, then uses a fatigue-adjusted optimization model to check whether the team's real minutes decisions matched what the data says was optimal.

**Live app:** https://nba-rotation-optimizer.streamlit.app

## What it does

**The season, as it happened.** Twelve story cards, opening night through the NBA Cup and the championship, every score and stat checked against basketball-reference box scores and AP/NBA.com coverage.

**Was the real rotation optimal?** For every game, a CVXPY optimizer re-allocates that game's real total minutes to maximize predicted value minus a fatigue penalty, and the app compares that allocation to what the coach actually did, for the season and for any single game. Real rotations trail the optimum by 5.5 units per game under Mike Brown (2025-26), mostly value left on the table, versus 9.0 under Tom Thibodeau (2024-25), almost all extra fatigue cost.

**Out-coach the coach.** Pick a real game, see each player's pre-game form and energy (exactly what the coach knew), split the minutes yourself, and get graded against the real coach and the model's best plan.

**2K Mode.** Player cards with each player's NBA 2K27 overall and six attribute ratings (`data/reference/nba2k27_ratings.csv`). Pick a starting five and bench, set minutes, watch the team rating and projected record update live, then sim one season or 200 seasons on the Knicks' real schedule against the real 2025-26 league, with injuries, the play-in and a full playoff bracket.

## How the data pipeline works

1. **Lineups from play-by-play** (`src/parse_lineups.py`, `src/build_ratings_and_schedule.py`). Every substitution event is walked in order to reconstruct which five players were on the court for every second. The lineup is re-detected at the top of every period, since a change between quarters isn't logged as a substitution.
2. **Box scores from play-by-play** (`src/build_boxscore_from_pbp_v2.py`). Points, rebounds, assists, steals, blocks, turnovers and fouls are aggregated event by event. Team points match the real final score in all 164 games, and every player's season per-game numbers match basketball-reference to within 0.1.
3. **Exact minutes** (`src/correct_minutes_with_official_box.py`). In 34 of 164 games the tracker lost a player who played a whole quarter without recording a stat. Those games use the official basketball-reference minutes and plus-minus (`data/reference/official_box_minutes_corrections.csv`), each checked so minutes sum to 5 × game length and plus-minus sums to 5 × the final margin. Every game now totals exactly 240 (265/290 in overtime), and every player's games played and minutes per game match basketball-reference. (This replaces the older season-total calibration in `calibrate_to_verified*.py`.)
4. **The optimizer** (`src/optimize_rotation.py`). Each player's value coefficient blends trailing (no-lookahead) on/off net rating with trailing Game Score per 48, shrunk toward a prior for small samples. A CVXPY program re-allocates each game's real total minutes to maximize value minus a fatigue penalty, with a 48-minute cap.
5. **The simulator** (`sim2k.py`). Player impact is real Box Plus/Minus shrunk toward replacement level by minutes. Team rating is calibrated so the Knicks' real 2025-26 minutes reproduce their real +6.5 net rating. Each simulated season adds random breakout/slump years for players and rating swings for the other 29 teams, so no two seasons are the same. Games use the real 2025-26 net ratings of all 30 teams, with a spread fitted so those ratings reproduce the real 2025-26 records.

Full formulas are in the app under "How this was built."

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
app.py              Streamlit app
sim2k.py            2K Mode season simulator (plain Python, testable on its own)
src/                data pipeline scripts
data/processed/     reconstructed lineups, box scores, corrected per-game ratings, optimizer results
data/reference/     official stats, schedules, team ratings and box-score corrections (basketball-reference)
```
