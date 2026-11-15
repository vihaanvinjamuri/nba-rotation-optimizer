"""
Replace the old "season-total calibration" with exact per-game corrections.

Background: the play-by-play lineup tracker is exact in most games, but in
34 of 164 games (24 in 2024-25, 10 in 2025-26) it lost track of one player
for a whole period (typically someone who played a full quarter without
recording a single box-score event and without being subbed, so nothing in
the log reveals he was on court). Those games came out 12-24 team-minutes
short (or, in one game, 12 over).

The old fix (calibrate_to_verified*.py) rescaled every game's minutes so
each player's SEASON total matched basketball-reference. That made season
totals look right, but it spread the error across all 82 games: in 2024-25
not a single game summed to the real 240 minutes afterwards (55 games summed
to ~245), and the games that were actually broken stayed broken.

This script does it properly:
  * games where the tracker's team-minutes total is already correct
    (240 / 265 / 290 within 1 minute) keep the reconstructed minutes and
    on-court plus-minus untouched;
  * the 34 broken games are replaced with the official per-player minutes
    and plus-minus from each game's basketball-reference box score
    (data/reference/official_box_minutes_corrections.csv). Each of those 34
    box scores was checked two ways before use: minutes sum to exactly
    5 x game length, and the players' plus-minus sums to exactly 5 x the
    final margin.

Output: data/processed/NYK_{season}_player_game_ratings_corrected.csv
"""
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
REF = ROOT / "data" / "reference"


def norm(name):
    return "".join(c for c in unicodedata.normalize("NFKD", str(name)) if not unicodedata.combining(c))


def run(season):
    df = pd.read_csv(PROC / f"NYK_{season}_player_game_ratings.csv", parse_dates=["game_date"])
    df["player_name"] = df["player_name"].map(norm)
    fix = pd.read_csv(REF / "official_box_minutes_corrections.csv", parse_dates=["game_date"])
    fix = fix[fix["season"] == season].copy()
    fix["player_name"] = fix["player_name"].map(norm)

    game_meta = df.groupby("game_id").agg(
        game_date=("game_date", "first"), opponent=("opponent", "first"),
        rest_days=("rest_days", "first"), back_to_back=("back_to_back", "first"),
    ).reset_index()

    fixed_dates = set(fix["game_date"])
    keep = df[~df["game_date"].isin(fixed_dates)].copy()
    new_rows = fix.merge(game_meta, on="game_date", how="left")
    assert new_rows["game_id"].notna().all(), "correction date with no matching game"
    new_rows["net_rating_per48"] = new_rows["plus_minus"] / new_rows["minutes"] * 48.0
    new_rows["minutes_source"] = "official box score"
    keep["minutes_source"] = "play-by-play reconstruction"

    # (player_id is dropped on purpose: the 2025-26 v3 substitution log attaches the
    # wrong personId to the incoming player, so ids in the raw file are unreliable.
    # Everything downstream keys on player_name, which is tracked correctly.)
    cols = ["game_id", "game_date", "opponent", "player_name", "minutes", "plus_minus",
            "net_rating_per48", "rest_days", "back_to_back", "minutes_source"]
    out = pd.concat([keep[cols], new_rows[cols]], ignore_index=True)
    out = out.sort_values(["player_name", "game_date"])
    out["rolling_7g_minutes"] = out.groupby("player_name")["minutes"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    out = out.sort_values(["game_date", "player_name"]).reset_index(drop=True)
    out.to_csv(PROC / f"NYK_{season}_player_game_ratings_corrected.csv", index=False)

    totals = out.groupby("game_id")["minutes"].sum()
    off = totals[(totals.round() % 5 != 0) | (totals < 239)]
    print(f"{season}: {out['game_id'].nunique()} games, {len(fixed_dates)} replaced from official box scores, "
          f"games whose minutes don't sum to a clean 240/265/290: {len(off)}")
    return out


if __name__ == "__main__":
    run("2024-25")
    run("2025-26")
