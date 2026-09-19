"""
Same anchoring approach as calibrate_to_verified.py, for the 2024-25
season. Player names in this season's derived data are already full names
(e.g. "Karl-Anthony Towns"), matching the reference file directly -- no
surname-to-full-name mapping needed, unlike the 2025-26 season's v3 feed.
"""
import pandas as pd
from pathlib import Path

PROC = Path(__file__).parent.parent / "data" / "processed"
REF = Path(__file__).parent.parent / "data" / "reference" / "nyk_2024_25_verified_stats.csv"


def run():
    df = pd.read_csv(PROC / "NYK_2024-25_player_game_ratings.csv", parse_dates=["game_date"])
    ref = pd.read_csv(REF).set_index("player_name")

    derived_totals = df.groupby("player_name")["minutes"].sum()
    derived_games = df.groupby("player_name")["game_id"].nunique()

    print(f"{'player':22s} {'derived_g':>9s} {'real_g':>7s} {'derived_mpg':>12s} {'real_mpg':>9s} {'scale':>7s}")
    scale_map = {}
    for name in ref.index:
        real_games, real_mpg = ref.loc[name, "games"], ref.loc[name, "mpg"]
        real_total = real_games * real_mpg
        d_total = derived_totals.get(name, 0)
        d_games = derived_games.get(name, 0)
        scale = real_total / d_total if d_total > 0 else 1.0
        scale_map[name] = scale
        d_mpg = d_total / d_games if d_games else 0
        print(f"{name:22s} {d_games:9d} {real_games:7d} {d_mpg:12.1f} {real_mpg:9.1f} {scale:7.2f}")

    missing_games_total = sum(max(ref.loc[n, "games"] - derived_games.get(n, 0), 0) for n in ref.index)
    print(f"\nTotal player-games present in real stats but missing entirely from derived data: {missing_games_total}")

    df["minutes_calibrated"] = df.apply(
        lambda r: r["minutes"] * scale_map.get(r["player_name"], 1.0), axis=1
    )
    df["net_rating_per48_calibrated"] = df.apply(
        lambda r: (r["plus_minus"] / (r["minutes_calibrated"] * 60) * 48 * 60)
        if r["minutes_calibrated"] > 0 else 0, axis=1
    )

    out_path = PROC / "NYK_2024-25_player_game_ratings_calibrated.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    run()
