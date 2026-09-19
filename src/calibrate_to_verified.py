"""
Anchor the 2025-26 play-by-play-derived minutes to verified real season
stats (games played, minutes per game) from basketball-reference, since the
raw substitution-log reconstruction was found to have real, non-trivial
errors (e.g. showing Karl-Anthony Towns at 21.9 MPG when he actually played
31.0). This rescales every player's per-game minutes so their season total
matches the verified real total exactly, and reports how many real games
are still missing entirely from the derived dataset (a genuine remaining
gap, not something rescaling can fix, since we have no row at all for a
game we never parsed).
"""
import pandas as pd
from pathlib import Path

PROC = Path(__file__).parent.parent / "data" / "processed"
REF = Path(__file__).parent.parent / "data" / "reference" / "nyk_2025_26_verified_stats.csv"

# Map the surname-only keys from the 2025-26 (v3 format) derived data to the
# verified reference's full names.
NAME_MAP = {
    "Brunson": "Jalen Brunson", "Towns": "Karl-Anthony Towns", "Anunoby": "OG Anunoby",
    "Bridges": "Mikal Bridges", "Hart": "Josh Hart", "McBride": "Miles McBride",
    "Shamet": "Landry Shamet", "Clarkson": "Jordan Clarkson", "Alvarado": "Jose Alvarado",
    "Robinson": "Mitchell Robinson", "Kolek": "Tyler Kolek", "Diawara": "Mohamed Diawara",
    "Sochan": "Jeremy Sochan", "Yabusele": "Guerschon Yabusele",
    "McCullar Jr.": "Kevin McCullar Jr.", "Hukporti": "Ariel Hukporti",
    "Dadiet": "Pacome Dadiet", "Jones": "Dillon Jones", "Jemison III": "Trey Jemison",
    "Evbuomwan": "Tosan Evbuomwan",
}


def run():
    df = pd.read_csv(PROC / "NYK_2025-26_player_game_ratings.csv", parse_dates=["game_date"])
    ref = pd.read_csv(REF).set_index("player_name")

    df["full_name"] = df["player_name"].map(NAME_MAP)
    unmapped = df[df["full_name"].isna()]["player_name"].unique()
    if len(unmapped):
        print("WARNING: unmapped names, left uncalibrated:", unmapped)

    derived_totals = df.groupby("full_name")["minutes"].sum()
    derived_games = df.groupby("full_name")["game_id"].nunique()

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
        lambda r: r["minutes"] * scale_map.get(r["full_name"], 1.0), axis=1
    )
    df["net_rating_per48_calibrated"] = df.apply(
        lambda r: (r["plus_minus"] / (r["minutes_calibrated"] * 60) * 48 * 60)
        if r["minutes_calibrated"] > 0 else 0, axis=1
    )

    out_path = PROC / "NYK_2025-26_player_game_ratings_calibrated.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    run()
