"""
Per-game rotation optimizer: given each available player's trailing
(pre-game, no-lookahead) net-rating and their trailing cumulative fatigue
load, allocate that game's total team-minutes to maximize expected point
differential minus a fatigue penalty.

Mirrors the EV charging paper's structure: maximize/minimize a weighted sum
of a "value" term and a "cost/degradation" term under a hard capacity
constraint (there: grid power cap; here: total team minutes), swept across
a tradeoff parameter lambda to build a Pareto frontier.
"""
import pandas as pd
import numpy as np
import cvxpy as cp
from pathlib import Path

PROC = Path(__file__).parent.parent / "data" / "processed"
MAX_MIN_PER_PLAYER = 48.0  # a real player can never exceed a full 48-minute game


SUFFIXES = {"Jr.", "Sr.", "II", "III", "IV"}


def _surname_key(full_name):
    """The season-over-season datasets key players differently: 2024-25
    uses full names ('Kevin McCullar Jr.'), 2025-26 uses surnames only
    ('McCullar Jr.'). Reduce a full name to the same surname-only key
    (keeping a trailing suffix like 'Jr.'/'III' attached) so a returning
    player's prior-season prior actually matches up with this season's row."""
    parts = full_name.split()
    if len(parts) > 1 and parts[-1] in SUFFIXES:
        return " ".join(parts[-2:])
    return parts[-1]


def prior_season_rates(prev_season_label):
    """Each returning player's own real (unshrunk) average net rating and
    Game Score per 48 from the PREVIOUS season, used as a warm-start prior
    for game 1 of the next season. Without this, every player's shrinkage
    weight is exactly 0 in their very first game (games_played_so_far=0),
    so value_per48 collapses to 0 for the entire roster and the optimizer
    has no basis to tell anyone apart — it just splits minutes evenly,
    which is mathematically consistent with 0 information but looks wrong
    for a known quantity like Jalen Brunson on opening night."""
    prev_path = PROC / f"NYK_{prev_season_label}_player_game_ratings_calibrated.csv"
    if not prev_path.exists():
        prev_path = PROC / f"NYK_{prev_season_label}_player_game_ratings.csv"
    if not prev_path.exists():
        return {}
    df = pd.read_csv(prev_path, parse_dates=["game_date"])
    if "net_rating_per48_calibrated" in df.columns:
        df["net_rating_per48"] = df["net_rating_per48_calibrated"]
    box = pd.read_csv(PROC / f"NYK_{prev_season_label}_player_game_boxscore.csv", parse_dates=["game_date"])
    df = df.merge(box[["game_id", "player_name", "game_score"]], on=["game_id", "player_name"], how="left")
    df["game_score"] = df["game_score"].fillna(0.0)
    minutes_col = df["minutes_calibrated"] if "minutes_calibrated" in df.columns else df["minutes"]
    df["gmsc_per48"] = np.where(minutes_col > 0, df["game_score"] / minutes_col * 48.0, 0.0)
    agg = df.groupby("player_name").agg(
        prior_net_rating=("net_rating_per48", "mean"), prior_gmsc=("gmsc_per48", "mean"),
        prior_games=("game_id", "nunique"),
    )
    # Shrink the prior itself toward 0 the same way in-season values are
    # shrunk: a player who logged 4 games last season (Kevin McCullar Jr.,
    # net rating +69 on that tiny sample) shouldn't hand this season's game
    # 1 a wildly overconfident starting point.
    K_PRIOR = 20
    w = agg["prior_games"] / (agg["prior_games"] + K_PRIOR)
    agg["prior_net_rating"] *= w
    agg["prior_gmsc"] *= w
    return {_surname_key(name): vals for name, vals in agg.to_dict(orient="index").items()}


def prep_season(season_label, prior_season_label=None):
    # Use the version calibrated against verified real season stats
    # (basketball-reference) when available: the raw play-by-play
    # reconstruction had real errors (e.g. Karl-Anthony Towns showing 21.9
    # MPG when he actually played 31.0), and those numbers feed everything
    # downstream, so this isn't a cosmetic fix.
    calibrated_path = PROC / f"NYK_{season_label}_player_game_ratings_calibrated.csv"
    if calibrated_path.exists():
        df = pd.read_csv(calibrated_path, parse_dates=["game_date"])
        df["minutes"] = df["minutes_calibrated"]
        df["net_rating_per48"] = df["net_rating_per48_calibrated"]
    else:
        df = pd.read_csv(PROC / f"NYK_{season_label}_player_game_ratings.csv", parse_dates=["game_date"])
    df = df.sort_values(["player_name", "game_date"])

    # Bring in the independently-reconstructed box score so the value
    # coefficient isn't ONLY on/off net rating. Net rating alone is a noisy,
    # team-context-dependent signal: a role player who enters games during
    # winning stretches inherits a lot of the team's own positive point
    # differential without individually producing much, which let small
    # bench pickups (e.g. a Feb trade-deadline addition with a hot 20-game
    # on/off stretch) show a near-identical value coefficient to a Finals
    # MVP-caliber starter. Blending in each player's own trailing box-score
    # production (Game Score, self-contained and already validated
    # elsewhere in this pipeline) anchors value to individual output, not
    # just who else was on the floor.
    box_path = PROC / f"NYK_{season_label}_player_game_boxscore.csv"
    box = pd.read_csv(box_path, parse_dates=["game_date"])
    df = df.merge(box[["game_id", "player_name", "game_score"]], on=["game_id", "player_name"], how="left")
    df["game_score"] = df["game_score"].fillna(0.0)
    df["gmsc_per48"] = np.where(df["minutes"] > 0, df["game_score"] / df["minutes"] * 48.0, 0.0)

    # Trailing (no-lookahead) inputs: expanding mean net rating and rolling
    # 7-game load, both computed using only games strictly BEFORE this one.
    df["trailing_net_rating_raw"] = df.groupby("player_name")["net_rating_per48"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["trailing_gmsc_raw"] = df.groupby("player_name")["gmsc_per48"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["games_played_so_far"] = df.groupby("player_name").cumcount()
    df["trailing_load_7g"] = df.groupby("player_name")["minutes"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).sum()
    )
    # First game of the season for a player has no trailing data; fall back
    # to a neutral prior (0 net rating, 0 prior load) rather than dropping them.
    df["trailing_net_rating_raw"] = df["trailing_net_rating_raw"].fillna(0.0)
    df["trailing_gmsc_raw"] = df["trailing_gmsc_raw"].fillna(0.0)
    df["trailing_load_7g"] = df["trailing_load_7g"].fillna(0.0)
    # Shrink small-sample net ratings toward a prior rather than trusting a
    # handful of noisy games at face value; shrinkage strength k controls
    # how many "prior" games of average performance we assume. The prior
    # itself is 0 (league-average) for a player with no prior-season data
    # (rookies, new additions), or that returning player's own real prior-
    # season rate — so game 1 of a new season reflects known quality
    # instead of blank-slate 0 for the whole roster at once.
    K_SHRINKAGE = 20
    priors = prior_season_rates(prior_season_label) if prior_season_label else {}
    prior_net = df["player_name"].map(lambda n: priors.get(n, {}).get("prior_net_rating", 0.0))
    prior_gmsc = df["player_name"].map(lambda n: priors.get(n, {}).get("prior_gmsc", 0.0))
    shrink_weight = df["games_played_so_far"] / (df["games_played_so_far"] + K_SHRINKAGE)
    df["trailing_net_rating"] = df["trailing_net_rating_raw"] * shrink_weight + prior_net * (1 - shrink_weight)
    df["trailing_gmsc"] = df["trailing_gmsc_raw"] * shrink_weight + prior_gmsc * (1 - shrink_weight)
    # Equal-weighted blend of team-context on/off impact and individual
    # box-score production, both already shrunk toward a neutral prior.
    df["value_per48"] = 0.5 * df["trailing_net_rating"] + 0.5 * df["trailing_gmsc"]
    return df.sort_values(["game_date", "player_name"])


GARBAGE_TIME_MINUTES = 4.0  # real coaches never "choose" a token cameo appearance;
                            # only players who actually got rotation-caliber minutes
                            # are treated as decisions the model gets to re-allocate.


def optimize_game(game_df, total_minutes, lam):
    n = len(game_df)
    coef_all = game_df["value_per48"].to_numpy() / 48.0  # value per minute
    load_all = game_df["trailing_load_7g"].to_numpy()
    actual_all = game_df["minutes"].to_numpy()

    is_rotation = actual_all >= GARBAGE_TIME_MINUTES
    result = np.array(actual_all, dtype=float)  # garbage-time players keep their real minutes
    if not is_rotation.any():
        return result

    coef = coef_all[is_rotation]
    load = load_all[is_rotation]
    remaining_minutes = total_minutes - actual_all[~is_rotation].sum()

    # Small floor before normalizing: a player with no game history has
    # coef=0 AND load=0, a perfectly flat slice of the objective, so
    # nothing stops the solver from over- or under-using them arbitrarily.
    # The floor gives everyone a nonzero marginal fatigue cost. It's set
    # fairly high (not just a tiny epsilon): a low FLOOR made the fatigue
    # cost nearly zero for any low-usage player (since load is close to 0
    # relative to a heavy-minutes teammate's load), so the model kept
    # pinning several players at the 48-minute hard cap every game — a
    # real player essentially never plays a full 48. A bigger floor keeps
    # the fatigue penalty from vanishing for low-load players, without
    # erasing the real differentiation between a heavily-used and a
    # lightly-used player.
    FATIGUE_FLOOR = 20.0
    load_norm = (load + FATIGUE_FLOOR) / (load.max() + FATIGUE_FLOOR + 1e-6)

    m = cp.Variable(is_rotation.sum(), nonneg=True)
    value_term = coef @ m
    fatigue_term = cp.sum(cp.multiply(load_norm, cp.square(m / MAX_MIN_PER_PLAYER)))
    objective = cp.Maximize(value_term - lam * fatigue_term)
    constraints = [cp.sum(m) == remaining_minutes, m <= MAX_MIN_PER_PLAYER]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if m.value is None:
        return None
    result[is_rotation] = np.clip(m.value, 0, None)
    return result


def run_season(season_label, lam=0.15, prior_season_label=None):
    df = prep_season(season_label, prior_season_label=prior_season_label)
    rows = []
    for gid, g in df.groupby("game_id"):
        raw_total = g["minutes"].sum()  # actual total = 5 * game length, handles OT
        if raw_total < 200:  # skip the handful of still-broken games from the parser
            continue
        # A real NBA game's total playable player-minutes is always a clean
        # multiple of 5 (5 players on court at once): 240 in regulation, plus
        # 25 per overtime period. The calibrated per-player minutes carry
        # tiny floating-point noise (summing to e.g. 240.9), so round the
        # TOTAL used for the optimizer's own constraint to the nearest clean
        # value rather than solving against a noisy target.
        total_minutes = round(raw_total / 5.0) * 5.0
        opt_minutes = optimize_game(g, total_minutes, lam)
        if opt_minutes is None:
            continue
        for i, (_, row) in enumerate(g.iterrows()):
            rows.append({
                "season": season_label, "game_id": gid, "game_date": row["game_date"],
                "opponent": row["opponent"],
                "player_name": row["player_name"], "actual_minutes": row["minutes"],
                "optimal_minutes": opt_minutes[i], "value_per48": row["value_per48"],
                "trailing_load_7g": row["trailing_load_7g"], "lambda": lam,
            })
    out = pd.DataFrame(rows)
    out_path = PROC / f"NYK_{season_label}_optimizer_results_lam{lam}.csv"
    out.to_csv(out_path, index=False)
    print(f"{season_label} (lambda={lam}): optimized {out['game_id'].nunique()} games, wrote {out_path}")
    return out


if __name__ == "__main__":
    # 2025-26 gets a real warm-start prior from 2024-25 (returning players'
    # actual rates), so opening night isn't a blank slate. 2024-25 is the
    # first season in this dataset, so it has no earlier season to draw a
    # prior from and still starts game 1 from a neutral 0 for everyone.
    run_season("2025-26", lam=8.0, prior_season_label="2024-25")
    run_season("2024-25", lam=8.0)
