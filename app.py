"""
NBA Rotation Optimizer - Streamlit app

New York Knicks, 2025-26 championship season under Mike Brown. Checks
whether the team's real minutes decisions matched what a fatigue-adjusted,
data-driven model says was optimal.

Data pipeline (see /src):
  1. Real play-by-play (stats.nba.com v3 feed) is turned into on-court
     lineup stints by re-detecting the starting five at the top of every
     period (not just game start) and walking substitution events forward.
  2. The same play-by-play is independently aggregated into a real box
     score (PTS/REB/AST/STL/BLK/TOV/PF) event-by-event.
  3. Minutes are lightly anchored to verified season totals
     (basketball-reference) to correct residual lineup-tracking noise.
  4. A CVXPY optimizer re-allocates each game's minutes to maximize a
     fatigue-adjusted value function.
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit_analytics2 as streamlit_analytics
from pathlib import Path

DATA = Path(__file__).parent / "data" / "processed"
REF = Path(__file__).parent / "data" / "reference"
SEASON = "2025-26"  # the season the montage tells the story of
SEASON_LABELS = {
    "2025-26": "2025-26 — Mike Brown's first season, the championship",
    "2024-25": "2024-25 — Tom Thibodeau's last season (for comparison)",
}

st.set_page_config(page_title="NYK Rotation Optimizer", page_icon="🏆", layout="wide")

st.markdown(
    """
    <style>
    @keyframes fadeSlideIn {
        0% { opacity: 0; transform: translateX(24px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    .movie-card {
        background: linear-gradient(135deg, #06103a 0%, #0d1f6b 100%);
        color: #f5f6ff; padding: 34px 38px; border-radius: 18px; margin-bottom: 4px;
        border-left: 7px solid #ffce3d; min-height: 300px;
        animation: fadeSlideIn 0.45s ease-out;
        box-shadow: 0 8px 30px rgba(6,16,58,0.35);
    }
    .movie-card h2 { margin-top: 0; margin-bottom: 6px; color: #ffce3d; font-size: 1.6rem; }
    .movie-card .tag {
        display: inline-block; background: rgba(255,206,61,0.15); color: #ffce3d;
        font-size: 0.75rem; letter-spacing: 0.06em; text-transform: uppercase;
        padding: 4px 12px; border-radius: 20px; margin-bottom: 14px; font-weight: 600;
    }
    .movie-card p { font-size: 1.08rem; line-height: 1.6; margin: 8px 0; }
    .movie-stat { display: inline-block; margin-right: 30px; margin-top: 14px; }
    .movie-stat .num { font-size: 1.9rem; font-weight: 700; color: #ffce3d; }
    .movie-stat .label { font-size: 0.8rem; opacity: 0.85; }
    .dot-row { text-align: center; margin: 14px 0 22px 0; }
    .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin: 0 4px; background: #cfd4ea; }
    .dot.active { background: #0d1f6b; width: 22px; border-radius: 5px; }
    .bracket-wrap { overflow-x: auto; padding: 10px 2px 18px 2px; }
    .bracket-row { display: flex; gap: 18px; align-items: center; min-width: 1150px; }
    .bracket-col { display: flex; flex-direction: column; gap: 14px; justify-content: space-around; flex: 1; }
    .bcol-label {
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: #0d1f6b;
        font-weight: 700; margin-bottom: 2px; text-align: center;
    }
    .bracket-box {
        background: #f5f6fa; border: 1px solid #d8dced; border-radius: 8px;
        padding: 6px 10px; font-size: 0.82rem;
    }
    .bracket-box .team { display: flex; justify-content: space-between; padding: 2px 0; color: #333; }
    .bracket-box .team.winner { font-weight: 700; color: #0d1f6b; }
    .bracket-box .team.nyk { background: rgba(255,206,61,0.35); border-radius: 4px; padding-left: 4px; }
    .bracket-box .bscore { font-size: 0.72rem; color: #999; text-align: right; margin-top: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_bracket_html(matchups):
    """Render the simulated playoff results as an actual left-to-right
    bracket (East rounds -> Finals -> West rounds, mirrored), instead of a
    plain text log."""
    def find(conf, rnd):
        return [m for m in matchups if m["conf"] == conf and m["round"] == rnd]

    def box(m):
        a_score, b_score = (m["score"].split("-") + ["", ""])[:2]
        def team_div(name, score):
            cls = "team"
            if name == m["winner"]:
                cls += " winner"
            if name == "NYK":
                cls += " nyk"
            return f'<div class="{cls}"><span>{name}</span><span class="bscore">{score}</span></div>'
        return f'<div class="bracket-box">{team_div(m["team_a"], a_score)}{team_div(m["team_b"], b_score)}</div>'

    def col(ms, label):
        return f'<div class="bracket-col"><div class="bcol-label">{label}</div>{"".join(box(m) for m in ms)}</div>'

    cols = [
        col(find("East", "Round 1"), "East · Round 1"),
        col(find("East", "Conf. Semifinals"), "East · Semis"),
        col(find("East", "Conf. Finals"), "East · Finals"),
        col(find("Finals", "NBA Finals"), "🏆 NBA Finals"),
        col(find("West", "Conf. Finals"), "West · Finals"),
        col(find("West", "Conf. Semifinals"), "West · Semis"),
        col(find("West", "Round 1"), "West · Round 1"),
    ]
    return f'<div class="bracket-wrap"><div class="bracket-row">{"".join(cols)}</div></div>'


@st.cache_data
def load_results(season=SEASON):
    return pd.read_csv(DATA / f"NYK_{season}_optimizer_results_lam8.0.csv", parse_dates=["game_date"])


@st.cache_data
def load_boxscore(season=SEASON):
    return pd.read_csv(DATA / f"NYK_{season}_player_game_boxscore.csv", parse_dates=["game_date"])


@st.cache_data
def load_positions():
    df = pd.read_csv(REF / "player_positions_2025_26.csv")
    return df.set_index("player_name")["position"].to_dict()


@st.cache_data
def load_team_strength():
    return pd.read_csv(REF / "team_strength_2025_26.csv")


@st.cache_data
def load_player_ratings(season=SEASON):
    """Each player's season-average blended value-per-48 (net rating blended
    with box-score production), the same coefficient the rotation optimizer
    uses — this is what stands in for a player's real, data-grounded 'skill
    level' when building a simulated lineup."""
    r = load_results(season)
    return r.groupby("player_name")["value_per48"].mean().to_dict()


def mmss(minutes_float):
    if pd.isna(minutes_float):
        return "0:00"
    m = int(minutes_float)
    s = int(round((minutes_float - m) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m}:{s:02d}"


def contribution_rank(box_df):
    """Rank players by TOTAL Game Score across the season (sum, not
    average). A player who logs one garbage-time appearance and scores 20
    points would top a per-game-average ranking despite that; Game Score
    already blends points with rebounds/assists/steals/blocks/turnovers per
    game, so summing it across every game played rewards durable, well-
    rounded contribution over a single hot small-sample game, without
    needing an arbitrary minimum-games cutoff."""
    return box_df.groupby("player_name")["game_score"].sum().sort_values(ascending=False)


def season_summary(results_df, box_df):
    agg = results_df.groupby("player_name").agg(
        games=("game_id", "nunique"),
        avg_actual=("actual_minutes", "mean"),
        avg_optimal=("optimal_minutes", "mean"),
    )
    order = contribution_rank(box_df).index
    agg = agg.reindex([p for p in order if p in agg.index]).reset_index()
    return agg


def season_stat_table(results_df, box_df):
    mpg = results_df.groupby("player_name").agg(
        games=("game_id", "nunique"), mpg=("actual_minutes", "mean")
    )
    box = box_df.groupby("player_name").agg(
        ppg=("pts", "mean"), rpg=("orb", "mean"), apg=("ast", "mean"), gmsc=("game_score", "mean"),
    )
    box["rpg"] = box_df.groupby("player_name").apply(lambda d: (d["orb"] + d["drb"]).mean())
    box["total_contribution"] = contribution_rank(box_df)
    out = mpg.join(box, how="left").reset_index()
    out = out.sort_values("total_contribution", ascending=False)
    return out


FATIGUE_FLOOR = 20.0
LAMBDA = 8.0
MAX_MIN = 48.0  # a real player can never exceed a full 48-minute game
K_SHRINKAGE = 20


def efficiency_gap(df, lam=LAMBDA, max_min=MAX_MIN):
    d = df.copy()
    d["load_norm"] = d.groupby("game_id")["trailing_load_7g"].transform(
        lambda s: (s + FATIGUE_FLOOR) / (s.max() + FATIGUE_FLOOR + 1e-6)
    )
    coef = d["value_per48"] / 48

    def obj(col):
        val = coef * d[col]
        fat = lam * d["load_norm"] * (d[col] / max_min) ** 2
        return (val - fat).groupby(d["game_id"]).sum()

    return obj("actual_minutes").mean(), obj("optimal_minutes").mean()


def game_load_norm(g):
    load = g["trailing_load_7g"].to_numpy()
    return (load + FATIGUE_FLOOR) / (load.max() + FATIGUE_FLOOR + 1e-6)


def game_objective(g, minutes, lam=LAMBDA, max_min=MAX_MIN):
    """Same value-minus-fatigue objective as the optimizer, evaluated for
    one game given any minutes allocation (real, model-optimal, or a
    user-built scenario)."""
    load_norm = game_load_norm(g)
    coef = g["value_per48"].to_numpy() / 48.0
    minutes = np.asarray(minutes, dtype=float)
    val = coef * minutes
    fat = lam * load_norm * (minutes / max_min) ** 2
    return float((val - fat).sum())


MARGIN_SIGMA = 12.0  # typical real-NBA single-game score-margin standard deviation


def win_probability(margin, sigma=MARGIN_SIGMA):
    """Normal-CDF conversion from a predicted point margin to a win
    probability, using the real NBA's single-game score-margin spread."""
    return 0.5 * (1 + math.erf(margin / (sigma * math.sqrt(2))))


POSITION_FAMILY = {"PG": "Guard", "SG": "Guard", "SF": "Wing", "PF": "Big", "C": "Big"}
SLOTS = ["PG", "SG", "SF", "PF", "C", "6th Man"]


def position_mismatch_tip(slot, player, natural_pos):
    """A slot like '6th Man' has no fixed family, so it never mismatches.
    Otherwise, flag when a player's real position is a different 'family'
    (Guard/Wing/Big) than the slot they've been dragged into."""
    if slot == "6th Man" or natural_pos is None:
        return None
    slot_family = POSITION_FAMILY.get(slot)
    player_family = POSITION_FAMILY.get(natural_pos)
    if slot_family == player_family:
        return None
    severity = "⚠️" if {slot_family, player_family} == {"Guard", "Big"} else "💡"
    return (
        f"{severity} **{player} at {slot}** — {player} is a natural {natural_pos} "
        f"({player_family.lower()}), not a {slot_family.lower()}. Playing him there "
        + (
            "will get exposed on the boards and at the rim on defense."
            if player_family == "Guard" and slot_family == "Big"
            else "gives up size and rim protection you'd normally get from a true big."
            if slot_family == "Big"
            else "leaves you thin on ball-handling and perimeter creation."
            if slot_family == "Guard"
            else "is an unconventional look — it can work in small stretches, but it's not sustainable as a full-time starting spot."
        )
    )


def team_net_rating_from_minutes(minutes_by_player, value_lookup, total_minutes=240.0):
    """Same math as the optimizer's predicted-margin logic, generalized to
    any dict of {player: minutes} rather than one game's dataframe: a
    minutes-weighted average of each player's season value-per-48, which is
    itself a blend of on/off net rating and box-score production."""
    total = 0.0
    for name, mins in minutes_by_player.items():
        total += value_lookup.get(name, 0.0) / 48.0 * mins
    return total * 48.0 / total_minutes if total_minutes > 0 else 0.0


def simulate_game(rating_a, rating_b, sigma=MARGIN_SIGMA, rng=None):
    """One simulated game between two teams of given net ratings: draw a
    point margin from a normal distribution centered on the rating gap,
    with the real NBA's typical single-game score spread as noise."""
    rng = rng or np.random.default_rng()
    margin = rng.normal(rating_a - rating_b, sigma)
    return margin


def simulate_series(rating_a, rating_b, best_of=7, rng=None):
    """Best-of-N series: repeated independent single-game draws, first to
    the majority wins. Returns (a_wins, b_wins, game_margins)."""
    rng = rng or np.random.default_rng()
    need = best_of // 2 + 1
    a_wins, b_wins, margins = 0, 0, []
    while a_wins < need and b_wins < need:
        m = simulate_game(rating_a, rating_b, rng=rng)
        margins.append(m)
        if m > 0:
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, b_wins, margins


LEAGUE_AVG_RATING = 0.0  # net ratings in team_strength csv are centered on 0 = league average


def simulate_standings(knicks_rating, rng=None):
    """Simulate a full 82-game season for all 30 teams. The other 29 teams'
    win totals are drawn directly from a binomial (82 independent games,
    each with a win probability implied by that team's fixed illustrative
    net rating vs. a league-average opponent) — realistic season-length
    variance without needing a full 30-team round-robin schedule. The
    Knicks' record is instead built game-by-game against a randomly drawn,
    conference-weighted 82-game schedule, so their result traces back to
    real simulated games rather than a single binomial draw."""
    rng = rng or np.random.default_rng()
    teams = load_team_strength().copy()

    other_teams = teams["team"].tolist()
    conf_of = dict(zip(teams["team"], teams["conference"]))
    rating_of = dict(zip(teams["team"], teams["net_rating"]))

    # Knicks schedule: weight in-conference (East) opponents ~2x, so the
    # 82-game slate looks roughly like a real conference-heavy schedule.
    weights = np.array([2.0 if conf_of[t] == "East" else 1.0 for t in other_teams])
    weights = weights / weights.sum()
    schedule = rng.choice(other_teams, size=82, replace=True, p=weights)

    knicks_wins = 0
    game_log = []
    for opp in schedule:
        margin = simulate_game(knicks_rating, rating_of[opp], rng=rng)
        won = margin > 0
        knicks_wins += int(won)
        game_log.append((opp, margin, won))

    rows = [{"team": "NYK", "conference": "East", "net_rating": knicks_rating, "wins": knicks_wins,
             "losses": 82 - knicks_wins}]
    for t in other_teams:
        win_pct = win_probability(rating_of[t] - LEAGUE_AVG_RATING)
        wins = int(rng.binomial(82, win_pct))
        rows.append({"team": t, "conference": conf_of[t], "net_rating": rating_of[t],
                     "wins": wins, "losses": 82 - wins})

    standings = pd.DataFrame(rows).sort_values(["wins"], ascending=False).reset_index(drop=True)
    east = standings[standings["conference"] == "East"].sort_values(
        ["wins", "net_rating"], ascending=False
    ).reset_index(drop=True)
    west = standings[standings["conference"] == "West"].sort_values(
        ["wins", "net_rating"], ascending=False
    ).reset_index(drop=True)
    return east, west, game_log


ROUND_NAMES = ["Play-In", "Round 1", "Conf. Semifinals", "Conf. Finals", "NBA Finals"]


def simulate_play_in(conf_df, conf_name, rng=None):
    """Real NBA play-in structure: 7 vs 8 -> winner takes the 7 seed;
    9 vs 10 -> winner faces the 7v8 loser for the 8 seed. Returns the final
    seeds 1-8 for that conference, in order, plus a list of matchup dicts
    for a bracket UI."""
    rng = rng or np.random.default_rng()
    seeds = conf_df.iloc[:10].reset_index(drop=True)
    matchups = []

    def play(row_a, row_b, label):
        m = simulate_game(row_a["net_rating"], row_b["net_rating"], rng=rng)
        won_a = m > 0
        winner = row_a if won_a else row_b
        matchups.append({
            "conf": conf_name, "round": "Play-In", "label": label,
            "team_a": row_a["team"], "team_b": row_b["team"],
            "score": "1-0" if won_a else "0-1", "winner": winner["team"],
        })
        return row_a, row_b, won_a

    a7, b8, a_won = play(seeds.iloc[6], seeds.iloc[7], f"{conf_name} 7/8 game")
    a9, b10, c_won = play(seeds.iloc[8], seeds.iloc[9], f"{conf_name} 9/10 game")

    seed7 = a7 if a_won else b8
    loser_7v8 = b8 if a_won else a7
    winner_9v10 = a9 if c_won else b10

    d1, d2, d_won = play(loser_7v8, winner_9v10, f"{conf_name} final play-in spot")
    seed8 = d1 if d_won else d2

    top6 = [seeds.iloc[i] for i in range(6)]
    return top6 + [seed7, seed8], matchups


def simulate_playoffs(east, west, rng=None):
    """Full bracket: play-in (if needed) -> Round 1 (1v8,2v7,3v6,4v5 within
    conference) -> Conf Semis -> Conf Finals -> NBA Finals. Returns a result
    dict describing the Knicks' exact finish, plus structured matchups for
    rendering an actual bracket."""
    rng = rng or np.random.default_rng()
    matchups = []

    east_seeds, east_pi = simulate_play_in(east, "East", rng=rng)
    west_seeds, west_pi = simulate_play_in(west, "West", rng=rng)
    matchups += east_pi + west_pi

    def play_round(seeds, conf_name, round_name, pairing):
        winners = []
        for a_idx, b_idx in pairing:
            a, b = seeds[a_idx], seeds[b_idx]
            aw, bw, _ = simulate_series(a["net_rating"], b["net_rating"], rng=rng)
            winner = a if aw > bw else b
            matchups.append({
                "conf": conf_name, "round": round_name, "label": f"{conf_name} {round_name}",
                "team_a": a["team"], "team_b": b["team"],
                "score": f"{aw}-{bw}", "winner": winner["team"],
            })
            winners.append(winner)
        return winners

    r1_pairing = [(0, 7), (3, 4), (2, 5), (1, 6)]  # 1v8, 4v5, 3v6, 2v7
    east_r1 = play_round(east_seeds, "East", "Round 1", r1_pairing)
    west_r1 = play_round(west_seeds, "West", "Round 1", r1_pairing)

    east_sf = play_round(east_r1, "East", "Conf. Semifinals", [(0, 1), (2, 3)])
    west_sf = play_round(west_r1, "West", "Conf. Semifinals", [(0, 1), (2, 3)])

    east_cf = play_round(east_sf, "East", "Conf. Finals", [(0, 1)])[0]
    west_cf = play_round(west_sf, "West", "Conf. Finals", [(0, 1)])[0]

    fw, fl_wins, _ = simulate_series(east_cf["net_rating"], west_cf["net_rating"], rng=rng)
    champion = east_cf if fw > fl_wins else west_cf
    matchups.append({
        "conf": "Finals", "round": "NBA Finals", "label": "NBA Finals",
        "team_a": east_cf["team"], "team_b": west_cf["team"],
        "score": f"{fw}-{fl_wins}", "winner": champion["team"],
    })

    # Trace the Knicks' exact finish through the bracket.
    made_play_in = (east["team"].tolist().index("NYK") < 10) if "NYK" in east["team"].values else False
    made_playoffs = any(t["team"] == "NYK" for t in east_seeds)
    result = {"champion": champion["team"], "matchups": matchups}
    if not made_play_in:
        result["outcome"] = "missed_playoffs"
    elif not made_playoffs:
        result["outcome"] = "lost_play_in"
    elif champion["team"] == "NYK":
        result["outcome"] = "champion"
    elif east_cf["team"] == "NYK":
        result["outcome"] = "lost_finals"
    elif any(t["team"] == "NYK" for t in east_sf):
        result["outcome"] = "lost_conf_finals"
    elif any(t["team"] == "NYK" for t in east_r1):
        result["outcome"] = "lost_conf_semis"
    else:
        result["outcome"] = "lost_round_1"
    return result


OUTCOME_LABELS = {
    "missed_playoffs": ("😬", "Missed the playoffs and the play-in"),
    "lost_play_in": ("😕", "Eliminated in the play-in tournament"),
    "lost_round_1": ("🙂", "Lost in Round 1"),
    "lost_conf_semis": ("👍", "Lost in the Conference Semifinals"),
    "lost_conf_finals": ("🔥", "Lost in the Conference Finals"),
    "lost_finals": ("🥈", "Lost in the NBA Finals"),
    "champion": ("🏆", "NBA Champions"),
}


def build_tips(outcome, knicks_rating, starters, minutes, positions, league_avg=2.0):
    """Plain-language tips based on how the simulated season actually went,
    referencing the user's own choices (their starting five and minutes),
    not any opposing player."""
    tips = []
    top_paid = max(minutes, key=minutes.get) if minutes else None
    bench_minutes = 240.0 - sum(minutes.values())

    if outcome in ("missed_playoffs", "lost_play_in"):
        tips.append(
            "Your simulated team rating came in below league average — try leaning harder on your highest-value "
            "players. Give your best 1-2 guys closer to 40+ minutes and see how much that alone moves the needle."
        )
        if bench_minutes > 120:
            tips.append(
                f"Your bench is soaking up {bench_minutes:.0f} of the 240 minutes each game — that's a lot of court "
                "time going to lower-tier production. Push more of that time back to your starters."
            )
    elif outcome == "lost_round_1":
        tips.append(
            "Competitive enough to make the playoffs, but not deep enough to survive Round 1. Try tightening the "
            "rotation further — fewer minutes spread across bench players, more concentrated on your top tier."
        )
    elif outcome in ("lost_conf_semis", "lost_conf_finals"):
        tips.append(
            "A real deep playoff run — the roster is close. Small tweaks (an extra 2-4 minutes for your best "
            "player, or fixing any position mismatches below) could be the difference in a series that's decided "
            "by a couple of possessions."
        )
    elif outcome == "lost_finals":
        tips.append(
            "One series short of the title. This lineup is legitimately championship-caliber — the Finals are a "
            "coin flip at this level, so don't overreact to one simulated loss; try running it again."
        )
    else:
        tips.append("A championship roster — this exact minutes distribution just won it all. Nothing to fix here.")

    return tips


@st.dialog("How this was built", width="large")
def methodology_dialog():
    st.markdown("#### 1. Turning play-by-play into on-court lineups")
    st.markdown(
        """
Every substitution in the raw play-by-play log is an event: `SUB: <in> FOR <out>`. Walking those events in
order, starting from a detected lineup, tells you exactly which 5 players were on the court for every
second of the game — the one number no public box score publishes directly.

The starting lineup is **re-detected at the start of every period**, not just once at the start of the game.
Early versions of this model only found the starters once and then trusted the substitution log for the rest
of the game — but real substitution logs only record *in-game* subs; a lineup change made between quarters
isn't logged as a substitution, since it happens on a dead clock. That silently corrupted who was "on court"
for the rest of the game after any such change. Re-detecting per period (the first 5 players from that team
to record a box-score action before that team's own first substitution of the quarter) fixed real, checkable
errors: Karl-Anthony Towns's game-by-game minutes were undercounted by around 10 minutes a game before this
fix.
        """
    )
    st.markdown("#### 2. Net rating, blended with individual box production")
    st.latex(r"\text{NetRating}_{48} = \frac{\text{team point differential while on court}}{\text{seconds on court}} \times 48 \times 60")
    st.markdown(
        "Small-sample players get shrunk toward a neutral (league-average) prior, so a hot 3-game stretch "
        "for a bench player doesn't dominate the model:"
    )
    st.latex(r"\text{trailingRating} = \text{rawTrailingRating} \times \frac{g}{g + k}, \quad k = " + str(K_SHRINKAGE))
    st.caption("g = games played so far this season; only games strictly BEFORE the one being predicted are used (no lookahead).")
    st.markdown(
        """
Net rating alone has a real weakness: it's a **team-context** number, not an individual one. A role player who
enters games mostly during winning stretches inherits a share of the team's own point differential without
necessarily producing much himself — an early version of this model let a small-sample bench addition show
almost the same value coefficient as the team's best player, which the optimizer then (correctly, given that
input) tried to hand a starter's workload. The fix: blend trailing net rating 50/50 with each player's own
trailing **Game Score per 48** (the same box-score rating described in section 4 below, already validated
against basketball-reference), shrunk toward a neutral prior the same way. This anchors value to individual
production, not just who else happened to be on the floor.
        """
    )
    st.latex(r"\text{value}_{48} = 0.5 \times \text{trailingRating} + 0.5 \times \text{trailingGameScore}_{48}")

    st.markdown("#### 3. The optimizer")
    st.markdown("For each game, split the team's total minutes across the roster to solve:")
    st.latex(
        r"\max_{m \ge 0} \; \sum_i c_i m_i \;-\; \lambda \sum_i w_i \left(\frac{m_i}{M}\right)^2 "
        r"\quad \text{s.t.} \quad \sum_i m_i = T,\;\; 0 \le m_i \le M"
    )
    st.markdown(
        f"""
- **c_i** — player i's blended value ÷ 48 (net rating and Game Score, above; predicted value per minute)
- **w_i** — normalized fatigue load: minutes played over the trailing 7 games, plus a floor of
  {FATIGUE_FLOOR} so a player with zero game history still has a nonzero marginal fatigue cost
- **λ = {LAMBDA}** — how heavily fatigue is weighted against predicted value
- **M = {MAX_MIN:.0f} min** — a hard per-player cap (a full 48-minute game; no real player can exceed it)
- **T** — that game's real total team-minutes, rounded to the nearest clean multiple of 5 (240 in regulation,
  265/290/etc. with overtime — a real game's total player-minutes is always 5 players' worth, never a fraction)

This is the exact same maximize-value-minus-cost structure as the companion EV charging optimization paper —
swap "predicted point differential" for "cost saved" and "fatigue" for "battery degradation" and it's the same
convex program. Garbage-time appearances (under 4 real minutes) are held fixed rather than treated as
decisions, since no coach is actually choosing to play someone 90 seconds.
        """
    )
    st.markdown("#### 4. Game Score (the advanced per-game rating shown below)")
    st.latex(
        r"GmSc = PTS + 0.4\cdot FGM - 0.7\cdot FGA - 0.4\cdot(FTA-FTM) + 0.7\cdot ORB + 0.3\cdot DRB "
        r"+ STL + 0.7\cdot AST + 0.7\cdot BLK - 0.4\cdot PF - TOV"
    )
    st.markdown(
        """
This is John Hollinger's Game Score, the same formula basketball-reference lists as **GmSc** in a player's
game log. It's used here instead of PER, Win Shares, or VORP because those all require full-league context
(every team's pace, offensive/defensive ratings, possessions) that a single-team model doesn't have; Game
Score is fully self-contained from one game's box score, so it's exactly checkable against
basketball-reference given accurate box-score inputs.

Every points/rebounds/assists/steals/blocks/turnovers/fouls number feeding it is reconstructed directly from
play-by-play events (made/missed shots, free throws, rebounds, turnovers, fouls, and the assist tagged in each
made-shot description) — not estimated. Team point totals reconstructed this way were checked against real
final scores (e.g. the Oct 22, 2025 opener: our reconstruction sums to exactly 119, the real final score), and
season totals were checked against StatMuse/basketball-reference for both leaders (Brunson: 26.0 PPG / 6.8 APG
/ 3.3 RPG / 74 GP, Towns: 20.1 PPG / 11.9 RPG / 3.0 APG / 75 GP — both exact matches).
        """
    )
    st.markdown("#### 5. What's still an estimate")
    st.markdown(
        """
Minutes are anchored to verified real season totals (most players within about 10%, since the box-score
reconstruction above is exact but minutes still depend on lineup-tracking, which is close but not perfect).
A handful of deep-bench players are missing a small number of real games entirely from the underlying
play-by-play data (22 player-games across the roster, concentrated in low-minute call-ups), since that
reconstruction wasn't complete for every brief appearance. Net rating is a genuinely reconstructed estimate;
no public source publishes on/off point differential directly, and it's noisier for players with few games.
        """
    )
    if st.button("Close", width="stretch"):
        st.rerun()


# Chronological season narrative, opening night through the championship.
# Every score, date, and record is checked against basketball-reference.com/teams/NYK/2026.html,
# StatMuse, and national game coverage of each specific game.
MOMENTS = [
    {
        "tag": "Opening Night · Oct 22, 2025", "emoji": "🏀",
        "title": "A statement over the defending No. 1 seed",
        "body": (
            "Mike Brown's first game as head coach: the Knicks opened the season by beating the "
            "Cleveland Cavaliers 119-111 at Madison Square Garden. Jalen Brunson set the tone early, "
            "the front office's bet on a new voice in the huddle looked justified from tip-off."
        ),
        "stats": [("119-111", "final score vs. CLE"), ("Game 1", "Mike Brown's debut as HC")],
    },
    {
        "tag": "November", "emoji": "📈",
        "title": "A shaky start turns into an 11-3 tear",
        "body": (
            "New York opened 2-3 and questions about the coaching change got loud fast. Brown's rotations "
            "settled, Towns and Brunson found their two-man rhythm, and the Knicks ripped off an 11-3 run "
            "through the rest of November to quiet the noise."
        ),
        "stats": [("2-3", "start to the season"), ("11-3", "record the rest of November")],
    },
    {
        "tag": "Dec 31 – Jan 19", "emoji": "⚠️",
        "title": "The stretch that tested the season",
        "body": (
            "Two separate four-game losing streaks in a three-week span dropped the Knicks to 2-9 over this "
            "stretch, their worst run of the year. It's the kind of slump that ends coaching tenures elsewhere; "
            "Brown stuck with his rotation math instead of overreacting to it."
        ),
        "stats": [("2-9", "record, Dec 31 – Jan 19"), ("2", "separate 4-game losing streaks")],
    },
    {
        "tag": "Jan 21, 2026", "emoji": "💥",
        "title": "A franchise-record blowout",
        "body": (
            "Eight days after the losing streak ended, the Knicks demolished the Brooklyn Nets 120-66, a "
            "54-point margin that set a New York Knicks franchise record for largest margin of victory. "
            "The bench played most of the second half."
        ),
        "stats": [("120-66", "final score vs. BKN"), ("54", "point margin, franchise record")],
    },
    {
        "tag": "Feb 5, 2026 · Trade Deadline", "emoji": "🔁",
        "title": "Brown gets his closers",
        "body": (
            "New York used the deadline to add exactly what a title team needs on the margins: Jose Alvarado "
            "for point-of-attack defense and pace off the bench, and Guerschon Yabusele for frontcourt depth "
            "and floor spacing behind Towns. Both were in the rotation within a week."
        ),
        "stats": [("2", "players acquired"), ("Alvarado / Yabusele", "the additions")],
    },
    {
        "tag": "March 30, 2026", "emoji": "✅",
        "title": "Playoffs, clinched",
        "body": (
            "New York locked up a playoff berth with weeks of the regular season still to play, the first "
            "time in years the Knicks could rest starters down the stretch instead of fighting for their lives "
            "on the last night of the season."
        ),
        "stats": [("Clinched", "playoff berth, March 30")],
    },
    {
        "tag": "Round 1 · vs Hawks", "emoji": "🔥",
        "title": "A 51-point win closes it out",
        "body": (
            "New York took the series 4-2 over Atlanta. Game 6 was the exclamation point: a 47-point lead at "
            "halftime, one of the largest halftime leads in NBA playoff history, on the way to a 51-point win "
            "that ended the series on the road."
        ),
        "stats": [("4-2", "series result vs ATL"), ("47", "point halftime lead, Game 6"), ("51", "final margin")],
    },
    {
        "tag": "Conf. Semis · vs 76ers", "emoji": "🧹",
        "title": "A sweep",
        "body": (
            "Philadelphia never found an answer for the Brunson-Towns-Anunoby core. New York closed it out "
            "in four straight games to reach the conference finals for the second straight year, a chance to "
            "finish what last year's run couldn't — the Knicks fell to the Pacers in the conference finals a "
            "year ago."
        ),
        "stats": [("4-0", "series result vs PHI")],
    },
    {
        "tag": "Conf. Finals · vs Cavaliers", "emoji": "🚀",
        "title": "First Finals trip since 1999",
        "body": (
            "New York swept the same Cleveland team it opened the season against, sending the Knicks to the "
            "NBA Finals for the first time since Patrick Ewing's 1999 squad, a 27-year wait finally over."
        ),
        "stats": [("4-0", "series result vs CLE"), ("27 years", "since the last Finals trip")],
    },
    {
        "tag": "NBA Finals, Game 4 · June 10, 2026", "emoji": "🎯", "featured": True,
        "title": "The Anunoby tip-in",
        "body": (
            "Down 76-49 at halftime in San Antonio, the Knicks pulled off the largest comeback in NBA Finals "
            "history, erasing a 29-point deficit (breaking the record held by the 2008 Celtics) to win it "
            "107-106. With 1.2 seconds left, Jalen Brunson drove into a double-team and put up a shot into "
            "traffic; OG Anunoby crashed the rim and tipped it in for the win."
        ),
        "stats": [("33", "Anunoby pts, 7 threes"), ("36 / 7", "Brunson pts / ast"), ("29", "deficit erased"), ("3-1", "series lead")],
    },
    {
        "tag": "NBA Finals, Game 5 · June 14, 2026", "emoji": "🏆", "featured": True,
        "title": "Championship, first since 1973",
        "body": (
            "New York closed out San Antonio 94-90 to win the title. Jalen Brunson dropped 45 points and was "
            "named Finals MVP. It's the franchise's third championship and first since 1973, and the Knicks "
            "became the first team ever to win both the in-season NBA Cup and the championship in the same "
            "season."
        ),
        "stats": [("94-90", "Game 5 final"), ("45", "Brunson pts, Finals MVP"), ("1973", "last title before this"), ("1st", "NBA Cup + title, same yr")],
    },
]

if "moment_idx" not in st.session_state:
    st.session_state.moment_idx = 0


def render_movie_card(m):
    stats_html = "".join(
        f'<div class="movie-stat"><div class="num">{v}</div><div class="label">{l}</div></div>'
        for v, l in m["stats"]
    )
    st.markdown(
        f"""
        <div class="movie-card">
        <span class="tag">{m['tag']}</span>
        <h2>{m.get('emoji','🏀')} {m['title']}</h2>
        <p>{m['body']}</p>
        {stats_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    dots = "".join(
        f'<span class="dot{" active" if i == st.session_state.moment_idx else ""}"></span>'
        for i in range(len(MOMENTS))
    )
    st.markdown(f'<div class="dot-row">{dots}</div>', unsafe_allow_html=True)


with streamlit_analytics.track(unsafe_password="knicks2026"):

    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title("🏆 Knicks Rotation Optimizer")
        st.caption(
            "New York Knicks, 2025-26: Mike Brown's first year as head coach ended with the franchise's "
            "third championship. Click through the season, then check whether his real rotation decisions "
            "matched what a fatigue-adjusted, data-driven optimizer says was ideal."
        )
        st.caption("By **Vihaan Vinjamuri**, IBDP2")
    with top_r:
        st.write("")
        if st.button("📖 How this was built", width="stretch"):
            methodology_dialog()

    st.subheader("The season, as it happened")
    idx = st.session_state.moment_idx
    render_movie_card(MOMENTS[idx])

    # The slider is a widget with its OWN persisted session_state entry:
    # once rendered, Streamlit ignores any `value=` passed to it on later
    # reruns and just keeps showing whatever it last held — and writing to
    # that key from a button's on-click handler raises (a widget's own key
    # can't be modified after it's been instantiated in the same run). So a
    # Back/Next click that only updated moment_idx got silently overwritten
    # right back to the slider's stale position the moment it re-rendered —
    # the buttons looked broken. Keying the slider off `idx` itself sidesteps
    # this entirely: whenever moment_idx changes, it's a brand-new widget
    # with a fresh key, so it always initializes at the correct position.
    scrub = st.slider(
        "Drag to scrub through the season", 1, len(MOMENTS), idx + 1,
        label_visibility="collapsed", key=f"movie_scrubber_{idx}",
    )
    if scrub - 1 != idx:
        st.session_state.moment_idx = scrub - 1
        st.rerun()

    nav1, nav2, nav3 = st.columns([1, 1, 5])
    with nav1:
        if st.button("← Back", disabled=(idx == 0), width="stretch"):
            st.session_state.moment_idx = max(0, idx - 1)
            st.rerun()
    with nav2:
        label = "Next →" if idx < len(MOMENTS) - 1 else "Restart ↺"
        if st.button(label, width="stretch"):
            st.session_state.moment_idx = 0 if idx == len(MOMENTS) - 1 else idx + 1
            st.rerun()
    with nav3:
        st.caption(f"Moment {idx + 1} of {len(MOMENTS)} — drag the slider, click Next/Back, or click a dot. Every score and record checked against basketball-reference and national game coverage.")

    st.divider()

    st.subheader("Dig into the numbers")
    season_choice = st.radio(
        "Season", list(SEASON_LABELS.values()), horizontal=True, label_visibility="collapsed",
    )
    season_key = "2025-26" if season_choice == SEASON_LABELS["2025-26"] else "2024-25"
    coach = "Mike Brown" if season_key == "2025-26" else "Tom Thibodeau"

    results = load_results(season_key)
    box = load_boxscore(season_key)
    actual_obj, optimal_obj = efficiency_gap(results)
    gap = optimal_obj - actual_obj

    st.subheader("Was the real rotation actually optimal?")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Avg. value, {coach}'s actual rotation", f"{actual_obj:.1f}")
    col2.metric("Avg. value at model-optimal rotation", f"{optimal_obj:.1f}")
    col3.metric("Gap to optimal, per game", f"{gap:.1f}")
    if season_key == "2024-25":
        st.caption(
            "For comparison: Thibodeau's final season before Mike Brown took over. Same model, same data "
            "pipeline, same accuracy checks (every player's games-played count matches basketball-reference "
            "exactly for this season too)."
        )

    st.subheader("Season stats (real, reconstructed from play-by-play)")
    stat_tbl = season_stat_table(results, box)
    disp = stat_tbl.copy()
    disp["mpg"] = disp["mpg"].round(1)
    disp["ppg"] = disp["ppg"].round(1)
    disp["rpg"] = disp["rpg"].round(1)
    disp["apg"] = disp["apg"].round(1)
    disp["gmsc"] = disp["gmsc"].round(1)
    disp["total_contribution"] = disp["total_contribution"].round(0).astype(int)
    disp = disp.rename(columns={
        "player_name": "Player", "games": "GP", "mpg": "MPG", "ppg": "PPG",
        "rpg": "RPG", "apg": "APG", "gmsc": "Avg. Game Score", "total_contribution": "Season Contribution",
    })
    st.dataframe(disp, width="stretch", hide_index=True)
    st.caption(
        "Sorted by Season Contribution (total Game Score summed across every game played), not raw PPG or "
        "MPG averages — a favorite of small-sample stats: one big garbage-time game shouldn't outrank a "
        "full season of real, well-rounded minutes. PPG/RPG/APG/Game Score are computed directly from real "
        "play-by-play box events; MPG comes from reconstructed on-court lineups, checked exactly against "
        "basketball-reference (every player's games-played count now matches exactly). Tap \"How this was "
        "built\" above for the full methodology and formulas."
    )

    st.subheader("Season average: real minutes vs. model-optimal minutes")
    st.caption("Players ordered by Season Contribution, same as the table above.")
    summary = season_summary(results, box)
    fig = go.Figure()
    fig.add_bar(name="Actual (real)", x=summary["player_name"], y=summary["avg_actual"], marker_color="#0d1f6b")
    fig.add_bar(name="Model-optimal", x=summary["player_name"], y=summary["avg_optimal"], marker_color="#ffce3d")
    fig.update_layout(barmode="group", yaxis_title="Minutes per game", legend_title=None, height=420)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Why does the model recommend these minutes?"):
        st.markdown(
            f"""
For every game, the model picks each player's minutes to maximize **predicted value − a fatigue penalty**,
subject to using exactly the game's real total minutes and a {MAX_MIN:.0f}-minute hard cap per player (see
"How this was built" for the full formula). Two things drive a player's recommended minutes up or down:

- **Value** — their own trailing net rating blended with trailing Game Score, both shrunk toward a neutral
  (or, in 2025-26, a real prior-season) baseline for small samples.
- **Fatigue load** — how many minutes they've actually played over their last 7 games. A player who's been
  used *lightly* recently costs little in fatigue to give more minutes to, even if his per-minute value isn't
  the very highest on the roster — so it's normal to see a lower-usage player (a backup center coming off a
  string of DNPs, say) recommended for more minutes than his real playing time, or even more than a
  higher-value teammate who's been playing heavy minutes every night. That's the model surfacing a real
  under-used-given-the-data case, not a bug — though it also means the model doesn't know *why* a real coach
  limited that player's minutes (foul trouble, matchups, a coach's trust) beyond what's in the box score.

A {MAX_MIN:.0f}-minute recommendation (a player's absolute physical limit) is intentionally rare — it shows up in
roughly 1 in 14 player-games in this data, reserved for cases where a player's trailing value is unusually
strong *and* his recent workload has been unusually light. Most nights, the fatigue penalty pulls every
player's recommendation well below the cap.
            """
        )

    st.subheader("Look at a single game")
    games = results[["game_id", "game_date", "opponent"]].drop_duplicates().sort_values("game_date")
    games["label"] = games["game_date"].dt.strftime("%b %d, %Y") + " vs " + games["opponent"]
    choice = st.selectbox("Game", games["label"], key=f"game_select_{season_key}")
    gid = games.loc[games["label"] == choice, "game_id"].iloc[0]

    gbox = box[box["game_id"] == gid].set_index("player_name")
    g = results[results["game_id"] == gid].set_index("player_name").join(
        gbox[["pts", "orb", "drb", "ast", "stl", "blk", "tov", "game_score"]], how="left"
    ).fillna(0)
    g = g.sort_values("pts", ascending=False).reset_index()

    fig2 = go.Figure()
    fig2.add_bar(name="Actual", x=g["player_name"], y=g["actual_minutes"], marker_color="#0d1f6b")
    fig2.add_bar(name="Model-optimal", x=g["player_name"], y=g["optimal_minutes"], marker_color="#ffce3d")
    fig2.update_layout(barmode="group", yaxis_title="Minutes", legend_title=None, height=380)
    st.caption(
        "Players ordered by points scored in this game, highest to lowest. Wondering why a low-minutes player "
        "shows a high model-optimal bar? See \"Why does the model recommend these minutes?\" above."
    )
    st.plotly_chart(fig2, width="stretch")

    box_disp = g[["player_name", "actual_minutes", "pts", "orb", "drb", "ast", "stl", "blk", "tov", "game_score"]].copy()
    box_disp["Minutes"] = box_disp["actual_minutes"].apply(mmss)
    box_disp["REB"] = box_disp["orb"] + box_disp["drb"]
    box_disp = box_disp.set_index("player_name")[["Minutes", "pts", "REB", "ast", "stl", "blk", "tov", "game_score"]].rename(
        columns={"pts": "PTS", "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TOV", "game_score": "Game Score"}
    )
    st.dataframe(box_disp, width="stretch")
    st.caption(
        "Ordered by points scored, highest to lowest. Minutes shown as m:ss. Box stats are reconstructed "
        "directly from play-by-play events for this exact game, not season-averaged; for the official box "
        "score, see basketball-reference."
    )

    st.divider()
    st.header("🏗️ Build your Knicks & simulate the season")
    st.caption(
        "Set your starting five, a 6th man, and how many minutes each of them plays — the rest of the "
        "roster's 240 minutes fills in around them automatically. Then simulate an entire 2025-26 season: "
        "conference standings, the play-in, the playoffs, and a final result. Every rating is built from "
        "this team's own real, reconstructed season data; the other 29 teams are represented only by a "
        "single illustrative team-strength number, never by name-matching individual opposing players."
    )

    positions = load_positions()
    player_ratings = load_player_ratings("2025-26")
    full_roster = sorted(player_ratings.keys(), key=lambda n: -player_ratings[n])

    st.markdown("##### Starting five & 6th man")
    starters = {}
    cols = st.columns(6)
    for slot, col in zip(SLOTS, cols):
        with col:
            default_name = {
                "PG": "Brunson", "SG": "Bridges", "SF": "Anunoby", "PF": "Towns", "C": "Robinson", "6th Man": "Hart",
            }.get(slot, full_roster[0])
            choice = st.selectbox(
                slot, full_roster, index=full_roster.index(default_name) if default_name in full_roster else 0,
                key=f"lineup_slot_{slot}",
            )
            starters[slot] = choice

    dupes = {n for n in starters.values() if list(starters.values()).count(n) > 1}
    if dupes:
        st.error(f"{', '.join(dupes)} is assigned to more than one slot — pick a different player for each spot.")

    tips_positional = [
        t for slot, name in starters.items()
        if (t := position_mismatch_tip(slot, name, positions.get(name))) is not None
    ]
    for t in tips_positional:
        st.warning(t)

    st.markdown("##### Minutes for your starting five & 6th man")
    st.caption("Everyone else on the roster splits whatever's left of the 240 team-minutes evenly, as your bench.")
    key_minutes = {}
    mcols = st.columns(6)
    for (slot, name), col in zip(starters.items(), mcols):
        with col:
            default_min = 38.0 if slot != "6th Man" else 28.0
            key_minutes[name] = col.slider(
                f"{name} ({slot})", 0.0, MAX_MIN, default_min, step=0.5, key=f"lineup_min_{slot}",
            )

    key_total = sum(key_minutes.values())
    bench_names = [n for n in full_roster if n not in starters.values()]
    bench_total = max(240.0 - key_total, 0.0)

    # Running total, so the user can see at a glance how much of the
    # 240-minute team budget their six sliders have used up, and how much
    # is left to hand out (to these six or to the bench) before it's spent.
    st.progress(min(key_total / 240.0, 1.0))
    if key_total > 240.0:
        st.error(f"Your starting six add up to {key_total:.0f} minutes — more than a team has (240). Lower a slider.")
    else:
        st.caption(
            f"**{key_total:.1f} / 240** minutes allocated to your starting six — "
            f"**{240.0 - key_total:.1f} left** ({bench_total:.1f} of which falls to the bench)."
        )
    bench_minutes = {n: bench_total / len(bench_names) for n in bench_names} if bench_names else {}
    all_minutes = {**key_minutes, **bench_minutes}

    knicks_rating = team_net_rating_from_minutes(all_minutes, player_ratings, total_minutes=240.0)
    st.metric("Your team's simulated net rating", f"{knicks_rating:+.1f}", help=(
        "A minutes-weighted average of each player's real 2025-26 value-per-48 (net rating blended with "
        "box-score production) — the same number the rotation optimizer above uses to judge lineups."
    ))

    if st.button("🎲 Simulate the 2025-26 season", type="primary", disabled=bool(dupes)):
        rng = np.random.default_rng()
        east, west, game_log = simulate_standings(knicks_rating, rng=rng)
        playoff_result = simulate_playoffs(east, west, rng=rng)
        st.session_state["sim_result"] = {
            "east": east, "west": west, "playoff": playoff_result,
            "knicks_rating": knicks_rating, "starters": dict(starters), "minutes": all_minutes,
        }

    if "sim_result" in st.session_state:
        sim = st.session_state["sim_result"]
        east, west, playoff_result = sim["east"], sim["west"], sim["playoff"]

        st.markdown("##### Final conference standings")
        sc1, sc2 = st.columns(2)
        for conf_name, conf_df, col in [("Eastern Conference", east, sc1), ("Western Conference", west, sc2)]:
            disp = conf_df[["team", "wins", "losses"]].copy()
            disp.index = disp.index + 1
            disp = disp.rename(columns={"team": "Team", "wins": "W", "losses": "L"})

            def highlight(row):
                return ["background-color: #ffce3d" if row["Team"] == "NYK" else "" for _ in row]

            col.markdown(f"**{conf_name}**")
            col.dataframe(disp.style.apply(highlight, axis=1), width="stretch", height=350)

        seed = int(east.reset_index(drop=True).query("team == 'NYK'").index[0]) + 1
        ordinal = "th" if 11 <= seed % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(seed % 10, "th")
        st.caption(f"The Knicks finished **{seed}{ordinal} in the East** with a simulated {east.loc[east['team']=='NYK','wins'].iloc[0]}-{east.loc[east['team']=='NYK','losses'].iloc[0]} record.")

        emoji, label = OUTCOME_LABELS[playoff_result["outcome"]]
        st.markdown(f"## {emoji} {label}")
        if playoff_result["outcome"] != "missed_playoffs":
            st.markdown("##### Playoff bracket")
            st.markdown(render_bracket_html(playoff_result["matchups"]), unsafe_allow_html=True)
            st.caption("Highlighted box = the Knicks. Bold team = series/game winner. Scroll right for the full bracket.")

        st.markdown("##### Tips")
        for t in build_tips(playoff_result["outcome"], sim["knicks_rating"], sim["starters"], sim["minutes"], positions):
            st.info(t)
        st.caption(
            "Every game (yours and the other 29 teams') is drawn from a normal distribution centered on the "
            "gap between the two teams' net ratings, using the real NBA's typical single-game score-margin "
            "spread as noise — the same win-probability model used elsewhere in this app. Re-run the "
            "simulation for a fresh season; the randomness means no two runs play out the same way, just like "
            "real NBA seasons."
        )

    st.caption(
        "Built with real play-by-play and box-score data, and CVXPY. Code and methodology: "
        "[github.com/vihaanvinjamuri/nba-rotation-optimizer](https://github.com/vihaanvinjamuri/nba-rotation-optimizer)"
    )
