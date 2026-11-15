"""
NBA Rotation Optimizer - Streamlit app

New York Knicks, 2025-26 championship season under Mike Brown (with 2024-25
under Tom Thibodeau for comparison). Checks whether the team's real minutes
decisions matched what a fatigue-adjusted, data-driven model says was optimal,
lets you try to out-coach the real coach game by game, and has a 2K-style
season simulator built on real 2025-26 player and team ratings.

Data pipeline (see /src):
  1. Real play-by-play (stats.nba.com) is turned into on-court lineup stints,
     re-detecting the lineup at the top of every period.
  2. The same play-by-play is aggregated into a box score event by event.
  3. In the 34 games where the lineup tracker lost a player for a period, the
     per-player minutes and plus-minus are replaced with the official
     basketball-reference box score (correct_minutes_with_official_box.py),
     so every one of the 164 games sums to exactly 240 (or 265/290 in OT).
  4. A CVXPY optimizer re-allocates each game's minutes to maximize a
     fatigue-adjusted value function (optimize_rotation.py).
"""
import math
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from pathlib import Path

import sim2k as S

DATA = Path(__file__).parent / "data" / "processed"
REF = Path(__file__).parent / "data" / "reference"
SEASON_LABELS = {
    "2025-26": "2025-26 · Mike Brown's first season, the championship",
    "2024-25": "2024-25 · Tom Thibodeau's last season (for comparison)",
}
COACH = {"2025-26": "Mike Brown", "2024-25": "Tom Thibodeau"}

st.set_page_config(page_title="NYK Rotation Optimizer", page_icon="🏆", layout="wide")

st.markdown(
    """
    <style>
    @keyframes fadeSlideIn { 0% { opacity: 0; transform: translateX(24px); } 100% { opacity: 1; transform: translateX(0); } }
    @keyframes pop { 0% { transform: scale(.96); opacity: .4; } 100% { transform: scale(1); opacity: 1; } }
    .movie-card {
        background: linear-gradient(135deg, #06103a 0%, #0d1f6b 100%);
        color: #f5f6ff; padding: 34px 38px; border-radius: 18px; margin-bottom: 4px;
        border-left: 7px solid #ffce3d; min-height: 300px;
        animation: fadeSlideIn 0.45s ease-out; box-shadow: 0 8px 30px rgba(6,16,58,0.35);
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
    .bracket-row { display: flex; gap: 14px; align-items: center; min-width: 1150px; }
    .bracket-col { display: flex; flex-direction: column; gap: 12px; justify-content: space-around; flex: 1; }
    .bcol-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: #8a93b8; font-weight: 700; text-align: center; }
    .bracket-box { background: #f5f6fa; border: 1px solid #d8dced; border-radius: 8px; padding: 6px 10px; font-size: 0.82rem; }
    .bracket-box .team { display: flex; justify-content: space-between; padding: 2px 0; color: #333; }
    .bracket-box .team.winner { font-weight: 700; color: #0d1f6b; }
    .bracket-box .team.nyk { background: rgba(255,206,61,0.45); border-radius: 4px; padding-left: 4px; }
    .bracket-box .bscore { font-size: 0.72rem; color: #777; }

    /* 2K-style player cards */
    .card2k { border-radius: 14px; padding: 12px 12px 10px 12px; color: #fff; position: relative;
              box-shadow: 0 6px 18px rgba(0,0,0,.35); animation: pop .25s ease-out; min-height: 270px; margin-bottom: 22px; }
    .t-diamond { background: linear-gradient(150deg,#5ee7df 0%,#7f53ac 55%,#1f1c4d 100%); }
    .t-amethyst { background: linear-gradient(150deg,#b06ab3 0%,#4527a0 60%,#1a1040 100%); }
    .t-ruby { background: linear-gradient(150deg,#ff5f6d 0%,#a8123e 60%,#3a0718 100%); }
    .t-sapphire { background: linear-gradient(150deg,#4facfe 0%,#1e3c96 60%,#0b1640 100%); }
    .t-emerald { background: linear-gradient(150deg,#43e97b 0%,#138a4b 60%,#06301b 100%); }
    .t-gold { background: linear-gradient(150deg,#f6d365 0%,#b8860b 60%,#4a3505 100%); }
    .card2k .c-top { display:flex; justify-content:space-between; align-items:flex-start; }
    .card2k .c-ovr { font-size: 2.3rem; font-weight: 800; line-height: 1; text-shadow: 0 2px 6px rgba(0,0,0,.4); }
    .card2k .c-ovrlbl { font-size: .62rem; letter-spacing: .12em; opacity: .85; }
    .card2k .c-slot { background: rgba(0,0,0,.35); border-radius: 6px; padding: 2px 8px; font-weight: 700; font-size: .85rem; }
    .card2k .c-name { font-weight: 800; font-size: 1.0rem; margin-top: 6px; line-height: 1.15; }
    .card2k .c-sub { font-size: .72rem; opacity: .9; margin-bottom: 6px; }
    .card2k .c-attr { display:flex; align-items:center; gap:6px; font-size:.7rem; margin: 2px 0; }
    .card2k .c-attr .lbl { width: 28px; opacity: .9; font-weight: 700; }
    .card2k .c-attr .bar { flex:1; height: 6px; background: rgba(255,255,255,.18); border-radius: 4px; overflow: hidden; }
    .card2k .c-attr .bar > div { height: 100%; background: #ffce3d; border-radius: 4px; }
    .card2k .c-attr .val { width: 22px; text-align: right; font-weight: 700; }
    .card2k .c-min { margin-top: 6px; font-size: .78rem; font-weight: 700; background: rgba(0,0,0,.3); border-radius: 6px; padding: 3px 6px; text-align:center; }
    .card2k .c-warn { margin-top: 5px; font-size: .68rem; font-weight: 800; background: #ff3b3b; border-radius: 5px; padding: 2px 6px; text-align:center; }
    .card2k .c-tired { margin-top: 5px; font-size: .68rem; font-weight: 800; background: #ff9f1a; color:#231200; border-radius: 5px; padding: 2px 6px; text-align:center; }
    .mini2k { border-radius: 10px; padding: 8px 10px; color:#fff; margin-bottom: 18px; animation: pop .25s ease-out; }
    .mini2k .row1 { display:flex; justify-content:space-between; font-weight:800; }
    .mini2k .row2 { font-size:.72rem; opacity:.9; }

    .team-banner { background: linear-gradient(90deg,#0b1640,#1e3c96); color:#fff; border-radius: 16px; padding: 16px 22px;
                   display:flex; gap: 26px; align-items:center; flex-wrap: wrap; box-shadow: 0 6px 20px rgba(11,22,64,.35); }
    .team-banner .big { font-size: 2.6rem; font-weight: 900; color:#ffce3d; line-height: 1; }
    .team-banner .lbl { font-size: .68rem; letter-spacing: .1em; opacity:.8; text-transform: uppercase; }
    .team-banner .val { font-size: 1.35rem; font-weight: 800; }
    .result-banner { border-radius: 16px; padding: 18px 24px; color: #fff; margin: 8px 0 12px 0; animation: pop .35s ease-out; }
    .result-banner h2 { margin: 0; color: #fff; }
    .rb-good { background: linear-gradient(90deg,#0d1f6b,#2a5bd7); border-left: 8px solid #ffce3d; }
    .rb-champ { background: linear-gradient(90deg,#7a5c00,#ffce3d); color:#1b1300; border-left: 8px solid #fff; }
    .rb-champ h2 { color:#1b1300; }
    .rb-bad { background: linear-gradient(90deg,#3a0718,#a8123e); border-left: 8px solid #ff9f9f; }
    .energy { height: 7px; background: rgba(255,255,255,.2); border-radius: 4px; overflow:hidden; margin-top: 3px; }
    .energy > div { height:100%; border-radius:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data
def load_results(season):
    return pd.read_csv(DATA / f"NYK_{season}_optimizer_results_lam8.0.csv", parse_dates=["game_date"])


@st.cache_data
def load_boxscore(season):
    import unicodedata
    df = pd.read_csv(DATA / f"NYK_{season}_player_game_boxscore.csv", parse_dates=["game_date"])
    df["player_name"] = df["player_name"].map(
        lambda n: "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining(c)))
    return df


@st.cache_data
def load_official(season):
    """Official per-game season stats from basketball-reference."""
    f = "nyk_2025_26_official_per_game.csv" if season == "2025-26" else "nyk_2024_25_official_per_game.csv"
    return pd.read_csv(REF / f)


@st.cache_data
def load_schedule(season):
    f = "nyk_2025_26_schedule.csv" if season == "2025-26" else "nyk_2024_25_schedule.csv"
    return pd.read_csv(REF / f, parse_dates=["date"])


@st.cache_data
def name_map(season):
    off = load_official(season)
    return dict(zip(off["key"], off["player_name"]))


def dname(season, key):
    return name_map(season).get(key, key)


@st.cache_resource
def sim_data():
    profiles = S.load_profiles()
    teams, sched = S.load_league()
    return profiles, teams, sched


def mmss(minutes_float):
    if pd.isna(minutes_float):
        return "0:00"
    m = int(minutes_float)
    s = int(round((minutes_float - m) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{m}:{s:02d}"


# --------------------------------------------------------------------------
# Optimizer objective (identical to src/optimize_rotation.py)
# --------------------------------------------------------------------------

FATIGUE_FLOOR = 20.0
LAMBDA = 8.0
MAX_MIN = 48.0
K_SHRINKAGE = 20
GARBAGE_TIME_MINUTES = 4.0


def _load_norm(d):
    return d.groupby("game_id")["trailing_load_7g"].transform(
        lambda s: (s + FATIGUE_FLOOR) / (s.max() + FATIGUE_FLOOR + 1e-6))


def efficiency_gap(df, lam=LAMBDA, max_min=MAX_MIN):
    d = df.copy()
    d["load_norm"] = _load_norm(d)
    coef = d["value_per48"] / 48

    def obj(col):
        return (coef * d[col] - lam * d["load_norm"] * (d[col] / max_min) ** 2).groupby(d["game_id"]).sum()

    a, o = obj("actual_minutes"), obj("optimal_minutes")
    value_part = (coef * (d["optimal_minutes"] - d["actual_minutes"])).groupby(d["game_id"]).sum().mean()
    fatigue_part = (lam * d["load_norm"] * ((d["actual_minutes"] / max_min) ** 2 - (d["optimal_minutes"] / max_min) ** 2)
                    ).groupby(d["game_id"]).sum().mean()
    return a.mean(), o.mean(), value_part, fatigue_part


def game_load_norm(g):
    load = g["trailing_load_7g"].to_numpy()
    return (load + FATIGUE_FLOOR) / (load.max() + FATIGUE_FLOOR + 1e-6)


def game_objective(g, minutes, lam=LAMBDA, max_min=MAX_MIN):
    """Value-minus-fatigue objective for one game, for any minutes allocation
    (the real coach's, the model's, or yours)."""
    load_norm = game_load_norm(g)
    coef = g["value_per48"].to_numpy() / 48.0
    minutes = np.asarray(minutes, dtype=float)
    return float((coef * minutes - lam * load_norm * (minutes / max_min) ** 2).sum())


def season_stat_table(season, box):
    """Official per-game stats (basketball-reference) next to the
    reconstructed Game Score numbers."""
    off = load_official(season).copy()
    gsum = box.groupby("player_name")["game_score"].sum()
    off["total_contribution"] = off["key"].map(gsum).fillna(0.0)
    off["gmsc"] = off["total_contribution"] / off["g"]
    off = off.sort_values("total_contribution", ascending=False)
    out = pd.DataFrame({
        "Player": off["player_name"], "GP": off["g"], "GS": off["gs"], "MPG": off["mpg"],
        "PPG": off["pts"], "RPG": off["trb"], "APG": off["ast"], "SPG": off["stl"], "BPG": off["blk"],
        "TOV": off["tov"], "FG%": (off["fg_pct"] * 100).round(1), "3P%": (off["fg3_pct"] * 100).round(1),
        "FT%": (off["ft_pct"] * 100).round(1), "Avg. Game Score": off["gmsc"].round(1),
        "Season Contribution": off["total_contribution"].round(0).astype(int),
    })
    return out


def season_summary(results, box, season):
    agg = results.groupby("player_name").agg(avg_actual=("actual_minutes", "mean"), avg_optimal=("optimal_minutes", "mean"))
    order = box.groupby("player_name")["game_score"].sum().sort_values(ascending=False).index
    agg = agg.reindex([p for p in order if p in agg.index]).reset_index()
    agg["name"] = agg["player_name"].map(lambda k: dname(season, k))
    return agg


# --------------------------------------------------------------------------
# Bracket + cards (HTML)
# --------------------------------------------------------------------------

def render_bracket_html(matchups):
    def find(conf, rnd):
        return [m for m in matchups if m["conf"] == conf and m["round"] == rnd]

    def box(m):
        a, b = (m["score"].split("-") + ["", ""])[:2] if "-" in m["score"] else ("", "")

        def team_div(name, seed, score):
            cls = "team" + (" winner" if name == m["winner"] else "") + (" nyk" if name == "NYK" else "")
            sd = f"<span style='opacity:.6'>{seed}</span> " if seed else ""
            return f'<div class="{cls}"><span>{sd}{name}</span><span class="bscore">{score}</span></div>'
        return f'<div class="bracket-box">{team_div(m["team_a"], m.get("seed_a"), a)}{team_div(m["team_b"], m.get("seed_b"), b)}</div>'

    def col(ms, label):
        return f'<div class="bracket-col"><div class="bcol-label">{label}</div>{"".join(box(m) for m in ms)}</div>'

    cols = [
        col(find("East", "Round 1"), "East · Round 1"), col(find("East", "Conf. Semifinals"), "East · Semis"),
        col(find("East", "Conf. Finals"), "East · Finals"), col(find("Finals", "NBA Finals"), "🏆 NBA Finals"),
        col(find("West", "Conf. Finals"), "West · Finals"), col(find("West", "Conf. Semifinals"), "West · Semis"),
        col(find("West", "Round 1"), "West · Round 1"),
    ]
    return f'<div class="bracket-wrap"><div class="bracket-row">{"".join(cols)}</div></div>'


def tier(ovr):
    if ovr >= 85:
        return "t-diamond"
    if ovr >= 80:
        return "t-amethyst"
    if ovr >= 75:
        return "t-ruby"
    if ovr >= 70:
        return "t-sapphire"
    if ovr >= 65:
        return "t-emerald"
    return "t-gold"


def card_html(p, slot, minutes, pos_pen, fat_pen):
    attrs = "".join(
        f'<div class="c-attr"><span class="lbl">{lbl}</span><div class="bar"><div style="width:{int(p[col])}%"></div></div>'
        f'<span class="val">{int(p[col])}</span></div>'
        for col, lbl in S.ATTRIBUTES
    )
    warn = f'<div class="c-warn">OUT OF POSITION −{pos_pen:.1f}</div>' if pos_pen > 0 else ""
    tired = f'<div class="c-tired">🔋 HEAVY LEGS −{fat_pen:.1f}</div>' if fat_pen > 0 else ""
    return (
        f'<div class="card2k {tier(p["ovr"])}"><div class="c-top"><div><div class="c-ovr">{p["ovr"]}</div>'
        f'<div class="c-ovrlbl">{"2K27 OVR" if p["ovr_is_2k"] else "OVR (est.)"}</div></div><span class="c-slot">{slot}</span></div>'
        f'<div class="c-name">{p["player_name"]}</div>'
        f'<div class="c-sub">{"/".join(p["playable"])} · Impact {p["impact"]:+.1f}</div>{attrs}'
        f'<div class="c-min">{minutes:.0f} MIN / GAME</div>{warn}{tired}</div>'
    )


def mini_html(p, minutes, fat_pen):
    tired = " · 🔋 tired" if fat_pen > 0 else ""
    return (f'<div class="mini2k {tier(p["ovr"])}"><div class="row1"><span>{p["player_name"]}</span>'
            f'<span>{p["ovr"]}</span></div><div class="row2">{"/".join(p["playable"])} · {minutes:.0f} min{tired}</div></div>')


# --------------------------------------------------------------------------
# Methodology dialog
# --------------------------------------------------------------------------

@st.dialog("How this was built", width="large")
def methodology_dialog():
    st.markdown("#### 1. Turning play-by-play into on-court lineups")
    st.markdown(
        """
Every substitution in the raw play-by-play log is an event (`SUB: <in> FOR <out>`). Walking those events in
order, starting from a detected lineup, tells you which 5 players were on the court for every second of the
game, the one number no public box score publishes directly.

The lineup is **re-detected at the start of every period**, because a change made between quarters happens on a
dead clock and is never logged as a substitution. That fix alone corrected Karl-Anthony Towns being undercounted
by about 10 minutes a game.

One gap remained: a player who plays an entire quarter without recording a single stat and without being subbed
leaves no trace in the log. That happened in **34 of the 164 games** (24 in 2024-25, 10 in 2025-26). For those 34
games the per-player minutes and plus-minus come straight from the official basketball-reference box score, and
each one was checked two ways before use: the minutes add up to exactly 5 × game length, and the players'
plus-minus adds up to exactly 5 × the final margin. Result: **every one of the 164 games now sums to exactly 240
minutes** (265 or 290 with overtime), and every player's games played and minutes per game match
basketball-reference.
        """
    )
    st.markdown("#### 2. Net rating, blended with individual box production")
    st.latex(r"\text{NetRating}_{48} = \frac{\text{team point differential while on court}}{\text{seconds on court}} \times 48 \times 60")
    st.markdown("Small samples get shrunk toward a prior, so a hot 3-game stretch for a bench player doesn't dominate:")
    st.latex(r"\text{trailingRating} = \text{rawTrailingRating} \times \frac{g}{g + k}, \quad k = " + str(K_SHRINKAGE))
    st.caption("g = games played so far this season; only games strictly BEFORE the one being predicted are used (no lookahead).")
    st.markdown(
        """
Net rating alone is a **team-context** number: a role player who enters during winning stretches inherits the team's
point differential. So it's blended 50/50 with each player's own trailing **Game Score per 48**, shrunk the same way.
        """
    )
    st.latex(r"\text{value}_{48} = 0.5 \times \text{trailingRating} + 0.5 \times \text{trailingGameScore}_{48}")

    st.markdown("#### 3. The optimizer")
    st.latex(
        r"\max_{m \ge 0} \; \sum_i c_i m_i \;-\; \lambda \sum_i w_i \left(\frac{m_i}{M}\right)^2 "
        r"\quad \text{s.t.} \quad \sum_i m_i = T,\;\; 0 \le m_i \le M"
    )
    st.markdown(
        f"""
- **c_i**: player i's blended value ÷ 48 (predicted value per minute)
- **w_i**: normalized fatigue load (minutes over the trailing 7 games, plus a floor of {FATIGUE_FLOOR:.0f})
- **λ = {LAMBDA}**: how heavily fatigue is weighted against value
- **M = {MAX_MIN:.0f} min**: hard per-player cap
- **T**: that game's real total team-minutes (240, or 265/290 with overtime)

The same maximize-value-minus-cost structure as the companion EV charging paper. Garbage-time cameos (under
{GARBAGE_TIME_MINUTES:.0f} real minutes) are held fixed, since no coach is really choosing to play someone 90 seconds.
        """
    )
    st.markdown("#### 4. Game Score")
    st.latex(
        r"GmSc = PTS + 0.4\cdot FGM - 0.7\cdot FGA - 0.4\cdot(FTA-FTM) + 0.7\cdot ORB + 0.3\cdot DRB "
        r"+ STL + 0.7\cdot AST + 0.7\cdot BLK - 0.4\cdot PF - TOV"
    )
    st.markdown(
        """
John Hollinger's Game Score (basketball-reference's **GmSc**), computed from a box score rebuilt event by event
from the play-by-play. Checks: the rebuilt team points equal the real final score in all 164 games, and every
player's season PPG, RPG, APG, SPG, BPG and turnovers match basketball-reference to within 0.1.
        """
    )
    st.markdown("#### 5. The 2K Mode simulator")
    st.markdown(
        f"""
- **Cards**: OVR and the six attributes (inside, outside, playmaking, athleticism, defending, rebounding) are each
  player's NBA 2K27 ratings. They're for display; results come from real data, below.
- **Player impact** = real 2025-26 Box Plus/Minus (basketball-reference), shrunk toward replacement level
  (−2.0) by minutes played, so a 2-game sample can't create a superstar.
- **Every season is different**: each player gets a random breakout or slump year (about ±1 point per 48), each
  other team's rating moves by about ±1.5, then every game, injury and playoff series is drawn.
- **Team rating** = Σ impact × minutes ÷ 48, plus one constant chosen so the Knicks' **real** 2025-26 minutes give
  their **real** net rating (+6.5). Unassigned minutes go to a G League call-up (−2.5).
- **Penalties**: starters out of position lose 1 / 3 / 5 / 7 points per 48 (1 to 4 spots away); every minute
  per game above 36 costs 0.22 per 48 and raises injury risk.
- **Games**: margin ~ Normal(rating gap ± 2 home court, {S.SIGMA}). The {S.SIGMA}-point spread was fitted so the
  30 real 2025-26 net ratings reproduce the 30 real records (about 3 wins of error per team). Your Knicks play the
  real 82-game schedule; the other 29 teams use their real net ratings; play-in and playoffs are played game by
  game with real 2-2-1-1-1 home court and a 13-point spread (full-strength teams).
        """
    )
    st.markdown("#### 6. What's still an estimate")
    st.markdown(
        """
Net rating is reconstructed, not published, and it is noisy for players with few games. The value coefficient is a
model choice (50/50 net rating and Game Score), and so are λ and the fatigue floor. The simulator's penalties are
assumptions chosen to be realistic, not fitted.
        """
    )
    if st.button("Close", width="stretch"):
        st.rerun()


# --------------------------------------------------------------------------
# Story cards
# --------------------------------------------------------------------------

# Every score, date, record and stat below was checked against
# basketball-reference.com/teams/NYK/2026_games.html, the individual box
# scores, and AP/NBA.com game coverage.
MOMENTS = [
    {
        "tag": "Opening Night · Oct 22, 2025", "emoji": "🏀",
        "title": "A statement over the defending No. 1 seed",
        "body": (
            "Mike Brown's first game as head coach: the Knicks beat the Cleveland Cavaliers, the East's No. 1 seed the "
            "year before, 119-111 at Madison Square Garden. OG Anunoby led the way with 24 points and 14 rebounds, and "
            "Jalen Brunson added 23."
        ),
        "stats": [("119-111", "final score vs. CLE"), ("24 / 14", "Anunoby pts / reb")],
    },
    {
        "tag": "November", "emoji": "📈",
        "title": "A shaky start turns into an 11-3 run",
        "body": (
            "New York opened 2-3, losing three straight on the road in Miami, Milwaukee and Chicago. Then it won 11 of "
            "its next 14 to finish November 13-6."
        ),
        "stats": [("2-3", "start to the season"), ("11-3", "the next 14 games"), ("13-6", "record on Nov 30")],
    },
    {
        "tag": "NBA Cup Final · Dec 16, 2025 · Las Vegas", "emoji": "🥇",
        "title": "The first trophy: the NBA Cup",
        "body": (
            "Down double digits to San Antonio late in the third quarter, the Knicks outscored the Spurs 35-19 in the "
            "fourth to win the Emirates NBA Cup final 124-113, the franchise's first Cup. Brunson was named Cup MVP, "
            "Anunoby led all scorers with 28, and Mitchell Robinson grabbed 15 rebounds, 10 of them offensive. Six months "
            "later, the same two teams would meet again in the NBA Finals."
        ),
        "stats": [("124-113", "final vs. SAS"), ("35-19", "4th quarter"), ("MVP", "Brunson, Cup MVP")],
    },
    {
        "tag": "Dec 31 – Jan 19", "emoji": "⚠️",
        "title": "The stretch that tested the season",
        "body": (
            "Two separate four-game losing streaks in less than three weeks dropped the Knicks to 2-9 over this stretch, "
            "their worst run of the year. Their answer: an eight-game winning streak from Jan 21 to Feb 4."
        ),
        "stats": [("2-9", "record, Dec 31 – Jan 19"), ("2", "four-game losing streaks"), ("8", "straight wins after")],
    },
    {
        "tag": "Jan 21, 2026", "emoji": "💥",
        "title": "The most lopsided win in Knicks history",
        "body": (
            "Two days after the second four-game skid, New York ended it in historic fashion: 120-66 over Brooklyn. The "
            "54-point margin is the largest win in franchise history (the old record was 48), and the Knicks led by as "
            "many as 59."
        ),
        "stats": [("120-66", "final score vs. BKN"), ("54", "point margin, franchise record"), ("59", "biggest lead")],
    },
    {
        "tag": "Feb 5, 2026 · Trade Deadline", "emoji": "🔁",
        "title": "Reshaping the bench",
        "body": (
            "New York traded for Jose Alvarado from New Orleans, adding point-of-attack defense and pace off the bench, "
            "and sent Guerschon Yabusele to Chicago in a separate deal. A week later, on Feb 13, the Knicks signed "
            "forward Jeremy Sochan after the Spurs released him."
        ),
        "stats": [("Alvarado", "acquired from NOP"), ("Yabusele", "traded to CHI"), ("Sochan", "signed Feb 13")],
    },
    {
        "tag": "March 30, 2026", "emoji": "✅",
        "title": "Playoffs, clinched",
        "body": (
            "New York locked up a top-six seed, and with it a spot in the playoffs without the play-in, on a night it "
            "didn't even play: Philadelphia's loss in Miami made it official. The Knicks were 48-27 at the time and "
            "finished 53-29, third in the East."
        ),
        "stats": [("Clinched", "March 30"), ("53-29", "final record"), ("3rd", "in the East")],
    },
    {
        "tag": "Round 1 · vs Hawks", "emoji": "🔥",
        "title": "A record halftime lead closes it out",
        "body": (
            "Down 2-1 in the series, New York won three straight to beat Atlanta 4-2. Game 6 in Atlanta was the "
            "exclamation point: a 47-point halftime lead, the largest in NBA playoff history, on the way to a 140-89 win."
        ),
        "stats": [("4-2", "series vs ATL"), ("47", "point halftime lead, a playoff record"), ("140-89", "Game 6")],
    },
    {
        "tag": "Conf. Semis · vs 76ers", "emoji": "🧹",
        "title": "A sweep",
        "body": (
            "New York swept Philadelphia in four straight, capped by a 144-114 win in Game 4, to reach the conference "
            "finals for the second straight year. A year earlier, the Pacers had ended the Knicks' season in that round."
        ),
        "stats": [("4-0", "series vs PHI"), ("144", "points in Game 4")],
    },
    {
        "tag": "Conf. Finals · vs Cavaliers", "emoji": "🚀",
        "title": "First Finals trip since 1999",
        "body": (
            "New York swept the same Cleveland team it beat on opening night, starting with a 115-104 overtime win in "
            "Game 1, and reached the NBA Finals for the first time since the 1999 team, a 27-year wait."
        ),
        "stats": [("4-0", "series vs CLE"), ("27 years", "since the last Finals trip")],
    },
    {
        "tag": "NBA Finals, Game 4 · June 10, 2026 · Madison Square Garden", "emoji": "🎯", "featured": True,
        "title": "The Anunoby tip-in",
        "body": (
            "Down 76-49 at halftime and by as many as 29, the Knicks pulled off the largest comeback in NBA Finals "
            "history (the old record was the 2008 Celtics' 24-point rally) to win 107-106. With 1.2 seconds left, "
            "Jalen Brunson's long pull-up came up short and OG Anunoby flew in to tip it home."
        ),
        "stats": [("33", "Anunoby pts, 7 threes"), ("36 / 7", "Brunson pts / ast"), ("29", "deficit erased"), ("3-1", "series lead")],
    },
    {
        "tag": "NBA Finals, Game 5 · June 13, 2026 · San Antonio", "emoji": "🏆", "featured": True,
        "title": "Champions, for the first time since 1973",
        "body": (
            "New York closed out San Antonio 94-90 on the road to win the series 4-1. Jalen Brunson scored 45, a Knicks "
            "Finals record, and was named Finals MVP. It's the franchise's third championship and first since 1973, and "
            "the Knicks became the first team to win the NBA Cup and the NBA title in the same season."
        ),
        "stats": [("94-90", "Game 5 final"), ("45", "Brunson pts, Finals MVP"), ("1973", "last title before this"), ("1st", "Cup + title, same season")],
    },
]

if "moment_idx" not in st.session_state:
    st.session_state.moment_idx = 0


def render_movie_card(m):
    stats_html = "".join(
        f'<div class="movie-stat"><div class="num">{v}</div><div class="label">{l}</div></div>' for v, l in m["stats"]
    )
    st.markdown(
        f'<div class="movie-card"><span class="tag">{m["tag"]}</span><h2>{m.get("emoji","🏀")} {m["title"]}</h2>'
        f'<p>{m["body"]}</p>{stats_html}</div>',
        unsafe_allow_html=True,
    )
    dots = "".join(f'<span class="dot{" active" if i == st.session_state.moment_idx else ""}"></span>' for i in range(len(MOMENTS)))
    st.markdown(f'<div class="dot-row">{dots}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Out-coach the coach
# --------------------------------------------------------------------------

COACH_LEVELS = [(0, "📼 Video coordinator"), (1, "📋 Assistant coach"), (3, "🧢 Associate head coach"),
                (5, "🎯 Head coach"), (10, "🏆 Coach of the Year")]


def grade_for(capture, beat):
    if capture is None:
        return ("A+", "Perfect read") if beat else ("C", "Even with the coach")
    if capture >= 90:
        return "A+", "Coach of the Year stuff"
    if capture >= 60:
        return "A", "You out-coached him"
    if capture >= 30:
        return "B", "Solid win on the whiteboard"
    if capture >= 0:
        return "C", "You edged him"
    if capture >= -50:
        return "D", "He had this one"
    return "F", "Back to the film room"


def render_challenge(season, results):
    coach = COACH[season]
    st.header(f"🧠 Out-coach {coach}")
    st.caption(
        f"Pick a real {season} game and split the minutes yourself, using only what {coach} knew before tip-off: each "
        "player's recent form and how tired he is. Then lock in and see whose plan scores higher on the same "
        "value-minus-fatigue model, and how close either of you got to the model's best possible plan."
    )
    sched = load_schedule(season)
    games = results[["game_id", "game_date", "opponent"]].drop_duplicates().sort_values("game_date").reset_index(drop=True)
    games["label"] = games["game_date"].dt.strftime("%b %d, %Y") + " vs " + games["opponent"]
    gkey, bkey = f"oc_game_{season}", f"oc_board_{season}"
    if gkey not in st.session_state:
        st.session_state[gkey] = int(games["game_id"].iloc[0])
    if bkey not in st.session_state:
        st.session_state[bkey] = []

    c1, c2, c3 = st.columns([1.2, 2.4, 1.4])
    with c1:
        if st.button("🎲 Deal me a random game", width="stretch", key=f"oc_deal_{season}"):
            st.session_state[gkey] = int(random.choice(games["game_id"].tolist()))
            st.rerun()
    with c2:
        labels = games["label"].tolist()
        idx = int(games.index[games["game_id"] == st.session_state[gkey]][0])
        pick = st.selectbox("Or pick a game", labels, index=idx, label_visibility="collapsed", key=f"oc_pick_{season}_{idx}")
        new_gid = int(games.loc[games["label"] == pick, "game_id"].iloc[0])
        if new_gid != st.session_state[gkey]:
            st.session_state[gkey] = new_gid
            st.rerun()
    board = st.session_state[bkey]
    beat = sum(1 for b in board if b["beat"])
    level = [lvl for n, lvl in COACH_LEVELS if beat >= n][-1]
    with c3:
        st.markdown(f"**{level}** · beat {coach} in **{beat}** of **{len(board)}**")

    gid = st.session_state[gkey]
    g = results[results["game_id"] == gid].copy().reset_index(drop=True)
    g["name"] = g["player_name"].map(lambda k: dname(season, k))
    total = round(g["actual_minutes"].sum() / 5.0) * 5.0
    rot_mask = g["actual_minutes"] >= GARBAGE_TIME_MINUTES
    fixed = g[~rot_mask]
    rot = g[rot_mask].copy()
    rot_total = total - fixed["actual_minutes"].sum()
    load_norm = game_load_norm(g)
    g["energy"] = 1.0 - (load_norm - load_norm.min()) / max(load_norm.max() - load_norm.min(), 1e-6) * 0.85
    gdate = g["game_date"].iloc[0]

    season_avg = results.groupby("player_name")["actual_minutes"].mean()
    rkey = f"oc_res_{season}_{gid}"
    st.markdown(
        f"#### {gdate:%b %d, %Y} · Knicks vs {g['opponent'].iloc[0]}  \n"
        f"You have **{total:.0f}** team minutes. **{rot_total:.1f}** are yours to hand out across these **{len(rot)}** "
        f"rotation players" + (f" (garbage-time cameos are fixed: {', '.join(fixed['name'])})." if len(fixed) else ".")
        + " The real result stays hidden until you lock in."
    )

    # defaults: each player's season-average minutes, scaled to fill the rotation minutes
    base = rot["player_name"].map(season_avg).fillna(10.0)
    base = base * rot_total / base.sum()
    for (_, r), b in zip(rot.iterrows(), base):
        k = f"oc_{season}_{gid}_{r['player_name']}"
        if k not in st.session_state:
            st.session_state[k] = float(min(48.0, round(b * 2) / 2))

    ncol = 4
    for start in range(0, len(rot), ncol):
        cols = st.columns(ncol)
        for col, (_, r) in zip(cols, rot.iloc[start:start + ncol].iterrows()):
            i = g.index[g["player_name"] == r["player_name"]][0]
            energy = float(g.loc[i, "energy"])
            form = float(r["value_per48"])
            color = "#43e97b" if energy > 0.66 else ("#ffce3d" if energy > 0.33 else "#ff5f6d")
            form_tag = "🔥" if form >= 10 else ("👍" if form >= 5 else ("😐" if form >= 0 else "🧊"))
            with col:
                st.markdown(
                    f'<div class="mini2k t-sapphire"><div class="row1"><span>{r["name"]}</span><span>{form_tag} {form:+.1f}</span></div>'
                    f'<div class="row2">Form (value per 48) · last 7 games: {r["trailing_load_7g"]:.0f} min</div>'
                    f'<div class="energy"><div style="width:{energy*100:.0f}%;background:{color}"></div></div>'
                    f'<div class="row2">Energy {energy*100:.0f}%</div></div>',
                    unsafe_allow_html=True,
                )
                st.slider("Minutes", 0.0, 48.0, step=0.5, key=f"oc_{season}_{gid}_{r['player_name']}",
                          label_visibility="collapsed")

    keys = [f"oc_{season}_{gid}_{p}" for p in rot["player_name"]]
    mine = np.array([st.session_state[k] for k in keys])
    diff = mine.sum() - rot_total

    def autofill():
        vals = np.array([st.session_state[k] for k in keys], dtype=float)
        vals = vals * rot_total / vals.sum() if vals.sum() > 0 else np.full(len(vals), rot_total / len(vals))
        vals = np.minimum(48.0, np.round(vals * 2) / 2)
        vals[np.argmax(vals)] += rot_total - vals.sum()
        for k, v in zip(keys, vals):
            st.session_state[k] = float(min(48.0, max(0.0, v)))

    b1, b2, b3 = st.columns([1.2, 1.2, 3])
    b1.button("⚖️ Auto-fill to the total", on_click=autofill, width="stretch", key=f"oc_fill_{season}_{gid}")
    lock = b2.button("🔒 Lock in my rotation", type="primary", width="stretch", key=f"oc_lock_{season}_{gid}")
    if abs(diff) < 0.26:
        b3.success(f"{mine.sum():.1f} / {rot_total:.1f} minutes: ready")
    else:
        b3.warning(f"{mine.sum():.1f} / {rot_total:.1f} minutes ({'over' if diff > 0 else 'short'} by {abs(diff):.1f}). "
                   "Locking in will scale everyone to fit.")

    if lock:
        scaled = mine * rot_total / mine.sum() if mine.sum() > 0 else np.full(len(mine), rot_total / len(mine))
        scaled = np.minimum(scaled, 48.0)
        full = g["actual_minutes"].to_numpy().copy()
        full[rot_mask.to_numpy()] = scaled
        you = game_objective(g, full)
        coach_v = game_objective(g, g["actual_minutes"])
        opt_v = game_objective(g, g["optimal_minutes"])
        capture = (you - coach_v) / (opt_v - coach_v) * 100 if opt_v - coach_v > 0.05 else None
        st.session_state[rkey] = {"you": you, "coach": coach_v, "opt": opt_v, "capture": capture, "mins": full}
        # one entry per game: re-trying a game replaces your earlier attempt
        board = [b for b in board if b["gid"] != gid] + [{"gid": gid, "beat": you > coach_v, "capture": capture}]
        st.session_state[bkey] = board
        st.rerun()

    res = st.session_state.get(rkey)
    if res:
        grade, msg = grade_for(res["capture"], res["you"] > res["coach"])
        good = res["you"] > res["coach"]
        cls = "rb-champ" if grade == "A+" else ("rb-good" if good else "rb-bad")
        sgame = sched[sched["date"] == gdate]
        real_line = ""
        if len(sgame):
            sg = sgame.iloc[0]
            real_line = f" · Real result: Knicks {'won' if sg['result'] == 'W' else 'lost'} {sg['nyk_pts']}-{sg['opp_pts']}"
        st.markdown(
            f'<div class="result-banner {cls}"><h2>Grade {grade}: {msg}</h2>'
            f'<div>Your plan {res["you"]:.1f} · {coach} {res["coach"]:.1f} · Model best {res["opt"]:.1f}'
            + (f' · You captured {res["capture"]:.0f}% of the possible improvement' if res["capture"] is not None else "")
            + f"{real_line}</div></div>",
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_bar(name="You", x=g["name"], y=res["mins"], marker_color="#2a5bd7")
        fig.add_bar(name=coach, x=g["name"], y=g["actual_minutes"], marker_color="#0d1f6b")
        fig.add_bar(name="Model best", x=g["name"], y=g["optimal_minutes"], marker_color="#ffce3d")
        fig.update_layout(barmode="group", yaxis_title="Minutes", legend_title=None, height=340, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Scores are in the model's value units (predicted value minus the fatigue penalty) for this exact game. "
            "Try again with different minutes, or deal a new game. Your record against the coach is kept for this session."
        )


# --------------------------------------------------------------------------
# 2K Mode
# --------------------------------------------------------------------------

PRESETS = {
    "🏆 Mike Brown's rotation": {
        "slots": {"PG": "Brunson", "SG": "Bridges", "SF": "Hart", "PF": "Anunoby", "C": "Towns"},
        "bench": ["Robinson", "McBride", "Shamet", "Clarkson", "Kolek"],
        "min": {"Brunson": 35, "Bridges": 33, "Hart": 30, "Anunoby": 33, "Towns": 31,
                "Robinson": 20, "McBride": 20, "Shamet": 16, "Clarkson": 14, "Kolek": 8},
    },
    "🔥 Ride the starters": {
        "slots": {"PG": "Brunson", "SG": "Bridges", "SF": "Hart", "PF": "Anunoby", "C": "Towns"},
        "bench": ["Robinson", "McBride"],
        "min": {"Brunson": 42, "Bridges": 42, "Hart": 42, "Anunoby": 42, "Towns": 42, "Robinson": 18, "McBride": 12},
    },
    "🐜 Small ball": {
        "slots": {"PG": "Brunson", "SG": "McBride", "SF": "Bridges", "PF": "Hart", "C": "Anunoby"},
        "bench": ["Towns", "Shamet", "Alvarado", "Clarkson"],
        "min": {"Brunson": 34, "McBride": 30, "Bridges": 34, "Hart": 34, "Anunoby": 34,
                "Towns": 24, "Shamet": 18, "Alvarado": 16, "Clarkson": 16},
    },
    "🪑 Bench mob": {
        "slots": {"PG": "Clarkson", "SG": "Dadiet", "SF": "McCullar Jr.", "PF": "Yabusele", "C": "Jemison III"},
        "bench": ["Hukporti", "Diawara", "Jones", "Evbuomwan"],
        "min": {"Clarkson": 36, "Dadiet": 34, "McCullar Jr.": 34, "Yabusele": 34, "Jemison III": 30,
                "Hukporti": 20, "Diawara": 20, "Jones": 18, "Evbuomwan": 14},
    },
}


def _apply_preset(name):
    pr = PRESETS[name]
    for s, k in pr["slots"].items():
        st.session_state[f"s2k_slot_{s}"] = k
    st.session_state["s2k_bench"] = list(pr["bench"])
    for k, m in pr["min"].items():
        st.session_state[f"s2k_min_{k}"] = int(m)
    st.session_state["s2k_prev_slots"] = dict(pr["slots"])
    st.session_state.pop("s2k_last", None)


def _apply_random():
    profiles, _, _ = sim_data()
    keys = list(profiles.index)
    random.shuffle(keys)
    for s, k in zip(S.SLOTS, keys[:5]):
        st.session_state[f"s2k_slot_{s}"] = k
    st.session_state["s2k_bench"] = keys[5:9]
    mins = [random.randint(26, 40) for _ in range(5)] + [random.randint(8, 22) for _ in range(4)]
    for k, m in zip(keys[:9], mins):
        st.session_state[f"s2k_min_{k}"] = m
    st.session_state["s2k_prev_slots"] = {s: st.session_state[f"s2k_slot_{s}"] for s in S.SLOTS}
    st.session_state.pop("s2k_last", None)


def _on_slot_change(slot):
    """2K-style swap: picking a player who is already in another slot (or on
    the bench) swaps him with whoever you replaced, instead of an error."""
    new = st.session_state[f"s2k_slot_{slot}"]
    prev = st.session_state["s2k_prev_slots"]
    old = prev[slot]
    for other in S.SLOTS:
        if other != slot and st.session_state[f"s2k_slot_{other}"] == new:
            st.session_state[f"s2k_slot_{other}"] = old
    bench = st.session_state.get("s2k_bench", [])
    if new in bench:
        st.session_state["s2k_bench"] = [old if b == new else b for b in bench]
    if f"s2k_min_{new}" not in st.session_state:
        st.session_state[f"s2k_min_{new}"] = 30
    st.session_state["s2k_prev_slots"] = {s: st.session_state[f"s2k_slot_{s}"] for s in S.SLOTS}


def _on_bench_change():
    for k in st.session_state.get("s2k_bench", []):
        if f"s2k_min_{k}" not in st.session_state:
            st.session_state[f"s2k_min_{k}"] = 14


def _current_spots():
    spots = [S.Spot(st.session_state[f"s2k_slot_{s}"], float(st.session_state[f"s2k_min_{st.session_state[f's2k_slot_{s}']}"]), s)
             for s in S.SLOTS]
    spots += [S.Spot(k, float(st.session_state[f"s2k_min_{k}"]), None) for k in st.session_state["s2k_bench"]]
    return spots


def _autofill_240():
    spots = _current_spots()
    tot = sum(s.minutes for s in spots)
    if tot <= 0:
        return
    vals = np.array([s.minutes * 240.0 / tot for s in spots])
    vals = np.minimum(48, np.round(vals))
    vals[np.argmax(vals)] += 240 - vals.sum()
    for s, v in zip(spots, vals):
        st.session_state[f"s2k_min_{s.key}"] = int(min(48, max(0, v)))


def render_2k_mode():
    profiles, teams, sched = sim_data()
    st.header("🎮 2K Mode: build your Knicks and sim the season")
    st.caption(
        "Player cards show each player's NBA 2K27 overall and attribute ratings. Pick a starting five and a bench, set the "
        "minutes, and watch your team rating update live. Then sim a full season on the Knicks' real schedule against the "
        "real 2025-26 league, or sim 200 seasons to see your title odds. Every sim is different: players have breakout or "
        "slump years, other teams rise and fall, and injuries happen. Game results are driven by each player's real "
        "2025-26 impact (Box Plus/Minus), so the outcomes stay grounded in what actually happened on the court."
    )
    if "s2k_slot_PG" not in st.session_state:
        _apply_preset("🏆 Mike Brown's rotation")

    pcols = st.columns(len(PRESETS) + 1)
    for col, name in zip(pcols, PRESETS):
        col.button(name, on_click=_apply_preset, args=(name,), width="stretch", key=f"preset_{name}")
    pcols[-1].button("🎲 Randomize", on_click=_apply_random, width="stretch", key="preset_random")

    roster = sorted(profiles.index, key=lambda k: -profiles.loc[k, "ovr"])
    fmt = lambda k: f"{profiles.loc[k, 'player_name']} · {profiles.loc[k, 'ovr']} OVR"

    st.markdown("##### Starting five")
    scols = st.columns(5)
    for slot, col in zip(S.SLOTS, scols):
        with col:
            st.selectbox(slot, roster, format_func=fmt, key=f"s2k_slot_{slot}", on_change=_on_slot_change, args=(slot,))
    starters = [st.session_state[f"s2k_slot_{s}"] for s in S.SLOTS]
    bench_opts = [k for k in roster if k not in starters]
    st.session_state["s2k_bench"] = [k for k in st.session_state.get("s2k_bench", []) if k in bench_opts][:5]
    st.multiselect("Bench rotation (up to 5, in order)", bench_opts, format_func=fmt, key="s2k_bench",
                   max_selections=5, on_change=_on_bench_change)
    for k in starters + st.session_state["s2k_bench"]:
        if f"s2k_min_{k}" not in st.session_state:
            st.session_state[f"s2k_min_{k}"] = 30 if k in starters else 14

    spots = _current_spots()
    # Cards + minute sliders for the starters
    ccols = st.columns(5)
    for sp, col in zip(spots[:5], ccols):
        p = profiles.loc[sp.key]
        _, pos_pen, fat_pen = S.player_effective_impact(p, sp.minutes, sp.slot)
        with col:
            st.markdown(card_html(p, sp.slot, sp.minutes, pos_pen, fat_pen), unsafe_allow_html=True)
            st.slider(f"{p['player_name']} minutes", 0, 48, step=1, key=f"s2k_min_{sp.key}", label_visibility="collapsed")
    if st.session_state["s2k_bench"]:
        bcols = st.columns(5)
        for sp, col in zip(spots[5:], bcols):
            p = profiles.loc[sp.key]
            _, _, fat_pen = S.player_effective_impact(p, sp.minutes, None)
            with col:
                st.markdown(mini_html(p, sp.minutes, fat_pen), unsafe_allow_html=True)
                st.slider(f"{p['player_name']} minutes", 0, 48, step=1, key=f"s2k_min_{sp.key}", label_visibility="collapsed")

    spots = _current_spots()
    norm, unassigned, scale = S.normalize_minutes(spots)
    tot = sum(s.minutes for s in spots)
    rating = S.team_rating(spots, profiles)
    ovr = S.team_ovr(spots, profiles)
    exp_w = S.expected_wins(rating, teams, sched)
    seed = S.projected_seed(exp_w, teams)
    ordinal = lambda n: f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

    m1, m2 = st.columns([3, 1.2])
    with m1:
        st.progress(min(tot / 240.0, 1.0))
        if tot > 240:
            st.caption(f"**{tot:.0f} / 240** minutes: over by {tot - 240:.0f}, so everyone is scaled down {100 * (1 - scale):.0f}% to fit.")
        elif unassigned > 0.5:
            st.caption(f"**{tot:.0f} / 240** minutes: the other **{unassigned:.0f}** go to a G League call-up (impact −2.5).")
        else:
            st.caption("**240 / 240** minutes: perfect.")
    with m2:
        st.button("⚖️ Auto-fill to 240", on_click=_autofill_240, width="stretch", key="s2k_fill")

    st.markdown(
        f'<div class="team-banner"><div><div class="big">{ovr}</div><div class="lbl">Team OVR</div></div>'
        f'<div><div class="val">{rating:+.1f}</div><div class="lbl">Net rating</div></div>'
        f'<div><div class="val">{exp_w:.0f}-{82 - exp_w:.0f}</div><div class="lbl">Projected record</div></div>'
        f'<div><div class="val">{ordinal(seed)} in East</div><div class="lbl">Projected seed</div></div>'
        f'<div><div class="val">+6.5 · 53-29</div><div class="lbl">Real 2025-26 Knicks</div></div></div>',
        unsafe_allow_html=True,
    )

    notes = []
    for sp in norm:
        p = profiles.loc[sp.key]
        _, pos_pen, fat_pen = S.player_effective_impact(p, sp.minutes, sp.slot)
        if pos_pen > 0:
            notes.append(f"⚠️ **{p['player_name']} at {sp.slot}**: he plays {'/'.join(p['playable'])}, so he loses "
                         f"{pos_pen:.1f} points per 48 there.")
        if sp.minutes >= 40:
            games_out = 82 * S.INJURY_BASE * (sp.minutes / 36) ** 3 * 7
            notes.append(f"🔋 **{p['player_name']} at {sp.minutes:.0f} min/night**: heavy legs cost {fat_pen:.1f} per 48, and "
                         f"he'll miss about {games_out:.0f} games to injury on average.")
        if sp.minutes >= 28 and p["impact"] < -1.5:
            notes.append(f"📉 **{p['player_name']}** (impact {p['impact']:+.1f}) is getting starter minutes. That drags the team down.")
    for n in notes[:6]:
        st.markdown(n)

    with st.expander("📋 Projected per-game stats for your rotation"):
        rows = []
        for sp in norm:
            p = profiles.loc[sp.key]
            f = sp.minutes / p["mpg"] if p["mpg"] > 0 else 0
            rows.append({"Player": p["player_name"], "Slot": sp.slot or "Bench", "MIN": round(sp.minutes),
                         "PTS": round(p["pts"] * f, 1), "REB": round(p["trb"] * f, 1), "AST": round(p["ast"] * f, 1),
                         "OVR": p["ovr"], "Impact": round(p["impact"], 1)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("His real 2025-26 per-minute production scaled to the minutes you gave him.")

    a, b = st.columns(2)
    if a.button("▶️ Sim the season", type="primary", width="stretch", key="s2k_sim1"):
        st.session_state["s2k_count"] = st.session_state.get("s2k_count", 0) + 1
        rng = np.random.default_rng()
        st.session_state["s2k_last"] = {"type": "one", "res": S.simulate_season(spots, profiles, teams, sched, rng),
                                        "rating": rating, "fresh": True}
    if b.button("⏩ Sim 200 seasons (title odds)", width="stretch", key="s2k_sim200"):
        with st.spinner("Simulating 200 seasons..."):
            many = S.simulate_many(spots, profiles, teams, sched, 200)
        st.session_state["s2k_last"] = {"type": "many", "res": many, "rating": rating, "fresh": True}

    last = st.session_state.get("s2k_last")
    if not last:
        return
    if last["type"] == "one":
        _render_one(last, profiles)
    else:
        _render_many(last)
    last["fresh"] = False


def _render_one(last, profiles):
    r = last["res"]
    emoji, label = S.OUTCOME_LABELS[r.outcome]
    seed_txt = f"{r.seed}{'st' if r.seed == 1 else 'nd' if r.seed == 2 else 'rd' if r.seed == 3 else 'th'} in the East"
    cls = "rb-champ" if r.outcome == "champion" else ("rb-bad" if r.outcome in ("missed_playoffs", "lost_play_in") else "rb-good")
    st.markdown(f'<div class="result-banner {cls}"><h2>{emoji} {label}</h2><div>Sim #{st.session_state.get("s2k_count", 1)} · '
                f'{r.wins}-{r.losses} · {seed_txt} · Champion: {r.champion} · Team rating this season {r.rating:+.1f}</div></div>',
                unsafe_allow_html=True)
    if r.outcome == "champion" and last["fresh"]:
        st.balloons()

    c1, c2 = st.columns([1.6, 1])
    with c1:
        log = r.game_log
        fig = go.Figure()
        fig.add_scatter(x=log["game"], y=log["wins"], name="Your Knicks", line=dict(color="#2a5bd7", width=3))
        fig.add_scatter(x=log["game"], y=log["real_wins"], name="Real 2025-26 Knicks", line=dict(color="#ffce3d", width=3, dash="dot"))
        fig.update_layout(height=320, xaxis_title="Game", yaxis_title="Wins", margin=dict(t=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        streak = best = cur = 0
        for w in log["won"]:
            cur = cur + 1 if w else 0
            best = max(best, cur)
        big = log.loc[log["margin"].idxmax()]
        bad = log.loc[log["margin"].idxmin()]
        st.markdown(f"**Longest win streak:** {best}")
        st.markdown(f"**Best win:** +{big['margin']} vs {big['opponent']} ({pd.Timestamp(big['date']):%b %d})")
        st.markdown(f"**Worst loss:** {bad['margin']} vs {bad['opponent']} ({pd.Timestamp(bad['date']):%b %d})")
        regulars = [f for f in r.form if f["minutes"] >= 12]
        if regulars:
            up = max(regulars, key=lambda f: f["form"])
            down = min(regulars, key=lambda f: f["form"])
            if up["form"] > 0.5:
                st.markdown(f"**📈 Breakout year:** {up['player']} ({up['form']:+.1f} impact)")
            if down["form"] < -0.5:
                st.markdown(f"**📉 Slump year:** {down['player']} ({down['form']:+.1f} impact)")
        if r.injuries:
            st.markdown("**Injury report:**")
            for inj in r.injuries[:5]:
                st.markdown(f"🚑 {inj['player']} out {inj['games_out']} games ({pd.Timestamp(inj['date']):%b %d})")
        else:
            st.markdown("**Injury report:** nobody got hurt. 🙏")
    if r.knicks_series:
        st.markdown("##### Your playoff run")
        for sr in r.knicks_series:
            opp = sr["team_b"] if sr["team_a"] == "NYK" else sr["team_a"]
            won = sr["winner"] == "NYK"
            score = sr["score"] if sr["team_a"] == "NYK" else "-".join(reversed(sr["score"].split("-")))
            st.markdown(f"{'✅' if won else '❌'} **{sr['round']}** vs {opp}: {'won' if won else 'lost'} "
                        + (score if "-" in sr["score"] else ""))
    if r.outcome != "missed_playoffs":
        with st.expander("🏀 Full playoff bracket", expanded=r.outcome in ("champion", "lost_finals")):
            st.markdown(render_bracket_html([m for m in r.matchups if m["round"] != "Play-In"]), unsafe_allow_html=True)
            st.caption("Highlighted = your Knicks. Bold = series winner. Small numbers = seeds.")
    with st.expander("📊 Final standings"):
        sc1, sc2 = st.columns(2)
        for conf_name, df, col in [("East", r.east, sc1), ("West", r.west, sc2)]:
            disp = df[["team", "wins", "losses"]].rename(columns={"team": "Team", "wins": "W", "losses": "L"})
            disp.index = disp.index + 1
            col.markdown(f"**{conf_name}**")
            col.dataframe(disp.style.apply(lambda row: ["background-color: #ffce3d; color: #000" if row["Team"] == "NYK" else "" for _ in row], axis=1),
                          width="stretch", height=560)
    with st.expander("🗓️ Game-by-game log"):
        lg = r.game_log.copy()
        lg["Date"] = pd.to_datetime(lg["date"]).dt.strftime("%b %d")
        lg["Result"] = np.where(lg["won"], "W", "L") + " (" + lg["margin"].map(lambda m: f"{m:+d}") + ")"
        lg["Record"] = lg["wins"].astype(str) + "-" + lg["losses"].astype(str)
        lg["Where"] = lg["site"].map({"H": "vs", "A": "@", "N": "neutral"})
        st.dataframe(lg[["game", "Date", "Where", "opponent", "Result", "Record", "out"]].rename(
            columns={"game": "G", "opponent": "Opp", "out": "Out injured"}), hide_index=True, width="stretch", height=380)


def _render_many(last):
    df = last["res"]
    n = len(df)
    order = S.OUTCOME_ORDER
    counts = df["outcome"].value_counts().reindex(order, fill_value=0)
    title = counts["champion"] / n * 100
    finals = (counts["champion"] + counts["lost_finals"]) / n * 100
    playoffs = counts[order[2:]].sum() / n * 100
    c = st.columns(4)
    c[0].metric("🏆 Title odds", f"{title:.0f}%")
    c[1].metric("Reach the Finals", f"{finals:.0f}%")
    c[2].metric("Make the playoffs", f"{playoffs:.0f}%")
    c[3].metric("Average record", f"{df['wins'].mean():.0f}-{82 - df['wins'].mean():.0f}")
    f1, f2 = st.columns(2)
    with f1:
        fig = go.Figure(go.Bar(x=[S.OUTCOME_LABELS[o][1] for o in order], y=counts.values / n * 100, marker_color="#2a5bd7"))
        fig.update_layout(height=320, yaxis_title="% of 200 seasons", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with f2:
        fig = go.Figure(go.Histogram(x=df["wins"], marker_color="#ffce3d", xbins=dict(size=2)))
        fig.add_vline(x=53, line_dash="dot", annotation_text="real 53 wins")
        fig.update_layout(height=320, xaxis_title="Wins", yaxis_title="Seasons", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    st.caption("Each of the 200 seasons is simulated from scratch: injuries, the other 29 teams' records, the play-in and every playoff game.")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

with streamlit_analytics.track(unsafe_password="knicks2026"):

    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.title("🏆 Knicks Rotation Optimizer")
        st.caption(
            "New York Knicks, 2025-26: Mike Brown's first year as head coach ended with the NBA Cup and the franchise's "
            "third championship. Click through the season, check whether his real rotation decisions matched what a "
            "fatigue-adjusted, data-driven optimizer says was ideal, try to out-coach him, then build your own Knicks in 2K Mode."
        )
        st.caption("By **Vihaan Vinjamuri**, IBDP2")
    with top_r:
        st.write("")
        if st.button("📖 How this was built", width="stretch"):
            methodology_dialog()

    st.subheader("The season, as it happened")
    idx = st.session_state.moment_idx
    render_movie_card(MOMENTS[idx])
    scrub = st.slider("Drag to scrub through the season", 1, len(MOMENTS), idx + 1,
                      label_visibility="collapsed", key=f"movie_scrubber_{idx}")
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
        st.caption(f"Moment {idx + 1} of {len(MOMENTS)}. Drag the slider or click Next/Back. Every score, date and stat "
                   "is checked against basketball-reference box scores and AP/NBA.com game coverage.")

    st.divider()
    st.subheader("Dig into the numbers")
    season_choice = st.radio("Season", list(SEASON_LABELS.values()), horizontal=True, label_visibility="collapsed")
    season = "2025-26" if season_choice == SEASON_LABELS["2025-26"] else "2024-25"
    coach = COACH[season]
    results = load_results(season)
    box = load_boxscore(season)
    actual_obj, optimal_obj, value_part, fatigue_part = efficiency_gap(results)
    gap = optimal_obj - actual_obj

    st.subheader("Was the real rotation actually optimal?")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Avg. value, {coach}'s actual rotation", f"{actual_obj:.1f}")
    col2.metric("Avg. value at model-optimal rotation", f"{optimal_obj:.1f}")
    col3.metric("Gap to optimal, per game", f"{gap:.1f}")
    st.caption(
        f"Of the {gap:.1f}-unit gap, {value_part:.1f} comes from value left on the table and {fatigue_part:.1f} from extra "
        f"fatigue cost. Units are the model's objective (value minus fatigue penalty), averaged over all 82 games."
        + (" Thibodeau's final season, before Mike Brown took over: same model, same data pipeline, same accuracy checks."
           if season == "2024-25" else "")
    )

    st.subheader("Season stats")
    st.dataframe(season_stat_table(season, box), width="stretch", hide_index=True)
    st.caption(
        "Official per-game stats from basketball-reference. Our own play-by-play rebuild reproduces every player's games "
        "played and minutes exactly and his PPG, RPG, APG, SPG, BPG and turnovers to within 0.1. Sorted by Season "
        "Contribution (total Game Score across the season), so one big garbage-time night can't outrank a full season."
    )

    st.subheader("Season average: real minutes vs. model-optimal minutes")
    summary = season_summary(results, box, season)
    fig = go.Figure()
    fig.add_bar(name="Actual (real)", x=summary["name"], y=summary["avg_actual"], marker_color="#0d1f6b")
    fig.add_bar(name="Model-optimal", x=summary["name"], y=summary["avg_optimal"], marker_color="#ffce3d")
    fig.update_layout(barmode="group", yaxis_title="Minutes per game played", legend_title=None, height=420)
    st.plotly_chart(fig, width="stretch")

    capped = (results["optimal_minutes"] >= 47.9).mean()
    with st.expander("Why does the model recommend these minutes?"):
        st.markdown(
            f"""
For every game, the model picks each player's minutes to maximize **predicted value minus a fatigue penalty**,
using exactly the game's real total minutes and a {MAX_MIN:.0f}-minute cap. Two things move a player's number:

- **Value**: his trailing net rating blended with trailing Game Score, both shrunk toward a baseline for small
  samples (in 2025-26 the baseline is his own 2024-25 rate).
- **Fatigue load**: minutes over his last 7 games. A player used lightly lately costs little in fatigue, so the model
  may hand him more minutes than he really got. The model doesn't know why a coach limited someone (foul trouble,
  matchups, trust) beyond what's in the data.

A full {MAX_MIN:.0f}-minute recommendation is rare: about 1 in {1 / capped:.0f} player-games this season, reserved for a
player whose trailing value is unusually strong and whose recent workload has been unusually light.
            """ if capped > 0 else "The model never recommends the full 48 minutes this season."
        )

    st.subheader("Look at a single game")
    games = results[["game_id", "game_date", "opponent"]].drop_duplicates().sort_values("game_date")
    games["label"] = games["game_date"].dt.strftime("%b %d, %Y") + " vs " + games["opponent"]
    choice = st.selectbox("Game", games["label"], key=f"game_select_{season}")
    gid = games.loc[games["label"] == choice, "game_id"].iloc[0]
    gbox = box[box["game_id"] == gid].set_index("player_name")
    g = results[results["game_id"] == gid].set_index("player_name").join(
        gbox[["pts", "orb", "drb", "ast", "stl", "blk", "tov", "game_score"]], how="left").fillna(0)
    g = g.sort_values("pts", ascending=False).reset_index()
    g["name"] = g["player_name"].map(lambda k: dname(season, k))
    fig2 = go.Figure()
    fig2.add_bar(name="Actual", x=g["name"], y=g["actual_minutes"], marker_color="#0d1f6b")
    fig2.add_bar(name="Model-optimal", x=g["name"], y=g["optimal_minutes"], marker_color="#ffce3d")
    fig2.update_layout(barmode="group", yaxis_title="Minutes", legend_title=None, height=380)
    st.caption("Players ordered by points scored in this game. Wondering why a low-minutes player shows a high "
               "model-optimal bar? See \"Why does the model recommend these minutes?\" above.")
    st.plotly_chart(fig2, width="stretch")
    sg = load_schedule(season)
    sg = sg[sg["date"] == g["game_date"].iloc[0]]
    if len(sg):
        r0 = sg.iloc[0]
        st.markdown(f"**Final: Knicks {r0['nyk_pts']}, {r0['opponent']} {r0['opp_pts']}** "
                    f"({'win' if r0['result'] == 'W' else 'loss'}{', ' + r0['ot'] if r0['ot'] != '-' else ''})")
    box_disp = g[["name", "actual_minutes", "pts", "orb", "drb", "ast", "stl", "blk", "tov", "game_score"]].copy()
    box_disp["Minutes"] = box_disp["actual_minutes"].apply(mmss)
    box_disp["REB"] = box_disp["orb"] + box_disp["drb"]
    box_disp = box_disp.set_index("name")[["Minutes", "pts", "REB", "ast", "stl", "blk", "tov", "game_score"]].rename(
        columns={"pts": "PTS", "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TOV", "game_score": "Game Score"})
    box_disp.index.name = "Player"
    st.dataframe(box_disp, width="stretch")
    st.caption("Box stats rebuilt from this game's play-by-play; minutes are exact (they sum to the game's real total).")

    st.divider()
    render_challenge(season, results)

    st.divider()
    render_2k_mode()

    st.divider()
    st.caption(
        "Built with real play-by-play and box-score data, basketball-reference, and CVXPY. Code and methodology: "
        "[github.com/vihaanvinjamuri/nba-rotation-optimizer](https://github.com/vihaanvinjamuri/nba-rotation-optimizer)"
    )
