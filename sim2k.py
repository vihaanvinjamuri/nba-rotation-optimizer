"""
2K Mode season simulator for the Knicks Rotation Optimizer app.

Everything here is plain Python (no Streamlit) so it can be tested on its own.

How a lineup becomes a team rating
----------------------------------
1. Each player's impact = his real 2025-26 Box Plus/Minus (basketball-reference),
   shrunk toward replacement level (-2.0) by playing time, so a 2-game sample
   can't produce a superstar:  impact = (BPM*MP + (-2.0)*250) / (MP + 250).
2. Two penalties a real coach would feel:
     * out of position: a starter in a slot he can't play loses 1.0 / 3.0 /
       5.0 / 7.0 points per 48 minutes (1 / 2 / 3 / 4 positions away);
     * fatigue: every minute per game above 36 costs 0.22 points per 48.
3. Team rating = sum(impact_i * minutes_i) / 48 + OFFSET. Minutes nobody is
   assigned are played by a G League call-up (impact -2.5). OFFSET is solved
   once so that the Knicks' REAL 2025-26 minutes distribution gives their
   REAL net rating (+6.5). So "play it like Mike Brown did" reproduces the
   real team, and five deep-bench players produce a lottery team.

How games are decided
---------------------
A game margin ~ Normal(rating_gap +/- 2.0 home court, 14.5). The 14.5 spread
was fitted so that the 30 real 2025-26 net ratings reproduce the 30 real
records (about 3 wins of error per team). The Knicks play their real 82-game
schedule (real opponents, real home/road); the other 29 teams' records are
drawn from their real net ratings; the play-in and every playoff series are
played game by game with real 2-2-1-1-1 home court (playoff games use a 13-point
spread, the typical single-game spread for full-strength teams).

Ratings are softly capped beyond -12 / +12 (real NBA teams essentially never
finish outside about -15 / +13), so an absurd lineup is historically bad,
not mathematically impossible.

Every simulated season is different: each player gets a random breakout or
slump year (impact +/- about 1 point per 48), every other team's rating moves
by about 1.5 points, and then every game, injury and playoff series is drawn.

Card OVR and attributes are NBA 2K27's ratings (data/reference/nba2k27_ratings.csv);
they are for display. Game results come from the real 2025-26 impact above.

Injuries: each game a player has a (0.8% x (minutes/36)^3) chance to get hurt
for 2-12 games, so riding your stars 44 minutes a night is a real risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path(__file__).parent / "data" / "reference"

SLOTS = ["PG", "SG", "SF", "PF", "C"]
SLOT_INDEX = {s: i for i, s in enumerate(SLOTS)}

REPLACEMENT = -2.0          # replacement-level BPM (the VORP definition)
SHRINK_MINUTES = 250.0      # prior strength for the BPM shrinkage
CALLUP_IMPACT = -2.5        # unassigned minutes go to a G League call-up
POSITION_PENALTY = {0: 0.0, 1: 1.0, 2: 3.0, 3: 5.0, 4: 7.0}
FATIGUE_FREE_MINUTES = 36.0
FATIGUE_PER_MINUTE = 0.22
SIGMA = 14.5
PLAYOFF_SIGMA = 13.0     # playoff games: full-strength rotations, the typical single-game spread
HOME_COURT = 2.0
INJURY_BASE = 0.008
PLAYER_FORM_SD = 1.0       # season-to-season swing in a player's impact (breakout / slump years)
TEAM_FORM_SD = 1.5         # season-to-season swing in the other 29 teams' net ratings
REAL_NET_RATING = 6.5
TEAM_MINUTES = 240.0


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Player profiles (2K-style cards)
# --------------------------------------------------------------------------

def _clip(x, lo=25, hi=99):
    return int(round(min(max(x, lo), hi)))


def load_profiles() -> pd.DataFrame:
    per = pd.read_csv(REF / "nyk_2025_26_official_per_game.csv")
    adv = pd.read_csv(REF / "nyk_2025_26_advanced.csv")
    df = per.merge(adv, on="key", how="left")
    df["playable"] = df["playable_positions"].str.split("/")

    mp = df["mp_total"].astype(float)
    df["impact"] = (df["bpm"] * mp + REPLACEMENT * SHRINK_MINUTES) / (mp + SHRINK_MINUTES)
    dbpm_s = (df["dbpm"] * mp) / (mp + SHRINK_MINUTES)

    # Per-36 rates, lightly shrunk toward the roster average for small samples.
    w = mp / (mp + 300.0)
    def per36(col):
        raw = np.where(df["mpg"] > 0, df[col] / df["mpg"] * 36.0, 0.0)
        team = np.average(raw, weights=mp)
        return w * raw + (1 - w) * team

    pts36, ast36, trb36 = per36("pts"), per36("ast"), per36("trb")
    stocks36 = per36("stl") + per36("blk")
    ts = df["ts_pct"].fillna(0.5)
    fg3 = df["fg3_pct"].fillna(0.0)
    fg3_s = w * fg3 + (1 - w) * 0.33

    df["att_scoring"] = [_clip(35 + 2.0 * p + 80 * (t - 0.56)) for p, t in zip(pts36, ts)]
    box = pd.read_csv(Path(__file__).parent / "data" / "processed" / "NYK_2025-26_player_game_boxscore.csv")
    fg3a_tot = box.groupby("player_name")["fg3a"].sum()
    fg3a_pg = df["key"].map(fg3a_tot).fillna(0) / df["g"]
    fg3a36 = np.where(df["mpg"] > 0, fg3a_pg / df["mpg"] * 36.0, 0.0) * w
    df["att_shooting"] = [_clip(25 + 300 * (f - 0.30) + 3.5 * v) if not np.isnan(x) else 25
                          for f, x, v in zip(fg3_s, df["fg3_pct"], fg3a36)]
    df["att_playmaking"] = [_clip(35 + 3.2 * a36 + 3.2 * a) for a36, a in zip(ast36, df["ast"])]
    df["att_rebounding"] = [_clip(30 + 2.4 * r36 + 1.6 * r) for r36, r in zip(trb36, df["trb"])]
    df["att_defense"] = [_clip(55 + 5 * d + 5 * s) for d, s in zip(dbpm_s, stocks36)]
    df["att_stamina"] = [_clip(35 + 1.7 * m) for m in df["mpg"]]
    est_ovr = [_clip(62 + 2.2 * i + 0.55 * p + 0.15 * m, 55, 99)
               for i, p, m in zip(df["impact"], df["pts"], df["mpg"])]

    # Card OVR and attributes come from NBA 2K27. A player with no 2K27 rating
    # found falls back to an estimate from his real 2025-26 stats (marked "est.").
    k2 = pd.read_csv(REF / "nba2k27_ratings.csv").drop(columns=["player_name", "source"])
    df = df.merge(k2, on="key", how="left")
    df["ovr_is_2k"] = df["ovr_2k27"].notna()
    df["ovr"] = np.where(df["ovr_is_2k"], df["ovr_2k27"], est_ovr).astype(int)
    fallback = {"inside": "att_scoring", "outside": "att_shooting", "playmaking": "att_playmaking",
                "athleticism": "att_stamina", "defending": "att_defense", "rebounding": "att_rebounding"}
    for k2col, est_col in fallback.items():
        df["att_" + k2col] = np.where(df[k2col].notna(), df[k2col], df[est_col]).astype(int)
    return df.set_index("key")


ATTRIBUTES = [("att_inside", "INS"), ("att_outside", "OUT"), ("att_playmaking", "PLY"),
              ("att_athleticism", "ATH"), ("att_defending", "DEF"), ("att_rebounding", "REB")]


# --------------------------------------------------------------------------
# Rotation -> team rating
# --------------------------------------------------------------------------

@dataclass
class Spot:
    key: str
    minutes: float
    slot: str | None = None      # PG..C for starters, None for bench


def position_distance(slot: str | None, playable: list[str]) -> int:
    if slot is None:
        return 0
    return min(abs(SLOT_INDEX[slot] - SLOT_INDEX[p]) for p in playable)


def player_effective_impact(prof: pd.Series, minutes: float, slot: str | None) -> tuple[float, float, float]:
    """Returns (effective impact, position penalty, fatigue penalty)."""
    pos_pen = POSITION_PENALTY[position_distance(slot, prof["playable"])]
    fat_pen = FATIGUE_PER_MINUTE * max(0.0, minutes - FATIGUE_FREE_MINUTES)
    return prof["impact"] - pos_pen - fat_pen, pos_pen, fat_pen


def normalize_minutes(spots: list[Spot]) -> tuple[list[Spot], float, float]:
    """Scale down if over 240. Returns (spots, unassigned minutes, scale applied)."""
    total = sum(s.minutes for s in spots)
    scale = 1.0
    if total > TEAM_MINUTES:
        scale = TEAM_MINUTES / total
        spots = [Spot(s.key, s.minutes * scale, s.slot) for s in spots]
        total = TEAM_MINUTES
    return spots, TEAM_MINUTES - total, scale


def raw_team_rating(spots: list[Spot], profiles: pd.DataFrame) -> float:
    spots, unassigned, _ = normalize_minutes(spots)
    total = 0.0
    for s in spots:
        if s.minutes <= 0:
            continue
        eff, _, _ = player_effective_impact(profiles.loc[s.key], s.minutes, s.slot)
        total += eff * s.minutes / 48.0
    total += CALLUP_IMPACT * unassigned / 48.0
    return total


def real_rotation_spots(profiles: pd.DataFrame) -> list[Spot]:
    """The Knicks' real 2025-26 minutes distribution: each player's season
    minutes divided by 82 games, rescaled to exactly 240."""
    mins = profiles["mp_total"] / 82.0
    mins = mins * TEAM_MINUTES / mins.sum()
    return [Spot(k, float(m)) for k, m in mins.items()]


_OFFSET_CACHE: dict = {}


def rating_offset(profiles: pd.DataFrame) -> float:
    key = id(profiles)
    if key not in _OFFSET_CACHE:
        _OFFSET_CACHE[key] = REAL_NET_RATING - raw_team_rating(real_rotation_spots(profiles), profiles)
    return _OFFSET_CACHE[key]


SOFT_CAP = 12.0


def soft_cap(r: float) -> float:
    if r > SOFT_CAP:
        return SOFT_CAP + (r - SOFT_CAP) * 0.5
    if r < -SOFT_CAP:
        return -SOFT_CAP + (r + SOFT_CAP) * 0.5
    return r


def team_rating(spots: list[Spot], profiles: pd.DataFrame) -> float:
    return soft_cap(raw_team_rating(spots, profiles) + rating_offset(profiles))


def team_ovr(spots: list[Spot], profiles: pd.DataFrame) -> int:
    spots, unassigned, _ = normalize_minutes(spots)
    num = sum(profiles.loc[s.key, "ovr"] * s.minutes for s in spots) + 65 * unassigned
    return int(round(num / TEAM_MINUTES))


# --------------------------------------------------------------------------
# League data
# --------------------------------------------------------------------------

def load_league():
    teams = pd.read_csv(REF / "team_strength_2025_26.csv")
    sched = pd.read_csv(REF / "nyk_2025_26_schedule.csv", parse_dates=["date"])
    return teams, sched


def site_edge(site: str) -> float:
    return HOME_COURT if site == "H" else (-HOME_COURT if site == "A" else 0.0)


def expected_wins(rating: float, teams: pd.DataFrame, sched: pd.DataFrame) -> float:
    r = dict(zip(teams["team"], teams["net_rating"]))
    return float(sum(phi((rating - r[o] + site_edge(s)) / SIGMA) for o, s in zip(sched["opponent"], sched["site"])))


def projected_seed(wins: float, teams: pd.DataFrame) -> int:
    east = teams[(teams["conference"] == "East") & (teams["team"] != "NYK")]
    return int((east["wins"] > wins).sum()) + 1


# --------------------------------------------------------------------------
# Season simulation
# --------------------------------------------------------------------------

@dataclass
class SeasonResult:
    wins: int
    losses: int
    seed: int
    outcome: str
    champion: str
    east: pd.DataFrame
    west: pd.DataFrame
    game_log: pd.DataFrame
    injuries: list = field(default_factory=list)
    matchups: list = field(default_factory=list)
    knicks_series: list = field(default_factory=list)
    form: list = field(default_factory=list)
    rating: float = 0.0


def _game(r_home_side: float, r_other: float, edge: float, rng, sigma: float = SIGMA) -> float:
    return rng.normal(r_home_side - r_other + edge, sigma)


def _spot_contrib(spots, profiles):
    """Per-spot contribution to the raw team rating (impact after penalties x minutes/48)."""
    return np.array([player_effective_impact(profiles.loc[s.key], s.minutes, s.slot)[0] * s.minutes / 48.0
                     for s in spots])


def simulate_season(spots: list[Spot], profiles: pd.DataFrame, teams: pd.DataFrame,
                    sched: pd.DataFrame, rng: np.random.Generator) -> SeasonResult:
    spots, _, _ = normalize_minutes(spots)
    spots = [s for s in spots if s.minutes > 0]
    n = len(spots)
    teams = teams.copy()
    teams["net_rating"] = teams["net_rating"] + rng.normal(0.0, TEAM_FORM_SD, len(teams))
    rate = dict(zip(teams["team"], teams["net_rating"]))
    offset = rating_offset(profiles)
    mins = np.array([s.minutes for s in spots])
    # Every simulated season is different: each player has a breakout or slump
    # year (his impact moves by a random amount), and so does every other team.
    form = rng.normal(0.0, PLAYER_FORM_SD, n)
    contrib = _spot_contrib(spots, profiles) + form * mins / 48.0
    callup_swap = contrib - CALLUP_IMPACT * mins / 48.0   # rating lost when this player sits
    full_raw = contrib.sum() + CALLUP_IMPACT * (TEAM_MINUTES - mins.sum()) / 48.0
    healthy_rating = soft_cap(full_raw + offset)
    inj_p = INJURY_BASE * (mins / 36.0) ** 3
    names = [profiles.loc[s.key, "player_name"] for s in spots]

    opp_r = np.array([rate[o] for o in sched["opponent"]])
    edges = np.array([site_edge(x) for x in sched["site"]])
    noise = rng.normal(0.0, SIGMA, len(sched))
    inj_draw = rng.random((len(sched), n))
    out_until = np.full(n, -1)
    injuries, nightly, margins, outs = [], [], [], []
    for gi in range(len(sched)):
        missing = out_until >= gi
        r_night = soft_cap(full_raw - callup_swap[missing].sum() + offset)
        nightly.append(r_night)
        margins.append(r_night - opp_r[gi] + edges[gi] + noise[gi])
        outs.append(", ".join(nm for nm, miss in zip(names, missing) if miss))
        hurt = (~missing) & (inj_draw[gi] < inj_p)
        for i in np.where(hurt)[0]:
            length = int(rng.integers(2, 13))
            out_until[i] = gi + length
            injuries.append({"player": names[i], "game": gi + 1, "date": sched["date"].iloc[gi],
                             "games_out": length, "minutes": float(mins[i])})
    margins = np.array(margins)
    won = margins > 0
    wins = int(won.sum())
    log = pd.DataFrame({"game": np.arange(1, len(sched) + 1), "date": sched["date"].to_numpy(),
                        "opponent": sched["opponent"].to_numpy(), "site": sched["site"].to_numpy(),
                        "margin": np.round(margins).astype(int), "won": won,
                        "wins": np.cumsum(won), "rating": nightly, "out": outs,
                        "real_wins": (sched["result"] == "W").cumsum().to_numpy()})
    log["losses"] = log["game"] - log["wins"]

    # Other 29 teams: binomial draw from their real net rating.
    others = teams[teams["team"] != "NYK"]
    ow = rng.binomial(82, [phi(r / SIGMA) for r in others["net_rating"]])
    league = pd.DataFrame({"team": ["NYK"] + others["team"].tolist(),
                           "name": ["New York Knicks (yours)"] + others["name"].tolist(),
                           "conference": ["East"] + others["conference"].tolist(),
                           "net_rating": [healthy_rating] + others["net_rating"].tolist(),
                           "wins": [wins] + list(ow)})
    league["losses"] = 82 - league["wins"]
    league["tiebreak"] = rng.random(len(league))
    east = league[league.conference == "East"].sort_values(["wins", "net_rating", "tiebreak"], ascending=False).reset_index(drop=True)
    west = league[league.conference == "West"].sort_values(["wins", "net_rating", "tiebreak"], ascending=False).reset_index(drop=True)

    # Players still hurt when the regular season ends miss the first round.
    hurt_at_end = out_until >= len(sched)
    r1_rating = soft_cap(full_raw - callup_swap[hurt_at_end].sum() + offset)

    seed = int(east.index[east.team == "NYK"][0]) + 1
    res = _playoffs(east, west, healthy_rating, r1_rating, rng)
    form_notes = [{"player": nm, "form": float(f), "minutes": float(m)} for nm, f, m in zip(names, form, mins)]
    return SeasonResult(wins=wins, losses=82 - wins, seed=seed, outcome=res["outcome"],
                        champion=res["champion"], east=east, west=west, game_log=log,
                        injuries=injuries, matchups=res["matchups"], knicks_series=res["knicks_series"],
                        form=form_notes, rating=healthy_rating)


def _playoffs(east, west, nyk_rating, nyk_r1_rating, rng):
    matchups, knicks_series = [], []

    def rating_of(row, rnd):
        if row["team"] == "NYK":
            return nyk_r1_rating if rnd in ("Play-In", "Round 1") else nyk_rating
        return row["net_rating"]

    def series(a, b, conf, rnd, best_of=7):
        # a has home court (better seed / record). 2-2-1-1-1.
        pattern = [1, 1, -1, -1, 1, -1, 1] if best_of == 7 else [1]
        need = best_of // 2 + 1
        aw = bw = 0
        i = 0
        ra, rb = rating_of(a, rnd), rating_of(b, rnd)
        while aw < need and bw < need:
            edge = HOME_COURT * pattern[i]
            m = _game(ra, rb, edge, rng, PLAYOFF_SIGMA)
            if m > 0:
                aw += 1
            else:
                bw += 1
            i += 1
        winner = a if aw > bw else b
        loser = b if aw > bw else a
        rec = {"conf": conf, "round": rnd, "team_a": a["team"], "team_b": b["team"],
               "seed_a": a.get("seed"), "seed_b": b.get("seed"),
               "score": f"{aw}-{bw}" if best_of > 1 else ("W" if aw else "L"), "winner": winner["team"]}
        matchups.append(rec)
        if "NYK" in (a["team"], b["team"]):
            knicks_series.append(rec)
        return winner, loser

    def conf_bracket(df, conf):
        rows = [dict(r, seed=i + 1) for i, r in enumerate(df.to_dict("records"))]
        w78, l78 = series(rows[6], rows[7], conf, "Play-In", 1)
        w910, _ = series(rows[8], rows[9], conf, "Play-In", 1)
        w8, _ = series(l78, w910, conf, "Play-In", 1)
        w78 = dict(w78, seed=7)
        w8 = dict(w8, seed=8)
        seeds = rows[:6] + [w78, w8]
        r1 = [series(seeds[a], seeds[b], conf, "Round 1")[0] for a, b in [(0, 7), (3, 4), (2, 5), (1, 6)]]
        sf = [series(*_order(r1[0], r1[1]), conf, "Conf. Semifinals")[0],
              series(*_order(r1[2], r1[3]), conf, "Conf. Semifinals")[0]]
        cf = series(*_order(sf[0], sf[1]), conf, "Conf. Finals")[0]
        return cf, seeds, r1, sf

    def _order(a, b):
        return (a, b) if (a["seed"], -a["wins"]) <= (b["seed"], -b["wins"]) else (b, a)

    e_champ, e_seeds, e_r1, e_sf = conf_bracket(east, "East")
    w_champ, *_ = conf_bracket(west, "West")
    fa, fb = (e_champ, w_champ) if e_champ["wins"] >= w_champ["wins"] else (w_champ, e_champ)
    champ, _ = series(fa, fb, "Finals", "NBA Finals")

    nyk_rank = int(east.index[east.team == "NYK"][0])
    if nyk_rank >= 10:
        outcome = "missed_playoffs"
    elif not any(s["team"] == "NYK" for s in e_seeds):
        outcome = "lost_play_in"
    elif champ["team"] == "NYK":
        outcome = "champion"
    elif e_champ["team"] == "NYK":
        outcome = "lost_finals"
    elif any(s["team"] == "NYK" for s in e_sf):
        outcome = "lost_conf_finals"
    elif any(s["team"] == "NYK" for s in e_r1):
        outcome = "lost_conf_semis"
    else:
        outcome = "lost_round_1"
    return {"outcome": outcome, "champion": champ["team"], "matchups": matchups, "knicks_series": knicks_series}


OUTCOME_ORDER = ["missed_playoffs", "lost_play_in", "lost_round_1", "lost_conf_semis",
                 "lost_conf_finals", "lost_finals", "champion"]
OUTCOME_LABELS = {
    "missed_playoffs": ("😬", "Missed the playoffs"),
    "lost_play_in": ("😕", "Lost in the play-in"),
    "lost_round_1": ("🙂", "Lost in Round 1"),
    "lost_conf_semis": ("👍", "Lost in the East semifinals"),
    "lost_conf_finals": ("🔥", "Lost in the East finals"),
    "lost_finals": ("🥈", "Lost in the NBA Finals"),
    "champion": ("🏆", "NBA Champions"),
}


def simulate_many(spots, profiles, teams, sched, n, seed=None):
    rng = np.random.default_rng(seed)
    wins, outcomes, seeds = [], [], []
    for _ in range(n):
        r = simulate_season(spots, profiles, teams, sched, rng)
        wins.append(r.wins)
        outcomes.append(r.outcome)
        seeds.append(r.seed)
    return pd.DataFrame({"wins": wins, "outcome": outcomes, "seed": seeds})
