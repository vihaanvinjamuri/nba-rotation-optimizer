"""
Build the two inputs the optimizer needs, per player per game:
  1. Net rating proxy: point differential accumulated while that player was
     on court (standard plus-minus), reconstructed stint-by-stint.
  2. Fatigue inputs: minutes played, rest days before the game, back-to-back
     flag, and a rolling cumulative-minutes load (last 7 games).

Reuses the validated lineup-tracking + duplicate-substitution guard from
parse_lineups.py, and adds score tracking so each stint carries a net
point differential we can attribute to every player in that stint.
"""
import pandas as pd
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from parse_lineups import (
    RAW, OUT, parse_v2_season, parse_v3_season, extract_starters_v2,
    game_absolute_seconds, pctime_to_seconds_elapsed, clock_to_seconds_elapsed,
    norm_name,
)

TEAM_ID = 1610612752
TEAM_ABBR = "NYK"


def score_at_row_v2(row):
    try:
        a, b = row["SCORE"].split(" - ")
        return int(a), int(b)  # (home, away)
    except Exception:
        return None


def _period_starters_v3(period_df, team_id):
    """First 5 distinct team players to record ANY event in this period
    before that team's own first substitution of the period."""
    team_subs_mask = (period_df["actionType"] == "Substitution") & (period_df["PLAYER1_TEAM_ID"] == team_id)
    cutoff = period_df.loc[team_subs_mask, "sec_elapsed"].min() if team_subs_mask.any() else period_df["sec_elapsed"].max()
    early = period_df[(period_df["sec_elapsed"] <= cutoff) & (period_df["PLAYER1_TEAM_ID"] == team_id)]
    seen = []
    for pid, name in zip(early["PLAYER1_ID"], early["PLAYER1_NAME"].map(norm_name)):
        if pd.isna(pid):
            continue
        if pid not in [p for p, _ in seen]:
            seen.append((pid, name))
        if len(seen) >= 5:
            break
    return seen[:5]


def _period_starters_v2(period_df, team_id):
    team_subs_mask = (period_df["EVENTMSGTYPE"] == 8) & (period_df["PLAYER1_TEAM_ID"] == team_id)
    cutoff = period_df.loc[team_subs_mask, "sec_elapsed"].min() if team_subs_mask.any() else period_df["sec_elapsed"].max()
    early = period_df[(period_df["PLAYER1_TEAM_ID"] == team_id) & (period_df["sec_elapsed"] <= cutoff)]
    seen = []
    for pid, name in zip(early["PLAYER1_ID"], early["PLAYER1_NAME"]):
        name = norm_name(name)
        if pid not in [p for p, _ in seen]:
            seen.append((pid, name))
        if len(seen) >= 5:
            break
    return seen[:5]


def _reconcile_starters(starters, subs, is_v3):
    """Expand the detected starting lineup to include any player who gets
    substituted OUT this period without ever having been recorded as
    subbed IN. You can only be subbed out if you were already on the
    court, so such a player must have started the period -- our early-
    event detection just never saw them do anything (no shot/rebound/foul)
    before the first sub. Without this, that substitution gets silently
    SKIPPED (the "out" name isn't in our tracked lineup), which drops
    both that player's minutes AND whoever subs in for them for the rest
    of the game. This is what caused several bench players (Diawara,
    Clarkson, Hukporti, Dadiet...) to go completely missing from specific
    games despite clearly appearing in the real box score.
    """
    starters = set(starters)
    current = set(starters)
    if is_v3:
        for _, row in subs.iterrows():
            desc = str(row["description"])
            m = re.match(r"SUB:\s*(.+?)\s+FOR\s+(.+)", desc)
            if not m:
                continue
            in_name, out_name = norm_name(m.group(1).strip()), norm_name(m.group(2).strip())
            cur_names = {p[1] for p in current}
            if out_name not in cur_names:
                phantom = (-1, out_name)
                starters.add(phantom)
                current.add(phantom)
            current = {p for p in current if p[1] != out_name}
            current.add((-1, in_name))
    else:
        for _, row in subs.iterrows():
            out_id, out_name = row["PLAYER1_ID"], norm_name(row["PLAYER1_NAME"])
            in_id, in_name = row["PLAYER2_ID"], norm_name(row["PLAYER2_NAME"])
            cur_ids = {p[0] for p in current}
            if out_id not in cur_ids:
                phantom = (out_id, out_name)
                starters.add(phantom)
                current.add(phantom)
            current = {p for p in current if p[0] != out_id}
            current.add((in_id, in_name))
    return starters


def build_game_stints_with_score(dfg, team_id, is_v3, is_home, skip_counter):
    """Returns list of dicts: lineup (set of (pid,name)), start, end, team_net_pts.

    KEY FIX: the on-court lineup is re-detected independently at the START OF
    EVERY PERIOD, not just carried forward from whatever the substitution log
    left it at. Real NBA play-by-play feeds only log "Substitution" actions
    for in-game subs; a lineup change made between quarters (very common:
    teams frequently start a quarter with a different 5 than closed the
    previous one) is NOT logged as a Substitution event, since it happens
    during the dead-clock intermission. Blindly carrying the lineup across
    the period boundary silently mis-tracked who was on court for the rest
    of the game after any such change, which is what caused real, systematic
    per-player minutes errors (Karl-Anthony Towns undercounted by ~10
    min/game, Mitchell Robinson overcounted, etc.) even though total team
    minutes for the game summed to the correct 240.
    """
    def score_now(sec_elapsed):
        """Last known (home,away) score at or before sec_elapsed."""
        window = dfg[dfg["sec_elapsed"] <= sec_elapsed]
        if is_v3:
            h = window["scoreHome"].dropna()
            a = window["scoreAway"].dropna()
            if len(h) == 0:
                return (0, 0)
            return (int(h.iloc[-1]), int(a.iloc[-1]))
        else:
            s = window["SCORE"].dropna()
            if len(s) == 0:
                return (0, 0)
            try:
                a, b = s.iloc[-1].split(" - ")
                return (int(a), int(b))
            except Exception:
                return (0, 0)

    stints = []
    lineup = set()
    periods = sorted(dfg["PERIOD"].unique())

    for period in periods:
        period_df = dfg[dfg["PERIOD"] == period]
        period_start = game_absolute_seconds(period, 0)
        period_end = period_df["sec_elapsed"].max()

        # IMPORTANT: always start each period from THIS period's own detected
        # starters, never from whatever lineup the previous period ended
        # with. Carrying the old lineup forward as a fallback when detection
        # found fewer than 5 seems safe, but it isn't: if the real lineup
        # actually changed between periods (the exact case this whole
        # per-period re-detection exists to catch), the carried-over players
        # who should have left never get removed, while _reconcile_starters
        # below still (correctly) adds whoever the new period's own subs
        # reveal -- so the two sets just pile up instead of one replacing
        # the other, producing "lineups" of 6, 7, even 9 players and wildly
        # inflated total minutes. Starting fresh from this period's own
        # detection (even partial) and letting reconciliation fill in the
        # rest from THIS period's own substitutions is self-contained and
        # doesn't have that failure mode.
        detected = _period_starters_v3(period_df, team_id) if is_v3 else _period_starters_v2(period_df, team_id)
        lineup = set(detected)

        if is_v3:
            subs = period_df[(period_df["actionType"] == "Substitution") & (period_df["PLAYER1_TEAM_ID"] == team_id)]
        else:
            subs = period_df[(period_df["EVENTMSGTYPE"] == 8) & (period_df["PLAYER1_TEAM_ID"] == team_id)]
        lineup = _reconcile_starters(lineup, subs, is_v3)

        stint_start = period_start

        if is_v3:
            for _, row in subs.iterrows():
                desc = str(row["description"])
                m = re.match(r"SUB:\s*(.+?)\s+FOR\s+(.+)", desc)
                if not m:
                    continue
                in_name, out_name = norm_name(m.group(1).strip()), norm_name(m.group(2).strip())
                names_on = {p[1] for p in lineup}
                if out_name not in names_on or in_name in names_on:
                    skip_counter["count"] += 1
                    continue
                stints.append({"lineup": frozenset(lineup), "start": stint_start, "end": row["sec_elapsed"]})
                lineup = {p for p in lineup if p[1] != out_name}
                lineup.add((row["PLAYER1_ID"], in_name))
                stint_start = row["sec_elapsed"]
        else:
            for _, row in subs.iterrows():
                out_player = (row["PLAYER1_ID"], norm_name(row["PLAYER1_NAME"]))
                in_player = (row["PLAYER2_ID"], norm_name(row["PLAYER2_NAME"]))
                currently_on = {p[0] for p in lineup}
                if out_player[0] not in currently_on or in_player[0] in currently_on:
                    skip_counter["count"] += 1
                    continue
                stints.append({"lineup": frozenset(lineup), "start": stint_start, "end": row["sec_elapsed"]})
                lineup = {p for p in lineup if p[0] != out_player[0]}
                lineup.add(in_player)
                stint_start = row["sec_elapsed"]

        stints.append({"lineup": frozenset(lineup), "start": stint_start, "end": period_end})

    # attach net point differential (for our team) during each stint
    out = []
    for s in stints:
        if s["end"] <= s["start"]:
            continue
        h0, a0 = score_now(s["start"])
        h1, a1 = score_now(s["end"])
        d_home, d_away = h1 - h0, a1 - a0
        team_net = (d_home - d_away) if is_home else (d_away - d_home)
        out.append({**s, "team_net_pts": team_net})
    return out


def run_season(season_label, csv_path, is_v3, schedule_path):
    print(f"Building ratings for {season_label}...")
    sched = pd.read_csv(schedule_path)
    sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE"], format="%Y%m%d")
    home_map = {int(r.GAME_ID): (r.HTM == TEAM_ABBR) for r in sched.itertuples()}
    date_map = {int(r.GAME_ID): r.GAME_DATE for r in sched.itertuples()}
    opp_map = {int(r.GAME_ID): (r.VTM if r.HTM == TEAM_ABBR else r.HTM) for r in sched.itertuples()}

    if is_v3:
        df, team_games = parse_v3_season(csv_path, TEAM_ID, TEAM_ABBR)
    else:
        df, team_games = parse_v2_season(csv_path, TEAM_ID, TEAM_ABBR)

    skip_counter = {"count": 0}
    player_rows = []
    for gid in team_games:
        gid = int(gid)
        if gid not in home_map:
            continue
        dfg = df[df["GAME_ID"] == gid]
        is_home = home_map[gid]
        stints = build_game_stints_with_score(dfg, TEAM_ID, is_v3, is_home, skip_counter)

        game_total_sec = dfg["sec_elapsed"].max()
        # Key on player NAME, not the (pid, name) pair: the newer season's
        # substitution log only gives a surname for the incoming player and
        # no reliable id, so the same real player was getting split across
        # several ids within one game. Names are unique within one roster.
        per_player = {}
        for s in stints:
            for pid, name in s["lineup"]:
                d = per_player.setdefault(name, {"seconds": 0.0, "net_pts": 0.0, "pid": pid})
                d["seconds"] += s["end"] - s["start"]
                d["net_pts"] += s["team_net_pts"]

        for name, d in per_player.items():
            player_rows.append({
                "game_id": gid, "game_date": date_map[gid], "opponent": opp_map[gid],
                "player_id": d["pid"], "player_name": name,
                "minutes": d["seconds"] / 60, "plus_minus": d["net_pts"],
                "net_rating_per48": (d["net_pts"] / d["seconds"] * 48 * 60) if d["seconds"] > 0 else 0,
            })

    pg = pd.DataFrame(player_rows).sort_values(["game_date", "player_name"])

    # rest days / back-to-back at the team level (dates of all NYK games this season)
    game_dates = sched.assign(is_nyk=lambda d: (d.HTM == TEAM_ABBR) | (d.VTM == TEAM_ABBR))
    game_dates = game_dates[game_dates.is_nyk].sort_values("GAME_DATE")
    game_dates["rest_days"] = game_dates["GAME_DATE"].diff().dt.days.sub(1).clip(lower=0)
    game_dates["back_to_back"] = game_dates["rest_days"] == 0
    rest_map = game_dates.set_index("GAME_ID")[["rest_days", "back_to_back"]]

    pg = pg.merge(rest_map, left_on="game_id", right_index=True, how="left")

    # rolling 7-game cumulative minutes load, per player, in date order
    pg = pg.sort_values(["player_name", "game_date"])
    pg["rolling_7g_minutes"] = pg.groupby("player_name")["minutes"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )

    out_path = OUT / f"{TEAM_ABBR}_{season_label}_player_game_ratings.csv"
    pg.to_csv(out_path, index=False)
    print(f"  wrote {len(pg)} rows -> {out_path}  ({skip_counter['count']} anomalous subs skipped)")
    return pg


if __name__ == "__main__":
    run_season("2024-25", RAW / "nbastats_2024.csv", False, RAW / "game_schedule_2024.csv")
    run_season("2025-26", RAW / "nbastatsv3_2025.csv", True, RAW / "game_schedule_2025.csv")
