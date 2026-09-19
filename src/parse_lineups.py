"""
Parse NBA play-by-play data into on-court lineup stints for a single team,
across two different source formats (stats.nba.com v2 and v3).

A "stint" = a contiguous stretch of game time during which the same 5 players
for our team were on the court together. We reconstruct stints purely from
substitution events (+ period-start events for the initial lineup), track
score during each stint (for a net-rating proxy), and roll everything up to
per-player, per-game minutes.

Output: data/processed/{team}_{season}_stints.csv and
        data/processed/{team}_{season}_player_game_minutes.csv
"""
import pandas as pd
import re
import unicodedata
from pathlib import Path


def norm_name(name) -> str:
    """Strip accents so the same player isn't split into two names when the
    raw feed spells them inconsistently (e.g. 'Pacome Dadiet' vs 'Pacôme Dadiet')."""
    if not isinstance(name, str):
        return name
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

NYK_ID_V2 = 1610612752
NYK_ID_V3 = 1610612752

SECONDS_PER_PERIOD = 12 * 60
SECONDS_PER_OT = 5 * 60


def period_length(period: int) -> int:
    return SECONDS_PER_PERIOD if period <= 4 else SECONDS_PER_OT


def pctime_to_seconds_elapsed(period: int, pctimestring: str) -> float:
    """v2 PCTIMESTRING is 'MM:SS' remaining in the period."""
    try:
        m, s = pctimestring.split(":")
        remaining = int(m) * 60 + int(s)
    except Exception:
        remaining = 0
    return period_length(period) - remaining


def clock_to_seconds_elapsed(period: int, clock: str) -> float:
    """v3 clock is ISO-8601-ish duration like 'PT11M32.00S' remaining."""
    if not isinstance(clock, str):
        return 0.0
    m = re.match(r"PT(\d+)M([\d.]+)S", clock)
    if not m:
        return 0.0
    remaining = int(m.group(1)) * 60 + float(m.group(2))
    return period_length(period) - remaining


def game_absolute_seconds(period: int, seconds_elapsed_in_period: float) -> float:
    """Convert (period, seconds elapsed in that period) to seconds elapsed since game start."""
    prior = sum(period_length(p) for p in range(1, period))
    return prior + seconds_elapsed_in_period


def parse_v2_season(csv_path: Path, team_id: int, team_abbr: str):
    usecols = [
        "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "PERIOD", "PCTIMESTRING",
        "SCORE", "SCOREMARGIN", "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ID",
        "PLAYER1_TEAM_ABBREVIATION", "PLAYER2_ID", "PLAYER2_NAME", "PLAYER2_TEAM_ID",
        "HOMEDESCRIPTION", "VISITORDESCRIPTION",
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    # keep only games that involve our team, but we need ALL events in those
    # games (both teams' actions affect score), so filter by game_id membership.
    team_games = df.loc[df["PLAYER1_TEAM_ID"] == team_id, "GAME_ID"].unique()
    df = df[df["GAME_ID"].isin(team_games)].copy()
    df["sec_elapsed"] = df.apply(
        lambda r: game_absolute_seconds(r["PERIOD"], pctime_to_seconds_elapsed(r["PERIOD"], r["PCTIMESTRING"])),
        axis=1,
    )
    df = df.sort_values(["GAME_ID", "sec_elapsed", "EVENTNUM"])
    return df, team_games


def parse_v3_season(csv_path: Path, team_id: int, team_abbr: str):
    usecols = [
        "gameId", "actionNumber", "actionType", "period", "clock",
        "scoreHome", "scoreAway", "personId", "playerName", "teamId", "teamTricode",
        "description",
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df.rename(columns={
        "gameId": "GAME_ID", "actionNumber": "EVENTNUM", "period": "PERIOD",
        "personId": "PLAYER1_ID", "playerName": "PLAYER1_NAME", "teamId": "PLAYER1_TEAM_ID",
        "teamTricode": "PLAYER1_TEAM_ABBREVIATION",
    })
    team_games = df.loc[df["PLAYER1_TEAM_ID"] == team_id, "GAME_ID"].unique()
    df = df[df["GAME_ID"].isin(team_games)].copy()
    df["sec_elapsed"] = df.apply(
        lambda r: game_absolute_seconds(r["PERIOD"], clock_to_seconds_elapsed(r["PERIOD"], r["clock"])),
        axis=1,
    )
    df["SCORE"] = df["scoreHome"].astype(str) + " - " + df["scoreAway"].astype(str)
    df = df.sort_values(["GAME_ID", "sec_elapsed", "EVENTNUM"])
    return df, team_games


def extract_starters_v2(df_game: pd.DataFrame, team_id: int) -> set:
    """First 5 distinct players from `team_id` to appear in any event before
    that team's first substitution, taken as the starting lineup."""
    first_sub_idx = df_game[(df_game["EVENTMSGTYPE"] == 8) & (df_game["PLAYER1_TEAM_ID"] == team_id)]
    cutoff = first_sub_idx["sec_elapsed"].min() if len(first_sub_idx) else df_game["sec_elapsed"].max()
    early = df_game[(df_game["PLAYER1_TEAM_ID"] == team_id) & (df_game["sec_elapsed"] <= cutoff)]
    seen = []
    for pid, name in zip(early["PLAYER1_ID"], early["PLAYER1_NAME"]):
        name = norm_name(name)
        if pid not in [p for p, _ in seen]:
            seen.append((pid, name))
        if len(seen) >= 5:
            break
    return set(seen)


def build_stints_for_game_v2(df_game: pd.DataFrame, team_id: int, skip_counter: dict):
    lineup = extract_starters_v2(df_game, team_id)
    stints = []
    stint_start = 0.0
    subs = df_game[(df_game["EVENTMSGTYPE"] == 8) & (df_game["PLAYER1_TEAM_ID"] == team_id)]
    for _, row in subs.iterrows():
        out_player = (row["PLAYER1_ID"], row["PLAYER1_NAME"])
        in_player = (row["PLAYER2_ID"], row["PLAYER2_NAME"])
        currently_on = {p[0] for p in lineup}
        # Guard against duplicate/erroneous substitution logging (e.g. replay
        # review corrections): only apply a swap if the outgoing player is
        # actually on court and the incoming one isn't already.
        if out_player[0] not in currently_on or in_player[0] in currently_on:
            skip_counter["count"] += 1
            continue
        stints.append({"lineup": frozenset(lineup), "start": stint_start, "end": row["sec_elapsed"]})
        lineup = {p for p in lineup if p[0] != out_player[0]}
        lineup.add(in_player)
        stint_start = row["sec_elapsed"]
    game_end = df_game["sec_elapsed"].max()
    stints.append({"lineup": frozenset(lineup), "start": stint_start, "end": game_end})
    return stints


def net_score_during(df_game: pd.DataFrame, team_id: int, start: float, end: float, is_v3: bool):
    """Point differential for `team_id` during [start,end) using SCORE snapshots."""
    window = df_game[(df_game["sec_elapsed"] >= start) & (df_game["sec_elapsed"] <= end)]
    if is_v3:
        home = window["scoreHome"].dropna()
        away = window["scoreAway"].dropna()
        if len(home) < 2 or len(away) < 2:
            return 0
        return None  # handled by caller with home/away context for v3
    scores = window["SCORE"].dropna()
    if len(scores) < 2:
        return 0
    def parse(s):
        try:
            a, b = s.split(" - ")
            return int(a), int(b)
        except Exception:
            return None
    first, last = parse(scores.iloc[0]), parse(scores.iloc[-1])
    if not first or not last:
        return 0
    return (last[0] - last[1]) - (first[0] - first[1])


def run(season_label, csv_path, team_id, team_abbr, is_v3):
    print(f"Parsing {season_label} ({'v3' if is_v3 else 'v2'}) for {team_abbr}...")
    if is_v3:
        df, team_games = parse_v3_season(csv_path, team_id, team_abbr)
    else:
        df, team_games = parse_v2_season(csv_path, team_id, team_abbr)

    print(f"  {len(team_games)} games found for {team_abbr}")

    skip_counter = {"count": 0}
    all_stints = []
    for gid in team_games:
        dfg = df[df["GAME_ID"] == gid]
        if is_v3:
            # v2-style substitution detection reused: actionType == 'Substitution'
            dfg2 = dfg.rename(columns={"actionType": "EVENTMSGTYPE_STR"})
            subs_mask = dfg2["EVENTMSGTYPE_STR"] == "Substitution"
            # v3 description format: "SUB: <in> FOR <out>"
            starters = []
            first_sub_time = dfg[subs_mask]["sec_elapsed"].min() if subs_mask.any() else dfg["sec_elapsed"].max()
            early = dfg[(dfg["sec_elapsed"] <= first_sub_time) & (dfg["PLAYER1_TEAM_ID"] == team_id)]
            seen = []
            for pid, name in zip(early["PLAYER1_ID"], early["PLAYER1_NAME"]):
                if pd.isna(pid):
                    continue
                if pid not in [p for p, _ in seen]:
                    seen.append((pid, name))
            lineup = set(seen[:5])
            stint_start = 0.0
            stints = []
            for _, row in dfg[subs_mask & (dfg["PLAYER1_TEAM_ID"] == team_id)].iterrows():
                desc = str(row["description"])
                min_ = re.match(r"SUB:\s*(.+?)\s+FOR\s+(.+)", desc)
                if not min_:
                    continue
                in_name, out_name = min_.group(1).strip(), min_.group(2).strip()
                names_on = {p[1] for p in lineup}
                # Same guard as v2: skip duplicate/erroneous substitution
                # events rather than let them inflate the on-court lineup.
                if out_name not in names_on or in_name in names_on:
                    skip_counter["count"] += 1
                    continue
                stints.append({"lineup": frozenset(lineup), "start": stint_start, "end": row["sec_elapsed"]})
                lineup = {p for p in lineup if p[1] != out_name}
                lineup.add((row["PLAYER1_ID"], in_name))
                stint_start = row["sec_elapsed"]
            game_end = dfg["sec_elapsed"].max()
            stints.append({"lineup": frozenset(lineup), "start": stint_start, "end": game_end})
        else:
            stints = build_stints_for_game_v2(dfg, team_id, skip_counter)

        for s in stints:
            if s["end"] <= s["start"]:
                continue
            all_stints.append({
                "game_id": gid, "team": team_abbr,
                "start_sec": s["start"], "end_sec": s["end"],
                "duration_sec": s["end"] - s["start"],
                "lineup_ids": "|".join(str(p[0]) for p in s["lineup"]),
                "lineup_names": "|".join(str(p[1]) for p in s["lineup"]),
            })

    stints_df = pd.DataFrame(all_stints)
    out_path = OUT / f"{team_abbr}_{season_label}_stints.csv"
    stints_df.to_csv(out_path, index=False)
    print(f"  wrote {len(stints_df)} stints -> {out_path}  ({skip_counter['count']} anomalous substitution events skipped)")

    # per-player per-game minutes = sum of stint durations they appear in
    rows = []
    for gid, grp in stints_df.groupby("game_id"):
        mins = {}
        for _, s in grp.iterrows():
            for pid, name in zip(s["lineup_ids"].split("|"), s["lineup_names"].split("|")):
                key = (pid, name)
                mins[key] = mins.get(key, 0) + s["duration_sec"]
        for (pid, name), sec in mins.items():
            rows.append({"game_id": gid, "player_id": pid, "player_name": name, "seconds": sec, "minutes": sec / 60})
    pg = pd.DataFrame(rows)
    out_path2 = OUT / f"{team_abbr}_{season_label}_player_game_minutes.csv"
    pg.to_csv(out_path2, index=False)
    print(f"  wrote {len(pg)} player-game rows -> {out_path2}")
    return stints_df, pg


if __name__ == "__main__":
    run("2024-25", RAW / "nbastats_2024.csv", NYK_ID_V2, "NYK", is_v3=False)
    run("2025-26", RAW / "nbastatsv3_2025.csv", NYK_ID_V3, "NYK", is_v3=True)
