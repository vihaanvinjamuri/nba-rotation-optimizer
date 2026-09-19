"""
Same idea as build_boxscore_from_pbp.py (real box score reconstructed from
play-by-play), but for the older stats.nba.com v2 event schema used for the
2024-25 season (EVENTMSGTYPE codes instead of named actionType strings).

v2 is actually a bit more reliable to parse than v3 for assists/steals/
blocks: the assisting/stealing/blocking player is given directly in a
PLAYER2_NAME/PLAYER3_NAME column with team info, not just a name buried in
free-text description, so no regex-matching against a roster is needed.
"""
import re
import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent.parent / "data" / "processed"
TEAM_ABBR = "NYK"


def norm_name(name):
    import unicodedata
    if not isinstance(name, str):
        return name
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))


def desc_of(row):
    for col in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"):
        v = row.get(col)
        if isinstance(v, str):
            return v
    return ""


def build_game_box(g):
    stats = {}

    def row(name):
        return stats.setdefault(name, {
            "pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "ftm": 0, "fta": 0,
            "orb": 0, "drb": 0, "ast": 0, "stl": 0, "blk": 0, "tov": 0, "pf": 0,
        })

    reb_running = {}

    for _, r in g.iterrows():
        et = r["EVENTMSGTYPE"]
        desc = desc_of(r)
        p1_team = r.get("PLAYER1_TEAM_ABBREVIATION")
        p1_name = norm_name(r["PLAYER1_NAME"]) if isinstance(r["PLAYER1_NAME"], str) else None

        if et == 1:  # made shot
            if p1_team == TEAM_ABBR and p1_name:
                d = row(p1_name)
                d["fga"] += 1
                d["fgm"] += 1
                is3 = "3PT" in desc
                d["pts"] += 3 if is3 else 2
                if is3:
                    d["fg3a"] += 1
                    d["fg3m"] += 1
            p2_team = r.get("PLAYER2_TEAM_ABBREVIATION")
            p2_name = norm_name(r["PLAYER2_NAME"]) if isinstance(r.get("PLAYER2_NAME"), str) else None
            if p2_team == TEAM_ABBR and p2_name:
                row(p2_name)["ast"] += 1
        elif et == 2:  # missed shot
            if p1_team == TEAM_ABBR and p1_name:
                d = row(p1_name)
                d["fga"] += 1
                if "3PT" in desc:
                    d["fg3a"] += 1
            p3_team = r.get("PLAYER3_TEAM_ABBREVIATION")
            p3_name = norm_name(r["PLAYER3_NAME"]) if isinstance(r.get("PLAYER3_NAME"), str) else None
            if p3_team == TEAM_ABBR and p3_name:
                row(p3_name)["blk"] += 1
        elif et == 3:  # free throw
            if p1_team == TEAM_ABBR and p1_name:
                d = row(p1_name)
                d["fta"] += 1
                if not desc.strip().startswith("MISS"):
                    d["ftm"] += 1
                    d["pts"] += 1
        elif et == 4:  # rebound
            if p1_team == TEAM_ABBR and p1_name:
                m = re.search(r"Off:(\d+)\s*Def:(\d+)", desc)
                if m:
                    off, deff = int(m.group(1)), int(m.group(2))
                    prev_off, prev_def = reb_running.get(p1_name, (0, 0))
                    d = row(p1_name)
                    if off > prev_off:
                        d["orb"] += 1
                    elif deff > prev_def:
                        d["drb"] += 1
                    reb_running[p1_name] = (off, deff)
        elif et == 5:  # turnover
            if p1_team == TEAM_ABBR and p1_name:
                row(p1_name)["tov"] += 1
            p2_team = r.get("PLAYER2_TEAM_ABBREVIATION")
            p2_name = norm_name(r["PLAYER2_NAME"]) if isinstance(r.get("PLAYER2_NAME"), str) else None
            if p2_team == TEAM_ABBR and p2_name:
                row(p2_name)["stl"] += 1
        elif et == 6:  # foul
            if p1_team == TEAM_ABBR and p1_name and "T.FOUL" not in desc and "Technical" not in desc:
                row(p1_name)["pf"] += 1

    return stats


def game_score(d):
    return (
        d["pts"] + 0.4 * d["fgm"] - 0.7 * d["fga"] - 0.4 * (d["fta"] - d["ftm"])
        + 0.7 * d["orb"] + 0.3 * d["drb"] + d["stl"] + 0.7 * d["ast"] + 0.7 * d["blk"]
        - 0.4 * d["pf"] - d["tov"]
    )


def run(season_label, csv_path, schedule_path):
    print(f"Building real box scores for {season_label} from play-by-play (v2 schema)...")
    sched = pd.read_csv(schedule_path)
    sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE"], format="%Y%m%d")
    date_map = {int(r.GAME_ID): r.GAME_DATE for r in sched.itertuples()}
    opp_map = {int(r.GAME_ID): (r.VTM if r.HTM == TEAM_ABBR else r.HTM) for r in sched.itertuples()}

    usecols = [
        "GAME_ID", "EVENTMSGTYPE", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION",
        "PLAYER1_NAME", "PLAYER1_TEAM_ABBREVIATION", "PLAYER2_NAME", "PLAYER2_TEAM_ABBREVIATION",
        "PLAYER3_NAME", "PLAYER3_TEAM_ABBREVIATION",
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    nyk_games = df.loc[df["PLAYER1_TEAM_ABBREVIATION"] == TEAM_ABBR, "GAME_ID"].unique()
    df = df[df["GAME_ID"].isin(nyk_games)]

    rows = []
    for gid, g in df.groupby("GAME_ID"):
        gid_int = int(gid)
        if gid_int not in date_map:
            continue
        box = build_game_box(g)
        for name, d in box.items():
            rows.append({
                "game_id": gid_int, "game_date": date_map[gid_int], "opponent": opp_map[gid_int],
                "player_name": name, "game_score": round(game_score(d), 1), **d,
            })

    out = pd.DataFrame(rows).sort_values(["game_date", "player_name"])
    out_path = OUT / f"NYK_{season_label}_player_game_boxscore.csv"
    out.to_csv(out_path, index=False)
    print(f"  wrote {len(out)} rows -> {out_path}")
    return out


if __name__ == "__main__":
    run("2024-25", RAW / "nbastats_2024.csv", RAW / "game_schedule_2024.csv")
