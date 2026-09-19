"""
Reconstruct a REAL per-player, per-game box score (points, shooting splits,
rebounds, assists, steals, blocks, turnovers, fouls) directly from the raw
play-by-play event log, and use it to compute Hollinger Game Score, a
standard, fully self-contained advanced box-score rating (no league-wide
context needed, unlike PER/Win Shares/VORP, which require full-league
possession and pace data we don't have loaded for every team).

Game Score = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB
             + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV

This is the same formula basketball-reference lists as "GmSc" in a player's
game log, so it's independently checkable there.
"""
import re
import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent.parent / "data" / "processed"
TEAM_ABBR = "NYK"

AST_RE = re.compile(r"\(([A-Za-z.\-' ]+?) (\d+) AST\)")


def norm_name(name):
    import unicodedata
    if not isinstance(name, str):
        return name
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))


def build_game_box(g):
    """g: all pbp rows for one game (all teams). Returns dict[player_name] -> stat dict for NYK players only."""
    stats = {}

    def row(name):
        return stats.setdefault(name, {
            "pts": 0, "fgm": 0, "fga": 0, "fg3m": 0, "fg3a": 0, "ftm": 0, "fta": 0,
            "orb": 0, "drb": 0, "ast": 0, "stl": 0, "blk": 0, "tov": 0, "pf": 0,
        })

    nyk = g[g["teamTricode"] == TEAM_ABBR]

    # running cumulative (off, def) rebound counters per player, to turn the
    # description's cumulative "(Off:X Def:Y)" annotation into a per-event
    # offensive/defensive classification.
    reb_running = {}

    for _, r in nyk.iterrows():
        name = norm_name(r["playerName"]) if isinstance(r["playerName"], str) else None
        desc = r["description"] if isinstance(r["description"], str) else ""
        at = r["actionType"] if isinstance(r["actionType"], str) else ""

        if at == "Made Shot":
            if name:
                d = row(name)
                d["fga"] += 1
                d["fgm"] += 1
                sv = r["shotValue"] if pd.notna(r["shotValue"]) else 2
                d["pts"] += int(sv)
                if int(sv) == 3:
                    d["fg3a"] += 1
                    d["fg3m"] += 1
            m = AST_RE.search(desc)
            if m:
                assist_name = norm_name(m.group(1).strip())
                row(assist_name)["ast"] += 1
        elif at == "Missed Shot":
            if name:
                d = row(name)
                d["fga"] += 1
                if pd.notna(r["shotValue"]) and int(r["shotValue"]) == 3:
                    d["fg3a"] += 1
        elif at == "Free Throw":
            if name:
                d = row(name)
                d["fta"] += 1
                if not desc.strip().startswith("MISS"):
                    d["ftm"] += 1
                    d["pts"] += 1
        elif at == "Rebound":
            if name:
                m = re.search(r"Off:(\d+)\s*Def:(\d+)", desc)
                if m:
                    off, deff = int(m.group(1)), int(m.group(2))
                    prev_off, prev_def = reb_running.get(name, (0, 0))
                    d = row(name)
                    if off > prev_off:
                        d["orb"] += 1
                    elif deff > prev_def:
                        d["drb"] += 1
                    reb_running[name] = (off, deff)
        elif at == "Turnover":
            if name:
                row(name)["tov"] += 1
        elif at == "Foul":
            sub = r["subType"] if isinstance(r["subType"], str) else ""
            if name and sub != "Technical":
                row(name)["pf"] += 1
        else:
            # steals/blocks are logged with actionType blank, only visible via description text
            if "STEAL" in desc and name:
                row(name)["stl"] += 1
            elif "BLOCK" in desc and name:
                row(name)["blk"] += 1

    return stats


def game_score(d):
    return (
        d["pts"] + 0.4 * d["fgm"] - 0.7 * d["fga"] - 0.4 * (d["fta"] - d["ftm"])
        + 0.7 * d["orb"] + 0.3 * d["drb"] + d["stl"] + 0.7 * d["ast"] + 0.7 * d["blk"]
        - 0.4 * d["pf"] - d["tov"]
    )


def run(season_label, csv_path, schedule_path):
    print(f"Building real box scores for {season_label} from play-by-play...")
    sched = pd.read_csv(schedule_path)
    sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE"], format="%Y%m%d")
    date_map = {int(r.GAME_ID): r.GAME_DATE for r in sched.itertuples()}
    opp_map = {int(r.GAME_ID): (r.VTM if r.HTM == TEAM_ABBR else r.HTM) for r in sched.itertuples()}

    usecols = [
        "gameId", "period", "clock", "teamTricode", "personId", "playerName",
        "description", "actionType", "subType", "shotValue",
    ]
    df = pd.read_csv(csv_path, usecols=usecols, dtype={"gameId": str})
    nyk_games = df.loc[df["teamTricode"] == TEAM_ABBR, "gameId"].unique()
    df = df[df["gameId"].isin(nyk_games)]

    rows = []
    for gid, g in df.groupby("gameId"):
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
    run("2025-26", RAW / "nbastatsv3_2025.csv", RAW / "game_schedule_2025.csv")
