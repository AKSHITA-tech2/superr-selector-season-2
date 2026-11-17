"""Load the official Data Pack (and nothing else)."""
import pandas as pd
from src.config import DATA_DIR


def _read(name):
    f = DATA_DIR / name
    if not f.exists() and (DATA_DIR / f"{name}.xz").exists():   # compressed copy (GitHub file-size limit)
        f = DATA_DIR / f"{name}.xz"
    return pd.read_csv(f, low_memory=False)


def load_raw():
    return {k: _read(f"{k}.csv") for k in
            ["ball_by_ball", "matches", "competitions", "teams", "playing_eleven",
             "competition_team_player_mapping"]}


def competition_type(name: str, fmt: str) -> str:
    if name.startswith("IPL"):
        return "IPL"
    if fmt == "T20I":
        return "T20I"
    return "OTHER"   # emerging / A-team tournaments: outside the brief's IPL+T20I scope


def load_matches():
    raw = load_raw()
    m = raw["matches"].merge(raw["competitions"], on="comp_id", how="left")
    m["comp_type"] = [competition_type(n, f) for n, f in zip(m.comp_name, m.match_format)]
    m["match_date"] = pd.to_datetime(m["match_date"])
    m["season"] = m["match_date"].dt.year
    res = m["result"].fillna("").str.lower()
    m["no_result"] = res.str.contains("no result|abandon") | m["win_team_id"].isna()
    return m, raw
