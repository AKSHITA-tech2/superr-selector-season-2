"""Reconstruct the match state BEFORE every delivery and attribute outcomes correctly.

Conventions (verified against the Data Pack in src/data/validation.py):
  runs          = runs off the bat (credited to the striker)
  extras1_runs  = extras of the type in `extras_type` (wide / no-ball penalty / byes / leg-byes)
  extras_2_runs = additional byes/leg-byes taken off a no-ball
  extras_type   'Overstep' and 'other' are both no-balls; 'wide' is a wide.
"""
import numpy as np
import pandas as pd
from src import config as C
from src.data.loader import load_matches


def build_deliveries(include_other=False):
    m, raw = load_matches()
    b = raw["ball_by_ball"]
    b = b[b.innings <= 2].copy()  # super overs (innings 3/4) are not part of the 20-over contest state
    keep = m if include_other else m[m.comp_type.isin(["IPL", "T20I"])]
    b = b.merge(keep[["match_id", "comp_id", "comp_name", "comp_type", "season", "match_date", "stage",
                      "knock_out", "win_team_id", "no_result", "first_inn_revised_overs",
                      "second_inn_revised_overs", "revised_target", "venue_country"]],
                on="match_id", how="inner")
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    et = b["extras_type"].fillna("None")
    b["extras_type"] = et
    b["is_legal"] = ~et.isin(C.ILLEGAL_EXTRAS)
    b["is_wide"] = et.eq("wide")
    b["is_noball"] = et.isin(C.NOBALL_EXTRAS)
    b["total_runs"] = b["runs"] + b["extras1_runs"] + b["extras_2_runs"]
    b["batter_runs"] = b["runs"]
    b["bowler_conceded"] = b["runs"] + np.where(et.isin(C.BOWLER_CHARGED_EXTRAS), b["extras1_runs"], 0)
    b["faced"] = ~b["is_wide"]                      # a batter faces every delivery except a wide

    dt = b["dismissal_type"].fillna("")
    b["team_wicket"] = b["is_wicket"].astype(bool) & dt.ne("")
    b["batter_out"] = (b["team_wicket"] & (b["dismissed_player_id"] == b["striker_id"])
                       & ~dt.isin(C.NON_DISMISSALS)).astype(int)
    b["bowler_wicket"] = (b["team_wicket"] & dt.isin(C.BOWLER_WICKETS)).astype(int)  # run-outs never credited

    g = b.groupby(["match_id", "innings"], sort=False)
    b["legal_before"] = g["is_legal"].cumsum() - b["is_legal"].astype(int)
    b["score_before"] = g["total_runs"].cumsum() - b["total_runs"]
    b["wkts_before"] = g["team_wicket"].cumsum() - b["team_wicket"].astype(int)
    b["innings_total"] = g["total_runs"].transform("sum")
    b["future_runs"] = b["innings_total"] - b["score_before"]

    # innings length (revised overs are stored in balls)
    inn1_max = b["first_inn_revised_overs"].fillna(120)
    inn2_max = b["second_inn_revised_overs"].fillna(inn1_max)
    b["max_balls"] = np.where(b.innings == 1, inn1_max, inn2_max).astype(float)
    b["max_balls"] = np.maximum(b["max_balls"], b.groupby(["match_id", "innings"])["is_legal"].transform("sum"))

    first_tot = b[b.innings == 1].groupby("match_id")["total_runs"].sum()
    b["target"] = np.where(b.innings == 2,
                           b["revised_target"].fillna(b["match_id"].map(first_tot) + 1), np.nan)

    b["balls_remaining"] = (b["max_balls"] - b["legal_before"]).clip(lower=1)
    b["wkts_in_hand"] = 10 - b["wkts_before"]
    b["runs_required"] = b["target"] - b["score_before"]
    b["rrr"] = np.where(b.innings == 2, b["runs_required"] * 6 / b["balls_remaining"], np.nan)
    b["crr"] = np.where(b["legal_before"] > 0, b["score_before"] * 6 / b["legal_before"].clip(lower=1), 0.0)
    b["phase"] = np.select([b.legal_before < C.POWERPLAY_END, b.legal_before >= C.DEATH_START],
                           ["powerplay", "death"], "middle")
    b["is_death"] = (b["phase"] == "death").astype(int)
    b["is_ipl"] = (b["comp_type"] == "IPL").astype(int)
    b["impact_era"] = ((b["comp_type"] == "IPL") & (b["season"] >= 2023)).astype(int)  # IPL Impact Player rule
    b["bat_won"] = np.where(b["no_result"], np.nan, (b["win_team_id"] == b["batting_team_id"]).astype(float))
    b["over"] = b["overid"]
    b["uid"] = np.arange(len(b))
    return b
