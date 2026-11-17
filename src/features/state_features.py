"""Pre-delivery feature matrices. ONLY information known before the ball is bowled."""
import numpy as np
import pandas as pd

# NOTE: current run rate is deliberately NOT a model feature. It is a deterministic function of score and
# balls bowled (both included) and is extremely volatile in the first overs (one boundary off the first
# ball = CRR 24), which made projected totals - and therefore leverage - jump artificially in the powerplay.
STATE_COLS = ["innings", "balls_remaining", "legal_before", "wkts_in_hand", "score_before",
              "runs_required_f", "rrr_f", "is_ipl", "impact_era"]


def state_frame(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["innings"] = d["innings"].astype(float)
    X["balls_remaining"] = d["balls_remaining"].astype(float)
    X["legal_before"] = d["legal_before"].astype(float)
    X["wkts_in_hand"] = d["wkts_in_hand"].astype(float)
    X["score_before"] = d["score_before"].astype(float)
    X["runs_required_f"] = np.where(d["innings"] == 2, d["runs_required"], -1.0)
    X["rrr_f"] = np.where(d["innings"] == 2, np.clip(d["rrr"], 0, 36), -1.0)
    X["is_ipl"] = d["is_ipl"].astype(float)
    X["impact_era"] = d["impact_era"].astype(float)
    return X[STATE_COLS]


def next_state_frame(d, runs_add, wkt_add, legal_add):
    """State after a hypothetical outcome (used for leverage and wicket value)."""
    n = d.copy()
    n["score_before"] = d["score_before"] + runs_add
    n["wkts_before"] = d["wkts_before"] + wkt_add
    n["wkts_in_hand"] = 10 - n["wkts_before"]
    n["legal_before"] = d["legal_before"] + legal_add
    n["balls_remaining"] = (d["max_balls"] - n["legal_before"]).clip(lower=0)
    n["runs_required"] = d["target"] - n["score_before"]
    br = n["balls_remaining"].clip(lower=1)
    n["rrr"] = np.where(n.innings == 2, n["runs_required"] * 6 / br, np.nan)
    n["crr"] = np.where(n["legal_before"] > 0, n["score_before"] * 6 / n["legal_before"].clip(lower=1), 0.0)
    return n
