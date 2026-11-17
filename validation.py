"""Data-pack audit: data dictionary + integrity checks (written to outputs/)."""
import pandas as pd
from src import config as C
from src.data.loader import load_raw

DESCRIPTIONS = {
    "match_id": "Match key (joins all tables)", "innings": "1/2 = regulation, 3/4 = super over (excluded)",
    "overid": "Over number 1-20", "over_ball_num": "Delivery index within over (includes extras)",
    "innings_ball_num": "Running delivery index within innings", "striker_id": "Batter on strike",
    "non_striker_id": "Batter at non-striker end", "bowler_id": "Bowler", "runs": "Runs off the bat",
    "is_wicket": "A dismissal occurred on the delivery", "dismissal_type": "Caught/Bowled/LBW/RunOut/Stump/...",
    "dismissed_player_id": "Player dismissed (may be the non-striker on run-outs)",
    "extras_type": "wide / Overstep (no-ball) / other (no-ball) / byes / legbyes / Penalty",
    "extras1_runs": "Runs of the extras_type", "extras_2_runs": "Byes/leg-byes run off a no-ball",
    "comp_id": "Competition key", "season": "Competition season label", "win_team_id": "Winner (NaN = no result)",
    "first_inn_revised_overs": "Revised innings length IN BALLS", "revised_target": "DLS target",
    "knock_out": "1 = knockout fixture", "is_impact": "IPL impact substitute flag",
}


def data_dictionary():
    rows = []
    for name, df in load_raw().items():
        for col in df.columns:
            s = df[col]
            rows.append(dict(table=name, column=col, dtype=str(s.dtype), n_rows=len(df),
                             missing=int(s.isna().sum()), n_unique=int(s.nunique()),
                             example=str(s.dropna().iloc[0]) if s.notna().any() else "",
                             description=DESCRIPTIONS.get(col, "")))
    return pd.DataFrame(rows)


def integrity_checks(d):
    """d = output of build_deliveries(). Returns a list of (check, value, passed)."""
    out = []
    legal_per_over = d.groupby(["match_id", "innings", "overid"])["is_legal"].sum()
    out.append(("share of overs with exactly 6 legal balls", float((legal_per_over == 6).mean()),
                (legal_per_over == 6).mean() > 0.98))
    ro = d[d.dismissal_type.fillna("") == "RunOut"]
    out.append(("run-outs credited to bowler", int(ro.bowler_wicket.sum()), ro.bowler_wicket.sum() == 0))
    out.append(("wides counted as balls faced", int((d.is_wide & d.faced).sum()), (d.is_wide & d.faced).sum() == 0))
    out.append(("byes/leg-byes charged to bowler",
                int(d[d.extras_type.isin(["byes", "legbyes"])].eval("bowler_conceded - runs").abs().sum()),
                d[d.extras_type.isin(["byes", "legbyes"])].eval("bowler_conceded - runs").abs().sum() == 0))
    out.append(("duplicate deliveries", int(d.duplicated(["match_id", "innings", "innings_ball_num"]).sum()),
                d.duplicated(["match_id", "innings", "innings_ball_num"]).sum() == 0))
    out.append(("negative balls remaining", int((d.balls_remaining < 1).sum()), (d.balls_remaining < 1).sum() == 0))
    out.append(("wickets before > 10", int((d.wkts_before > 10).sum()), (d.wkts_before > 10).sum() == 0))
    return pd.DataFrame(out, columns=["check", "value", "passed"])
