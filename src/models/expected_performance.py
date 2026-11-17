"""Contextual expectations for an AVERAGE player in the exact pre-delivery state.

xR_bat  = E[runs off the bat | S_t]            (deliveries faced: everything except wides)
xW_bat  = P(striker dismissed | S_t)           (includes striker run-outs; excludes retirements)
xR_bowl = E[runs charged to bowler | S_t]      (bat runs + wides + no-balls; byes/leg-byes excluded)
xW_bowl = P(bowler-credited wicket | S_t)      (run-outs are never bowler wickets)
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from src.config import HGB_PARAMS
from src.features.state_features import state_frame


class ExpectedPerformance:
    def fit(self, d):
        f = d[d["faced"]]
        X, Xf = state_frame(d), state_frame(f)
        p = {**HGB_PARAMS, "max_iter": 200}
        self.xr_bat = HistGradientBoostingRegressor(**p, loss="poisson").fit(Xf, f["batter_runs"])
        self.xw_bat = HistGradientBoostingClassifier(**p).fit(Xf, f["batter_out"])
        self.xr_bowl = HistGradientBoostingRegressor(**p, loss="poisson").fit(X, d["bowler_conceded"])
        self.xw_bowl = HistGradientBoostingClassifier(**p).fit(X, d["bowler_wicket"])
        return self

    def predict(self, d):
        X = state_frame(d)
        return dict(xr_bat=self.xr_bat.predict(X), xw_bat=self.xw_bat.predict_proba(X)[:, 1],
                    xr_bowl=self.xr_bowl.predict(X), xw_bowl=self.xw_bowl.predict_proba(X)[:, 1])
