"""Wicket value in runs.

State-dependent (Model C):  WV_t = E[FutureRuns | S_t, no wicket] - E[FutureRuns | S_t, wicket]
estimated with a monotone remaining-runs model F(state) evaluated at the two post-delivery states.
Phase-dependent (Model B): training mean of WV_t within powerplay / middle / death.
Fixed (Model A): training mean of WV_t.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import HGB_PARAMS
from src.features.state_features import state_frame, next_state_frame, STATE_COLS

MONO = {"balls_remaining": 1, "legal_before": -1, "wkts_in_hand": 1, "score_before": 1}


class FutureRunsModel:
    def fit(self, d):
        X = state_frame(d)
        self.m = HistGradientBoostingRegressor(
            **{**HGB_PARAMS, "max_iter": 250}, loss="poisson",
            monotonic_cst=[MONO.get(c, 0) for c in STATE_COLS]).fit(X, d["future_runs"].clip(lower=0))
        return self

    def predict(self, d):
        out = self.m.predict(state_frame(d))
        out = np.where(d["wkts_in_hand"].values <= 0, 0.0, out)
        out = np.where(d["balls_remaining"].values <= 0, 0.0, out)
        if "runs_required" in d:  # a chase can never produce more than the runs still required (+ a boundary)
            rr = d["runs_required"].values
            cap = np.where(d["innings"].values == 2, np.clip(rr + 3, 0, None), np.inf)
            out = np.minimum(out, cap)
        return out

    def wicket_value(self, d):
        keep = next_state_frame(d, 0, 0, 1)
        lose = next_state_frame(d, 0, 1, 1)
        return np.clip(self.predict(keep) - self.predict(lose), 0, None)


def phase_and_fixed(train_d, wv_train):
    s = pd.Series(wv_train, index=train_d.index)
    return s.groupby(train_d["phase"]).mean().to_dict(), float(s.mean())
