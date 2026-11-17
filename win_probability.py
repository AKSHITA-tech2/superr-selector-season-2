"""Pre-delivery win-probability model, one monotone gradient-boosting model per innings.

Monotone constraints encode cricket logic so the model cannot learn absurd shapes from noise:
  1st innings: WP rises with runs scored, wickets in hand and balls remaining.
  2nd innings: WP falls with runs required / required rate, rises with wickets in hand and balls remaining.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from src.config import HGB_PARAMS
from src.features.state_features import state_frame, STATE_COLS

MONO = {
    1: {"balls_remaining": 1, "legal_before": -1, "wkts_in_hand": 1, "score_before": 1},
    2: {"balls_remaining": 1, "wkts_in_hand": 1, "runs_required_f": -1, "rrr_f": -1},
}
FEATS = {1: ["balls_remaining", "legal_before", "wkts_in_hand", "score_before", "is_ipl", "impact_era"],
         2: ["balls_remaining", "wkts_in_hand", "runs_required_f", "rrr_f", "is_ipl", "impact_era"]}


SIG_FEATS = ["balls_remaining", "legal_before", "wkts_in_hand", "is_ipl", "impact_era"]
SIG_MONO = {"balls_remaining": 1, "legal_before": -1}


class WinProbModel:
    """2nd innings: direct monotone model of P(chase succeeds | state).
    1st innings: 'projected-total' framework (no RRR exists yet):
        final total  T ~ Normal( score + mu(S), sigma(S)^2 )      mu, sigma from a remaining-runs model
        WP1(S) = 1 - E_T[ WP2(chase start, target = T + 1) ]
    i.e. the batting-first side wins iff the chasing side fails to chase what it is projected to post.
    This is smooth, uses only pre-delivery state and reuses the well-calibrated chase model.
    (A direct 1st-innings classifier is kept as a benchmark: `mode='direct'`.)"""

    def __init__(self, mode="projected"):
        self.mode = mode

    def fit(self, d):
        from src.models.wicket_value import FutureRunsModel
        d = d[d["bat_won"].notna()]
        self.m = {}
        if self.mode == "projected":
            i1 = d[d.innings == 1]
            self.fr = FutureRunsModel().fit(i1)
            resid = np.abs(i1["future_runs"].values - self.fr.predict(i1))
            # spread of the final total depends only on remaining RESOURCES (smooth, monotone)
            self.sig = HistGradientBoostingRegressor(
                **{**HGB_PARAMS, "max_iter": 150, "min_samples_leaf": 300},
                monotonic_cst=[SIG_MONO.get(c, 0) for c in SIG_FEATS]).fit(
                state_frame(i1)[SIG_FEATS], resid * np.sqrt(np.pi / 2))
        for inn in (1, 2):
            s = d[d.innings == inn]
            X = state_frame(s)[FEATS[inn]]
            cst = [MONO[inn].get(c, 0) for c in FEATS[inn]]
            self.m[inn] = HistGradientBoostingClassifier(
                **{**HGB_PARAMS, "max_iter": 250, "min_samples_leaf": 200}, monotonic_cst=cst
            ).fit(X, s["bat_won"].astype(int))
        return self

    def predict(self, d) -> np.ndarray:
        """P(batting side wins | state). Terminal 2nd-innings states are resolved exactly."""
        X = state_frame(d)
        wp = np.full(len(d), np.nan)
        for inn in (1, 2):
            idx = (d["innings"].values == inn)
            if idx.any():
                wp[idx] = self.m[inn].predict_proba(X.loc[idx, FEATS[inn]])[:, 1]
        idx1 = d["innings"].values == 1
        if self.mode == "projected" and idx1.any():
            wp[idx1] = self._wp1_projected(d[idx1])
        inn2 = d["innings"].values == 2
        rr = d["runs_required"].values
        done_won = inn2 & (rr <= 0)
        done_lost = inn2 & (rr > 0) & ((d["wkts_in_hand"].values <= 0) | (d["balls_remaining"].values <= 0))
        tie = inn2 & (rr == 1) & ((d["wkts_in_hand"].values <= 0) | (d["balls_remaining"].values <= 0))
        wp[done_won] = 1.0
        wp[done_lost] = 0.0
        wp[tie] = 0.5   # scores level when the chase ends -> tie
        # 1st innings ending (all out / overs done): the model is evaluated at the boundary state
        return np.clip(wp, 0.0, 1.0)

    def _chase_curve(self, is_ipl, era, max_balls):
        """P(chase succeeds) at the start of the chase for targets 1..400 (cached)."""
        key = (is_ipl, era, max_balls)
        if not hasattr(self, "_cc"):
            self._cc = {}
        if key not in self._cc:
            T = np.arange(1, 401, dtype=float)
            X = pd.DataFrame({"balls_remaining": max_balls, "wkts_in_hand": 10.0, "runs_required_f": T,
                              "rrr_f": np.clip(T * 6 / max_balls, 0, 36), "is_ipl": float(is_ipl),
                              "impact_era": float(era)})[FEATS[2]]
            self._cc[key] = self.m[2].predict_proba(X)[:, 1]
        return self._cc[key]

    def _wp1_projected(self, d):
        live = (d["wkts_in_hand"].values > 0) & (d["balls_remaining"].values > 0)
        mu = d["score_before"].values + np.where(live, self.fr.predict(d), 0.0)
        sig = np.where(live, np.maximum(self.sig.predict(state_frame(d)[SIG_FEATS]), 2.0), 1.0)
        grid = np.arange(0, 400)            # possible final totals
        out = np.empty(len(d))
        keys = list(zip(d["is_ipl"].values, d["impact_era"].values, d["max_balls"].values))
        kser = pd.Series(keys)
        for key in kser.unique():
            rows = np.flatnonzero((kser == key).values)
            chase = self._chase_curve(*key)[grid]   # chase[t] = P(chasing target t+1 succeeds)
            for s in range(0, len(rows), 20000):
                r = rows[s:s + 20000]
                z = (grid[None, :] - mu[r, None]) / sig[r, None]
                w = np.exp(-0.5 * z ** 2)
                w /= w.sum(axis=1, keepdims=True)
                out[r] = 1.0 - w @ chase
        return out


def evaluate(wp, y):
    y = np.asarray(y).astype(int)
    bins = pd.cut(wp, np.linspace(0, 1, 11), include_lowest=True)
    cal = pd.DataFrame({"wp": wp, "y": y, "bin": bins}).groupby("bin", observed=True).agg(
        pred=("wp", "mean"), obs=("y", "mean"), n=("y", "size")).reset_index()
    cal["bin"] = cal["bin"].astype(str)
    ece = float((cal.n * (cal.pred - cal.obs).abs()).sum() / cal.n.sum())
    return dict(brier=float(brier_score_loss(y, wp)), auc=float(roc_auc_score(y, wp)),
                logloss=float(log_loss(y, np.clip(wp, 1e-4, 1 - 1e-4))), ece=ece), cal
