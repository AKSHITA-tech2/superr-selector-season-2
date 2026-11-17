"""Multinomial delivery-outcome model p(o | S_t) used for leverage."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.config import HGB_PARAMS
from src.features.state_features import state_frame

CLASSES = ["0", "1", "2", "3", "4", "6", "W", "WD", "NB"]


def outcome_label(d: pd.DataFrame) -> pd.Series:
    r = d["total_runs"].clip(upper=6).replace({5: 4})
    lab = r.astype(int).astype(str)
    lab = np.where(d["is_legal"] & d["team_wicket"], "W", lab)
    lab = np.where(d["is_wide"], "WD", lab)
    lab = np.where(d["is_noball"], "NB", lab)
    return pd.Series(lab, index=d.index)


class OutcomeModel:
    def fit(self, d):
        y = outcome_label(d)
        self.m = HistGradientBoostingClassifier(**{**HGB_PARAMS, "max_iter": 200}).fit(state_frame(d), y)
        # mean team runs added by each outcome class (for the next-state transition)
        self.runs_by_class = d.groupby(y)["total_runs"].mean().reindex(CLASSES).fillna(0).to_dict()
        return self

    def predict_proba(self, d) -> pd.DataFrame:
        p = self.m.predict_proba(state_frame(d))
        P = pd.DataFrame(p, columns=self.m.classes_, index=d.index)
        return P.reindex(columns=CLASSES, fill_value=0.0)

    @staticmethod
    def transition(c):
        """(wicket_add, legal_add) for an outcome class."""
        return {"W": (1, 1), "WD": (0, 0), "NB": (0, 0)}.get(c, (0, 1))
