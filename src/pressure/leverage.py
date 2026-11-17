"""Leverage index: expected absolute swing in win probability caused by the next delivery.

    LI_t = sum_o p(o | S_t) * | WP(S_t (+) o) - WP(S_t) |

Pressure P_t is the percentile of LI_t within the TRAINING-period LI distribution (0..1), so it is
continuous, bounded, comparable across innings/competitions, and never calibrated on future data.
"""
import numpy as np
import pandas as pd
from src.features.state_features import next_state_frame
from src.pressure.outcome_model import CLASSES, OutcomeModel


def leverage(d, outcome: OutcomeModel, wpm):
    P = outcome.predict_proba(d)
    wp0 = wpm.predict(d)
    li = np.zeros(len(d))
    for c in CLASSES:
        wk, lg = OutcomeModel.transition(c)
        nxt = next_state_frame(d, outcome.runs_by_class[c], wk, lg)
        li += P[c].values * np.abs(wpm.predict(nxt) - wp0)
    return wp0, li, P


class PressureScaler:
    """Empirical CDF of training-period leverage -> pressure percentile in [0,1]."""

    def fit(self, li_train):
        self.ref = np.sort(np.asarray(li_train))
        self.mean_li = float(np.mean(li_train))
        return self

    def transform(self, li):
        return np.searchsorted(self.ref, np.asarray(li), side="right") / len(self.ref)

    def li_norm(self, li):
        """Leverage relative to the training average (1.0 = average delivery) — baseball LI convention."""
        return np.asarray(li) / self.mean_li
