"""Hand-built composite pressure index used by Model 0 / Model 1 (mirrors typical public solutions).

2nd innings: PI = 0.50*sigmoid((RRR-8.5)/1.5) + 0.25*death + 0.15*resource_lost + 0.10*sigmoid(...)*death
1st innings: PI = 0.25*death + 0.40*resource_lost + 0.35*collapse
  resource_lost = wickets lost / 10 ; collapse = min(wickets lost in previous 18 legal balls / 3, 1)
High pressure = PI >= 75th percentile of the TRAINING-period PI distribution.
The weights are hand-set (that is the point of the benchmark): the research PDF lists this as a weakness.
"""
import numpy as np
import pandas as pd


def composite_index(d: pd.DataFrame) -> np.ndarray:
    sig = 1 / (1 + np.exp(-(np.clip(d["rrr"].fillna(0), 0, 36) - 8.5) / 1.5))
    death = d["is_death"].values
    res = d["wkts_before"].values / 10
    # wickets in the previous 18 deliveries of the innings (pre-delivery, no look-ahead)
    w = d.groupby(["match_id", "innings"])["team_wicket"].transform(
        lambda s: s.astype(int).shift(1, fill_value=0).rolling(18, min_periods=1).sum()).values
    collapse = np.minimum(w / 3, 1)
    inn2 = 0.50 * sig + 0.25 * death + 0.15 * res + 0.10 * sig * death
    inn1 = 0.25 * death + 0.40 * res + 0.35 * collapse
    return np.where(d["innings"].values == 2, inn2, inn1)
