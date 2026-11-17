"""Leakage-controlled generation of delivery-level expectations, win probability and leverage.

Two schemes:

FORWARD fold  (used for model selection and the untouched final test)
  train window W (e.g. 2022-24), target season T (e.g. 2025)
  * deliveries in T   : predicted by models fitted on ALL of W            (strictly out-of-time)
  * deliveries in W   : 3-fold cross-fitting grouped by MATCH inside W   (never scored by a model that saw
                        the same match; nothing from T or later is used)
  * pressure percentiles, leverage normalisation, fixed/phase wicket values: fitted on W only.

RETROSPECTIVE scheme (used only for the final descriptive ranking over 2022-26)
  * leave-one-season-out cross-fitting: each season is scored by models fitted on the other seasons,
    so no delivery is ever scored by a model that saw its own match;
  * pressure percentiles and fixed/phase wicket values are still calibrated on 2022-24 (training) only.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src import config as C
from src.pressure.outcome_model import OutcomeModel
from src.pressure.win_probability import WinProbModel
from src.pressure.leverage import leverage, PressureScaler
from src.models.expected_performance import ExpectedPerformance
from src.models.wicket_value import FutureRunsModel, phase_and_fixed
from src.pressure.composite_index import composite_index
from src.models.vaa import vaa_bat, vaa_bowl

PRED_COLS = ["xr_bat", "xw_bat", "xr_bowl", "xw_bowl", "wp", "li", "wv_state", "fr_pred"]


def fit_predict(d, fit_mask, pred_mask):
    tr, te = d[fit_mask], d[pred_mask]
    om = OutcomeModel().fit(tr)
    wpm = WinProbModel().fit(tr)
    ep = ExpectedPerformance().fit(tr)
    fr = FutureRunsModel().fit(tr)
    wp, li, _ = leverage(te, om, wpm)
    out = pd.DataFrame(ep.predict(te), index=te.index)
    out["wp"], out["li"] = wp, li
    out["wv_state"] = fr.wicket_value(te)
    out["fr_pred"] = fr.predict(te)
    return out, dict(om=om, wpm=wpm, ep=ep, fr=fr)


def _crossfit(d, mask, n_splits=3):
    idx = d.index[mask]
    parts = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=n_splits).split(idx, groups=d.loc[idx, "match_id"])):
        fm = d.index.isin(idx[a]); pm = d.index.isin(idx[b])
        p, _ = fit_predict(d, fm, pm)
        parts.append(p)
    return pd.concat(parts)


def finalize(d, preds, calib_mask):
    """Attach predictions, pressure and VAA variants. calib_mask = rows defining the training calibration."""
    x = d.join(preds[PRED_COLS])
    sc = PressureScaler().fit(x.loc[calib_mask, "li"])
    x["pressure"] = sc.transform(x["li"])
    x["li_norm"] = sc.li_norm(x["li"])
    phase_wv, fixed_wv = phase_and_fixed(x[calib_mask], x.loc[calib_mask, "wv_state"])
    x["wv_fixed"] = fixed_wv
    x["wv_phase"] = x["phase"].map(phase_wv)
    pi = composite_index(x)
    x["pi_composite"] = pi
    x["hp_composite"] = (pi >= np.quantile(pi[calib_mask.values], C.HIGH_P)).astype(int)
    for v in ["fixed", "phase", "state"]:
        wv = x[f"wv_{v}"]
        x[f"vaa_bat_{v}"] = np.where(x["faced"], vaa_bat(x.batter_runs, x.xr_bat, x.batter_out, x.xw_bat, wv), np.nan)
        x[f"vaa_bowl_{v}"] = vaa_bowl(x.bowler_conceded, x.xr_bowl, x.bowler_wicket, x.xw_bowl, wv)
    meta = dict(fixed_wv=fixed_wv, phase_wv=phase_wv, mean_li=sc.mean_li,
                li_q75=float(np.quantile(x.loc[calib_mask, "li"], C.HIGH_P)))
    return x, meta


def forward_fold(d, train_seasons, target_season):
    wmask = d.season.isin(train_seasons)
    tmask = d.season.eq(target_season)
    within = _crossfit(d, wmask)
    target, models = fit_predict(d, wmask, tmask)
    preds = pd.concat([within, target])
    sub = d[wmask | tmask]
    x, meta = finalize(sub, preds, wmask[wmask | tmask])
    return x, meta, models


def retrospective(d):
    parts = []
    for s in C.ALL_SEASONS:
        p, _ = fit_predict(d, d.season.ne(s), d.season.eq(s))
        parts.append(p)
    preds = pd.concat(parts)
    return finalize(d, preds, d.season.isin(C.TRAIN_SEASONS))
