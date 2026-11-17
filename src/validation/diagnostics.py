"""Out-of-sample diagnostics of the delivery-level models (all fitted on train seasons, scored on holdout)."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error
from src import config as C
from src.pressure.win_probability import WinProbModel, evaluate
from src.pressure.outcome_model import OutcomeModel, outcome_label, CLASSES
from src.models.expected_performance import ExpectedPerformance
from src.models.wicket_value import FutureRunsModel
from src.features.state_features import next_state_frame


def run(d, train_seasons, hold_season):
    tr, te = d[d.season.isin(train_seasons)], d[d.season.eq(hold_season)]
    rows, cal_out = [], []
    # --- win probability (projected vs direct 1st-innings framework) ---
    for mode in ("projected", "direct"):
        w = WinProbModel(mode).fit(tr)
        v = te[te.bat_won.notna()]
        for inn in (1, 2):
            s = v[v.innings == inn]
            m, cal = evaluate(w.predict(s), s.bat_won)
            rows.append(dict(model="Win probability", variant=f"{mode} / innings {inn}", **m))
            if mode == "projected":
                cal["innings"] = inn
                cal_out.append(cal)
    # --- outcome model vs base-rate ---
    om = OutcomeModel().fit(tr)
    y = outcome_label(te)
    P = om.predict_proba(te)
    base = y.value_counts(normalize=True).reindex(CLASSES).fillna(1e-6)
    base_tr = outcome_label(tr).value_counts(normalize=True).reindex(CLASSES).fillna(1e-6)
    lab = sorted(CLASSES)   # sklearn expects probability columns in sorted label order
    rows.append(dict(model="Outcome model (9 classes)", variant="state model",
                     logloss=log_loss(y, P[lab].values, labels=lab)))
    rows.append(dict(model="Outcome model (9 classes)", variant="train base rates",
                     logloss=log_loss(y, np.tile(base_tr[lab].values, (len(y), 1)), labels=lab)))
    # --- expected performance vs naive (train mean) ---
    ep = ExpectedPerformance().fit(tr)
    pr = ep.predict(te)
    f = te.faced.values
    for name, target, pred, mask, kind in [
            ("xR batter", te.batter_runs, pr["xr_bat"], f, "reg"), ("xW batter", te.batter_out, pr["xw_bat"], f, "clf"),
            ("xR bowler", te.bowler_conceded, pr["xr_bowl"], None, "reg"), ("xW bowler", te.bowler_wicket, pr["xw_bowl"], None, "clf")]:
        yt = target.values if mask is None else target.values[mask]
        yp = pred if mask is None else pred[mask]
        trm = (tr[tr.faced] if mask is not None else tr)[target.name].mean()
        if kind == "reg":
            rows.append(dict(model=name, variant="state model", mae=mean_absolute_error(yt, yp),
                             rmse=float(np.sqrt(np.mean((yt - yp) ** 2))), calib_ratio=float(yp.mean() / yt.mean())))
            rows.append(dict(model=name, variant="train mean", mae=mean_absolute_error(yt, np.full_like(yp, trm)),
                             rmse=float(np.sqrt(np.mean((yt - trm) ** 2)))))
        else:
            rows.append(dict(model=name, variant="state model", logloss=log_loss(yt, np.clip(yp, 1e-5, 1 - 1e-5)),
                             calib_ratio=float(yp.mean() / yt.mean())))
            rows.append(dict(model=name, variant="train mean", logloss=log_loss(yt, np.full_like(yp, trm))))
    # --- wicket value: which version best predicts the runs actually lost after a wicket? ---
    fr = FutureRunsModel().fit(tr)
    keep = fr.predict(next_state_frame(te, 0, 0, 1))
    wv_state = fr.wicket_value(te)
    trw = pd.Series(fr.wicket_value(tr), index=tr.index)
    phase_wv = trw.groupby(tr.phase).mean()
    variants = {"A fixed": np.full(len(te), trw.mean()), "B phase": te.phase.map(phase_wv).values, "C state": wv_state}
    wk = te.team_wicket.values & te.is_legal.values
    actual_after = (te.future_runs - te.total_runs).values
    for k, wv in variants.items():
        pred = keep - wv
        e = actual_after[wk] - pred[wk]
        rows.append(dict(model="Wicket value", variant=k, rmse=float(np.sqrt(np.mean(e ** 2))),
                         bias=float(np.mean(e)), mae=float(np.mean(np.abs(e)))))
    rows.append(dict(model="Remaining-runs model", variant="state model",
                     rmse=float(np.sqrt(np.mean((te.future_runs - fr.predict(te)) ** 2))),
                     r2=float(1 - np.mean((te.future_runs - fr.predict(te)) ** 2) / np.var(te.future_runs))))
    out = pd.DataFrame(rows)
    out.insert(0, "holdout", hold_season)
    return out, pd.concat(cal_out)
