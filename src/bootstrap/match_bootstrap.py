"""Match-level bootstrap of the final ranking (stratified by competition x season).

Each replicate resamples MATCHES with replacement inside every competition-season stratum, then re-runs the
complete player-level estimation (per-player regressions, EB hyper-parameters, eligibility, ranking).
Delivery-level expectations are held fixed (see Limitations: model-refit uncertainty is not propagated).
"""
import numpy as np
import pandas as pd
from src import config as C
from src.ranking.candidates import score_model


def stratified_weights(t, rng):
    m = t.drop_duplicates("match_id")[["match_id", "comp_type", "season"]]
    w = pd.Series(0.0, index=m.match_id.values)
    for _, g in m.groupby(["comp_type", "season"]):
        ids = g.match_id.values
        draw = rng.integers(0, len(ids), len(ids))
        w.loc[ids] += np.bincount(draw, minlength=len(ids))
    return t["match_id"].map(w).values


def bootstrap_ranks(tables, spec, role, p_high, n_boot=C.N_BOOT, seed=C.SEED):
    rng = np.random.default_rng(seed)
    t = next(iter(tables.values()))
    ranks, scores = [], []
    for b in range(n_boot):
        mw = stratified_weights(t, rng)
        out, _ = score_model(tables, spec, role, p_high, mw=mw)
        e = out[out.eligible & out.score.notna()]
        r = e["score"].rank(ascending=False, method="first")
        ranks.append(r.rename(b)); scores.append(e["score"].rename(b))
    R = pd.concat(ranks, axis=1)   # NaN = not eligible in that replicate
    S = pd.concat(scores, axis=1)
    return R, S


def summarise(R, n_boot):
    big = len(R) + 1
    Rf = R.fillna(big)
    s = pd.DataFrame(index=R.index)
    s["boot_mean_rank"] = R.mean(axis=1)
    s["boot_median_rank"] = Rf.median(axis=1)
    s["boot_rank_sd"] = R.std(axis=1)
    s["boot_rank_lo"] = Rf.quantile(0.10, axis=1)
    s["boot_rank_hi"] = Rf.quantile(0.90, axis=1)
    for k in (1, 3, 5, 10, 20):
        s[f"p_top{k}"] = (Rf <= k).sum(axis=1) / n_boot
    s["p_eligible"] = R.notna().sum(axis=1) / n_boot
    return s


def pairwise_beat(R, ids):
    """P(row player ranked above column player) among the listed players."""
    Rf = R.loc[ids].fillna(len(R) + 1).values
    P = (Rf[:, None, :] < Rf[None, :, :]).mean(axis=2)
    return pd.DataFrame(P, index=ids, columns=ids)
