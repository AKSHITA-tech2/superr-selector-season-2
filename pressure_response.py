"""Player-level estimators built on per-(player, match) sufficient statistics.

Everything is computed from match-level sums, so (a) standard errors are clustered by MATCH, and
(b) the match-level bootstrap is just a re-weighting of matches (no refitting of delivery models).

Estimators
----------
weighted_mean_eb : y_bar_i = sum w*y / sum w  (w = 1 on high-pressure balls, or w = leverage)
                   normal-normal Empirical Bayes:  post_i = (y_i/V_i + mu/tau2) / (1/V_i + 1/tau2)
response_eb      : VAA_it = alpha_i + beta_i (P_t - 0.5) + e_it   (per-player OLS, match-clustered covariance V_i)
                   bivariate EB:  theta_i ~ N(mu, T);  post cov S_i = (T^-1 + V_i^-1)^-1,
                                  post mean m_i = S_i (T^-1 mu + V_i^-1 theta_hat_i)
                   score_i = m_i[0] + m_i[1] * (P_high - 0.5)
"""
import numpy as np
import pandas as pd

SUMS = ["n", "Sx", "Sxx", "Sy", "Sxy", "Syy", "Sw", "Sw2", "Swy", "Nh", "Nhi", "Shiy", "Nlo", "Sloy"]


def player_match_table(x, role, ycol, wcol, hp_level=0.75):
    """Aggregate deliveries to (player, match) sums. role: 'bat' (striker, faced) or 'bowl' (bowler)."""
    if role == "bat":
        r = x[x["faced"]]
        pid, pname = r["striker_id"], r["striker_name"]
    else:
        r = x
        pid, pname = r["bowler_id"], r["bowler_name"]
    y = r[ycol].values.astype(float)
    p = r["pressure"].values
    c = p - 0.5
    w = r[wcol].values.astype(float)
    # pressure opportunities are counted under the SAME pressure definition the model scores:
    # binary composite-index models -> composite high-pressure balls; leverage models -> top-quartile leverage
    binary = np.isin(w, (0.0, 1.0)).all()
    hp = w if binary else (p >= hp_level).astype(float)
    df = pd.DataFrame({
        "player_id": pid.values, "match_id": r["match_id"].values,
        "n": 1.0, "Sx": c, "Sxx": c * c, "Sy": y, "Sxy": c * y, "Syy": y * y, "Sw": w, "Sw2": w * w, "Swy": w * y,
        "Nh": hp,
        "Nhi": (p >= 0.9).astype(float), "Shiy": np.where(p >= 0.9, y, 0.0),
        "Nlo": (p <= 0.3).astype(float), "Sloy": np.where(p <= 0.3, y, 0.0)})
    t = df.groupby(["player_id", "match_id"], sort=True)[SUMS].sum().reset_index()
    names = pd.Series(pname.values, index=pid.values)
    meta = r.drop_duplicates("match_id").set_index("match_id")[["comp_type", "season"]]
    t = t.join(meta, on="match_id")
    t.attrs["names"] = names[~names.index.duplicated()].to_dict()
    return t


def pooled_var(t):
    """Per-ball variance of y pooled over all deliveries in the table (unweighted)."""
    N = t["n"].sum()
    return float(t["Syy"].sum() / N - (t["Sy"].sum() / N) ** 2)


def _wsum(t, cols, mw):
    """Per-player sums of match sums, weighted by bootstrap multiplicity mw (aligned to t rows)."""
    z = t[cols].mul(mw, axis=0)
    z["player_id"] = t["player_id"].values
    z["G"] = np.asarray(mw, dtype=float)   # matches counted with bootstrap multiplicity (= distinct matches when mw = 1)
    return z.groupby("player_id", sort=True)[cols + ["G"]].sum()


def eb_normal(est, var, elig, min_tau2=1e-6):
    """Method-of-moments normal-normal EB fitted on eligible players."""
    e, v = est[elig], var[elig]
    mu = np.average(e, weights=1 / v)
    tau2 = max(np.var(e, ddof=1) - np.mean(v), min_tau2)
    post = (est / var + mu / tau2) / (1 / var + 1 / tau2)
    pvar = 1 / (1 / var + 1 / tau2)
    return post, pvar, dict(mu=float(mu), tau2=float(tau2))


def weighted_mean_eb(t, mw, wtype, elig_mask_fn):
    """wtype: 'w' (leverage or hp indicator stored in Sw/Swy)."""
    s = _wsum(t, SUMS, mw)
    mean = s["Swy"] / s["Sw"].replace(0, np.nan)
    # clustered variance: sum_m (Swy_m - mean*Sw_m)^2 / (sum Sw)^2 * G/(G-1)
    mrow = t["player_id"].map(mean).values
    u = (t["Swy"].values - mrow * t["Sw"].values) ** 2 * mw
    meat = pd.Series(u).groupby(t["player_id"].values).sum().reindex(s.index)
    G = s["G"]
    var_cl = meat / s["Sw"] ** 2 * G / (G - 1).clip(lower=1)
    # floor: homoskedastic variance with the pooled per-ball variance (clustering may only ADD uncertainty)
    sig2 = pooled_var(t)
    var_h = sig2 * s["Sw2"] / s["Sw"] ** 2
    var = np.maximum(var_cl, var_h)
    elig = elig_mask_fn(s)
    ok = mean.notna() & var.notna()
    post, pvar, hyper = eb_normal(mean[ok].values, var[ok].values, elig[ok].values)
    out = pd.DataFrame(index=s.index)
    out["raw"] = mean
    out["raw_se"] = np.sqrt(var)
    out.loc[ok, "score"] = post
    out.loc[ok, "score_se"] = np.sqrt(pvar)
    return out, s, hyper


def response_eb(t, mw, p_high, elig_mask_fn):
    s = _wsum(t, SUMS, mw)
    n, Sx, Sxx, Sy, Sxy = (s[k].values for k in ["n", "Sx", "Sxx", "Sy", "Sxy"])
    det = n * Sxx - Sx ** 2
    det = np.where(det <= 1e-9, np.nan, det)
    a = (Sxx * Sy - Sx * Sxy) / det
    b = (n * Sxy - Sx * Sy) / det
    # match-clustered sandwich covariance
    idx = s.index.get_indexer(t["player_id"].values)
    am, bm = a[idx], b[idx]
    s0 = t["Sy"].values - am * t["n"].values - bm * t["Sx"].values
    s1 = t["Sxy"].values - am * t["Sx"].values - bm * t["Sxx"].values
    M00 = np.bincount(idx, s0 * s0 * mw, len(s)); M01 = np.bincount(idx, s0 * s1 * mw, len(s))
    M11 = np.bincount(idx, s1 * s1 * mw, len(s))
    Ainv = np.stack([np.stack([Sxx, -Sx], -1), np.stack([-Sx, n], -1)], -2) / det[:, None, None]
    Meat = np.stack([np.stack([M00, M01], -1), np.stack([M01, M11], -1)], -2)
    G = s["G"].values
    Vc = Ainv @ Meat @ Ainv * (G / np.clip(G - 1, 1, None))[:, None, None]
    # floor: never report more precision than a homoskedastic fit with the pooled per-ball variance;
    # keep the clustered correlation (clipped) and take the larger standard error for each coefficient.
    Vh = pooled_var(t) * Ainv
    sd = np.sqrt(np.maximum(np.stack([Vc[:, 0, 0], Vc[:, 1, 1]], -1), np.stack([Vh[:, 0, 0], Vh[:, 1, 1]], -1)))
    rho = np.clip(Vc[:, 0, 1] / np.sqrt(np.abs(Vc[:, 0, 0] * Vc[:, 1, 1]) + 1e-12), -0.95, 0.95)
    V = np.empty_like(Vc)
    V[:, 0, 0], V[:, 1, 1] = sd[:, 0] ** 2, sd[:, 1] ** 2
    V[:, 0, 1] = V[:, 1, 0] = rho * sd[:, 0] * sd[:, 1]
    theta = np.stack([a, b], -1)
    elig = elig_mask_fn(s).values & np.isfinite(a) & np.isfinite(b)
    mu = theta[elig].mean(0)
    T = np.cov(theta[elig].T) - V[elig].mean(0)
    w, U = np.linalg.eigh(T)
    T = U @ np.diag(np.maximum(w, [1e-5, 1e-4])) @ U.T     # positive-definite projection
    Tinv = np.linalg.inv(T)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(V).all(axis=(1, 2)) & (G >= 3)
    post_m = np.full((len(s), 2), np.nan); post_S = np.full((len(s), 2, 2), np.nan)
    Vinv = np.linalg.inv(V[ok])
    S_ = np.linalg.inv(Tinv[None] + Vinv)
    post_S[ok] = S_
    post_m[ok] = (S_ @ (Tinv @ mu + (Vinv @ theta[ok][..., None])[..., 0])[..., None])[..., 0]
    c = np.array([1.0, p_high - 0.5])
    out = pd.DataFrame(index=s.index)
    out["alpha_raw"], out["beta_raw"] = a, b
    out["alpha_se"], out["beta_se"] = np.sqrt(V[:, 0, 0]), np.sqrt(V[:, 1, 1])
    out["alpha"], out["beta"] = post_m[:, 0], post_m[:, 1]
    out["alpha_post_se"], out["beta_post_se"] = np.sqrt(post_S[:, 0, 0]), np.sqrt(post_S[:, 1, 1])
    out["score"] = post_m @ c
    out["score_se"] = np.sqrt(np.einsum("i,nij,j->n", c, post_S, c))
    out["raw"] = theta @ c
    hyper = dict(mu_alpha=float(mu[0]), mu_beta=float(mu[1]), T=T.tolist())
    return out, s, hyper
