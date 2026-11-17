"""Chronological model selection.

For a forward fold (train window W, holdout season H):
  * each model scores players using W deliveries only (cross-fitted expectations inside W);
  * the holdout TARGET is the player's realised pressure performance in H, computed with expectations from
    models fitted on W only (strictly out-of-time). Two targets are used so no model grades itself:
      T_lev : leverage-weighted mean contextual VAA (state wicket value)
      T_raw : mean raw outcome on top-quartile COMPOSITE-index balls (the baseline's own definition)
  * metric: Spearman(train score, holdout target) over players eligible in W with >= MIN_HOLDOUT balls in H,
    plus Top-20 overlap and the RMSE of predicting holdout leverage-weighted VAA (for VAA-scale models).
  * uncertainty: 1,000 paired bootstrap resamples of HOLDOUT MATCHES (targets recomputed each time).

Pre-registered selection rule: walk the ladder M0 -> M4; a richer model replaces the current choice only
if its mean Spearman improvement (averaged over role x target) has a 90% bootstrap interval above zero.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src import config as C
from src.ranking.candidates import MODELS, build_tables, score_model, ycol
from src.models.pressure_response import player_match_table

MIN_HOLDOUT = 60


def holdout_targets(xh, role):
    t_lev = player_match_table(xh, role, f"vaa_{role}_state", "li_norm")
    t_raw = player_match_table(xh, role, f"raw_{role}_fixed", "hp_composite")
    return t_lev, t_raw


def _target_vec(t, mw):
    z = t[["Sw", "Swy", "n"]].mul(mw, axis=0)
    z["player_id"] = t["player_id"].values
    s = z.groupby("player_id").sum()
    return s["Swy"] / s["Sw"].replace(0, np.nan), s["n"]


def evaluate_fold(x, train_seasons, hold_season, p_high, n_boot=1000, seed=C.SEED):
    xw = x[x.season.isin(train_seasons)]
    xh = x[x.season.eq(hold_season)]
    rows, boot = [], {}
    rng = np.random.default_rng(seed)
    for role in ["bat", "bowl"]:
        tabs = build_tables(xw, role)
        scores = {k: score_model(tabs, sp, role, p_high)[0] for k, sp in MODELS.items()}
        t_lev, t_raw = holdout_targets(xh, role)
        matches = np.sort(t_lev["match_id"].unique())
        mi_lev = np.searchsorted(matches, t_lev["match_id"].values)
        mi_raw = np.searchsorted(matches, t_raw["match_id"].values)

        def metrics(mw_lev, mw_raw):
            out = {}
            for tname, t, mw in [("T_lev", t_lev, mw_lev), ("T_raw", t_raw, mw_raw)]:
                tgt, nh = _target_vec(t, mw)
                for k, sc in scores.items():
                    e = sc[sc.eligible].join(tgt.rename("tgt")).join(nh.rename("nh"))
                    e = e[(e.nh >= MIN_HOLDOUT) & e.tgt.notna() & e.score.notna()]
                    out[(k, tname)] = (spearmanr(e.score, e.tgt).correlation, len(e), e)
            return out

        base = metrics(np.ones(len(t_lev)), np.ones(len(t_raw)))
        for (k, tname), (rho, n, e) in base.items():
            top_s = set(e.nlargest(20, "score").index); top_t = set(e.nlargest(20, "tgt").index)
            rmse = float(np.sqrt(np.mean((e.score - e.tgt) ** 2))) if MODELS[k]["contextual"] and tname == "T_lev" else np.nan
            rows.append(dict(role=role, model=k, target=tname, spearman=rho, n_players=n,
                             top20_overlap=len(top_s & top_t), rmse=rmse))
        bs = []
        for b in range(n_boot):
            cnt = np.bincount(rng.integers(0, len(matches), len(matches)), minlength=len(matches)).astype(float)
            m = metrics(cnt[mi_lev], cnt[mi_raw])
            bs.append({key: v[0] for key, v in m.items()})
        boot[role] = pd.DataFrame(bs)
    return pd.DataFrame(rows), boot


def select(res, boot, order=("M0", "M1", "M2", "M3", "M4"), level=0.90):
    """Apply the pre-registered ladder rule. Returns (selected, decision log)."""
    def mean_rho(df_or_boot, k):
        cols = [(k, t) for t in ("T_lev", "T_raw")]
        return pd.concat([boot[r][cols] for r in ("bat", "bowl")], axis=1).mean(axis=1)

    cur, log = order[0], []
    for k in order[1:]:
        diff = mean_rho(boot, k) - mean_rho(boot, cur)
        lo, hi = np.quantile(diff, [(1 - level) / 2, 1 - (1 - level) / 2])
        point = res[res.model == k].spearman.mean() - res[res.model == cur].spearman.mean()
        accept = lo > 0
        log.append(dict(candidate=k, versus=cur, mean_improvement=point, ci_lo=lo, ci_hi=hi,
                        p_better=float((diff > 0).mean()), accepted=accept))
        if accept:
            cur = k
    return cur, pd.DataFrame(log)
