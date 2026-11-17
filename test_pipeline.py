"""Automated checks: data attribution, signs, leverage, leakage, shrinkage, ranking, bootstrap."""
import numpy as np
import pandas as pd
import pytest
from src import config as C
from src.data.state_reconstructor import build_deliveries
from src.models.vaa import vaa_bat, vaa_bowl
from src.pressure.leverage import PressureScaler
from src.models.pressure_response import weighted_mean_eb, response_eb, player_match_table
from src.bootstrap.match_bootstrap import stratified_weights


@pytest.fixture(scope="module")
def d():
    return build_deliveries()


# ---------------- data ----------------
def test_only_official_pack_and_scope(d):
    assert set(d.comp_type.unique()) == {"IPL", "T20I"}
    assert d.innings.isin([1, 2]).all()
    assert str(C.DATA_DIR).endswith("Data Pack")


def test_legal_balls_per_over(d):
    per_over = d.groupby(["match_id", "innings", "overid"]).is_legal.sum()
    assert (per_over <= 6).mean() > 0.99 and (per_over == 6).mean() > 0.97


def test_state_is_pre_delivery(d):
    first = d.groupby(["match_id", "innings"]).head(1)
    assert (first.score_before == 0).all() and (first.wkts_before == 0).all() and (first.legal_before == 0).all()
    nxt = d.groupby(["match_id", "innings"]).score_before.shift(-1)
    ok = nxt.notna()
    assert np.allclose(nxt[ok], (d.score_before + d.total_runs)[ok])


def test_run_outs_never_bowler_wickets(d):
    ro = d.dismissal_type.fillna("").isin(["RunOut", "Obstracting the field", "Obstructing the field", "Retire Out"])
    assert d.loc[ro, "bowler_wicket"].sum() == 0
    assert d.loc[d.dismissal_type.eq("Bowled"), "bowler_wicket"].eq(1).all()


def test_extras_attribution(d):
    assert not (d.is_wide & d.faced).any()
    lb = d.extras_type.isin(["byes", "legbyes"])
    assert (d.loc[lb, "bowler_conceded"] == d.loc[lb, "runs"]).all()
    wd = d.is_wide
    assert (d.loc[wd, "bowler_conceded"] >= 1).all()


def test_target_and_rrr(d):
    i2 = d[d.innings == 2]
    assert i2.target.notna().all()
    s = i2.iloc[0]
    assert np.isclose(s.rrr, s.runs_required * 6 / s.balls_remaining)


# ---------------- VAA signs ----------------
def test_vaa_sign_conventions():
    # batter: a six above expectation is good, a dismissal is bad
    assert vaa_bat(6, 1.3, 0, 0.05, 6) > 0
    assert vaa_bat(0, 1.3, 1, 0.05, 6) < 0
    # bowler: a dot ball and a wicket are good, a boundary is bad
    assert vaa_bowl(0, 1.3, 0, 0.05, 6) > 0
    assert vaa_bowl(0, 1.3, 1, 0.05, 6) > vaa_bowl(0, 1.3, 0, 0.05, 6)
    assert vaa_bowl(6, 1.3, 0, 0.05, 6) < 0
    # expected-value centring: if outcomes equal expectations on average, mean VAA is 0
    rng = np.random.default_rng(0)
    xr, xw, wv = 1.3, 0.05, 6.0
    runs = rng.poisson(xr, 200000); out = rng.random(200000) < xw
    assert abs(vaa_bat(runs, xr, out, xw, wv).mean()) < 0.02


# ---------------- pressure ----------------
def test_pressure_scaler_uses_training_reference_only():
    tr = np.linspace(0, 1, 1001)
    sc = PressureScaler().fit(tr)
    assert sc.transform([0.5])[0] == pytest.approx(0.5, abs=0.002)
    assert sc.transform([5.0])[0] == 1.0           # future extremes are capped, not re-scaled
    p = sc.transform(np.array([0.1, 0.2, 0.9]))
    assert np.all(np.diff(p) > 0)


# ---------------- shrinkage & ranking ----------------
def _toy(n_players=60, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        n_matches = 5 if i < 5 else 40
        true = rng.normal(0, 0.05)
        for m in range(n_matches):
            k = 20
            p = rng.random(k)
            y = true + rng.normal(0, 1.5, k)
            for pp, yy in zip(p, y):
                rows.append(dict(striker_id=i, striker_name=f"P{i}", match_id=i * 1000 + m, faced=True, y=yy,
                                 pressure=pp, li_norm=1.0 + pp, comp_type="IPL", season=2024))
    return pd.DataFrame(rows)


def test_eb_shrinks_small_samples_more():
    x = _toy()
    t = player_match_table(x, "bat", "y", "li_norm")
    elig = lambda s: s["n"] >= 0
    out, s, h = weighted_mean_eb(t, np.ones(len(t)), "w", elig)
    gap = (out.raw - h["mu"]).abs() - (out.score - h["mu"]).abs()
    keep = (out.score - h["mu"]) / (out.raw - h["mu"])
    assert keep[out.index < 5].mean() < keep[out.index >= 5].mean()
    assert (gap >= -1e-9).all()


def test_response_eb_finite_and_ranked():
    x = _toy()
    t = player_match_table(x, "bat", "y", "li_norm")
    out, s, h = response_eb(t, np.ones(len(t)), C.P_HIGH, lambda s: s["n"] >= 0)
    assert out.score.notna().all() and np.isfinite(out.score_se).all()
    assert out.index.is_unique


# ---------------- bootstrap ----------------
def test_bootstrap_resamples_matches_within_strata():
    x = _toy()
    t = player_match_table(x, "bat", "y", "li_norm")
    w = stratified_weights(t, np.random.default_rng(0))
    # every row of the same match gets the same weight
    assert pd.Series(w).groupby(t.match_id.values).nunique().max() == 1
    # number of resampled matches equals the number of matches (per stratum)
    per_match = pd.Series(w, index=t.match_id.values).groupby(level=0).first()
    assert per_match.sum() == t.match_id.nunique()


# ---------------- leakage ----------------
def test_forward_fold_never_trains_on_holdout(monkeypatch):
    import src.validation.temporal_split as TS
    seen = []
    real = TS.fit_predict

    def spy(d, fit_mask, pred_mask):
        seen.append((set(d.loc[fit_mask, "season"]), set(d.loc[pred_mask, "season"])))
        raise StopIteration

    monkeypatch.setattr(TS, "fit_predict", spy)
    d = build_deliveries()
    with pytest.raises(StopIteration):
        TS.forward_fold(d, [2022, 2023, 2024], 2025)
    fit_seasons, pred_seasons = seen[0]
    assert 2025 not in fit_seasons and 2026 not in fit_seasons


def test_outputs_if_present():
    f = C.OUT_DIR / "final_batter_rankings.csv"
    if not f.exists():
        pytest.skip("pipeline not run yet")
    for name in ("final_batter_rankings.csv", "final_bowler_rankings.csv"):
        r = pd.read_csv(C.OUT_DIR / name)
        assert len(r) == 10 and r.player_id.is_unique and list(r["rank"]) == list(range(1, 11))
        e = C.ELIG["bat" if "batter" in name else "bowl"]
        assert (r.hp_balls >= e["min_hp_balls"]).all() and (r.matches >= e["min_matches"]).all()
