"""Post-run QA: data integrity, output consistency, dashboard pages and report generation.

Run after `python run_pipeline.py` (tests skip cleanly if outputs are absent).
"""
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from src import config as C
from src.data.loader import load_matches
from src.data.state_reconstructor import build_deliveries

O = C.OUT_DIR
HAVE = (O / "final_batter_rankings.csv").exists()
needs_outputs = pytest.mark.skipif(not HAVE, reason="pipeline outputs not generated yet")


@pytest.fixture(scope="module")
def d():
    return build_deliveries()


# ---------------- data integrity ----------------
def test_competition_classification():
    m, _ = load_matches()
    ipl = m[m.comp_name.str.startswith("IPL")]
    assert (ipl.comp_type == "IPL").all()
    assert (m[m.match_format == "T20I"].comp_type == "T20I").all()
    assert set(m.comp_type) <= {"IPL", "T20I", "OTHER"}


def test_match_join_complete(d):
    m, raw = load_matches()
    keep = m[m.comp_type.isin(["IPL", "T20I"])]
    bb = raw["ball_by_ball"]
    expected = bb[bb.match_id.isin(keep.match_id) & (bb.innings <= 2)]
    assert len(d) == len(expected)            # inner join lost nothing
    assert d.season.notna().all() and d.comp_type.notna().all()


def test_no_impossible_states(d):
    assert (d.wkts_before.between(0, 10)).all()
    assert (d.legal_before >= 0).all() and (d.legal_before <= d.max_balls).all()
    assert (d.score_before >= 0).all()
    assert d.loc[d.innings == 2, "target"].gt(0).all()
    assert (d.total_runs >= 0).all() and (d.batter_runs <= 7).all()


def test_no_nans_in_model_features(d):
    from src.features.state_features import state_frame
    X = state_frame(d)
    assert not X.isna().any().any()


def test_player_names_consistent(d):
    """One name per player id (no conflicting spellings within the Data Pack)."""
    for idc, nc in (("striker_id", "striker_name"), ("bowler_id", "bowler_name")):
        assert d.groupby(idc)[nc].nunique().max() == 1


# ---------------- outputs ----------------
@needs_outputs
def test_rankings_valid_and_consistent():
    S = json.load(open(O / "dataset_summary.json"))
    P = pd.read_csv(O / "player_metrics.csv")
    for role, f in (("bat", "final_batter_rankings.csv"), ("bowl", "final_bowler_rankings.csv")):
        r = pd.read_csv(O / f)
        assert len(r) == 10 and r.player_id.is_unique and r.player.is_unique
        assert list(r["rank"]) == list(range(1, 11))
        assert r.score_rel100.is_monotonic_decreasing
        top = P[P.role == role].sort_values("rank").head(10)
        assert list(top.player_id) == list(r.player_id)          # final list = top of the full table
        assert not r[["score_rel100", "p_top10", "boot_median_rank", "hp_balls"]].isna().any().any()
        e = S["elig"][role]
        assert (r.hp_balls >= e["min_hp_balls"]).all() and (r.matches >= e["min_matches"]).all()
        assert (r.p_top10.between(0, 1)).all()


@needs_outputs
def test_bootstrap_count_and_selection_logged():
    S = json.load(open(O / "dataset_summary.json"))
    assert S["n_boot"] == 1000
    rs = pd.read_csv(O / "bootstrap_rank_samples_bat.csv")
    assert len(rs) == 1000
    log = pd.read_csv(O / "selection_log.csv")
    assert set(log.candidate) == {"M1", "M2", "M3", "M4"}
    mc = pd.read_csv(O / "model_comparison.csv")
    assert mc.selected.sum() == 1 and mc[mc.selected].model.iloc[0] == S["selected_model"]


@needs_outputs
def test_why_text_generated_for_every_top10():
    w = json.load(open(O / "why_this_player.json"))
    for role, f in (("bat", "final_batter_rankings.csv"), ("bowl", "final_bowler_rankings.csv")):
        ids = pd.read_csv(O / f).player_id.astype(str)
        assert all(i in w[role] and len(w[role][i]["bullets"]) >= 4 for i in ids)


# ---------------- dashboard ----------------
@needs_outputs
def test_dashboard_every_page_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(C.ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    pages = at.radio(key="page").options
    assert len(pages) == 12
    for p in pages:
        at.radio(key="page").set_value(p).run()
        assert not at.exception, p


# ---------------- report ----------------
@needs_outputs
def test_report_builds_and_matches_rankings():
    r = subprocess.run([sys.executable, str(C.ROOT / "report" / "build_report.py")], capture_output=True, text=True, cwd=C.ROOT)
    assert r.returncode == 0, r.stderr
    md = (C.ROOT / "report" / "methodology_report.md").read_text()
    for f in ("final_batter_rankings.csv", "final_bowler_rankings.csv"):
        for name in pd.read_csv(O / f).player:
            assert name in md
    assert (C.ROOT / "report" / "final_methodology_report.pdf").stat().st_size > 100_000
