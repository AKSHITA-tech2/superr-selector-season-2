# SupeRR Selector Season 2 — Round 1 · Pressure Lab

**Task (official brief):** rank the 10 most reliable batters and 10 most reliable bowlers *under pressure*, comparing
high-pressure performance with usual performance, using IPL + T20I data from 2022 onwards, and explain pressure,
ratings, fairness (sample size, eligibility, IPL/T20I), evidence and reliability.

> **All model training and player rankings are based exclusively on the competition-provided Data Pack. External sources
> are used only for qualitative research, previous-winner analysis and supporting video evidence.**

## Quick start (Windows / macOS / Linux)

```bash
cd C:\superr                      # or wherever you unzip
python -m venv .venv && .venv\Scripts\activate      # (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python run_pipeline.py --rebuild  # full analysis from the raw Data Pack (~10-12 min on 2 cores)
python -m src.visualization.figures
python report/build_report.py     # -> report/final_methodology_report.pdf
streamlit run dashboard/app.py    # -> http://localhost:8501
python -m pytest -q tests         # automated checks
```
`python run_pipeline.py` without `--rebuild` reuses `outputs/cache/frames.pkl` if present (not shipped in the zip: it is
regenerated on first run).

## Official dataset (`Data Pack/`)
`ball_by_ball.csv` (200,999 deliveries), `matches.csv` (883), `competitions.csv`, `teams.csv`, `playing_eleven.csv`,
`competition_team_player_mapping.csv`. A full data dictionary is written to `outputs/data_dictionary.csv`; integrity checks
to `outputs/integrity_checks.csv`. Scope: IPL + T20I matches (emerging/A-team tournaments excluded), innings 1–2 (super overs excluded).

## Methodology (one paragraph)
1. **Match state** before every delivery (balls left, wickets in hand, score, target, RRR, phase, competition, Impact-Player era),
   with correct handling of wides, no-balls (`Overstep`/`other`), byes, leg-byes, penalties, run-outs (never bowler wickets) and DLS.
2. **Win probability**: monotone gradient boosting for chases; first innings via a projected-total framework.
3. **Pressure**: (a) composite index (RRR, death overs, wickets lost, collapse) — top quartile of 2022–24; (b) leverage
   `LI = Σ_o p(o|S)|WP(S⊕o) − WP(S)|`, as a 2022–24 percentile.
4. **Expected performance** `xR`, `xW` and **state-dependent wicket value** `WV = E[FR|no wkt] − E[FR|wkt]`.
5. **VAA** (higher = better for both roles): bat `R − xR − WV(Out − xW)`; bowl `(xR − R) + WV(Wkt − xW)`.
6. **Player models** M0–M4 (baseline → full research methodology), all with empirical-Bayes shrinkage and match-clustered SEs.
7. **Selection**: train 2022–24 → validate 2025 with a pre-registered ladder rule; untouched 2026 test.
8. **Reliability**: 1,000 stratified match-level bootstraps (rank intervals, Top-3/5/10 probabilities), sensitivity analysis.

The selected model and every number are in `outputs/dataset_summary.json` and the report.

## Architecture
```
run_pipeline.py                     one-command orchestration (audit -> models -> selection -> test -> ranking -> bootstrap)
src/config.py                       every tunable choice (seed, split, eligibility, thresholds)
src/data/                           loader, pre-delivery state reconstruction, data audit
src/features/state_features.py      pre-delivery feature matrices (no future information)
src/pressure/                       composite index, 9-outcome model, win probability, leverage
src/models/                         expected performance (xR/xW), wicket value, VAA, EB & pressure-response estimators
src/ranking/                        candidate models M0-M4, player profiles, "why selected" generator
src/validation/                     temporal split & cross-fitting, diagnostics, model selection, sensitivity
src/bootstrap/match_bootstrap.py    stratified match-level bootstrap, rank intervals, pairwise probabilities
src/evidence/video_evidence.py      qualitative public sources (never a model input)
src/visualization/figures.py        15 publication figures
dashboard/app.py                    12-page Streamlit dashboard (reads outputs only)
report/build_report.py              22-section Markdown + PDF report (numbers read from outputs)
tests/                              23 tests: attribution, signs, leakage, shrinkage, bootstrap, rankings, dashboard, report
```

## Outputs
`outputs/final_batter_rankings.csv`, `final_bowler_rankings.csv`, `player_metrics.csv`, `bootstrap_results.csv`,
`model_comparison.csv`, `validation_results.csv`, `test_results.csv`, `selection_log.csv`, `sensitivity_results.csv`,
`delivery_model_diagnostics.csv`, `pairwise_beat_*.csv`, `why_this_player.json`, `run_manifest.json`;
baseline preserved in `outputs/baseline/`; `evidence/player_match_evidence.csv`, `evidence/player_video_evidence.csv`,
`evidence/previous_winners.json`; `figures/*.png`; `report/final_methodology_report.pdf`, `report/methodology_report.md`.

## Validation & leakage control
* Forward-only: models used to score 2025 / 2026 are fitted on earlier seasons only; inside a training window,
  expectations are cross-fitted by match.
* Pressure percentiles, leverage normalisation, composite threshold and fixed/phase wicket values are calibrated on 2022–24.
* The final retrospective ranking uses leave-one-season-out expectations (no delivery scored by a model that saw its match).
* `tests/test_pipeline.py::test_forward_fold_never_trains_on_holdout` asserts the split.

## Reproducibility
Seed 2026 everywhere; 1,000 bootstraps. A clean `--rebuild` from the raw Data Pack and a cached run give identical ranking files (SHA-256 in `outputs/run_manifest.json`).

Scope note: per the organisers' clarification, the lists rank pressure performance; they are not retention or auction recommendations.
