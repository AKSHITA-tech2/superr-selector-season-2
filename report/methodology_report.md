# SupeRR Selector Season 2 — Round 1: Most Reliable Players Under Pressure

|   Rank | Batter           |   Score | Top-10 prob.   | Bowler              |   Score | Top-10 prob.   |
|-------:|:-----------------|--------:|:---------------|:--------------------|--------:|:---------------|
|      1 | Tim David        |    26.6 | 89%            | Mohammad Nawaz      |    24   | 92%            |
|      2 | Shashank Singh   |    20.4 | 76%            | Sunil Narine        |    19.7 | 61%            |
|      3 | Suryakumar Yadav |    19.8 | 58%            | Abrar Ahmed         |    18.9 | 75%            |
|      4 | Tristan Stubbs   |    15.5 | 47%            | Mitchell Santner    |    16.9 | 33%            |
|      5 | Naman Dhir       |    15   | 53%            | Shaheen Afridi      |    15.8 | 44%            |
|      6 | Rinku Singh      |    12.6 | 25%            | Matheesha Pathirana |    15.7 | 24%            |
|      7 | Nicholas Pooran  |    11.5 | 22%            | Jasprit Bumrah      |    15.4 | 39%            |
|      8 | Dhruv Jurel      |    11   | 28%            | Matthew Humphreys   |    13.5 | 33%            |
|      9 | Faf du Plessis   |    10.9 | 41%            | Taskin Ahmed        |    12.8 | 37%            |
|     10 | Jos Buttler      |     9.8 | 19%            | Mustafizur Rahman   |    12.5 | 17%            |

Data: official SupeRR Data Pack only (837 IPL + T20I matches, 193,988 deliveries, 2022–2026). Selected model: **M0 — PE-EB baseline**. Seed 2026; 1,000 stratified match-level bootstraps.


## 1. Executive summary

We built a leakage-controlled pipeline that reconstructs the match state before every delivery, defines pressure, measures each delivery's value relative to an average player in the same state, and estimates each player's pressure performance with empirical-Bayes shrinkage. Five candidate models — from a simple baseline to the full research-PDF methodology (leverage, contextual VAA, state-dependent wicket value, hierarchical pressure response) — were compared on a forward-only split: fit on 2022–24, select on 2025, report an untouched 2026 test.

**Result of the model contest.** No richer model beat the baseline by a margin we could trust. Mean validation Spearman: M0 0.148, M1 0.124, M2 0.043, M3 0.070, M4 0.077. Under our pre-registered rule (a richer model must improve with a 90% bootstrap interval above zero) the simplest model, **M0**, produces the final ranking. On the 2026 test (run once, after selection) the context-aware models did slightly better (M0 0.070, M1 0.008, M2 0.149, M3 0.132, M4 0.098), but again within noise. We report this honestly rather than promote the more impressive-sounding model.

**Main substantive finding.** One-season pressure performance is noisy for every model, and the spread of individual 'pressure slopes' is almost entirely sampling noise once shrunk. Reliability therefore comes from consistent high-pressure output over many matches, not from a player's personal improvement under pressure — which is why every pick is reported with a posterior interval, a bootstrap Top-10 probability and model-agreement count.

**Stability.** The final Top 10 sets have an average bootstrap Top-10 probability of 46% (batters) and 46% (bowlers). The strongest picks are Tim David (89%) and Mohammad Nawaz (92%). Eligibility thresholds and competition weighting barely move the lists; the definition of pressure moves them most (Section 15).


## 2. Problem definition

Objective (official brief): identify and rank the 10 batters and 10 bowlers who deliver most consistently in high-pressure situations relative to their usual level, with a pressure definition that works for batting and bowling, fair handling of sample size, eligibility, context and IPL/T20I data, and evidence plus reliability checks.

Pressure is deliberately left for entrants to define. Following the organisers' clarification email (17 Nov 2026), these lists rank **pressure performance**; they are not retention or auction recommendations. Rankings come only from the supplied data, never reputation.


## 3. Dataset

|   Season |   IPL |   T20I |
|---------:|------:|-------:|
|     2022 |    74 |    106 |
|     2023 |    74 |     66 |
|     2024 |    71 |    106 |
|     2025 |    74 |    122 |
|     2026 |    74 |     70 |

The Data Pack holds 883 matches. We analyse the 837 IPL and T20I matches that have ball-by-ball data (367 IPL, 470 T20I); 30 emerging/A-team tournament matches are excluded because the brief scopes IPL and T20I. 25 no-result matches contribute deliveries to expectation models but not to win-probability training. 824 batters and 596 bowlers appear; 131 and 172 meet eligibility. Super overs (innings 3/4) are excluded.


## 4. Data processing

**Attribution rules (unit-tested).** 'Overstep' and 'other' extras are no-balls; wides and no-balls are not legal balls; a batter faces every delivery except a wide; byes, leg-byes and penalty runs are not charged to the bowler; run-outs, obstruction and retirements are never bowler wickets; a batter is 'out' only when he is the dismissed striker; revised (DLS) innings lengths are stored in balls and respected.

| check                                     |   value | passed   |
|:------------------------------------------|--------:|:---------|
| share of overs with exactly 6 legal balls |   0.982 | True     |
| run-outs credited to bowler               |   0     | True     |
| wides counted as balls faced              |   0     | True     |
| byes/leg-byes charged to bowler           |   0     | True     |
| duplicate deliveries                      |   0     | True     |
| negative balls remaining                  |   0     | True     |
| wickets before > 10                       |   0     | True     |


## 5. Match-state reconstruction and win probability

For every delivery we compute, before the ball: legal balls elapsed and remaining (respecting revised overs), score, wickets lost and in hand, target, runs required, CRR, RRR, phase (powerplay overs 1–6, middle, death 17–20), competition and an IPL Impact-Player era flag (2023+).

**Second innings:** monotone gradient boosting for P(chase succeeds) (WP falls with runs required and RRR, rises with wickets in hand and balls remaining). **First innings:** no RRR exists yet, so we project the final total T ~ N(score + μ(S), σ(S)²) from a monotone remaining-runs model and set WP₁ = 1 − E_T[WP₂(start of chase, T+1)].

|   holdout | variant               |   brier |   auc |   logloss |   ece |
|----------:|:----------------------|--------:|------:|----------:|------:|
|      2025 | projected / innings 1 |   0.212 | 0.726 |     0.611 | 0.039 |
|      2025 | projected / innings 2 |   0.12  | 0.913 |     0.371 | 0.016 |
|      2025 | direct / innings 1    |   0.214 | 0.725 |     0.619 | 0.056 |
|      2025 | direct / innings 2    |   0.12  | 0.913 |     0.371 | 0.016 |
|      2026 | projected / innings 1 |   0.244 | 0.66  |     0.689 | 0.111 |
|      2026 | projected / innings 2 |   0.129 | 0.913 |     0.398 | 0.074 |
|      2026 | direct / innings 1    |   0.243 | 0.665 |     0.688 | 0.112 |
|      2026 | direct / innings 2    |   0.129 | 0.913 |     0.398 | 0.074 |

All rows are strictly out-of-time (fit on earlier seasons only). The projected first-innings framework beats a direct classifier on 2025 (Brier, log-loss, ECE) and is used. The 2026 first-innings calibration drift (ECE) reflects 2026's higher scoring environment — a known limitation (Section 20).

**Two corrections found by inspecting the leverage surface.** (i) Current run rate was removed as a model feature (it is a function of score and balls and swings wildly in the first overs); (ii) monotone constraints were added on balls bowled and on the projected total's spread, after a single in the powerplay was seen to move win probability by about ±0.018 with a mean near zero — a jagged-surface artefact.


## 6. Pressure definition

We built and compared two pressure definitions; both use only information known before the ball is bowled.


### 6.1 Composite pressure index (used by the selected model)


    2nd innings: PI = 0.50·σ((RRR−8.5)/1.5) + 0.25·death + 0.15·wkts_lost/10 + 0.10·σ(·)·death


    1st innings: PI = 0.25·death + 0.40·wkts_lost/10 + 0.35·min(wkts in previous 18 balls / 3, 1)

A delivery is **high pressure** when PI ≥ the 75th percentile of the 2022–24 (training) distribution — 48,624 deliveries. It captures what the brief names explicitly: required rate, wickets in hand, balls remaining and stage (death overs, collapses). Its weights are hand-set; this is its acknowledged weakness.


### 6.2 Win-probability leverage (research-PDF definition)


    LI_t = Σ_o p(o | S_t) · | WP(S_t ⊕ o) − WP(S_t) |      P_t = F_train(LI_t)  ∈ [0, 1]

o ranges over nine outcomes (0,1,2,3,4,6, wicket, wide, no-ball) from a multinomial gradient-boosting model. Leverage is the expected absolute swing in win probability; P is its percentile within 2022–24 only (mean training leverage 0.0158). Leverage is near zero when a match is decided and peaks in close finishes — it addresses the composite index's habit of treating hopeless chases as maximal pressure. 45,277 deliveries are top-quartile by leverage.


![01_pressure_distribution.png](../figures/01_pressure_distribution.png)


![02_leverage_distribution.png](../figures/02_leverage_distribution.png)


## 7. Expected performance


    xR_t = E[Runs_t | S_t]        xW_t = P(Wicket_t = 1 | S_t)

Four gradient-boosting models fitted only on training seasons: runs off the bat and striker-dismissal probability (deliveries faced), runs charged to the bowler and bowler-credited wicket probability (all deliveries). They describe an _average_ player in the identical state.

|   holdout | model                     | variant          |      mae |     rmse |   logloss |   calib_ratio |
|----------:|:--------------------------|:-----------------|---------:|---------:|----------:|--------------:|
|      2025 | Outcome model (9 classes) | state model      | nan      | nan      |    1.6193 |      nan      |
|      2025 | Outcome model (9 classes) | train base rates | nan      | nan      |    1.6573 |      nan      |
|      2025 | xR batter                 | state model      |   1.31   |   1.7414 |  nan      |        0.9836 |
|      2025 | xR batter                 | train mean       |   1.3275 |   1.7591 |  nan      |      nan      |
|      2025 | xW batter                 | state model      | nan      | nan      |    0.2116 |        0.9701 |
|      2025 | xW batter                 | train mean       | nan      | nan      |    0.2148 |      nan      |
|      2025 | xR bowler                 | state model      |   1.2804 |   1.7177 |  nan      |        0.9845 |
|      2025 | xR bowler                 | train mean       |   1.2947 |   1.7337 |  nan      |      nan      |
|      2025 | xW bowler                 | state model      | nan      | nan      |    0.2027 |        0.9647 |
|      2025 | xW bowler                 | train mean       | nan      | nan      |    0.205  |      nan      |
|      2026 | Outcome model (9 classes) | state model      | nan      | nan      |    1.6405 |      nan      |
|      2026 | Outcome model (9 classes) | train base rates | nan      | nan      |    1.68   |      nan      |
|      2026 | xR batter                 | state model      |   1.3801 |   1.806  |  nan      |        0.9551 |
|      2026 | xR batter                 | train mean       |   1.3788 |   1.8269 |  nan      |      nan      |
|      2026 | xW batter                 | state model      | nan      | nan      |    0.209  |        0.9951 |
|      2026 | xW batter                 | train mean       | nan      | nan      |    0.2121 |      nan      |
|      2026 | xR bowler                 | state model      |   1.3458 |   1.7782 |  nan      |        0.9589 |
|      2026 | xR bowler                 | train mean       |   1.3405 |   1.7968 |  nan      |      nan      |
|      2026 | xW bowler                 | state model      | nan      | nan      |    0.1993 |        0.9961 |
|      2026 | xW bowler                 | train mean       | nan      | nan      |    0.2015 |      nan      |


![14_expected_vs_actual.png](../figures/14_expected_vs_actual.png)

Expected vs actual by decile of the leave-one-season-out prediction.


## 8. Contextual value (VAA)


    Batter:  VAA_t = Runs_t − xR_t − WV_t·(Out_t − xW_t)


    Bowler:  VAA_t = (xR_t − Conceded_t) + WV_t·(Wicket_t − xW_t)

Higher is better for both roles; automated tests verify the signs and that VAA averages zero when outcomes match expectations. The baseline model uses the context-free analogue: Runs − WV·Out (batters) and WV·Wicket − Conceded (bowlers).


![03_pressure_vs_vaa.png](../figures/03_pressure_vs_vaa.png)


## 9. Wicket value


    WV_t = E[FutureRuns | S_t, no wicket] − E[FutureRuns | S_t, wicket]

Estimated with a monotone remaining-runs model evaluated at the two post-delivery states (training R² 0.84 out of time). Training means: fixed 5.98 runs; powerplay 8.64, middle 5.64, death 2.18.

|   holdout | variant   |   rmse |    mae |   bias |
|----------:|:----------|-------:|-------:|-------:|
|      2025 | A fixed   | 20.988 | 14.557 | -1.283 |
|      2025 | B phase   | 20.778 | 14.28  | -1.874 |
|      2025 | C state   | 20.625 | 13.65  | -1.914 |
|      2026 | A fixed   | 23.177 | 15.838 | -0.449 |
|      2026 | B phase   | 23.139 | 15.682 | -0.82  |
|      2026 | C state   | 22.977 | 15.055 | -0.522 |

Out of time, the state-dependent value predicts the runs actually lost after a wicket best (lowest RMSE/MAE in both holdout seasons), but the gain over a fixed value is small.


## 10. Empirical-Bayes shrinkage


    Mean models:  θ̂_EB = (θ̂/V + μ/τ²) / (1/V + 1/τ²),   τ² = max(Var(θ̂) − mean V, ε)


    Response models:  Σ_i = (T⁻¹ + V_i⁻¹)⁻¹,  m_i = Σ_i (T⁻¹μ + V_i⁻¹ θ̂_i),  T = Cov(θ̂) − mean V  (PSD)

Priors are estimated by method of moments on eligible players. V is the match-clustered variance, so a player with 30 pressure balls is pulled strongly to the average while one with 500 consistent balls keeps most of his estimate. Every ranked player carries a posterior SE and a 90% interval.


![07_uncertainty.png](../figures/07_uncertainty.png)


## 11. Candidate models and player pressure response

**M0** raw outcome + composite pressure (top quartile) + EB · **M1** + contextual VAA · **M2** + continuous leverage weights · **M3** + per-player pressure response · **M4** + state-dependent wicket value. Each rung adds exactly one idea, so validation shows what each idea buys.


    VAA_it = α_i + β_i·(P_t − 0.5) + ε_it

Per-player least squares with match-clustered (sandwich) covariance, floored at the homoskedastic variance so that a player whose pressure balls sit in one match cannot appear spuriously precise. α is performance at median pressure; β is pressure elasticity; model lift = β·(0.9 − 0.3). Observed lift = mean VAA at P ≥ 0.9 minus P ≤ 0.3.


![04_response_curves.png](../figures/04_response_curves.png)


![15_pressure_lift.png](../figures/15_pressure_lift.png)

Raw lifts spread widely; after empirical-Bayes shrinkage the batting slopes collapse to almost zero — evidence that individual 'clutch response' is mostly noise in this data.


## 12. Model comparison

| model   | name                       | pressure                         | contextual_vaa   | pressure_response   | wicket_value   |   valid_mean_spearman |   test_mean_spearman | selected   |
|:--------|:---------------------------|:---------------------------------|:-----------------|:--------------------|:---------------|----------------------:|---------------------:|:-----------|
| M0      | PE-EB baseline             | Composite index (top 25%)        | False            | False               | fixed          |                 0.148 |                0.07  | True       |
| M1      | Contextual VAA + EB        | Composite index (top 25%)        | True             | False               | fixed          |                 0.124 |                0.008 | False      |
| M2      | Leverage-weighted VAA + EB | Leverage (continuous weight)     | True             | False               | fixed          |                 0.043 |                0.149 | False      |
| M3      | Pressure response + EB     | Leverage percentile (continuous) | True             | True                | fixed          |                 0.07  |                0.132 | False      |
| M4      | Full: response + state WV  | Leverage percentile (continuous) | True             | True                | state          |                 0.077 |                0.098 | False      |

**Ablation ladder.** M0 raw outcome + composite pressure + EB; M1 adds contextual VAA; M2 replaces the composite indicator with continuous leverage weights; M3 models the player's pressure response; M4 adds the state-dependent wicket value.

**Validation design.** Each model scores players on 2022–24; the target is the player's realised 2025 pressure performance computed with expectations fitted on 2022–24 only. Two targets prevent a model from grading itself: leverage-weighted contextual VAA (favours M2–M4's view) and raw composite-pressure outcome (favours M0's view). Metric: Spearman correlation among players eligible in training with ≥60 holdout balls; uncertainty from 1,000 resamples of 2025 matches.

| candidate   | versus   |   mean_improvement |   ci_lo |   ci_hi |   p_better | accepted   |
|:------------|:---------|-------------------:|--------:|--------:|-----------:|:-----------|
| M1          | M0       |             -0.024 |  -0.035 |   0.023 |      0.371 | False      |
| M2          | M0       |             -0.105 |  -0.145 |   0.024 |      0.119 | False      |
| M3          | M0       |             -0.079 |  -0.115 |   0.033 |      0.186 | False      |
| M4          | M0       |             -0.071 |  -0.105 |   0.034 |      0.2   | False      |


![09_model_comparison.png](../figures/09_model_comparison.png)

**Why M0 won.** It has the highest mean validation Spearman and no richer model's improvement over it has a 90% interval above zero. The richer models do better on the context-adjusted bowling target (e.g. M3 0.169 vs M0 0.065) but worse on raw batting outcomes, and the pre-registered rule averages both. Because the evidence is close, we report the context-aware Top 10s alongside (dashboard 'Model Comparison') and quantify how much they overlap (Section 15.1).


## 13. Temporal validation and leakage control

**Split:** train 2022–24 · validate 2025 (selection only) · test 2026 (run once after selection). **Within the training window**, expectations are 3-fold cross-fitted by match, so no delivery is scored by a model that saw its own match. **Holdout seasons** are scored by models fitted on earlier seasons only. Pressure percentiles, leverage normalisation, composite threshold and fixed/phase wicket values are calibrated on the training window. The final descriptive ranking (2022–26) uses leave-one-season-out expectations with pressure still calibrated on 2022–24. A unit test asserts the forward fold never fits on 2025 or 2026.

| fold            | role   | target   |   spearman |   n_players |   top20_overlap |
|:----------------|:-------|:---------|-----------:|------------:|----------------:|
| Validation 2025 | bat    | T_lev    |      0.144 |          67 |               6 |
| Validation 2025 | bat    | T_raw    |      0.163 |          65 |               6 |
| Validation 2025 | bowl   | T_lev    |      0.065 |          85 |               5 |
| Validation 2025 | bowl   | T_raw    |      0.221 |          85 |               7 |
| Test 2026       | bat    | T_lev    |     -0.124 |          76 |               3 |
| Test 2026       | bat    | T_raw    |      0.128 |          73 |               5 |
| Test 2026       | bowl   | T_lev    |      0.275 |          94 |               4 |
| Test 2026       | bowl   | T_raw    |      0.001 |          93 |               6 |


## 14. Bootstrap reliability

1,000 resamples of **matches** (never deliveries), stratified by competition × season. Each replicate re-runs the whole player-level estimation — clustered variances, empirical-Bayes priors, eligibility and ranking. Matches drawn twice count twice toward eligibility. We report mean/median rank, rank SD, 80% rank interval and Top-3/5/10 probabilities, plus pairwise 'beats' probabilities (outputs/pairwise_beat_*.csv).


![08_bootstrap_ranks.png](../figures/08_bootstrap_ranks.png)


## 15. Fairness and sensitivity

**Eligibility** (fixed before ranking): batters ≥250 balls faced, ≥60 high-pressure balls, ≥12 matches; bowlers ≥300 balls, ≥60 high-pressure balls, ≥12 matches. High-pressure balls are counted under the selected model's own pressure definition.

**IPL vs T20I.** Competition and the IPL Impact-Player era are features of every expectation model, so a T20I ball is judged against T20I par. The baseline outcome is not context-adjusted, so we test IPL-only, T20I-only and equal-weight variants (Section 15.1). **Context:** phase, wickets, target and resources enter every expectation; spinners and pacers are compared through the same state-conditioned par.


![10_ipl_t20i_sensitivity.png](../figures/10_ipl_t20i_sensitivity.png)


![11_eligibility_sensitivity.png](../figures/11_eligibility_sensitivity.png)


### 15.1 Sensitivity of the Top 10

| Role    | Group               | Variant                                |   Top-10 overlap (of 10) |   Eligible players |
|:--------|:--------------------|:---------------------------------------|-------------------------:|-------------------:|
| Batters | Eligibility         | x0.5 thresholds (125/30/6)             |                        9 |                216 |
| Batters | Eligibility         | x0.75 thresholds (187/45/9)            |                        9 |                166 |
| Batters | Eligibility         | x1.0 thresholds (250/60/12)            |                       10 |                131 |
| Batters | Eligibility         | x1.5 thresholds (375/90/18)            |                        9 |                 94 |
| Batters | Eligibility         | x2.0 thresholds (500/120/24)           |                        9 |                 73 |
| Batters | Competition         | IPL only (thresholds x0.6)             |                        9 |                 98 |
| Batters | Competition         | T20I only (thresholds x0.6)            |                        2 |                128 |
| Batters | Competition         | Equal IPL/T20I weighting               |                       10 |                131 |
| Batters | Wicket value        | fixed wicket value                     |                       10 |                131 |
| Batters | Wicket value        | phase wicket value                     |                        7 |                131 |
| Batters | Wicket value        | state wicket value                     |                        7 |                131 |
| Batters | Pressure definition | composite index, top 33% of balls      |                        7 |                159 |
| Batters | Pressure definition | composite index, top 25% of balls      |                       10 |                131 |
| Batters | Pressure definition | composite index, top 20% of balls      |                        8 |                108 |
| Batters | Pressure definition | leverage top 25% (same outcome metric) |                        1 |                162 |
| Batters | Pressure definition | leverage-weighted, continuous          |                        2 |                162 |
| Batters | Model               | M0 PE-EB baseline                      |                       10 |                131 |
| Batters | Model               | M1 Contextual VAA + EB                 |                        7 |                131 |
| Batters | Model               | M2 Leverage-weighted VAA + EB          |                        3 |                162 |
| Batters | Model               | M3 Pressure response + EB              |                        3 |                162 |
| Batters | Model               | M4 Full: response + state WV           |                        2 |                162 |
| Batters | Era                 | 2024-2026 only (thresholds x0.6)       |                        6 |                138 |
| Bowlers | Eligibility         | x0.5 thresholds (150/30/6)             |                        9 |                240 |
| Bowlers | Eligibility         | x0.75 thresholds (225/45/9)            |                        9 |                198 |
| Bowlers | Eligibility         | x1.0 thresholds (300/60/12)            |                       10 |                172 |
| Bowlers | Eligibility         | x1.5 thresholds (450/90/18)            |                        9 |                129 |
| Bowlers | Eligibility         | x2.0 thresholds (600/120/24)           |                        9 |                 98 |
| Bowlers | Competition         | IPL only (thresholds x0.6)             |                        4 |                117 |
| Bowlers | Competition         | T20I only (thresholds x0.6)            |                        6 |                157 |
| Bowlers | Competition         | Equal IPL/T20I weighting               |                        9 |                170 |
| Bowlers | Wicket value        | fixed wicket value                     |                       10 |                172 |
| Bowlers | Wicket value        | phase wicket value                     |                        6 |                172 |
| Bowlers | Wicket value        | state wicket value                     |                        6 |                172 |
| Bowlers | Pressure definition | composite index, top 33% of balls      |                        8 |                178 |
| Bowlers | Pressure definition | composite index, top 25% of balls      |                       10 |                172 |
| Bowlers | Pressure definition | composite index, top 20% of balls      |                        8 |                160 |
| Bowlers | Pressure definition | leverage top 25% (same outcome metric) |                        2 |                173 |
| Bowlers | Pressure definition | leverage-weighted, continuous          |                        2 |                173 |
| Bowlers | Model               | M0 PE-EB baseline                      |                       10 |                172 |
| Bowlers | Model               | M1 Contextual VAA + EB                 |                        7 |                172 |
| Bowlers | Model               | M2 Leverage-weighted VAA + EB          |                        2 |                173 |
| Bowlers | Model               | M3 Pressure response + EB              |                        4 |                173 |
| Bowlers | Model               | M4 Full: response + state WV           |                        4 |                173 |
| Bowlers | Era                 | 2024-2026 only (thresholds x0.6)       |                        8 |                166 |

**Reading.** The lists are robust to eligibility thresholds and to equal IPL/T20I weighting, and largely robust to the composite cut-off and wicket-value choice. They are **not** robust to replacing the composite pressure definition with leverage (overlap 1 batters / 2 bowlers): the two definitions describe different situations — chase/death/collapse pressure versus win-probability volatility, which is high in first-innings powerplays. The T20I-only batting list differs strongly, showing that the batting ranking leans on IPL evidence.


![12_season_stability.png](../figures/12_season_stability.png)


![13_match_consistency.png](../figures/13_match_consistency.png)


## 16. Top 10 batters

Score = net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter; empirical-Bayes shrunk; 90% interval in brackets. Expected/actual are runs per 100 high-pressure balls (scored); IPL/T20I = balls in each competition.

|   # | Player           | Score [90%]          |   Pressure VAA | Exp / Act   |   Lift |   HP balls | P(Top10)   | Median rank (80%)   | IPL / T20I balls   |
|----:|:-----------------|:---------------------|---------------:|:------------|-------:|-----------:|:-----------|:--------------------|:-------------------|
|   1 | Tim David        | +26.6 [+12.2, +41.1] |           52   | 162 / 194   |   -1.8 |        588 | 89%        | 3 (1-11)            | 647 / 579          |
|   2 | Shashank Singh   | +20.4 [+3.3, +37.4]  |           54.4 | 159 / 187   |   18   |        322 | 76%        | 4 (1-20)            | 560 / 0            |
|   3 | Suryakumar Yadav | +19.8 [+5.9, +33.7]  |           36.3 | 155 / 180   |  -19.3 |        543 | 58%        | 9 (3-25)            | 1358 / 1401        |
|   4 | Tristan Stubbs   | +15.5 [-0.5, +31.5]  |           35.8 | 160 / 181   |  -19.2 |        454 | 47%        | 11 (3-34)           | 648 / 433          |
|   5 | Naman Dhir       | +15.0 [-2.7, +32.6]  |           45.9 | 160 / 199   |   26   |        212 | 53%        | 9 (2-37)            | 433 / 82           |
|   6 | Rinku Singh      | +12.6 [-2.2, +27.4]  |           30   | 160 / 171   |   29.6 |        545 | 25%        | 19 (6-47)           | 879 / 405          |
|   7 | Nicholas Pooran  | +11.5 [-3.8, +26.8]  |           23.7 | 161 / 178   |   -9.6 |        436 | 22%        | 21 (7-53)           | 1149 / 823         |
|   8 | Dhruv Jurel      | +11.0 [-5.6, +27.6]  |           23.3 | 166 / 179   |   -2.3 |        312 | 28%        | 20 (5-54)           | 775 / 23           |
|   9 | Faf du Plessis   | +10.9 [-9.1, +30.9]  |           39   | 169 / 212   |  -45.7 |         81 | 41%        | 16 (2-260)          | 1276 / 0           |
|  10 | Jos Buttler      | +9.8 [-6.9, +26.6]   |           18.7 | 167 / 183   |  -16   |        251 | 19%        | 24 (7-63)           | 1791 / 1038        |


![05_top10_batters.png](../figures/05_top10_batters.png)


## 17. Top 10 bowlers

Score = net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler; empirical-Bayes shrunk; 90% interval in brackets. Expected/actual are runs per 100 high-pressure balls (conceded); IPL/T20I = balls in each competition.

|   # | Player              | Score [90%]         |   Pressure VAA | Exp / Act   |   Lift |   HP balls | P(Top10)   | Median rank (80%)   | IPL / T20I balls   |
|----:|:--------------------|:--------------------|---------------:|:------------|-------:|-----------:|:-----------|:--------------------|:-------------------|
|   1 | Mohammad Nawaz      | +24.0 [+6.2, +41.8] |           66.5 | 122 / 87    |  -18.7 |        166 | 92%        | 2 (1-8)             | 0 / 1017           |
|   2 | Sunil Narine        | +19.7 [+4.6, +34.8] |           45.5 | 141 / 98    |   25.4 |        322 | 61%        | 9 (3-20)            | 1553 / 0           |
|   3 | Abrar Ahmed         | +18.9 [+1.7, +36.2] |           37.2 | 122 / 92    |  -15   |        208 | 75%        | 5 (1-19)            | 0 / 776            |
|   4 | Mitchell Santner    | +16.9 [+2.5, +31.2] |           22.7 | 132 / 116   |   15.2 |        406 | 33%        | 15 (5-43)           | 574 / 1219         |
|   5 | Shaheen Afridi      | +15.8 [-0.2, +31.9] |           33   | 146 / 122   |  -14.8 |        278 | 44%        | 12 (4-34)           | 0 / 1036           |
|   6 | Matheesha Pathirana | +15.7 [+2.5, +29.0] |           33.8 | 154 / 124   |  -15   |        560 | 24%        | 18 (7-44)           | 797 / 482          |
|   7 | Jasprit Bumrah      | +15.4 [-0.4, +31.2] |           39.9 | 150 / 112   |  -39.2 |        557 | 39%        | 13 (4-37)           | 1262 / 518         |
|   8 | Matthew Humphreys   | +13.5 [-5.9, +32.9] |           72.9 | 124 / 90    |  -50.7 |         63 | 33%        | 266 (2-266)         | 0 / 315            |
|   9 | Taskin Ahmed        | +12.8 [-4.3, +29.9] |           24.8 | 137 / 130   |   46.6 |        263 | 37%        | 15 (4-44)           | 0 / 1083           |
|  10 | Mustafizur Rahman   | +12.5 [-2.0, +27.0] |           25.1 | 149 / 121   |  -21.4 |        533 | 17%        | 24 (8-56)           | 532 / 1114         |


![06_top10_bowlers.png](../figures/06_top10_bowlers.png)


## 18. Why each player was selected

Every sentence below is generated from the player's computed metrics (outputs/why_this_player.json); the key match comes from the Data Pack. Public video/report links (Section 18.3) are qualitative context only and were consulted after the ranking was final.


### 18.1 Batters

**Why #1 — Tim David** (Finisher, Australia, Mumbai Indians, Royal Challengers Bengaluru).<br/>• Highest shrunk pressure score among 131 eligible players (+26.6)<br/>• Under pressure scored 194 runs/100 balls vs 162 expected (+32)<br/>• 588 high-pressure balls in 98 matches (more than 97% of eligible players)<br/>• Top-10 in 89% of 1,000 match bootstraps; median rank 3 (80% interval 1–11)<br/>• Positive pressure value in 85% of 39 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +26.6 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval +12.2 to +41.1). Context-adjusted VAA on his 588 high-pressure balls: +52.0 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +38.1; overall +34.8 across 1226 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope +0.97 per unit of pressure (raw -3.2): performance improves as leverage rises. Model-implied lift f(0.9)-f(0.3) = +0.58.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -1.8 runs per 100 balls (182 vs 455 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 98 matches, 588 high-pressure balls; posterior SE 8.8. Empirical-Bayes shrinkage kept 58% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 85% of 39 matches with 6+ high-pressure balls; best single match = 8% of positive pressure value; without his best match the score is +24.9.<br/>_Model agreement:_ Top-20 in 5 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 89%, Top-5 in 74%, median rank 3 (80% rank interval 1-11).<br/>_Cricket interpretation:_ A finisher who strikes at 194 in high-leverage moments (overall 178) and is dismissed once every 24 high-pressure balls with evidence from both IPL (647 balls) and T20Is (579). In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2025-04-18, IPL 2025 v Punjab Kings: 50 off 26 balls (not out). 1st innings: 33/5 after 6.1 ov; side's win prob. 10%. Team result: Lost.

**Why #2 — Shashank Singh** (Finisher, Punjab Kings, Sunrisers Hyderabad).<br/>• Shrunk pressure score +20.4 — rank 2 of 131 eligible<br/>• Under pressure scored 187 runs/100 balls vs 159 expected (+28)<br/>• 322 high-pressure balls in 42 matches (more than 80% of eligible players)<br/>• Top-10 in 76% of 1,000 match bootstraps; median rank 4 (80% interval 1–20)<br/>• Positive pressure value in 76% of 21 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +20.4 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval +3.3 to +37.4). Context-adjusted VAA on his 322 high-pressure balls: +54.4 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +25.8; overall +21.1 across 560 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.39 per unit of pressure (raw +6.7): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.24.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = +18.0 runs per 100 balls (62 vs 238 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 42 matches, 322 high-pressure balls; posterior SE 10.3. Empirical-Bayes shrinkage kept 43% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 76% of 21 matches with 6+ high-pressure balls; best single match = 14% of positive pressure value; without his best match the score is +16.9.<br/>_Model agreement:_ Top-20 in 3 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 76%, Top-5 in 61%, median rank 4 (80% rank interval 1-20).<br/>_Cricket interpretation:_ A finisher who strikes at 187 in high-leverage moments (overall 162) and is dismissed once every 36 high-pressure balls. In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2024-04-04, IPL 2024 v Gujarat Titans: 61 off 29 balls (not out). Chasing 200: 70/4 after 8.4 ov, needed 130 off 68 (RRR 11.5); side's win prob. 17%. Team result: Won.

**Why #3 — Suryakumar Yadav** (Middle-order batter, India, Mumbai Indians).<br/>• Shrunk pressure score +19.8 — rank 3 of 131 eligible<br/>• Under pressure scored 180 runs/100 balls vs 155 expected (+25)<br/>• 543 high-pressure balls in 139 matches (more than 94% of eligible players)<br/>• Top-10 in 58% of 1,000 match bootstraps; median rank 9 (80% interval 3–25)<br/>• Positive pressure value in 62% of 26 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +19.8 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval +5.9 to +33.7). Context-adjusted VAA on his 543 high-pressure balls: +36.3 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +31.2; overall +27.8 across 2759 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope +1.10 per unit of pressure (raw -11.3): performance improves as leverage rises. Model-implied lift f(0.9)-f(0.3) = +0.66.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -19.3 runs per 100 balls (214 vs 889 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 139 matches, 543 high-pressure balls; posterior SE 8.4. Empirical-Bayes shrinkage kept 62% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 62% of 26 matches with 6+ high-pressure balls; best single match = 22% of positive pressure value; without his best match the score is +16.2.<br/>_Model agreement:_ Top-20 in 5 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 58%, Top-5 in 29%, median rank 9 (80% rank interval 3-25).<br/>_Cricket interpretation:_ A middle-order batter who strikes at 180 in high-leverage moments (overall 166) and is dismissed once every 24 high-pressure balls with evidence from both IPL (1358 balls) and T20Is (1401). In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2022-07-10, India in England T20I Series 2022 v England: 117 off 55 balls (dismissed). Chasing 216: 13/2 after 2.4 ov, needed 203 off 104 (RRR 11.7); side's win prob. 8%. Team result: Lost.

**Why #4 — Tristan Stubbs** (Middle-order batter, Delhi Capitals, South Africa, Mumbai Indians).<br/>• Shrunk pressure score +15.5 — rank 4 of 131 eligible<br/>• Under pressure scored 181 runs/100 balls vs 160 expected (+20)<br/>• 454 high-pressure balls in 69 matches (more than 89% of eligible players)<br/>• Top-10 in 47% of 1,000 match bootstraps; median rank 11 (80% interval 3–34)<br/>• Positive pressure value in 83% of 24 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +15.5 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval -0.5 to +31.5). Context-adjusted VAA on his 454 high-pressure balls: +35.8 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +11.7; overall +14.2 across 1081 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.37 per unit of pressure (raw -0.9): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.22.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -19.2 runs per 100 balls (137 vs 501 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 69 matches, 454 high-pressure balls; posterior SE 9.7. Empirical-Bayes shrinkage kept 49% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 83% of 24 matches with 6+ high-pressure balls; best single match = 17% of positive pressure value; without his best match the score is +12.8.<br/>_Model agreement:_ Top-20 in 2 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 47%, Top-5 in 24%, median rank 11 (80% rank interval 3-34).<br/>_Cricket interpretation:_ A middle-order batter who strikes at 181 in high-leverage moments (overall 149) and is dismissed once every 23 high-pressure balls with evidence from both IPL (648 balls) and T20Is (433). In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2024-04-07, IPL 2024 v Mumbai Indians: 71 off 25 balls (not out). Chasing 235: 110/2 after 11.5 ov, needed 125 off 49 (RRR 15.3); side's win prob. 2%. Team result: Lost.

**Why #5 — Naman Dhir** (Middle-order batter, Mumbai Indians, India).<br/>• Shrunk pressure score +15.0 — rank 5 of 131 eligible<br/>• Under pressure scored 199 runs/100 balls vs 160 expected (+39)<br/>• 212 high-pressure balls in 42 matches (more than 58% of eligible players)<br/>• Top-10 in 53% of 1,000 match bootstraps; median rank 10 (80% interval 2–37)<br/>• Positive pressure value in 65% of 17 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +15.0 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval -2.7 to +32.6). Context-adjusted VAA on his 212 high-pressure balls: +45.9 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +30.9; overall +20.3 across 515 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.32 per unit of pressure (raw -21.7): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.19.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = +26.0 runs per 100 balls (43 vs 169 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 42 matches, 212 high-pressure balls; posterior SE 10.7. Empirical-Bayes shrinkage kept 38% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 65% of 17 matches with 6+ high-pressure balls; best single match = 22% of positive pressure value; without his best match the score is +10.8.<br/>_Model agreement:_ Top-20 in 4 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 53%, Top-5 in 33%, median rank 10 (80% rank interval 2-37).<br/>_Cricket interpretation:_ A middle-order batter who strikes at 199 in high-leverage moments (overall 172) and is dismissed once every 16 high-pressure balls with evidence from both IPL (433 balls) and T20Is (82). In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2024-05-17, IPL 2024 v Lucknow Super Giants: 62 off 28 balls (not out). Chasing 215: 120/5 after 14.2 ov, needed 95 off 34 (RRR 16.8); side's win prob. 1%. Team result: Lost.

**Why #6 — Rinku Singh** (Finisher, Kolkata Knight Riders, India).<br/>• Shrunk pressure score +12.6 — rank 6 of 131 eligible<br/>• Under pressure scored 171 runs/100 balls vs 160 expected (+11)<br/>• 545 high-pressure balls in 83 matches (more than 95% of eligible players)<br/>• Top-10 in 25% of 1,000 match bootstraps; median rank 19 (80% interval 6–47)<br/>• Positive pressure value in 66% of 35 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +12.6 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval -2.2 to +27.4). Context-adjusted VAA on his 545 high-pressure balls: +30.0 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +14.7; overall +17.7 across 1284 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.01 per unit of pressure (raw +5.0): performance stays essentially flat as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.01.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = +29.6 runs per 100 balls (84 vs 513 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 83 matches, 545 high-pressure balls; posterior SE 9.0. Empirical-Bayes shrinkage kept 56% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 66% of 35 matches with 6+ high-pressure balls; best single match = 11% of positive pressure value; without his best match the score is +10.7.<br/>_Model agreement:_ Top-20 in 4 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 25%, Top-5 in 9%, median rank 19 (80% rank interval 6-47).<br/>_Cricket interpretation:_ A finisher who strikes at 171 in high-leverage moments (overall 152) and is dismissed once every 23 high-pressure balls with evidence from both IPL (879 balls) and T20Is (405). In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2023-05-20, IPL 2023 v Lucknow Super Giants: 67 off 33 balls (not out). Chasing 177: 108/4 after 14.0 ov, needed 69 off 36 (RRR 11.5); side's win prob. 45%. Team result: Lost.

**Why #7 — Nicholas Pooran** (Middle-order batter, Lucknow Super Giants, West Indies, Sunrisers Hyderabad).<br/>• Shrunk pressure score +11.5 — rank 7 of 131 eligible<br/>• Under pressure scored 178 runs/100 balls vs 161 expected (+17)<br/>• 436 high-pressure balls in 115 matches (more than 88% of eligible players)<br/>• Top-10 in 22% of 1,000 match bootstraps; median rank 21 (80% interval 7–53)<br/>• Positive pressure value in 73% of 26 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +11.5 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval -3.8 to +26.8). Context-adjusted VAA on his 436 high-pressure balls: +23.7 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +15.3; overall +16.5 across 1972 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.07 per unit of pressure (raw +6.6): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.04.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -9.6 runs per 100 balls (183 vs 576 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 115 matches, 436 high-pressure balls; posterior SE 9.3. Empirical-Bayes shrinkage kept 53% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 73% of 26 matches with 6+ high-pressure balls; best single match = 13% of positive pressure value; without his best match the score is +8.8.<br/>_Model agreement:_ Top-20 in 4 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 22%, Top-5 in 8%, median rank 21 (80% rank interval 7-53).<br/>_Cricket interpretation:_ A middle-order batter who strikes at 178 in high-leverage moments (overall 157) and is dismissed once every 17 high-pressure balls with evidence from both IPL (1149 balls) and T20Is (823). In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2022-05-01, IPL 2022 v Chennai Super Kings: 64 off 33 balls (not out). Chasing 203: 88/3 after 9.5 ov, needed 115 off 61 (RRR 11.3); side's win prob. 34%. Team result: Lost.

**Why #8 — Dhruv Jurel** (Middle-order batter, Rajasthan Royals, India).<br/>• Shrunk pressure score +11.0 — rank 8 of 131 eligible<br/>• Under pressure scored 179 runs/100 balls vs 166 expected (+13)<br/>• 312 high-pressure balls in 54 matches (more than 79% of eligible players)<br/>• Top-10 in 28% of 1,000 match bootstraps; median rank 20 (80% interval 5–54)<br/>• Positive pressure value in 72% of 18 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +11.0 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval -5.6 to +27.6). Context-adjusted VAA on his 312 high-pressure balls: +23.3 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +1.7; overall +1.5 across 798 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -1.29 per unit of pressure (raw -7.0): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.77.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -2.3 runs per 100 balls (159 vs 243 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 54 matches, 312 high-pressure balls; posterior SE 10.1. Empirical-Bayes shrinkage kept 45% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 72% of 18 matches with 6+ high-pressure balls; best single match = 17% of positive pressure value; without his best match the score is +7.9.<br/>_Model agreement:_ Top-20 in 2 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 28%, Top-5 in 12%, median rank 20 (80% rank interval 5-54).<br/>_Cricket interpretation:_ A middle-order batter who strikes at 179 in high-leverage moments (overall 151) and is dismissed once every 18 high-pressure balls with evidence from both IPL (775 balls) and T20Is (23). In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2024-05-24, IPL 2024 v Sunrisers Hyderabad: 56 off 35 balls (not out). Chasing 176: 82/5 after 12.3 ov, needed 94 off 45 (RRR 12.5); side's win prob. 13%. Team result: Lost.

**Why #9 — Faf du Plessis** (Top-order batter, Royal Challengers Bengaluru, Delhi Capitals). <font color='#C2412D'>**Caution:** eligible in only 70% of bootstrap resamples — the least robust pick in this list.</font><br/>• Shrunk pressure score +10.9 — rank 9 of 131 eligible<br/>• Under pressure scored 212 runs/100 balls vs 169 expected (+44)<br/>• 81 high-pressure balls in 54 matches (more than 11% of eligible players)<br/>• Top-10 in 41% of 1,000 match bootstraps; median rank 16 (80% interval 2–260)<br/>• Positive pressure value in 80% of 5 pressure-heavy matches<br/>• Caution: meets eligibility in only 70% of resamples (marginal sample)<br/>_Pressure performance:_ Shrunk pressure score +10.9 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval -9.1 to +30.9). Context-adjusted VAA on his 81 high-pressure balls: +39.0 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +11.5; overall -5.5 across 1276 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -2.32 per unit of pressure (raw -62.7): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -1.39.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -45.7 runs per 100 balls (74 vs 229 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 54 matches, 81 high-pressure balls; posterior SE 12.2. Empirical-Bayes shrinkage kept 21% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 80% of 5 matches with 6+ high-pressure balls; best single match = 45% of positive pressure value; without his best match the score is +4.8.<br/>_Model agreement:_ Top-20 in 1 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 41%, Top-5 in 27%, median rank 16 (80% rank interval 2-260).<br/>_Cricket interpretation:_ A top-order batter who strikes at 212 in high-leverage moments (overall 144) and is dismissed once every 16 high-pressure balls. In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2023-04-17, IPL 2023 v Chennai Super Kings: 62 off 33 balls (dismissed). Chasing 227: 15/2 after 2.0 ov, needed 212 off 108 (RRR 11.8); side's win prob. 5%. Team result: Lost.

**Why #10 — Jos Buttler** (Top-order batter, Rajasthan Royals, England, Gujarat Titans).<br/>• Shrunk pressure score +9.8 — rank 10 of 131 eligible<br/>• Under pressure scored 183 runs/100 balls vs 167 expected (+16)<br/>• 251 high-pressure balls in 121 matches (more than 69% of eligible players)<br/>• Top-10 in 19% of 1,000 match bootstraps; median rank 24 (80% interval 7–63)<br/>• Positive pressure value in 73% of 15 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +9.8 net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter (90% interval -6.9 to +26.6). Context-adjusted VAA on his 251 high-pressure balls: +18.7 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +21.7; overall +12.4 across 2829 balls faced.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.38 per unit of pressure (raw -36.6): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.23.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -16.0 runs per 100 balls (213 vs 675 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 121 matches, 251 high-pressure balls; posterior SE 10.2. Empirical-Bayes shrinkage kept 45% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 73% of 15 matches with 6+ high-pressure balls; best single match = 41% of positive pressure value; without his best match the score is +3.9.<br/>_Model agreement:_ Top-20 in 3 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 19%, Top-5 in 7%, median rank 24 (80% rank interval 7-63).<br/>_Cricket interpretation:_ A top-order batter who strikes at 183 in high-leverage moments (overall 155) and is dismissed once every 16 high-pressure balls with evidence from both IPL (1791 balls) and T20Is (1038). In context he adds value relative to the situation: the model credits runs only above what an average batter scores in the identical state.<br/>_Key pressure match (Data Pack):_ 2024-04-16, IPL 2024 v Kolkata Knight Riders: 107 off 60 balls (not out). Chasing 224: 101/4 after 9.0 ov, needed 123 off 66 (RRR 11.2); side's win prob. 18%. Team result: Won.


### 18.2 Bowlers

**Why #1 — Mohammad Nawaz** (Spin bowler, Pakistan).<br/>• Highest shrunk pressure score among 172 eligible players (+24.0)<br/>• Under pressure conceded 87 runs/100 balls vs 122 expected (-34)<br/>• 166 high-pressure balls in 52 matches (more than 43% of eligible players)<br/>• Top-10 in 92% of 1,000 match bootstraps; median rank 2 (80% interval 1–8)<br/>• Positive pressure value in 67% of 15 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +24.0 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval +6.2 to +41.8). Context-adjusted VAA on his 166 high-pressure balls: +66.5 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +34.2; overall +18.6 across 1017 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -2.28 per unit of pressure (raw -54.7): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -1.37.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -18.7 runs per 100 balls (92 vs 364 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 52 matches, 166 high-pressure balls; posterior SE 10.8. Empirical-Bayes shrinkage kept 28% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 67% of 15 matches with 6+ high-pressure balls; best single match = 24% of positive pressure value; without his best match the score is +21.0.<br/>_Model agreement:_ Top-20 in 4 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 92%, Top-5 in 84%, median rank 2 (80% rank interval 1-8).<br/>_Cricket interpretation:_ A spin bowler conceding 5.34 an over under high pressure (overall 7.12), with 19 high-pressure wickets. Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2026-02-01, Australia in Pakistan T20I Series 2026 v Australia: 5/18 from 4.0 overs. Chasing 208: 48/3 after 7.0 ov, needed 160 off 78 (RRR 12.3); side's win prob. 97%. Team result: Won.

**Why #2 — Sunil Narine** (Spin bowler, Kolkata Knight Riders).<br/>• Shrunk pressure score +19.7 — rank 2 of 172 eligible<br/>• Under pressure conceded 98 runs/100 balls vs 141 expected (-43)<br/>• 322 high-pressure balls in 67 matches (more than 77% of eligible players)<br/>• Top-10 in 61% of 1,000 match bootstraps; median rank 9 (80% interval 3–20)<br/>• Positive pressure value in 86% of 22 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +19.7 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval +4.6 to +34.8). Context-adjusted VAA on his 322 high-pressure balls: +45.5 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +24.7; overall +27.1 across 1553 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.64 per unit of pressure (raw +11.1): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.38.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = +25.4 runs per 100 balls (147 vs 537 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 67 matches, 322 high-pressure balls; posterior SE 9.2. Empirical-Bayes shrinkage kept 49% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 86% of 22 matches with 6+ high-pressure balls; best single match = 12% of positive pressure value; without his best match the score is +18.1.<br/>_Model agreement:_ Top-20 in 5 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 61%, Top-5 in 28%, median rank 9 (80% rank interval 3-20).<br/>_Cricket interpretation:_ A spin bowler conceding 6.04 an over under high pressure (overall 6.87), with 19 high-pressure wickets. Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2024-05-05, IPL 2024 v Lucknow Super Giants: 1/22 from 4.0 overs. Chasing 236: 43/1 after 4.0 ov, needed 193 off 96 (RRR 12.1); side's win prob. 94%. Team result: Won.

**Why #3 — Abrar Ahmed** (Spin bowler, Pakistan).<br/>• Shrunk pressure score +18.9 — rank 3 of 172 eligible<br/>• Under pressure conceded 92 runs/100 balls vs 122 expected (-30)<br/>• 208 high-pressure balls in 35 matches (more than 56% of eligible players)<br/>• Top-10 in 75% of 1,000 match bootstraps; median rank 5 (80% interval 1–19)<br/>• Positive pressure value in 75% of 16 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +18.9 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval +1.7 to +36.2). Context-adjusted VAA on his 208 high-pressure balls: +37.2 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +28.3; overall +21.7 across 776 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -2.19 per unit of pressure (raw -23.6): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -1.31.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -15.0 runs per 100 balls (76 vs 312 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 35 matches, 208 high-pressure balls; posterior SE 10.5. Empirical-Bayes shrinkage kept 33% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 75% of 16 matches with 6+ high-pressure balls; best single match = 28% of positive pressure value; without his best match the score is +17.5.<br/>_Model agreement:_ Top-20 in 4 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 75%, Top-5 in 52%, median rank 5 (80% rank interval 1-19).<br/>_Cricket interpretation:_ A spin bowler conceding 5.51 an over under high pressure (overall 6.68), with 16 high-pressure wickets. Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2025-09-04, United Arab Emirates T20I Tri-Series 2025 v United Arab Emirates: 4/9 from 4.0 overs. Chasing 172: 84/2 after 12.2 ov, needed 88 off 46 (RRR 11.5); side's win prob. 56%. Team result: Won.

**Why #4 — Mitchell Santner** (Spin bowler, New Zealand, Mumbai Indians, Chennai Super Kings).<br/>• Shrunk pressure score +16.9 — rank 4 of 172 eligible<br/>• Under pressure conceded 116 runs/100 balls vs 132 expected (-16)<br/>• 406 high-pressure balls in 83 matches (more than 85% of eligible players)<br/>• Top-10 in 33% of 1,000 match bootstraps; median rank 15 (80% interval 5–43)<br/>• Positive pressure value in 70% of 30 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +16.9 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval +2.5 to +31.2). Context-adjusted VAA on his 406 high-pressure balls: +22.7 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +17.1; overall +13.5 across 1793 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -1.05 per unit of pressure (raw -8.7): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.63.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = +15.2 runs per 100 balls (120 vs 735 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 83 matches, 406 high-pressure balls; posterior SE 8.7. Empirical-Bayes shrinkage kept 54% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 70% of 30 matches with 6+ high-pressure balls; best single match = 11% of positive pressure value; without his best match the score is +14.6.<br/>_Model agreement:_ Top-20 in 5 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 33%, Top-5 in 12%, median rank 15 (80% rank interval 5-43).<br/>_Cricket interpretation:_ A spin bowler conceding 7.36 an over under high pressure (overall 7.44), with 30 high-pressure wickets, across IPL (574 balls) and T20Is (1219). Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2025-05-21, IPL 2025 v Delhi Capitals: 3/11 from 4.0 overs. Chasing 181: 82/5 after 11.0 ov, needed 99 off 54 (RRR 11.0); side's win prob. 91%. Team result: Won.

**Why #5 — Shaheen Afridi** (Pace bowler, Pakistan).<br/>• Shrunk pressure score +15.8 — rank 5 of 172 eligible<br/>• Under pressure conceded 122 runs/100 balls vs 146 expected (-24)<br/>• 278 high-pressure balls in 47 matches (more than 72% of eligible players)<br/>• Top-10 in 44% of 1,000 match bootstraps; median rank 12 (80% interval 4–34)<br/>• Positive pressure value in 67% of 33 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +15.8 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval -0.2 to +31.9). Context-adjusted VAA on his 278 high-pressure balls: +33.0 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +23.3; overall +8.1 across 1036 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.93 per unit of pressure (raw -35.7): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.56.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -14.8 runs per 100 balls (117 vs 319 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 47 matches, 278 high-pressure balls; posterior SE 9.8. Empirical-Bayes shrinkage kept 42% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 67% of 33 matches with 6+ high-pressure balls; best single match = 12% of positive pressure value; without his best match the score is +14.2.<br/>_Model agreement:_ Top-20 in 3 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 44%, Top-5 in 20%, median rank 12 (80% rank interval 4-34).<br/>_Cricket interpretation:_ A pace bowler conceding 7.80 an over under high pressure (overall 8.01), with 26 high-pressure wickets. Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2024-05-25, Pakistan in England T20I Series 2024 v England: 3/36 from 4.0 overs. 1st innings: 164/5 after 18.0 ov; side's win prob. 35%. Team result: Lost.

**Why #6 — Matheesha Pathirana** (Pace bowler (death specialist), Chennai Super Kings, Sri Lanka, Kolkata Knight Riders).<br/>• Shrunk pressure score +15.7 — rank 6 of 172 eligible<br/>• Under pressure conceded 124 runs/100 balls vs 154 expected (-30)<br/>• 560 high-pressure balls in 54 matches (more than 94% of eligible players)<br/>• Top-10 in 24% of 1,000 match bootstraps; median rank 18 (80% interval 7–44)<br/>• Positive pressure value in 71% of 34 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +15.7 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval +2.5 to +29.0). Context-adjusted VAA on his 560 high-pressure balls: +33.8 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +31.8; overall +21.4 across 1279 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -2.75 per unit of pressure (raw -37.7): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -1.65.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -15.0 runs per 100 balls (128 vs 591 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 54 matches, 560 high-pressure balls; posterior SE 8.0. Empirical-Bayes shrinkage kept 61% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 71% of 34 matches with 6+ high-pressure balls; best single match = 9% of positive pressure value; without his best match the score is +13.5.<br/>_Model agreement:_ Top-20 in 4 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 24%, Top-5 in 7%, median rank 18 (80% rank interval 7-44).<br/>_Cricket interpretation:_ A pace bowler (death specialist) conceding 8.57 an over under high pressure (overall 8.83), with 44 high-pressure wickets, across IPL (797 balls) and T20Is (482). Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2023-05-20, IPL 2023 v Delhi Capitals: 2/22 from 4.0 overs. Chasing 224: 76/4 after 11.0 ov, needed 148 off 54 (RRR 16.4); side's win prob. 99%. Team result: Won.

**Why #7 — Jasprit Bumrah** (Pace bowler, Mumbai Indians, India).<br/>• Shrunk pressure score +15.4 — rank 7 of 172 eligible<br/>• Under pressure conceded 112 runs/100 balls vs 150 expected (-38)<br/>• 557 high-pressure balls in 75 matches (more than 93% of eligible players)<br/>• Top-10 in 39% of 1,000 match bootstraps; median rank 13 (80% interval 4–37)<br/>• Positive pressure value in 70% of 50 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +15.4 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval -0.4 to +31.2). Context-adjusted VAA on his 557 high-pressure balls: +39.9 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +31.6; overall +26.5 across 1780 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -2.02 per unit of pressure (raw -30.2): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -1.21.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -39.2 runs per 100 balls (180 vs 589 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 75 matches, 557 high-pressure balls; posterior SE 9.6. Empirical-Bayes shrinkage kept 44% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 70% of 50 matches with 6+ high-pressure balls; best single match = 9% of positive pressure value; without his best match the score is +13.8.<br/>_Model agreement:_ Top-20 in 5 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 39%, Top-5 in 17%, median rank 13 (80% rank interval 4-37).<br/>_Cricket interpretation:_ A pace bowler conceding 6.94 an over under high pressure (overall 7.27), with 41 high-pressure wickets, across IPL (1262 balls) and T20Is (518). Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2022-05-09, IPL 2022 v Kolkata Knight Riders: 5/10 from 4.0 overs. 1st innings: 139/5 after 14.5 ov; side's win prob. 57%. Team result: Lost.

**Why #8 — Matthew Humphreys** (Spin bowler, Ireland). <font color='#C2412D'>**Caution:** eligible in only 42% of bootstrap resamples — the least robust pick in this list.</font><br/>• Shrunk pressure score +13.5 — rank 8 of 172 eligible<br/>• Under pressure conceded 90 runs/100 balls vs 124 expected (-34)<br/>• 63 high-pressure balls in 14 matches (more than 1% of eligible players)<br/>• Top-10 in 33% of 1,000 match bootstraps; median rank 266 (80% interval 2–266)<br/>• Positive pressure value in 67% of 6 pressure-heavy matches<br/>• Caution: meets eligibility in only 42% of resamples (marginal sample)<br/>_Pressure performance:_ Shrunk pressure score +13.5 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval -5.9 to +32.9). Context-adjusted VAA on his 63 high-pressure balls: +72.9 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +42.9; overall +21.4 across 315 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -0.88 per unit of pressure (raw -96.6): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.53.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -50.7 runs per 100 balls (51 vs 119 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 14 matches, 63 high-pressure balls; posterior SE 11.8. Empirical-Bayes shrinkage kept 15% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 67% of 6 matches with 6+ high-pressure balls; best single match = 46% of positive pressure value; without his best match the score is +6.0.<br/>_Model agreement:_ Top-20 in 2 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 33%, Top-5 in 26%, median rank 266 (80% rank interval 2-266).<br/>_Cricket interpretation:_ A spin bowler conceding 5.90 an over under high pressure (overall 7.14), with 8 high-pressure wickets. Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2025-11-27, Ireland in Bangladesh T20I Series 2025 v Bangladesh: 4/13 from 4.0 overs. Chasing 182: 38/4 after 8.0 ov, needed 144 off 72 (RRR 12.0); side's win prob. 98%. Team result: Won.

**Why #9 — Taskin Ahmed** (Pace bowler, Bangladesh).<br/>• Shrunk pressure score +12.8 — rank 9 of 172 eligible<br/>• Under pressure conceded 130 runs/100 balls vs 137 expected (-7)<br/>• 263 high-pressure balls in 46 matches (more than 68% of eligible players)<br/>• Top-10 in 37% of 1,000 match bootstraps; median rank 15 (80% interval 4–44)<br/>• Positive pressure value in 65% of 26 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +12.8 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval -4.3 to +29.9). Context-adjusted VAA on his 263 high-pressure balls: +24.8 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is -8.1; overall +8.0 across 1083 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope +2.31 per unit of pressure (raw +58.2): performance improves as leverage rises. Model-implied lift f(0.9)-f(0.3) = +1.38.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = +46.6 runs per 100 balls (82 vs 319 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 46 matches, 263 high-pressure balls; posterior SE 10.4. Empirical-Bayes shrinkage kept 34% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 65% of 26 matches with 6+ high-pressure balls; best single match = 21% of positive pressure value; without his best match the score is +11.3.<br/>_Model agreement:_ Top-20 in 2 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 37%, Top-5 in 16%, median rank 15 (80% rank interval 4-44).<br/>_Cricket interpretation:_ A pace bowler conceding 8.14 an over under high pressure (overall 7.98), with 28 high-pressure wickets. Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2023-03-27, Ireland in Bangladesh T20I Series 2023 v Ireland: 4/16 from 2.0 overs. Chasing 104: 37/1 after 3.0 ov, needed 67 off 30 (RRR 13.4); side's win prob. 81%. Team result: Won.

**Why #10 — Mustafizur Rahman** (Pace bowler (death specialist), Bangladesh, Delhi Capitals, Chennai Super Kings).<br/>• Shrunk pressure score +12.5 — rank 10 of 172 eligible<br/>• Under pressure conceded 121 runs/100 balls vs 149 expected (-28)<br/>• 533 high-pressure balls in 71 matches (more than 92% of eligible players)<br/>• Top-10 in 17% of 1,000 match bootstraps; median rank 24 (80% interval 8–56)<br/>• Positive pressure value in 65% of 52 pressure-heavy matches<br/>_Pressure performance:_ Shrunk pressure score +12.5 net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler (90% interval -2.0 to +27.0). Context-adjusted VAA on his 533 high-pressure balls: +25.1 runs per 100 balls vs an average player in the same state.<br/>_Normal performance:_ In low-pressure situations (P < 0.5) the raw VAA is +19.3; overall +14.2 across 1646 balls bowled.<br/>_Pressure response:_ Pressure-response model (VAA = alpha + beta*P): shrunk slope -1.17 per unit of pressure (raw -14.3): performance declines as leverage rises. Model-implied lift f(0.9)-f(0.3) = -0.70.<br/>_Pressure lift:_ Observed lift (P>=0.9 vs P<=0.3) = -21.4 runs per 100 balls (138 vs 483 balls) — descriptive only; it is noisy and heavily shrunk in the score.<br/>_Reliability:_ 71 matches, 533 high-pressure balls; posterior SE 8.8. Empirical-Bayes shrinkage kept 53% of his raw distance from the average.<br/>_Consistency:_ Positive pressure-VAA in 65% of 52 matches with 6+ high-pressure balls; best single match = 8% of positive pressure value; without his best match the score is +11.1.<br/>_Model agreement:_ Top-20 in 4 of 5 candidate models (M0-M4).<br/>_Bootstrap stability:_ Bootstrap (1,000 match resamples): Top-10 in 17%, Top-5 in 5%, median rank 24 (80% rank interval 8-56).<br/>_Cricket interpretation:_ A pace bowler (death specialist) conceding 7.68 an over under high pressure (overall 7.79), with 37 high-pressure wickets, across IPL (532 balls) and T20Is (1114). Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par.<br/>_Key pressure match (Data Pack):_ 2022-04-28, IPL 2022 v Kolkata Knight Riders: 3/18 from 4.0 overs. 1st innings: 118/6 after 17.0 ov; side's win prob. 84%. Team result: Won.


### 18.3 Qualitative match and video evidence (supporting only)

These public sources were consulted only **after** the ranking was final and are not model inputs. Each item links a data-derived high-pressure match to a public report or video and records an observable behaviour.

| player           | match                                                 | observation                                                                                                                                                                                                                                      | source_title                                                                                      |
|:-----------------|:------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|
| Jos Buttler      | KKR v RR, IPL 2024 (16 Apr 2024)                      | Chasing 224 (data: 101/4 after 9 overs, needing 123 off 66, win prob. 18%), Buttler made 107* off 60 and RR won by 2 wickets in the final over - chase acceleration while wickets fell at the other end.                                         | IPL 2024: Jos Buttler's Brilliant Century Ensures Historic 2-Wicket Win for RR Against KKR        |
| Suryakumar Yadav | England v India, 3rd T20I, Trent Bridge (10 Jul 2022) | 117 off 55 chasing 216 from an early hole (data: 13/2 after 2.4 overs, RRR 11.7): sustained boundary hitting at a required rate above 11; India still lost (our data: high VAA, team result 'Lost').                                             | Suryakumar Yadav's Incredible 117 off 55 Balls at Trent Bridge! (official ECB channel)            |
| Rinku Singh      | KKR v LSG, IPL 2023 (20 May 2023)                     | 67* off 33 from 108/4 needing 69 off 36; KKR lost by 1 run - a near-miss finish that our VAA credits even though the team lost.                                                                                                                  | KKR vs LSG Highlights, IPL 2023: Rinku Singh's Fifty in Vain as Lucknow Clinches Narrow 1-Run Win |
| Tim David        | RCB v PBKS, IPL 2025 (18 Apr 2025, rain-shortened)    | Came in during a collapse (data: 33/5 after 6.1 overs) and made 50* in a 14-over game, lifting RCB to 95/9 - recovery batting when resources were depleted.                                                                                      | IPL: RCB Lose Third Straight Home Game As Punjab Kings Dominate In Rain Clash                     |
| Shashank Singh   | GT v PBKS, IPL 2024 (4 Apr 2024)                      | 61* off 29 (6x4, 4x6) chasing 200 after the top order fell (data: 70/4 after 8.4 overs, RRR 11.5) - PBKS won by 3 wickets with a ball to spare.                                                                                                  | IPL 2024 Match Recap: Shashank Guides PBKS to Thrilling Win over GT                               |
| Tristan Stubbs   | MI v DC, IPL 2024 (7 Apr 2024)                        | 71* off 25 chasing 235 from a near-hopeless position (data: win prob. 2%); DC lost by 29 - high individual value in a low-probability chase.                                                                                                     | MI vs DC Highlights: Stubbs' 71 In Vain, Mumbai Indians Beat Delhi Capitals By 29 Runs            |
| Dhruv Jurel      | SRH v RR, Qualifier 2, IPL 2024 (24 May 2024)         | Knockout chase of 176: 56* from 82/5 needing 94 off 45 (data); RR fell short - match-level pressure response, not a match-winning outcome.                                                                                                       | SRH vs RR - IPL 2024 Qualifier 2 Match Highlights                                                 |
| Jasprit Bumrah   | MI v KKR, IPL 2022 (9 May 2022)                       | 5/10 in 4 overs at the death of the first innings (data: 139/5 after 14.5) - wickets when the batting side tried to accelerate.                                                                                                                  | Jasprit Bumrah 5/10 (4) vs Kolkata Knight Riders IPL 2022 (Ball-by-Ball)                          |
| Mohammad Nawaz   | Pakistan v Australia, 3rd T20I, Lahore (1 Feb 2026)   | Career-best 5/18 as Australia, chasing 208, were bowled out for 96. Note: the composite index flags this as pressure (high RRR) although leverage says the match was largely settled - an honest limitation of the selected pressure definition. | PCB: Nawaz's 5-18, Babar, Saim fifties help Pakistan complete T20I series sweep over Australia    |
| Abrar Ahmed      | Pakistan v UAE, T20I tri-series, Sharjah (4 Sep 2025) | 4/9 in 4 overs with 14 dot balls defending 172; took two wickets in the 13th over with UAE 84/4 - wickets in the middle overs of a live chase.                                                                                                   | ESPNcricinfo: Pakistan seal final berth as Fakhar and Abrar headline commanding win               |
| Mitchell Santner | MI v DC, IPL 2025 (21 May 2025)                       | 3/11 in 4 overs in a must-win playoff-qualification game; DC 121 all out chasing 181 - control on a gripping pitch while the chase was still live.                                                                                               | ESPNcricinfo: Suryakumar, Santner and Bumrah lead MI into playoffs                                |

Source URLs: Jos Buttler: https://www.youtube.com/watch?v=oW-zgZWL7yE; Suryakumar Yadav: https://www.youtube.com/watch?v=8S5lApwP3yo; Rinku Singh: https://www.youtube.com/watch?v=OfX5BTU0j2A; Tim David: https://www.youtube.com/watch?v=vzFlE5xRZrI; Shashank Singh: https://www.youtube.com/watch?v=kfyG0SrIsdU; Tristan Stubbs: https://www.youtube.com/watch?v=LP55jzoUaAs; Dhruv Jurel: https://www.youtube.com/watch?v=7TC0FX5HO1k; Jasprit Bumrah: https://www.youtube.com/watch?v=lFW8_4PlD1I; Mohammad Nawaz: https://www.pcb.com.pk/news-detail/nawaz-s-5-18-babar-saim-fifties-help-pakistan-complete-t20i-series-sweep-over-australia.html; Abrar Ahmed: https://www.espncricinfo.com/ci/content/story/1501585.html; Mitchell Santner: https://www.espncricinfo.com/ci/content/story/1486873.html

Note on Mohammad Nawaz: the source confirms the spell, but leverage shows that match was largely decided — an example of where the composite pressure definition is weaker than leverage.


## 19. Previous winners research

Web searches for Season 1 winners, presentations, YouTube walkthroughs, GitHub/Kaggle/LinkedIn work. No video, slide deck, repository or methodology write-up by the Season 1 winners was found publicly; the tools used here also cannot play video, so nothing below is claimed from watching footage. Only statements traceable to the sources listed are recorded.

• Season 1 drew about 7,500 participants from 39 countries and about 1,500 valid submissions. <font color='#6B6F80'>[https://www.rajasthanroyals.com/latest-news/superr-selector-hackathon-7500-participants-1500-submissions]</font>

• 16 finalists (students, young professionals, early-career analysts) attended a day-long in-person analytics challenge at the Royals' Mumbai office, working through real-world T20 strategy scenarios with guidance from the Royals' analytics team, then presented to a panel including senior franchise members. <font color='#6B6F80'>[https://www.rajasthanroyals.com/latest-news/superr-selector-hackathon-7500-participants-1500-submissions]</font>

• Winners: Abhishek Akula (IIT Madras), Nitin Rajasekar (IIIT Hyderabad), Sainath Reddy (IIIT Raipur), selected for 'clarity of thought, analytical depth and innovative approach'. <font color='#6B6F80'>[https://www.rajasthanroyals.com/latest-news/superr-selector-hackathon-7500-participants-1500-submissions]</font>

• Winners travelled to Abu Dhabi as fan representatives at the IPL auction and presented case studies to Kumar Sangakkara and Giles Lindsay (Director of Strategy & Analytics). <font color='#6B6F80'>[https://www.crictracker.com/cricket-news/ipl-2026-auction-rajasthan-royals-reveal-their-auction-playbook-with-fans-1469]</font>

• An RR article describes a hackathon in which participants developed models and recommendations for an ideal 25-member squad, could use generative AI / OpenAI APIs, top entrants presented to cricket leadership including Sangakkara, and some recommendations later influenced decisions; RR stresses that technology should augment, not replace, human judgement. <font color='#6B6F80'>[https://www.rajasthanroyals.com/rajasthan-royals-open-ai-transforms-cricket-content-fan-experience-vaibhav-sooryavanshi]</font>

• Vignesh Puthur: IPL lists him as a left-arm wrist spinner, released by MI and signed by RR for INR 30 lakh ahead of 2026. No public source found linking him to SupeRR Selector recommendations. <font color='#6B6F80'>[https://www.iplt20.com/players/vignesh-puthur/22266]</font>

| Aspect                 | Previous / public approaches                                                 | Our solution                                                                                                                               |
|:-----------------------|:-----------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| Pressure definition    | Not public (S1). S2 public repos: hand-weighted composite index + threshold. | Leverage = expected |change in win probability| from a validated pre-delivery WP model; continuous percentile, calibrated on 2022-24 only. |
| Performance metric     | Not public (S1). S2 repos: raw SR / economy deltas.                          | Contextual VAA in runs: actual minus what an average player produces in the identical state, wickets valued by state.                      |
| Ranking                | Not public (S1). S2 repos: weighted multi-metric 0-100 composites.           | One validated quantity: posterior expected VAA at high pressure; no hand-set weights.                                                      |
| Reliability            | S2 repos: n/(n+k) with hand-picked k; match bootstrap.                       | Empirical Bayes with data-estimated priors, match-clustered SEs, 1,000 stratified match bootstraps.                                        |
| Validation             | Not public.                                                                  | Forward-only: select on 2025 using 2022-24, untouched 2026 test; pre-registered ladder rule.                                               |
| Dashboard              | Not public (S1).                                                             | Streamlit scouting dashboard; every number read from pipeline outputs.                                                                     |
| Evidence               | Case studies presented in person (content not public).                       | Data-derived high-leverage match evidence per player + qualitative public links (never used for ranking).                                  |
| Cricket interpretation | Valued explicitly by the judges ('clarity of thought').                      | Auto-generated 'Why this player?' built only from the player's own statistics.                                                             |

**What we learned / changed:** Judges rewarded clarity and depth together: one defensible headline metric beats a dashboard of unexplained composites. The franchise frames the work as decision support for auctions: every recommendation should carry how confident we are. RR emphasises human judgement: evidence (match situations) should let a coach check the numbers quickly.


## 20. Limitations

• No ground truth for 'clutch'. Validation measures persistence of pressure performance into a later season; all models show weak persistence (|ρ| mostly < 0.3), so rankings carry real uncertainty.

• The selected pressure definition is hand-weighted and can label decided chases as pressure; leverage fixes this but did not validate better on our pre-registered targets, and it changes the lists substantially.

• The baseline outcome is not adjusted for match state; batters who face high-pressure balls in high-scoring IPL conditions are advantaged. Contextual VAA and IPL/T20I sensitivity are reported to expose this.

• Opponent quality is not modelled (an elite death bowler and an associate-nation bowler are the same 'state'); T20Is include associate nations.

• The bootstrap re-runs all player-level estimation but holds delivery-level expectation models fixed, so it does not propagate expectation-model uncertainty.

• 2026 is a higher-scoring season; first-innings win-probability calibration drifts out of time.

• Eligibility thresholds are judgement calls; near-threshold players (flagged in red) are less robust.

• Previous Season 1 methodology is not public; YouTube videos could not be watched by our tools, so only verifiable text sources are cited.


## 21. Reproducibility

**One command:** <font face='Courier'>python run_pipeline.py</font> (add <font face='Courier'>--rebuild</font> to refit every delivery model), then <font face='Courier'>python -m src.visualization.figures</font> and <font face='Courier'>python report/build_report.py</font>. Dashboard: <font face='Courier'>streamlit run dashboard/app.py</font>. Seed 2026; 1,000 bootstraps; split 2022–24 / 2025 / 2026; Python packages in requirements.txt. Two full runs produce identical ranking files (SHA-256 fingerprint in outputs/run_manifest.json). All model training and player rankings use exclusively the competition-provided Data Pack; external sources are used only for qualitative research, previous-winner analysis and supporting video evidence.


## 22. Conclusion

We defined pressure from pre-ball match state, measured performance against the situation, shrank every estimate by its evidence, and let a forward-only test choose between five models. The simplest model, M0, was not beaten out of sample, so it produces the final lists; the more sophisticated leverage/pressure-response machinery is retained as evidence and as an honest counterpoint. The headline names — Tim David, Shashank Singh and Suryakumar Yadav with the bat; Mohammad Nawaz, Sunil Narine and Abrar Ahmed with the ball — are the most stable in 1,000 match resamples, while picks lower in each list should be read with their rank intervals. The clearest lesson from the data is that reliability under pressure is about sustained high-pressure output across many matches; individual 'clutch' effects are too noisy to rank on.
