"""SupeRR Selector Season 2 - Round 1: one command reproduces the full analysis.

    python run_pipeline.py            # full run (uses cached delivery frames if present)
    python run_pipeline.py --rebuild  # refit every delivery-level model from scratch

All model training and player rankings use ONLY the competition-provided Data Pack.
"""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from src import config as C
from src.data.loader import load_matches, load_raw
from src.data.validation import data_dictionary, integrity_checks
from src.validation.frames import build_all
from src.validation import diagnostics, model_selection, sensitivity
from src.bootstrap import match_bootstrap as bootstrap
from src.ranking.candidates import MODELS, build_tables, score_model, rank_eligible, ycol
from src.ranking import profiles as PR

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main(rebuild=False):
    np.random.seed(C.SEED)
    C.OUT_DIR.mkdir(exist_ok=True); (C.OUT_DIR / "baseline").mkdir(exist_ok=True); C.EVIDENCE_DIR.mkdir(exist_ok=True)
    O = C.OUT_DIR

    # ---------------- 1. data audit ----------------
    log("Data audit")
    raw = load_raw()
    data_dictionary().to_csv(O / "data_dictionary.csv", index=False)
    frames = build_all(force=rebuild)
    d = frames["deliveries"]
    integrity_checks(d).to_csv(O / "integrity_checks.csv", index=False)
    m, _ = load_matches()
    team_names = raw["teams"].drop_duplicates("team_id").set_index("team_id").team_name.to_dict()
    summary = dict(
        matches_in_pack=int(len(m)), matches_analysed=int(d.match_id.nunique()),
        ipl_matches=int(d[d.comp_type == "IPL"].match_id.nunique()),
        t20i_matches=int(d[d.comp_type == "T20I"].match_id.nunique()),
        excluded_other_matches=int((m.comp_type == "OTHER").sum()),
        deliveries=int(len(d)), legal_deliveries=int(d.is_legal.sum()),
        competitions=int(d.comp_id.nunique()), seasons=sorted(map(int, d.season.unique())),
        batters=int(d.striker_id.nunique()), bowlers=int(d.bowler_id.nunique()),
        no_result_matches=int(d.drop_duplicates("match_id").no_result.sum()),
        matches_by_season_comp=d.drop_duplicates("match_id").groupby(["season", "comp_type"]).size()
        .unstack(fill_value=0).to_dict(orient="index"))

    # ---------------- 2. delivery-model diagnostics ----------------
    log("Delivery-model diagnostics (forward, out-of-time)")
    dv, calv = diagnostics.run(d, C.TRAIN_SEASONS, C.VALID_SEASON)
    dt, _ = diagnostics.run(d, C.TRAIN_SEASONS + [C.VALID_SEASON], C.TEST_SEASON)
    pd.concat([dv, dt]).to_csv(O / "delivery_model_diagnostics.csv", index=False)
    calv.to_csv(O / "wp_calibration_2025.csv", index=False)

    # ---------------- 3. model selection on 2025 ----------------
    log("Model selection: train 2022-24 -> validate 2025 (1,000 match bootstraps)")
    xv, mv = frames["valid"]
    res_v, boot_v = model_selection.evaluate_fold(xv, C.TRAIN_SEASONS, C.VALID_SEASON, C.P_HIGH, n_boot=C.N_BOOT)
    selected, sel_log = model_selection.select(res_v, boot_v)
    res_v.to_csv(O / "validation_results.csv", index=False)
    sel_log.to_csv(O / "selection_log.csv", index=False)
    ci_v = {r: boot_v[r].quantile([0.05, 0.95]).T for r in boot_v}
    log(f"Selected model: {selected}")

    # ---------------- 4. untouched final test 2026 ----------------
    log("Final test: train 2022-25 -> test 2026 (run once, after selection)")
    xt, mt = frames["test"]
    res_t, boot_t = model_selection.evaluate_fold(xt, C.TRAIN_SEASONS + [C.VALID_SEASON], C.TEST_SEASON, C.P_HIGH,
                                                   n_boot=C.N_BOOT)
    res_t.to_csv(O / "test_results.csv", index=False)
    comp_rows = []
    for k, sp in MODELS.items():
        v = res_v[res_v.model == k]; t = res_t[res_t.model == k]
        def ci(boot, role, tgt):
            q = boot[role][(k, tgt)].quantile([0.05, 0.95]).values
            return q
        row = dict(model=k, name=sp["name"], pressure=sp["pressure"], contextual_vaa=sp["contextual"],
                   pressure_response=sp["response"], empirical_bayes=True, wicket_value=sp["wv"],
                   leakage_status="Forward-only expectations; pressure calibrated on train window",
                   complexity={"M0": 1, "M1": 2, "M2": 3, "M3": 4, "M4": 5}[k],
                   valid_mean_spearman=v.spearman.mean(), test_mean_spearman=t.spearman.mean(),
                   selected=(k == selected))
        for role in ("bat", "bowl"):
            for tgt in ("T_lev", "T_raw"):
                row[f"valid_{role}_{tgt}"] = float(v[(v.role == role) & (v.target == tgt)].spearman.iloc[0])
                lo, hi = ci(boot_v, role, tgt)
                row[f"valid_{role}_{tgt}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
                row[f"test_{role}_{tgt}"] = float(t[(t.role == role) & (t.target == tgt)].spearman.iloc[0])
            row[f"valid_{role}_top20_overlap"] = float(v[(v.role == role) & (v.target == "T_lev")].top20_overlap.iloc[0])
        comp_rows.append(row)
    model_comp = pd.DataFrame(comp_rows)

    # ---------------- 5. final retrospective ranking (2022-26) ----------------
    log("Final ranking on 2022-26 (leave-one-season-out expectations)")
    xr, mr = frames["retro"]
    xr["hp_lev"] = (xr.pressure >= C.HIGH_P)
    spec = MODELS[selected]
    hp_col = "hp_composite" if spec["w"] == "hp_composite" else "hp_lev"
    metric_desc = {
        "bat": ("net runs per 100 high-pressure balls (runs minus wicket value x dismissals) above the average eligible batter"
                if not spec["contextual"] else "runs per 100 balls above an average batter in the same match state, at high pressure"),
        "bowl": ("net runs saved per 100 high-pressure balls (wicket value x wickets minus runs conceded) above the average eligible bowler"
                 if not spec["contextual"] else "runs saved per 100 balls versus an average bowler in the same match state, at high pressure")}
    resp_spec = MODELS["M4"] if spec["wv"] == "state" else MODELS["M3"]
    final, players_all, boots, curves, evid, why_all, sens_all, stab_rows, brk = {}, [], {}, [], [], {}, [], [], []
    for role in ("bat", "bowl"):
        tabs = build_tables(xr, role)
        names = next(iter(tabs.values())).attrs["names"]
        outs = {k: score_model(tabs, sp, role, C.P_HIGH) for k, sp in MODELS.items()}
        top20 = {k: set(rank_eligible(o).index[:20]) for k, (o, _) in outs.items()}
        top10_by_model = {k: list(rank_eligible(o).index[:10]) for k, (o, _) in outs.items()}
        out, hyper = outs[selected]
        resp, rhyper = outs["M4" if resp_spec is MODELS["M4"] else "M3"]
        ranked = rank_eligible(out)
        yc = ycol(spec, role) if spec["contextual"] else f"vaa_{role}_{spec['wv']}"
        prof = PR.profile_table(xr, role, yc, team_names, hp_col=hp_col)
        df = ranked.join(prof, how="left")
        df["player"] = [names.get(i, str(i)) for i in df.index]
        df["model_agreement_top20"] = [sum(i in s for s in top20.values()) for i in df.index]
        # pressure-response diagnostics always come from the contextual response model (VAA scale)
        for c in ["lift_raw", "n_hi", "n_lo"]:
            df[c] = resp.loc[df.index, c]
        for c in ["alpha", "beta", "alpha_raw", "beta_raw", "alpha_post_se", "beta_post_se"]:
            df[c if c not in ("beta", "alpha") else f"{c}_post"] = resp.loc[df.index, c]
        df["lift_model"] = df["beta_post"] * (C.LIFT_HI - C.LIFT_LO) * PR.PER100
        prior = (hyper["mu"] if "mu" in hyper else hyper["mu_alpha"] + hyper["mu_beta"] * (C.P_HIGH - 0.5))
        df["shrink_keep"] = ((df.score - prior) / (df.raw - prior)).clip(0, 1)
        for c in ["score", "raw", "score_se"]:
            df[c + "100"] = df[c] * PR.PER100
        df["score_lo100"] = (df.score - 1.645 * df.score_se) * PR.PER100
        df["score_hi100"] = (df.score + 1.645 * df.score_se) * PR.PER100
        df["prior_mean100"] = prior * PR.PER100
        for c in ["score", "score_lo", "score_hi"]:
            df[c + "_rel100"] = df[c + "100"] - prior * PR.PER100
        # bootstrap (selected model)
        log(f"  {role}: 1,000 stratified match-level bootstraps")
        R, S = bootstrap.bootstrap_ranks(tabs, spec, role, C.P_HIGH)
        bs = bootstrap.summarise(R, C.N_BOOT)
        df = df.join(bs)
        boots[role] = R
        # leave-best-match-out robustness for the top 20
        t = tabs[(ycol(spec, role), spec["w"])]
        drop = {}
        for pid in df.index[:20]:
            tp = t[t.player_id == pid]
            lv = (tp.Swy if spec["kind"] == "mean" else tp.Sy)
            best = tp.index[np.argmax(lv.values)]
            mw = np.ones(len(t)); mw[t.index.get_loc(best)] = 0
            o2, _ = score_model(tabs, spec, role, C.P_HIGH, mw=mw)
            drop[pid] = o2.loc[pid, "score"] * PR.PER100
        df["score_drop_best100"] = pd.Series(drop)
        df["score_drop_best_rel100"] = df["score_drop_best100"] - prior * PR.PER100
        df["role"] = role
        df["rank"] = range(1, len(df) + 1)
        players_all.append(df)
        top = df.head(10).copy()
        why = {}
        for pid, p in top.iterrows():
            why[int(pid)] = PR.why_text(p, role, int(p.model_agreement_top20), len(MODELS), metric_desc[role])
            why[int(pid)]["bullets"] = PR.why_bullets(p, role, df)
        brk.append(PR.breakdowns(xr, role, yc, df.index[:30]).assign(role=role))
        why_all[role] = why
        final[role] = top
        curves.append(PR.pressure_curve(xr, role, yc, df.index[:30]).assign(role=role))
        ev = PR.match_evidence(xr, role, yc, top.index, team_names, wcol=spec["w"], hp_col=hp_col)
        ev["player"] = ev.player_id.map(names); ev["role"] = role
        evid.append(ev)
        pw = bootstrap.pairwise_beat(R, list(df.index[:15]))
        pw.index = [names[i] for i in pw.index]; pw.columns = [names[i] for i in pw.columns]
        pw.to_csv(O / f"pairwise_beat_{role}.csv")
        rs = R.loc[df.index[:30]].copy(); rs.index = [names[i] for i in rs.index]
        rs.T.to_csv(O / f"bootstrap_rank_samples_{role}.csv", index=False)
        # top-10 by model for agreement display
        pd.DataFrame({k: [names[i] for i in v] for k, v in top10_by_model.items()}).to_csv(
            O / f"top10_by_model_{role}.csv", index=False)
        # baseline preserved separately
        b0 = rank_eligible(outs["M0"][0]).head(25).copy(); b0["player"] = [names[i] for i in b0.index]
        b0.to_csv(O / "baseline" / f"M0_baseline_{role}_ranking.csv")
        # stability of the final top 10
        stab_rows.append(dict(role=role, mean_p_top10_of_top10=float(top.p_top10.mean()),
                              expected_overlap=float(top.p_top10.sum())))
        log(f"  {role}: sensitivity analysis")
        sens = sensitivity.run(xr, selected, role, C.P_HIGH, list(top.index))
        sens["top10"] = sens.top10.map(lambda s: " | ".join(names.get(int(i), i) for i in s.split(" | ") if i))
        sens_all.append(sens)

    # ---------------- 6. write outputs ----------------
    log("Writing outputs")
    players = pd.concat(players_all)
    players.index.name = "player_id"
    players.reset_index().to_csv(O / "player_metrics.csv", index=False)
    cols = ["rank", "player", "role_label", "teams", "score_rel100", "score_lo_rel100", "score_hi_rel100", "score_se100",
            "score100", "prior_mean100",
            "alpha_post", "beta_post", "lift_model", "lift_raw", "vaa_pressure", "vaa_normal", "vaa_overall",
            "hp_balls", "n_balls", "matches", "pressure_matches", "consistency", "p_top10", "p_top5", "p_top3",
            "boot_median_rank", "boot_rank_lo", "boot_rank_hi", "model_agreement_top20", "ipl_balls", "t20i_balls",
            "hp_actual_runs_100", "hp_expected_runs_100", "actual_runs_100", "expected_runs_100", "p_eligible"]
    for role, fname in (("bat", "final_batter_rankings.csv"), ("bowl", "final_bowler_rankings.csv")):
        f = final[role].reset_index().rename(columns={"index": "player_id"})
        f[["player_id"] + cols].to_csv(O / fname, index=False)
    players.reset_index()[["player_id", "role", "player", "rank"] + [c for c in players.columns if c.startswith(("boot_", "p_top"))]] \
        .to_csv(O / "bootstrap_results.csv", index=False)
    pd.concat(sens_all).to_csv(O / "sensitivity_results.csv", index=False)
    pd.concat(curves).to_csv(O / "pressure_curves.csv", index=False)
    pd.concat(brk).to_csv(O / "player_breakdowns.csv", index=False)
    pd.concat(evid).to_csv(C.EVIDENCE_DIR / "player_match_evidence.csv", index=False)
    model_comp.to_csv(O / "model_comparison.csv", index=False)
    json.dump(why_all, open(O / "why_this_player.json", "w"), indent=1)

    # pressure analysis data for the dashboard (aggregates + a small sample; no external data)
    pa = xr[["innings", "phase", "comp_type", "season", "pressure", "li", "li_norm", "wp", "rrr", "wkts_in_hand",
             "balls_remaining", "legal_before", "vaa_bat_state", "vaa_bowl_state", "pi_composite", "hp_composite"]]
    pa.sample(25000, random_state=C.SEED).to_csv(O / "pressure_sample.csv.gz", index=False)
    pa.assign(hp=pa.pressure >= C.HIGH_P).groupby(["comp_type", "season", "innings", "phase"]).agg(
        deliveries=("pressure", "size"), mean_pressure=("pressure", "mean"), mean_li=("li_norm", "mean"),
        hp_share=("hp", "mean"), hp_comp_share=("hp_composite", "mean")).reset_index().to_csv(O / "pressure_summary.csv", index=False)

    summary.update(dict(
        selected_model=selected, selected_model_name=MODELS[selected]["name"], metric_desc=metric_desc,
        hp_definition=hp_col,
        p_high=C.P_HIGH, hp_threshold_percentile=C.HIGH_P,
        pressure_deliveries=int(xr[hp_col].sum()),
        pressure_deliveries_leverage=int((xr.pressure >= C.HIGH_P).sum()),
        mean_li_train=mr["mean_li"], li_q75_train=mr["li_q75"], fixed_wv=mr["fixed_wv"], phase_wv=mr["phase_wv"],
        eligible_batters=int((players.role == "bat").sum()), eligible_bowlers=int((players.role == "bowl").sum()),
        top10_stability=stab_rows, seed=C.SEED, n_boot=C.N_BOOT, elig=C.ELIG,
        validation_mean_spearman={k: float(res_v[res_v.model == k].spearman.mean()) for k in MODELS},
        test_mean_spearman={k: float(res_t[res_t.model == k].spearman.mean()) for k in MODELS},
        runtime_sec=round(time.time() - T0, 1)))
    json.dump(summary, open(O / "dataset_summary.json", "w"), indent=1, default=str)
    # reproducibility fingerprint of the final rankings
    fp = hashlib.sha256(open(O / "final_batter_rankings.csv", "rb").read() +
                        open(O / "final_bowler_rankings.csv", "rb").read()).hexdigest()
    json.dump(dict(rankings_sha256=fp, seed=C.SEED, n_boot=C.N_BOOT), open(O / "run_manifest.json", "w"), indent=1)
    log(f"Done. Rankings fingerprint {fp[:16]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    main(rebuild=ap.parse_args().rebuild)
