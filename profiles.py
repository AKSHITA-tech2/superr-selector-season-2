"""Player profiles, consistency, pressure curves, match evidence and the dynamic 'Why this player?' text."""
import numpy as np
import pandas as pd
from src import config as C

PER100 = 100.0   # report VAA per 100 balls


def role_rows(x, role):
    if role == "bat":
        r = x[x["faced"]].copy()
        r["pid"], r["pname"], r["team"], r["opp"] = r.striker_id, r.striker_name, r.batting_team_id, r.bowling_team_id
    else:
        r = x.copy()
        r["pid"], r["pname"], r["team"], r["opp"] = r.bowler_id, r.bowler_name, r.bowling_team_id, r.batting_team_id
    return r


def profile_table(x, role, ycol, team_names, hp_col="hp_lev"):
    """hp_col: boolean column defining 'high pressure' under the SELECTED model's pressure definition."""
    r = role_rows(x, role)
    r["y"] = r[ycol]
    r["is_hp"] = r[hp_col].astype(bool)
    r["is_norm"] = r.pressure < 0.5
    g = r.groupby("pid")
    prof = pd.DataFrame({
        "player": g.pname.first(),
        "deliveries": g.size(),
        "matches_all": g.match_id.nunique(),
        "seasons": g.season.agg(lambda s: ", ".join(map(str, sorted(s.unique())))),
        "ipl_balls": g.apply(lambda s: int((s.comp_type == "IPL").sum()), include_groups=False),
        "t20i_balls": g.apply(lambda s: int((s.comp_type == "T20I").sum()), include_groups=False),
        "vaa_overall": g.y.mean() * PER100,
        "vaa_pressure": r[r.is_hp].groupby("pid").y.mean() * PER100,
        "vaa_normal": r[r.is_norm].groupby("pid").y.mean() * PER100,
        "lev_weighted_vaa": (r.y * r.li_norm).groupby(r.pid).sum() / r.li_norm.groupby(r.pid).sum() * PER100,
        "mean_pressure": g.pressure.mean(),
        "share_hp": g.is_hp.mean(),
    })
    if role == "bat":
        r["act"], r["exp"] = r.batter_runs, r.xr_bat
        r["out_a"], r["out_e"] = r.batter_out, r.xw_bat
    else:
        r["act"], r["exp"] = r.bowler_conceded, r.xr_bowl
        r["out_a"], r["out_e"] = r.bowler_wicket, r.xw_bowl
    hpg = r[r.is_hp].groupby("pid")
    prof["actual_runs_100"] = g.act.mean() * PER100
    prof["expected_runs_100"] = g.exp.mean() * PER100
    prof["hp_actual_runs_100"] = hpg.act.mean() * PER100
    prof["hp_expected_runs_100"] = hpg.exp.mean() * PER100
    prof["hp_actual_wkts_100"] = hpg.out_a.mean() * PER100
    prof["hp_expected_wkts_100"] = hpg.out_e.mean() * PER100
    teams = r.groupby("pid").team.agg(lambda s: s.value_counts().index.tolist())
    prof["teams"] = teams.map(lambda ids: ", ".join(team_names.get(i, str(i)) for i in ids[:4]))
    if role == "bat":
        prof["runs"] = g.batter_runs.sum(); prof["outs"] = g.batter_out.sum()
        prof["strike_rate"] = prof.runs / prof.deliveries * 100
        hp = r[r.is_hp].groupby("pid")
        prof["hp_strike_rate"] = hp.batter_runs.sum() / hp.size() * 100
        prof["hp_balls_per_dismissal"] = hp.size() / hp.batter_out.sum().replace(0, np.nan)
        prof["death_share"] = g.is_death.mean()
        prof["role_label"] = np.where(prof.death_share > 0.3, "Finisher", np.where(
            r.groupby("pid").legal_before.mean() < 45, "Top-order batter", "Middle-order batter"))
    else:
        prof["wickets"] = g.bowler_wicket.sum(); prof["conceded"] = g.bowler_conceded.sum()
        legal = r.groupby("pid").is_legal.sum()
        prof["economy"] = prof.conceded / legal * 6
        hp = r[r.is_hp].groupby("pid")
        prof["hp_economy"] = hp.bowler_conceded.sum() / hp.is_legal.sum() * 6
        prof["hp_wickets"] = hp.bowler_wicket.sum()
        prof["death_share"] = g.is_death.mean()
        style = r.groupby("pid").pace_or_spin.agg(lambda s: s.mode().iat[0])
        prof["role_label"] = style.map({"Pace": "Pace bowler", "Spin": "Spin bowler"}) + np.where(
            prof.death_share > 0.3, " (death specialist)", np.where(r.groupby("pid").legal_before.mean() < 40,
                                                                      " (new ball)", ""))
    # match-level consistency: matches with >= 6 high-pressure balls
    mm = r[r.is_hp].groupby(["pid", "match_id"]).agg(n=("y", "size"), v=("y", "sum"))
    mm = mm[mm.n >= 6]
    prof["pressure_matches"] = mm.groupby("pid").size()
    prof["consistency"] = (mm.v > 0).groupby("pid").mean()
    prof["best_match_share"] = (mm.v.groupby("pid").max() / mm.v.clip(lower=0).groupby("pid").sum().replace(0, np.nan))
    return prof


def breakdowns(x, role, ycol, ids):
    """VAA and actual-vs-expected by phase, competition and pressure band for the listed players."""
    r = role_rows(x, role)
    r = r[r.pid.isin(ids)].copy()
    act, exp = ("batter_runs", "xr_bat") if role == "bat" else ("bowler_conceded", "xr_bowl")
    r["band"] = np.where(r.pressure >= C.HIGH_P, "high (P>=0.75)", np.where(r.pressure < 0.5, "normal (P<0.5)", "moderate"))
    out = []
    for dim, col in (("phase", "phase"), ("competition", "comp_type"), ("pressure band", "band"), ("innings", "innings")):
        g = r.groupby(["pid", col]).agg(balls=(ycol, "size"), vaa=(ycol, "mean"), actual=(act, "mean"), expected=(exp, "mean")).reset_index()
        g = g.rename(columns={col: "level"}); g["dimension"] = dim
        for c in ("vaa", "actual", "expected"):
            g[c] *= PER100
        out.append(g)
    b = pd.concat(out)
    b["level"] = b.level.astype(str)
    return b


def why_bullets(p, role, pool):
    """Short evidence bullets for a Top-10 card. Thresholds are relative to the eligible pool, never hand-tuned per player."""
    out = []
    if p["rank"] == 1:
        out.append(f"Highest shrunk pressure score among {len(pool)} eligible players ({p.score_rel100:+.1f})")
    else:
        out.append(f"Shrunk pressure score {p.score_rel100:+.1f} — rank {int(p['rank'])} of {len(pool)} eligible")
    diff = p.hp_actual_runs_100 - p.hp_expected_runs_100
    if role == "bat":
        out.append(f"Under pressure scored {p.hp_actual_runs_100:.0f} runs/100 balls vs {p.hp_expected_runs_100:.0f} expected ({diff:+.0f})")
    else:
        out.append(f"Under pressure conceded {p.hp_actual_runs_100:.0f} runs/100 balls vs {p.hp_expected_runs_100:.0f} expected ({diff:+.0f})")
    q = (pool.hp_balls < p.hp_balls).mean()
    out.append(f"{int(p.hp_balls)} high-pressure balls in {int(p.matches)} matches (more than {q:.0%} of eligible players)")
    out.append(f"Top-10 in {p.p_top10:.0%} of 1,000 match bootstraps; median rank {p.boot_median_rank:.0f} "
               f"(80% interval {p.boot_rank_lo:.0f}–{p.boot_rank_hi:.0f})")
    if pd.notna(p.consistency):
        out.append(f"Positive pressure value in {p.consistency:.0%} of {int(p.pressure_matches)} pressure-heavy matches")
    if p.get("p_eligible", 1) < 0.8:
        out.append(f"Caution: meets eligibility in only {p.p_eligible:.0%} of resamples (marginal sample)")
    return out


def pressure_curve(x, role, ycol, ids, bins=(0, .2, .4, .6, .75, .9, 1.0)):
    r = role_rows(x, role)
    r = r[r.pid.isin(ids)]
    r["pbin"] = pd.cut(r.pressure, bins, include_lowest=True)
    c = r.groupby(["pid", "pbin"], observed=True).agg(vaa=(ycol, "mean"), n=(ycol, "size"),
                                                       p=("pressure", "mean")).reset_index()
    c["vaa"] *= PER100
    c["pbin"] = c.pbin.astype(str)
    return c


def match_evidence(x, role, ycol, ids, team_names, k=3, wcol="li_norm", hp_col="hp_lev"):
    """Top-k matches per player by pressure-weighted VAA (weights = the selected model's pressure weights)."""
    r = role_rows(x, role)
    r = r[r.pid.isin(ids)].copy()
    r["lv"] = r[ycol] * r[wcol]
    agg = dict(lev_vaa=("lv", "sum"), vaa=(ycol, "sum"), balls=(ycol, "size"), max_p=("pressure", "max"),
               mean_p=("pressure", "mean"), date=("match_date", "first"), comp=("comp_name", "first"),
               stage=("stage", "first"), opp=("opp", "first"), team=("team", "first"), innings=("innings", "first"),
               won=("bat_won", "first"))
    if role == "bat":
        agg.update(runs=("batter_runs", "sum"), outs=("batter_out", "sum"))
    else:
        agg.update(runs=("bowler_conceded", "sum"), wkts=("bowler_wicket", "sum"), legal=("is_legal", "sum"))
    m = r.groupby(["pid", "match_id"]).agg(**agg).reset_index()
    first_hp = r[r[hp_col].astype(bool)].sort_values("uid").groupby(["pid", "match_id"]).first()
    out = []
    for pid, g in m.groupby("pid"):
        for _, row in g.nlargest(k, "lev_vaa").iterrows():
            st = first_hp.loc[(pid, row.match_id)] if (pid, row.match_id) in first_hp.index else None
            team_won = row.won if role == "bat" else (1 - row.won if pd.notna(row.won) else np.nan)
            if role == "bat":
                act = f"{int(row.runs)} off {int(row.balls)} balls" + (" (dismissed)" if row.outs else " (not out)")
            else:
                act = f"{int(row.wkts)}/{int(row.runs)} from {int(row.legal) // 6}.{int(row.legal) % 6} overs"
            if st is not None:
                if st.innings == 2:
                    state = (f"Chasing {int(st.target)}: {int(st.score_before)}/{int(st.wkts_before)} after "
                             f"{int(st.legal_before)//6}.{int(st.legal_before)%6} ov, needed {int(st.runs_required)} off "
                             f"{int(st.balls_remaining)} (RRR {st.rrr:.1f})")
                else:
                    state = (f"1st innings: {int(st.score_before)}/{int(st.wkts_before)} after "
                             f"{int(st.legal_before)//6}.{int(st.legal_before)%6} ov")
                wp = st.wp if role == "bat" else 1 - st.wp
                state += f"; side's win prob. {wp:.0%}"
            else:
                state = "no top-quartile pressure ball in this match"
            out.append(dict(player_id=pid, match_id=row.match_id, date=str(pd.Timestamp(row.date).date()),
                            competition=row.comp, stage=row.stage, team=team_names.get(row.team, row.team),
                            opponent=team_names.get(row.opp, row.opp), match_state=state,
                            peak_pressure=row.max_p, mean_pressure=row.mean_p, action=act,
                            vaa=row.vaa, pressure_weighted_vaa=row.lev_vaa,
                            team_result="Won" if team_won == 1 else ("Lost" if team_won == 0 else "No result")))
    return pd.DataFrame(out)


def _pct(v):
    return f"{v:.0%}"


def why_text(p, role, n_models_top20, n_models, metric_desc):
    """Statistics -> sentences. Every number comes from the player's row `p`."""
    unit = "runs per 100 balls"
    s = {}
    s["pressure"] = (f"Shrunk pressure score {p.score_rel100:+.1f} {metric_desc} (90% interval "
                     f"{p.score_lo_rel100:+.1f} to {p.score_hi_rel100:+.1f}). Context-adjusted VAA on his "
                     f"{int(p.hp_balls)} high-pressure balls: {p.vaa_pressure:+.1f} {unit} vs an average player in the same state.")
    s["normal"] = (f"In low-pressure situations (P < 0.5) the raw VAA is {p.vaa_normal:+.1f}; overall {p.vaa_overall:+.1f} "
                   f"across {int(p.deliveries)} {'balls faced' if role == 'bat' else 'balls bowled'}.")
    trend = "improves" if p.beta_post > 0.0005 else ("declines" if p.beta_post < -0.0005 else "stays essentially flat")
    s["response"] = (f"Pressure-response model (VAA = alpha + beta*P): shrunk slope {p.beta_post * PER100:+.2f} per unit of pressure (raw {p.beta_raw * PER100:+.1f}): "
                     f"performance {trend} as leverage rises. Model-implied lift f(0.9)-f(0.3) = {p.lift_model:+.2f}.")
    s["lift"] = (f"Observed lift (P>=0.9 vs P<=0.3) = {p.lift_raw * PER100:+.1f} {unit} "
                 f"({int(p.n_hi)} vs {int(p.n_lo)} balls) — descriptive only; it is noisy and heavily shrunk in the score.")
    s["reliability"] = (f"{int(p.matches)} matches, {int(p.hp_balls)} high-pressure balls; posterior SE "
                        f"{p.score_se100:.1f}. Empirical-Bayes shrinkage kept {p.shrink_keep:.0%} of his raw distance from the average.")
    if pd.notna(p.consistency):
        s["consistency"] = (f"Positive pressure-VAA in {_pct(p.consistency)} of {int(p.pressure_matches)} matches with 6+ "
                            f"high-pressure balls; best single match = {_pct(p.best_match_share if pd.notna(p.best_match_share) else 0)} "
                            f"of positive pressure value; without his best match the score is {p.score_drop_best_rel100:+.1f}.")
    else:
        s["consistency"] = "Too few matches with 6+ high-pressure balls to assess match-level consistency."
    s["agreement"] = f"Top-20 in {n_models_top20} of {n_models} candidate models (M0-M4)."
    s["stability"] = (f"Bootstrap (1,000 match resamples): Top-10 in {_pct(p.p_top10)}, Top-5 in {_pct(p.p_top5)}, "
                      f"median rank {p.boot_median_rank:.0f} (80% rank interval {p.boot_rank_lo:.0f}-{p.boot_rank_hi:.0f}).")
    # cricket interpretation, generated from the numbers
    if role == "bat":
        hp_sr, sr = p.hp_strike_rate, p.strike_rate
        bits = [f"{p.role_label.lower()} who strikes at {hp_sr:.0f} in high-leverage moments (overall {sr:.0f})"]
        if pd.notna(p.hp_balls_per_dismissal):
            bits.append(f"and is dismissed once every {p.hp_balls_per_dismissal:.0f} high-pressure balls")
        if p.ipl_balls > 0 and p.t20i_balls > 0:
            bits.append(f"with evidence from both IPL ({int(p.ipl_balls)} balls) and T20Is ({int(p.t20i_balls)})")
        val = "adds value relative to the situation" if p.score100 > 0 else "holds par"
        s["cricket"] = "A " + " ".join(bits) + f". In context he {val}: the model credits runs only above what an average batter scores in the identical state."
    else:
        bits = [f"{p.role_label.lower()} conceding {p.hp_economy:.2f} an over under high pressure (overall {p.economy:.2f})",
                f"with {int(p.hp_wickets)} high-pressure wickets"]
        if p.ipl_balls > 0 and p.t20i_balls > 0:
            bits.append(f"across IPL ({int(p.ipl_balls)} balls) and T20Is ({int(p.t20i_balls)})")
        s["cricket"] = "A " + ", ".join(bits) + ". Value is measured against an average bowler in the same state, so death-over economy is judged against death-over par, not whole-innings par."
    return s
