"""Sensitivity / fairness analysis: does the Top 10 survive reasonable methodological changes?"""
import numpy as np
import pandas as pd
from src import config as C
from src.ranking.candidates import MODELS, build_tables, score_model, rank_eligible
from src.models.pressure_response import player_match_table


def _top(out, k=10):
    return list(rank_eligible(out).index[:k])


def overlap(a, b):
    return len(set(a) & set(b))


def run(x, sel, role, p_high, base_top):
    spec = MODELS[sel]
    rows = []

    def add(group, variant, out, note=""):
        top = _top(out)
        rows.append(dict(role=role, group=group, variant=variant, top10_overlap=overlap(top, base_top),
                         n_eligible=int(out.eligible.sum()), top10=" | ".join(map(str, top)), note=note))
        return out

    tabs = build_tables(x, role, {sel: spec})
    # 1) eligibility thresholds
    e0 = C.ELIG[role]
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0):
        el = {k: int(v * mult) for k, v in e0.items()}
        out, _ = score_model(tabs, spec, role, p_high, elig=el)
        add("Eligibility", f"x{mult} thresholds ({el['min_balls']}/{el['min_hp_balls']}/{el['min_matches']})", out)
    # 2) competition subsets / weighting
    for comp in ("IPL", "T20I"):
        xs = x[x.comp_type == comp]
        out, _ = score_model(build_tables(xs, role, {sel: spec}), spec, role, p_high,
                             elig={k: max(1, int(v * 0.6)) for k, v in e0.items()})
        add("Competition", f"{comp} only (thresholds x0.6)", out)
    t = next(iter(tabs.values()))
    share = t.groupby("comp_type")["n"].transform("sum") / t["n"].sum()
    mw = (0.5 / share).values   # equal total weight to IPL and T20I deliveries
    out, _ = score_model(tabs, spec, role, p_high, mw=mw)
    add("Competition", "Equal IPL/T20I weighting", out)
    # 3) wicket value (applied to the selected model's own outcome definition)
    for wv in ("fixed", "phase", "state"):
        sp = {**spec, "wv": wv}
        if spec["contextual"]:
            sp["y"] = f"vaa_{{role}}_{wv}"
        out, _ = score_model(build_tables(x, role, {"v": sp}), sp, role, p_high)
        add("Wicket value", f"{wv} wicket value", out)
    # 4) pressure definition
    train = x.season.isin(C.TRAIN_SEASONS)
    if spec["kind"] == "mean":
        for q in (0.67, 0.75, 0.80):
            col = f"hp_comp_{int(q * 100)}"
            x[col] = (x.pi_composite >= np.quantile(x.loc[train, "pi_composite"], q)).astype(int)
            sp = {**spec, "w": col}
            out, _ = score_model(build_tables(x, role, {"v": sp}), sp, role, p_high)
            add("Pressure definition", f"composite index, top {100 - int(q * 100)}% of balls", out)
        for col, lab in (("hp_lev", "leverage top 25% (same outcome metric)"), ("li_norm", "leverage-weighted, continuous")):
            sp = {**spec, "w": col}
            out, _ = score_model(build_tables(x, role, {"v": sp}), sp, role, p_high)
            add("Pressure definition", lab, out)
    else:
        for ph in (0.75, 0.80, p_high, 0.95):
            out, _ = score_model(tabs, spec, role, ph)
            add("Pressure definition", f"score at P = {ph:.3f}", out)
    for k in MODELS:
        out, _ = score_model(build_tables(x, role, {k: MODELS[k]}), MODELS[k], role, p_high)
        add("Model", f"{k} {MODELS[k]['name']}", out)
    # 5) era: IPL Impact-Player era only (2023+) and recent seasons only
    xs = x[x.season >= 2024]
    out, _ = score_model(build_tables(xs, role, {sel: spec}), spec, role, p_high,
                         elig={k: max(1, int(v * 0.6)) for k, v in e0.items()})
    add("Era", "2024-2026 only (thresholds x0.6)", out)
    return pd.DataFrame(rows)
