"""Candidate models (an ablation ladder: each rung adds exactly one idea).

M0  PE-EB baseline     : raw outcome (runs - WV_fixed*out), composite hand-built pressure (top quartile), EB mean
M1  Contextual VAA+EB  : contextual VAA (fixed WV), same composite pressure, EB mean
M2  Leverage VAA+EB    : contextual VAA (fixed WV), LEVERAGE-weighted mean (continuous pressure), EB
M3  Pressure response  : VAA (fixed WV) = alpha_i + beta_i(P-0.5); bivariate EB; score at high pressure
M4  Full               : M3 with STATE-dependent wicket value
"""
import numpy as np
import pandas as pd
from src import config as C
from src.models.pressure_response import player_match_table, weighted_mean_eb, response_eb

MODELS = {
    "M0": dict(name="PE-EB baseline", y="raw_{wv}", w="hp_composite", kind="mean", wv="fixed",
               pressure="Composite index (top 25%)", contextual=False, response=False),
    "M1": dict(name="Contextual VAA + EB", y="vaa_{role}_fixed", w="hp_composite", kind="mean", wv="fixed",
               pressure="Composite index (top 25%)", contextual=True, response=False),
    "M2": dict(name="Leverage-weighted VAA + EB", y="vaa_{role}_fixed", w="li_norm", kind="mean", wv="fixed",
               pressure="Leverage (continuous weight)", contextual=True, response=False),
    "M3": dict(name="Pressure response + EB", y="vaa_{role}_fixed", w="li_norm", kind="response", wv="fixed",
               pressure="Leverage percentile (continuous)", contextual=True, response=True),
    "M4": dict(name="Full: response + state WV", y="vaa_{role}_state", w="li_norm", kind="response", wv="state",
               pressure="Leverage percentile (continuous)", contextual=True, response=True),
}


def add_raw_outcomes(x):
    """Context-free outcome for the baseline (still sign-consistent: higher = better), for each wicket-value variant."""
    for v in ("fixed", "phase", "state"):
        wv = x[f"wv_{v}"]
        x[f"raw_bat_{v}"] = np.where(x["faced"], x["batter_runs"] - wv * x["batter_out"], np.nan)
        x[f"raw_bowl_{v}"] = -x["bowler_conceded"] + wv * x["bowler_wicket"]
    if "hp_lev" not in x:
        x["hp_lev"] = (x["pressure"] >= C.HIGH_P).astype(int)
    return x


def elig_fn(role, elig=None):
    e = elig or C.ELIG[role]
    return lambda s: (s["n"] >= e["min_balls"]) & (s["Nh"] >= e["min_hp_balls"]) & (s["G"] >= e["min_matches"])


def ycol(spec, role):
    if spec["y"].startswith("raw_"):
        return f"raw_{role}_{spec['wv']}"
    return spec["y"].format(role=role, wv=spec["wv"])


def build_tables(x, role, models=MODELS):
    """One (player, match) table per distinct (y, w) combination."""
    cache = {}
    for k, sp in models.items():
        key = (ycol(sp, role), sp["w"])
        if key not in cache:
            cache[key] = player_match_table(x, role, key[0], key[1])
    return cache


def score_model(tables, spec, role, p_high, mw=None, elig=None):
    t = tables[(ycol(spec, role), spec["w"])]
    mw = np.ones(len(t)) if mw is None else mw
    f = elig_fn(role, elig)
    if spec["kind"] == "mean":
        out, s, hyper = weighted_mean_eb(t, mw, "w", f)
    else:
        out, s, hyper = response_eb(t, mw, p_high, f)
    out["n_balls"], out["hp_balls"], out["matches"] = s["n"], s["Nh"], s["G"]
    out["eligible"] = f(s)
    out["lift_raw"] = s["Shiy"] / s["Nhi"].replace(0, np.nan) - s["Sloy"] / s["Nlo"].replace(0, np.nan)
    out["n_hi"], out["n_lo"] = s["Nhi"], s["Nlo"]
    return out, hyper


def rank_eligible(out):
    e = out[out["eligible"] & out["score"].notna()].copy()
    e["rank"] = e["score"].rank(ascending=False, method="first").astype(int)
    return e.sort_values("rank")
