"""Publication figures for the report (PNG, 200 dpi). Every value is read from pipeline outputs / cached frames."""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C

PURPLE, GOLD, BLUE, ROSE, NAVY, MUTED, GRID = "#6A4BC4", "#B07C10", "#1F7FB8", "#B5487A", "#121935", "#6B6F80", "#ECEAE4"
GOOD, BAD = "#2E7D5B", "#B23A3A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#C9C4B8", "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
                     "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 200,
                     "legend.frameon": False})


def _save(fig, name):
    C.FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all():
    O = C.OUT_DIR
    S = json.load(open(O / "dataset_summary.json"))
    P = pd.read_csv(O / "player_metrics.csv")
    x = pickle.load(open(C.CACHE_DIR / "frames.pkl", "rb"))["retro"][0]
    fb, fw = pd.read_csv(O / "final_batter_rankings.csv"), pd.read_csv(O / "final_bowler_rankings.csv")

    # 1 pressure distribution
    fig, ax = plt.subplots(figsize=(6.4, 3))
    for inn, c in ((1, PURPLE), (2, GOLD)):
        ax.hist(x.loc[x.innings == inn, "pressure"], bins=40, alpha=.75, color=c, label=f"Innings {inn}")
    ax.set(xlabel="Pressure (percentile of 2022-24 leverage)", ylabel="Deliveries", title="01  Pressure distribution by innings")
    ax.legend(); _save(fig, "01_pressure_distribution.png")
    # 2 leverage distribution
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.hist(x.li_norm.clip(upper=12), bins=80, color=PURPLE)
    ax.set_yscale("log"); ax.axvline(1, color=NAVY, ls=":", lw=1)
    ax.set(xlabel="Leverage index (1.0 = average delivery, capped at 12)", ylabel="Deliveries (log)", title="02  Leverage distribution")
    _save(fig, "02_leverage_distribution.png")
    # 3 pressure vs VAA (all deliveries, binned)
    fig, ax = plt.subplots(figsize=(6.4, 3))
    b = pd.cut(x.pressure, np.linspace(0, 1, 11), include_lowest=True)
    for col, c, lab in (("vaa_bat_state", PURPLE, "Batters"), ("vaa_bowl_state", GOLD, "Bowlers")):
        g = x.groupby(b, observed=True)[col].agg(["mean", "sem"])
        mids = [iv.mid for iv in g.index]
        ax.errorbar(mids, g["mean"] * 100, yerr=1.96 * g["sem"] * 100, color=c, marker="o", ms=5, lw=2, capsize=2, label=lab)
    ax.axhline(0, color="#C9C4B8", lw=1)
    ax.set(xlabel="Pressure decile", ylabel="Mean VAA per 100 balls", title="03  League-wide VAA vs pressure (centred by construction)")
    ax.legend(); _save(fig, "03_pressure_vs_vaa.png")
    # 4 response curves for top 5 of each role
    cur = pd.read_csv(O / "pressure_curves.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (role, fr) in zip(axs, (("bat", fb), ("bowl", fw))):
        for i, (pid, name) in enumerate(zip(fr.player_id[:5], fr.player[:5])):
            q = cur[(cur.role == role) & (cur.pid == pid)]
            ax.plot(q.p, q.vaa, marker="o", ms=4, lw=1.6, label=name, color=[PURPLE, GOLD, BLUE, ROSE, NAVY][i])
        ax.axhline(0, color="#C9C4B8", lw=1)
        ax.set(xlabel="Pressure band (mean percentile)", title=f"04  Pressure-response, top 5 {'batters' if role == 'bat' else 'bowlers'}")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("VAA per 100 balls"); _save(fig, "04_response_curves.png")
    # 5/6 top-10 rankings with intervals
    for fr, role, name in ((fb, "bat", "05_top10_batters.png"), (fw, "bowl", "06_top10_bowlers.png")):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        y = np.arange(len(fr))[::-1]
        ax.errorbar(fr.score_rel100, y, xerr=[fr.score_rel100 - fr.score_lo_rel100, fr.score_hi_rel100 - fr.score_rel100],
                    fmt="o", color=PURPLE if role == "bat" else GOLD, ms=7, lw=2, capsize=0)
        ax.set_yticks(y, [f"{r}. {p}" for r, p in zip(fr["rank"], fr.player)])
        ax.axvline(0, color="#C9C4B8", lw=1)
        ax.set(xlabel="Shrunk pressure score vs average eligible player (per 100 balls), 90% interval",
               title=f"{name[:2]}  Final Top 10 {'batters' if role == 'bat' else 'bowlers'}")
        _save(fig, name)
    # 7 uncertainty: posterior SE vs pressure balls
    fig, ax = plt.subplots(figsize=(6.4, 3))
    for role, c in (("bat", PURPLE), ("bowl", GOLD)):
        q = P[P.role == role]
        ax.scatter(q.hp_balls, q.score_se100, s=12, color=c, alpha=.7, label="Batters" if role == "bat" else "Bowlers")
    ax.set_xscale("log")
    ax.set(xlabel="High-pressure balls (log)", ylabel="Posterior SE (per 100 balls)", title="07  Uncertainty shrinks with evidence")
    ax.legend(); _save(fig, "07_uncertainty.png")
    # 8 bootstrap rank distributions (top 5 each)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, role in zip(axs, ("bat", "bowl")):
        rs = pd.read_csv(O / f"bootstrap_rank_samples_{role}.csv")
        fr = fb if role == "bat" else fw
        data = [rs[n].fillna(60).clip(upper=60).values for n in fr.player[:10]]
        bp = ax.boxplot(data, orientation="horizontal", widths=.6, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set(facecolor=PURPLE if role == "bat" else GOLD, alpha=.55, edgecolor="none")
        ax.set_yticks(range(1, 11), list(fr.player[:10]))
        ax.invert_yaxis(); ax.axvline(10.5, color=NAVY, ls=":", lw=1)
        ax.set(xlabel="Rank across 1,000 match bootstraps (capped 60)", title=f"08  Bootstrap ranks, {'batters' if role == 'bat' else 'bowlers'}")
    _save(fig, "08_bootstrap_ranks.png")
    # 9 model comparison
    vr = pd.read_csv(O / "validation_results.csv"); tr = pd.read_csv(O / "test_results.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (d, t) in zip(axs, ((vr, "2025 validation"), (tr, "2026 final test"))):
        piv = d.pivot_table(index="model", columns=["role", "target"], values="spearman")
        w = .2
        for i, (col, c) in enumerate(zip(piv.columns, [PURPLE, "#A08BE0", GOLD, "#D9B566"])):
            ax.bar(np.arange(len(piv)) + (i - 1.5) * w, piv[col], w, color=c,
                   label=f"{'Bat' if col[0] == 'bat' else 'Bowl'} - {'lev-VAA' if col[1] == 'T_lev' else 'raw'} target")
        ax.set_xticks(range(len(piv)), piv.index); ax.axhline(0, color="#C9C4B8", lw=1)
        ax.set(title=f"09  Spearman, {t}")
    axs[0].set_ylabel("Spearman (train score vs holdout)"); axs[1].legend(fontsize=7)
    _save(fig, "09_model_comparison.png")
    # 10/11 sensitivity (IPL/T20I + eligibility)
    sr = pd.read_csv(O / "sensitivity_results.csv")
    for grp, name, ttl in (("Competition", "10_ipl_t20i_sensitivity.png", "10  IPL vs T20I sensitivity"),
                           ("Eligibility", "11_eligibility_sensitivity.png", "11  Eligibility sensitivity")):
        fig, axs = plt.subplots(1, 2, figsize=(9, 2.8))
        for ax, role in zip(axs, ("bat", "bowl")):
            q = sr[(sr.group == grp) & (sr.role == role)]
            y = np.arange(len(q))
            ax.barh(y, q.top10_overlap, color=PURPLE if role == "bat" else GOLD)
            for yy, v in zip(y, q.top10_overlap):
                ax.text(v + .15, yy, str(int(v)), va="center", fontsize=8, color=MUTED)
            ax.set_yticks(y, q.variant.str.slice(0, 38), fontsize=7); ax.set_xlim(0, 10.8)
            ax.set(xlabel="Players shared with final Top 10", title=f"{ttl} - {'batters' if role == 'bat' else 'bowlers'}")
        _save(fig, name)
    # 12 season stability: top-10 players' high-pressure VAA by season
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    hpc = "hp_composite" if S["hp_definition"] == "hp_composite" else "hp_lev"
    for ax, (role, fr) in zip(axs, (("bat", fb), ("bowl", fw))):
        r = x[x.faced] if role == "bat" else x
        pidc, yc = ("striker_id", "vaa_bat_fixed") if role == "bat" else ("bowler_id", "vaa_bowl_fixed")
        r = r[r[pidc].isin(fr.player_id) & (r[hpc] == 1)]
        g = r.groupby([pidc, "season"])[yc].agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= 20]
        for pid, q in g.groupby(pidc):
            ax.plot(q.season, q["mean"] * 100, color=PURPLE if role == "bat" else GOLD, alpha=.35, lw=1, marker="o", ms=3)
        m = g.groupby("season")["mean"].mean() * 100
        ax.plot(m.index, m.values, color=NAVY, lw=2.5, marker="o", label="Top-10 average")
        ax.axhline(0, color="#C9C4B8", lw=1); ax.set_xticks(C.ALL_SEASONS)
        ax.set(title=f"12  Season stability, {'batters' if role == 'bat' else 'bowlers'} (seasons with 20+ HP balls)")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("High-pressure VAA per 100 balls"); _save(fig, "12_season_stability.png")
    # 13 match-level consistency
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    both = pd.concat([fb.assign(r="bat"), fw.assign(r="bowl")])
    pm = P.set_index(["role", "player_id"])
    cons = [pm.loc[(r, i), "consistency"] for r, i in zip(both.r, both.player_id)]
    y = np.arange(len(both))[::-1]
    ax.barh(y, cons, color=[PURPLE if r == "bat" else GOLD for r in both.r])
    ax.set_yticks(y, both.player, fontsize=7); ax.axvline(.5, color=NAVY, ls=":", lw=1); ax.set_xlim(0, 1)
    ax.set(xlabel="Share of matches (6+ high-pressure balls) with positive VAA", title="13  Match-level consistency, final 20")
    _save(fig, "13_match_consistency.png")
    # 14 expected vs actual (calibration of xR, out-of-season predictions)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    for ax, (pred, act, mask, t) in zip(axs, (("xr_bat", "batter_runs", x.faced, "Batter runs"), ("xw_bowl", "bowler_wicket", None, "Bowler wicket"))):
        q = x if mask is None else x[mask]
        bins = pd.qcut(q[pred], 10, duplicates="drop")
        g = q.groupby(bins, observed=True)[[pred, act]].mean()
        ax.plot(g[pred], g[act], marker="o", color=PURPLE, lw=2)
        lim = [g.values.min(), g.values.max()]
        ax.plot(lim, lim, color="#C9C4B8", ls=":")
        ax.set(xlabel="Expected (leave-one-season-out)", ylabel="Actual", title=f"14  Expected vs actual: {t}")
    _save(fig, "14_expected_vs_actual.png")
    # 15 pressure-lift distribution
    fig, ax = plt.subplots(figsize=(6.4, 3))
    for role, c in (("bat", PURPLE), ("bowl", GOLD)):
        q = P[P.role == role]
        ax.hist(q.lift_raw * 100, bins=30, alpha=.65, color=c, label="Batters (raw)" if role == "bat" else "Bowlers (raw)")
        ax.hist(q.lift_model, bins=30, alpha=.9, color=NAVY if role == "bat" else "#7C5A10", histtype="step", lw=2,
                label=("Batters" if role == "bat" else "Bowlers") + " (EB-shrunk model)")
    ax.set(xlabel="Pressure lift, VAA per 100 balls (P>=0.9 minus P<=0.3)", ylabel="Players",
           title="15  Pressure lift: raw spread is mostly noise")
    ax.legend(fontsize=7); _save(fig, "15_pressure_lift.png")


if __name__ == "__main__":
    make_all()
