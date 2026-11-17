"""SupeRR Selector S2 · Pressure-Based T20 Player Intelligence — interactive dashboard.

Run:  streamlit run dashboard/app.py
Every number shown is read from files written by run_pipeline.py (outputs/, evidence/). Nothing is hard-coded.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT, EVD = ROOT / "outputs", ROOT / "evidence"

# ---------------- identity (validated categorical: royal blue · amber · royals pink) ----------------
NAVY, ROYAL, AMBER, PINK = "#0B1F3A", "#2450C8", "#B7791F", "#D6337A"
ROYAL_T, AMBER_T = "#8FA8EA", "#DDB97A"
INK, MUTED, LINE, BG, CARD = "#141A2A", "#667085", "#E6E9F0", "#F5F7FB", "#FFFFFF"
GOOD, BAD = "#1E8E5A", "#C2412D"
ROLE_C = {"bat": ROYAL, "bowl": AMBER}
ROLE_W = {"bat": "Batter", "bowl": "Bowler"}

st.set_page_config(page_title="SupeRR Selector S2 · Pressure Intelligence", page_icon="◆", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: {BG}; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 1.1rem; max-width: 1440px; }}
section[data-testid="stSidebar"] > div {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #DCE3F5 !important; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {{ padding: 8px 12px; border-radius: 9px; margin: 1px 0; width: 100%; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover {{ background: rgba(255,255,255,.06); }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {{ background: rgba(36,80,200,.45); box-shadow: inset 3px 0 0 {PINK}; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display:none !important; }}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{ font-size: .9rem; font-weight: 500; }}
.navsec {{ font-size: .68rem; letter-spacing: .12em; opacity: .55; margin: 14px 0 4px 4px; }}
.brand {{ font-weight: 800; font-size: 1.12rem; letter-spacing: -.02em; }}
.brand .dot {{ color: {PINK} !important; }}
.topbar {{ display:flex; align-items:center; gap:10px; padding: 4px 0 14px 0; border-bottom: 1px solid {LINE}; margin-bottom: 16px; }}
.crumb {{ color: {MUTED}; font-size: .82rem; }}
.crumb b {{ color: {INK}; }}
.chip {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: .72rem; font-weight: 600; border: 1px solid {LINE}; background: {CARD}; color: {INK}; }}
.chip.blue {{ background: #EAF0FF; border-color: #C9D6FB; color: {ROYAL}; }}
.chip.pink {{ background: #FCEAF2; border-color: #F4C2D8; color: {PINK}; }}
.chip.amber {{ background: #FBF2E3; border-color: #EBD3A8; color: #8A5A10; }}
.chip.red {{ background: #FBECE9; border-color: #F2C6BD; color: {BAD}; }}
.spacer {{ flex: 1; }}
h1.title {{ font-size: 1.85rem; font-weight: 800; color: {INK}; letter-spacing: -.03em; margin: 0; }}
.subtitle {{ color: {MUTED}; font-size: .95rem; margin: 2px 0 14px 0; }}
.card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 16px; height: 100%; }}
.kl {{ font-size: .74rem; color: {MUTED}; font-weight: 500; }}
.kv {{ font-size: 1.55rem; font-weight: 800; color: {INK}; letter-spacing: -.03em; line-height: 1.25; }}
.ks {{ font-size: .7rem; color: {MUTED}; }}
.sect {{ font-size: 1.02rem; font-weight: 700; color: {INK}; margin: 18px 0 8px 0; }}
.lb {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; overflow: hidden; }}
.lbrow {{ display:grid; grid-template-columns: 36px 1.6fr 0.7fr 0.8fr 0.8fr 0.9fr; align-items:center; padding: 9px 14px; border-top: 1px solid {LINE}; font-size: .84rem; }}
.lbrow.h {{ border-top: none; background: #FAFBFD; color: {MUTED}; font-size: .7rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
.lbrow .rk {{ font-weight: 800; color: {MUTED}; }}
.lbrow.t3 .rk {{ color: {PINK}; }}
.nm {{ font-weight: 650; color: {INK}; }}
.sm {{ font-size: .72rem; color: {MUTED}; }}
.bar {{ height: 6px; background: #EEF1F7; border-radius: 4px; overflow: hidden; margin-top: 3px; }}
.bar > div {{ height: 100%; border-radius: 4px; }}
.pc {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; }}
.pc.t1 {{ border-color: {PINK}; box-shadow: 0 0 0 3px rgba(214,51,122,.08); }}
.pchead {{ display:flex; align-items:baseline; gap: 10px; }}
.pcrank {{ font-size: 1.35rem; font-weight: 800; color: {ROYAL}; }}
.pc.t1 .pcrank {{ color: {PINK}; }}
.pcname {{ font-size: 1.05rem; font-weight: 750; color: {INK}; }}
.pcscore {{ margin-left:auto; font-size: 1.25rem; font-weight: 800; color: {INK}; }}
.stats {{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 10px 0; }}
.stats div {{ background: #F8FAFD; border-radius: 9px; padding: 6px 8px; }}
.stats b {{ display:block; font-size: .95rem; color: {INK}; }}
.stats span {{ font-size: .66rem; color: {MUTED}; }}
.why {{ font-size: .8rem; color: {INK}; margin: 0; padding-left: 16px; }}
.why li {{ margin-bottom: 2px; }}
.side h4 {{ margin: 0 0 6px 0; font-size: .9rem; color: {INK}; }}
.side p {{ font-size: .8rem; color: {INK}; margin: 0 0 8px 0; }}
.flow {{ display:flex; flex-direction: column; align-items:center; gap: 4px; }}
.fbox {{ background: {CARD}; border: 1px solid {LINE}; border-left: 4px solid {ROYAL}; border-radius: 10px; padding: 8px 14px; width: 100%; max-width: 640px; }}
.fbox b {{ color: {INK}; }} .fbox span {{ font-size: .78rem; color: {MUTED}; }}
.arrow {{ color: {MUTED}; font-size: 1rem; line-height: 1; }}
.tl {{ display:grid; grid-template-columns: 3fr 1fr 1fr; gap: 6px; }}
.tl div {{ border-radius: 10px; padding: 12px; color: white; font-weight: 600; font-size: .85rem; }}
.tl small {{ display:block; font-weight: 400; opacity: .85; font-size: .72rem; }}
.lbl {{ display:inline-block; font-size: .66rem; font-weight: 700; padding: 1px 7px; border-radius: 6px; margin-right: 6px; }}
.lbl.v {{ background: #E6F4EC; color: {GOOD}; }} .lbl.i {{ background: #FBF2E3; color: #8A5A10; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    D = {}
    for f in ["final_batter_rankings", "final_bowler_rankings", "player_metrics", "model_comparison", "validation_results",
              "test_results", "selection_log", "sensitivity_results", "pressure_curves", "pressure_sample",
              "pressure_summary", "delivery_model_diagnostics", "wp_calibration_2025", "player_breakdowns"]:
        fp = OUT / f"{f}.csv"
        D[f] = pd.read_csv(fp if fp.exists() else OUT / f"{f}.csv.gz")
    for r in ("bat", "bowl"):
        D[f"rs_{r}"] = pd.read_csv(OUT / f"bootstrap_rank_samples_{r}.csv")
        D[f"t10_{r}"] = pd.read_csv(OUT / f"top10_by_model_{r}.csv")
    D["S"] = json.load(open(OUT / "dataset_summary.json"))
    D["why"] = json.load(open(OUT / "why_this_player.json"))
    D["manifest"] = json.load(open(OUT / "run_manifest.json"))
    D["evidence"] = pd.read_csv(EVD / "player_match_evidence.csv")
    vf = EVD / "player_video_evidence.csv"
    D["video"] = pd.read_csv(vf) if vf.exists() else pd.DataFrame(columns=["player"])
    D["prev"] = json.load(open(EVD / "previous_winners.json"))
    return D


D = load()
S, P = D["S"], D["player_metrics"]
FINAL = {"bat": D["final_batter_rankings"], "bowl": D["final_bowler_rankings"]}


def style(fig, h=340, legend=True):
    fig.update_layout(template="none", height=h, margin=dict(l=8, r=8, t=38, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color=INK, size=12),
                      title_font=dict(size=13.5, color=INK), showlegend=legend,
                      legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", title_text=""),
                      hoverlabel=dict(bgcolor="white", font_size=12))
    fig.update_xaxes(gridcolor="#EEF1F6", zerolinecolor="#D5DAE5", linecolor="#D5DAE5")
    fig.update_yaxes(gridcolor="#EEF1F6", zerolinecolor="#D5DAE5", linecolor="#D5DAE5")
    return fig


def card_chart(fig, h=340, legend=True):
    with st.container(border=True):
        st.plotly_chart(style(fig, h, legend), use_container_width=True, config={"displayModeBar": False})


def kpi(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="kl">{label}</div><div class="kv">{value}</div><div class="ks">{sub}</div></div>',
                unsafe_allow_html=True)


def f1(v, d=1, sign=True):
    return "—" if pd.isna(v) else (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def topbar(section, title, subtitle):
    st.markdown(f"""<div class="topbar"><span class="crumb">SupeRR Selector S2 / <b>{section}</b></span><span class="spacer"></span>
      <span class="chip blue">Model {S['selected_model']} · {S['selected_model_name']}</span>
      <span class="chip">{S['n_boot']:,} match bootstraps</span><span class="chip pink">Official Data Pack only</span></div>
      <h1 class="title">{title}</h1><div class="subtitle">{subtitle}</div>""", unsafe_allow_html=True)


def leaderboard(role):
    fr = FINAL[role]; mx = fr.score_rel100.max()
    rows = "".join(
        f"""<div class="lbrow {'t3' if r['rank'] <= 3 else ''}"><div class="rk">{int(r['rank'])}</div>
        <div><div class="nm">{r.player}</div><div class="sm">{r.role_label}</div></div>
        <div><b>{f1(r.score_rel100)}</b><div class="bar"><div style="width:{max(4, 100 * r.score_rel100 / mx):.0f}%;background:{ROLE_C[role]}"></div></div></div>
        <div><b>{r.p_top10:.0%}</b></div>
        <div>{int(r.boot_median_rank)}<div class="sm">{int(r.boot_rank_lo)}–{int(r.boot_rank_hi)}</div></div>
        <div>{int(r.hp_balls)}<div class="sm">{int(r.matches)} matches</div></div></div>""" for _, r in fr.iterrows())
    st.markdown(f"""<div class="lb"><div class="lbrow h"><div>#</div><div>Player</div><div>Score</div><div>P(Top 10)</div>
        <div>Median rank</div><div>Pressure balls</div></div>{rows}</div>""", unsafe_allow_html=True)


def spark(role, name):
    rs = D[f"rs_{role}"]
    if name not in rs.columns:
        return None
    v = rs[name].fillna(61).clip(upper=61)
    c = np.bincount(v.astype(int), minlength=62)[1:]
    f = go.Figure(go.Bar(x=np.arange(1, 62), y=c, marker_color=[ROLE_C[role] if i <= 10 else "#CBD2E1" for i in range(1, 62)],
                         marker_line_width=0, hovertemplate="rank %{x}: %{y} resamples<extra></extra>"))
    f.update_layout(height=70, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, bargap=.1, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return f


def goto_profile(role, pid):
    st.session_state.prof_role, st.session_state.prof_pid = role, int(pid)
    st.session_state.page = "👤  Player Profiles"


NAV = [("ANALYSIS", ["🏠  Overview", "🏏  Top 10 Batters", "🎯  Top 10 Bowlers", "👤  Player Profiles", "📊  Pressure Analysis"]),
       ("EVIDENCE", ["🔬  Model Comparison", "📈  Validation", "⚖️  Sensitivity & Fairness", "🎥  Match Evidence"]),
       ("CONTEXT", ["🏆  Previous Winners", "🧠  Methodology", "ℹ️  About / Reproducibility"])]
PAGES = [p for _, ps in NAV for p in ps]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
with st.sidebar:
    st.markdown('<div class="brand">SupeRR Selector <span class="dot">◆</span> S2</div>'
                '<div style="font-size:.72rem;opacity:.6">Pressure-Based T20 Player Intelligence</div>', unsafe_allow_html=True)
    page = st.radio("nav", PAGES, key="page", label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"<div style='font-size:.72rem;opacity:.7'>Selected model <b>{S['selected_model']}</b><br>"
                f"Seed {S['seed']} · run {D['manifest']['rankings_sha256'][:10]}</div>", unsafe_allow_html=True)
SEC = page.split("  ", 1)[1]

# =================================================================== OVERVIEW
if SEC == "Overview":
    topbar("Overview", "SupeRR Selector S2", "Pressure-Based T20 Player Intelligence — who delivers most reliably when the game is on the line, "
                                              "measured only from the official IPL + T20I Data Pack (2022–2026).")
    k = st.columns(7)
    vals = [("Matches analysed", f"{S['matches_analysed']:,}", f"{S['ipl_matches']} IPL · {S['t20i_matches']} T20I"),
            ("Deliveries analysed", f"{S['deliveries']:,}", f"{S['legal_deliveries']:,} legal"),
            ("Eligible batters", S["eligible_batters"], "after sample-size rules"),
            ("Eligible bowlers", S["eligible_bowlers"], "after sample-size rules"),
            ("High-pressure opportunities", f"{S['pressure_deliveries']:,}", "top 25% pressure (2022–24 cut-off)"),
            ("Model selected", S["selected_model"], S["selected_model_name"]),
            ("Bootstrap iterations", f"{S['n_boot']:,}", "match-level, stratified")]
    for c, (a, b, s_) in zip(k, vals):
        with c: kpi(a, b, s_)
    main, side = st.columns([3.2, 1])
    with main:
        a, b = st.columns(2)
        with a:
            st.markdown('<div class="sect">Top 10 Batters</div>', unsafe_allow_html=True); leaderboard("bat")
        with b:
            st.markdown('<div class="sect">Top 10 Bowlers</div>', unsafe_allow_html=True); leaderboard("bowl")
        st.markdown(f"<div class='sm' style='margin-top:6px'>Score = {S['metric_desc']['bat']} / {S['metric_desc']['bowl']}; "
                    "empirical-Bayes shrunk. Median rank and 80% interval from 1,000 match bootstraps.</div>", unsafe_allow_html=True)
    with side:
        st.markdown('<div class="sect">At a glance</div>', unsafe_allow_html=True)
        vm = S["validation_mean_spearman"]; stb = {d["role"]: d for d in S["top10_stability"]}
        sr = D["sensitivity_results"]; grp = sr[sr.group != "Model"].groupby("group").top10_overlap.mean().sort_values()
        fb, fw = FINAL["bat"].iloc[0], FINAL["bowl"].iloc[0]
        st.markdown(f"""<div class="card side">
        <h4>Model decision</h4><p>{S['selected_model']} won the 2025 out-of-time test (mean Spearman {vm[S['selected_model']]:.3f}); no richer model
        cleared the pre-registered bar. Range across models: {min(vm.values()):.3f}–{max(vm.values()):.3f}.</p>
        <h4>Most stable picks</h4><p>{fb.player} (Top-10 in {fb.p_top10:.0%} of resamples) and {fw.player} ({fw.p_top10:.0%}).
        Average Top-10 probability of the lists: {stb['bat']['mean_p_top10_of_top10']:.0%} / {stb['bowl']['mean_p_top10_of_top10']:.0%}.</p>
        <h4>Assumption that matters most</h4><p>{grp.index[0]} (mean overlap {grp.iloc[0]:.1f}/10). Least: {grp.index[-1]} ({grp.iloc[-1]:.1f}/10).</p>
        <h4>Key finding</h4><p>Individual "pressure response" slopes shrink almost to zero — reliability comes from consistent high-pressure
        output across many matches, not from a personal clutch effect.</p></div>""", unsafe_allow_html=True)
        st.write("")
        f = go.Figure()
        for role in ("bat", "bowl"):
            fr = FINAL[role]
            f.add_trace(go.Bar(y=[p.split()[-1] for p in fr.player], x=fr.p_top10, orientation="h", name=ROLE_W[role] + "s",
                               marker_color=ROLE_C[role], hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
        f.update_layout(title="Top-10 probability", barmode="group", xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"))
        style(f, 460); f.update_layout(legend=dict(orientation="h", y=-0.08, x=0))
        with st.container(border=True):
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})

# =================================================================== TOP 10 PAGES
elif SEC in ("Top 10 Batters", "Top 10 Bowlers"):
    role = "bat" if SEC == "Top 10 Batters" else "bowl"
    fr = FINAL[role]
    topbar(SEC, SEC, f"Ranked by {S['metric_desc'][role]} (empirical-Bayes shrunk). Each card's 'Why selected?' is generated from that player's computed metrics.")
    cols = st.columns(2)
    for i, (_, r) in enumerate(fr.iterrows()):
        why = D["why"][role].get(str(int(r.player_id)), {})
        bl = "".join(f"<li>{b}</li>" for b in why.get("bullets", [])[:4])
        with cols[i % 2]:
            st.markdown(f"""<div class="pc {'t1' if r['rank'] == 1 else ''}"><div class="pchead"><span class="pcrank">#{int(r['rank'])}</span>
            <span class="pcname">{r.player}</span><span class="chip">{r.role_label}</span><span class="pcscore">{f1(r.score_rel100)}</span></div>
            <div class="stats"><div><b>{r.p_top10:.0%}</b><span>Top-10 probability</span></div><div><b>{int(r.hp_balls)}</b><span>pressure balls · {int(r.matches)} m</span></div>
            <div><b>{int(r.boot_median_rank)}</b><span>median rank</span></div><div><b>{int(r.boot_rank_lo)}–{int(r.boot_rank_hi)}</b><span>80% rank interval</span></div></div>
            <div class="sm" style="font-weight:600;margin-bottom:2px">Why selected?</div><ul class="why">{bl}</ul></div>""", unsafe_allow_html=True)
            sp = spark(role, r.player)
            if sp is not None:
                st.plotly_chart(sp, use_container_width=True, config={"displayModeBar": False}, key=f"sp{role}{i}")
            st.button(f"Open {r.player}'s profile", key=f"b{role}{i}", on_click=goto_profile, args=(role, r.player_id))
    st.markdown('<div class="sect">Expected vs actual under pressure</div>', unsafe_allow_html=True)
    c = st.columns(2)
    with c[0]:
        f = go.Figure()
        y = list(fr.player)[::-1]
        f.add_trace(go.Scatter(x=fr.hp_expected_runs_100[::-1], y=y, mode="markers", name="Expected (average player, same state)",
                               marker=dict(color="#9AA4B8", size=10, symbol="line-ns-open", line=dict(width=3))))
        f.add_trace(go.Scatter(x=fr.hp_actual_runs_100[::-1], y=y, mode="markers", name="Actual",
                               marker=dict(color=ROLE_C[role], size=11, line=dict(color="white", width=2))))
        for _, r in fr.iterrows():
            f.add_shape(type="line", x0=r.hp_expected_runs_100, x1=r.hp_actual_runs_100, y0=r.player, y1=r.player,
                        line=dict(color=ROLE_C[role], width=2))
        f.update_layout(title=("Runs scored" if role == "bat" else "Runs conceded") + " per 100 high-pressure balls",
                        xaxis_title="runs per 100 balls")
        card_chart(f, 420)
    with c[1]:
        rs = D[f"rs_{role}"]
        names = [n for n in fr.player if n in rs.columns]
        long = rs[names].melt(var_name="player", value_name="rank").dropna()
        f = px.box(long, x="rank", y="player", color_discrete_sequence=[ROLE_C[role]], points=False,
                   category_orders={"player": names})
        f.add_vline(x=10.5, line_dash="dot", line_color=MUTED, annotation_text="Top 10")
        f.update_layout(title="Bootstrap rank distribution (1,000 match resamples)", xaxis_title="rank in resample", yaxis_title="")
        card_chart(f, 420, False)
    show = fr[["rank", "player", "role_label", "score_rel100", "score_lo_rel100", "score_hi_rel100", "vaa_pressure", "vaa_normal",
               "lift_raw", "hp_balls", "matches", "ipl_balls", "t20i_balls", "p_top10", "boot_median_rank"]].copy()
    show["lift_raw"] *= 100
    show.columns = ["Rank", "Player", "Role", "Score", "90% low", "90% high", "Pressure VAA", "Normal VAA", "Lift", "Pressure balls",
                    "Matches", "IPL balls", "T20I balls", "P(Top 10)", "Median rank"]
    with st.container(border=True):
        st.dataframe(show, hide_index=True, use_container_width=True,
                     column_config={**{k: st.column_config.NumberColumn(format="%+.1f") for k in ["Score", "90% low", "90% high", "Pressure VAA", "Normal VAA", "Lift"]},
                                    "P(Top 10)": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1)})

# =================================================================== PLAYER PROFILES
elif SEC == "Player Profiles":
    topbar("Player Profiles", "Player Profiles", "Every eligible player — open any Top-10 card to jump here.")
    roles = {"Batter": "bat", "Bowler": "bowl"}
    c = st.columns([1, 3])
    d_role = st.session_state.get("prof_role", "bat")
    rl = c[0].radio("Role", list(roles), index=0 if d_role == "bat" else 1, horizontal=True)
    role = roles[rl]
    pool = P[P.role == role].set_index("player_id")
    ids = list(pool.index)
    pid0 = st.session_state.get("prof_pid", ids[0])
    pid = c[1].selectbox("Player (all eligible)", ids, index=ids.index(pid0) if pid0 in ids else 0,
                         format_func=lambda i: f"#{int(pool.loc[i, 'rank'])}  {pool.loc[i, 'player']}")
    r = pool.loc[pid]
    st.markdown(f"""<div class="card" style="margin:6px 0 12px 0"><span class="pcrank">#{int(r['rank'])}</span> <span class="pcname" style="font-size:1.4rem">{r.player}</span>
    <span class="chip">{r.role_label}</span><div class="sm">#{int(r['rank'])} of {len(pool)} eligible {rl.lower()}s · {r.teams} · seasons {r.seasons}</div></div>""",
                unsafe_allow_html=True)
    k = st.columns(8)
    for c_, (a, b, s_) in zip(k, [
        ("Final rank", f"#{int(r['rank'])}", f"median {r.boot_median_rank:.0f} · {r.boot_rank_lo:.0f}–{r.boot_rank_hi:.0f}"),
        ("Final score", f1(r.score_rel100), f"90%: {f1(r.score_lo_rel100)} to {f1(r.score_hi_rel100)}"),
        ("Pressure VAA", f1(r.vaa_pressure), "per 100 high-pressure balls"),
        ("Normal VAA", f1(r.vaa_normal), "per 100 balls, P < 0.5"),
        ("Pressure lift", f1(r.lift_raw * 100), f"{int(r.n_hi)} vs {int(r.n_lo)} balls"),
        ("Expected → actual", f"{r.hp_expected_runs_100:.0f} → {r.hp_actual_runs_100:.0f}", "runs /100 HP balls" + (" conceded" if role == "bowl" else "")),
        ("Sample size", f"{int(r.hp_balls)}", f"pressure balls of {int(r.n_balls)}"),
        ("Bootstrap stability", f"{r.p_top10:.0%}", f"Top-10 · Top-3 {r.p_top3:.0%}")]):
        with c_: kpi(a, b, s_)
    br = D["player_breakdowns"]; br = br[(br.role == role) & (br.pid == pid)]
    c = st.columns(2)
    with c[0]:
        cur = D["pressure_curves"]; cur = cur[(cur.role == role) & (cur.pid == pid)]
        f = go.Figure()
        if len(cur):
            f.add_trace(go.Scatter(x=cur.p, y=cur.vaa, mode="markers+lines", name="Observed by pressure band", line=dict(color=ROLE_C[role], width=2),
                                   marker=dict(size=np.clip(cur.n / 15, 8, 24), line=dict(color="white", width=2)), customdata=cur.n,
                                   hovertemplate="P≈%{x:.2f}: %{y:+.1f} (%{customdata} balls)<extra></extra>"))
        pp = np.linspace(0, 1, 21)
        f.add_trace(go.Scatter(x=pp, y=(r.alpha_post + r.beta_post * (pp - .5)) * 100, mode="lines", name="EB-shrunk response",
                               line=dict(color=NAVY, dash="dash", width=2)))
        f.add_hline(y=0, line_color="#C9D0DE")
        f.update_layout(title="Pressure response (VAA per 100 balls vs pressure percentile)", xaxis_title="pressure percentile", yaxis_title="VAA /100")
        card_chart(f, 340)
    with c[1]:
        sp = D[f"rs_{role}"]
        if r.player in sp.columns:
            v = sp[r.player].fillna(len(pool) + 1)
            f = px.histogram(v, nbins=int(min(40, v.max())), color_discrete_sequence=[ROLE_C[role]])
            f.add_vline(x=10.5, line_dash="dot", line_color=MUTED, annotation_text="Top 10")
            f.update_layout(title="Rank distribution across 1,000 match bootstraps", xaxis_title="rank", yaxis_title="resamples")
            card_chart(f, 340, False)
        else:
            st.info("Rank samples are stored for the top 30 players per role.")
    if len(br):
        c = st.columns(3)
        for col, dim, ttl in ((c[0], "phase", "By phase"), (c[1], "competition", "IPL vs T20I"), (c[2], "pressure band", "Expected vs actual by pressure band")):
            q = br[br.dimension == dim]
            with col:
                f = go.Figure()
                if dim == "pressure band":
                    f.add_trace(go.Bar(x=q.level, y=q.expected, name="Expected", marker_color="#C3CBDB"))
                    f.add_trace(go.Bar(x=q.level, y=q.actual, name="Actual", marker_color=ROLE_C[role]))
                    f.update_layout(barmode="group", yaxis_title="runs /100 balls" + (" conceded" if role == "bowl" else ""))
                else:
                    f.add_trace(go.Bar(x=q.level, y=q.vaa, marker_color=[GOOD if v > 0 else BAD for v in q.vaa],
                                       text=[f"{v:+.0f}<br>({n} b)" for v, n in zip(q.vaa, q.balls)], textposition="outside"))
                    f.update_layout(yaxis_title="VAA /100 balls", showlegend=False)
                f.update_layout(title=ttl)
                card_chart(f, 300, dim == "pressure band")
    else:
        st.caption("Phase / competition breakdowns are stored for the top 30 players per role.")
    why = D["why"][role].get(str(int(pid)))
    if why:
        st.markdown('<div class="sect">Why this player was selected</div>', unsafe_allow_html=True)
        lab = dict(pressure="Pressure performance", normal="Normal performance", response="Pressure response", lift="Pressure lift",
                   reliability="Reliability", consistency="Consistency", agreement="Model agreement", stability="Bootstrap stability",
                   cricket="Cricket interpretation")
        a, b = st.columns([1, 1.6])
        with a:
            st.markdown("<div class='card'><ul class='why'>" + "".join(f"<li>{x}</li>" for x in why.get("bullets", [])) + "</ul></div>",
                        unsafe_allow_html=True)
        with b:
            st.markdown("<div class='card'>" + "".join(f"<p style='font-size:.82rem;margin:0 0 6px 0'><b>{lab[k_]}.</b> {v_}</p>"
                                                       for k_, v_ in why.items() if k_ in lab) + "</div>", unsafe_allow_html=True)
        ev = D["evidence"]; ev = ev[(ev.role == role) & (ev.player_id == pid)]
        if len(ev):
            st.markdown('<div class="sect">Match-level evidence (Data Pack)</div>', unsafe_allow_html=True)
            st.dataframe(ev[["date", "competition", "opponent", "match_state", "action", "vaa", "pressure_weighted_vaa", "team_result"]],
                         hide_index=True, use_container_width=True,
                         column_config={"vaa": st.column_config.NumberColumn("VAA", format="%+.1f"),
                                        "pressure_weighted_vaa": st.column_config.NumberColumn("Pressure-weighted VAA", format="%+.1f")})
    else:
        st.caption("The 'Why this player was selected' panel is generated for the final Top 10 in each role.")

# =================================================================== PRESSURE ANALYSIS
elif SEC == "Pressure Analysis":
    topbar("Pressure Analysis", "Pressure Analysis", "Two pressure signals, both built only from pre-delivery state: the composite pressure index "
           "(used by the selected model) and win-probability leverage (research-PDF definition).")
    ps, sm = D["pressure_sample"], D["pressure_summary"]
    ps = ps.assign(inn=ps.innings.map({1: "1st innings", 2: "2nd innings"}))
    c = st.columns(2)
    with c[0]:
        f = px.histogram(ps, x="pi_composite", color="inn", nbins=50, barmode="overlay", opacity=.75, color_discrete_sequence=[ROYAL, AMBER])
        f.update_layout(title="Composite pressure index distribution", xaxis_title="composite index (0–1)", yaxis_title="deliveries")
        card_chart(f)
    with c[1]:
        f = px.histogram(ps, x="li_norm", nbins=80, log_y=True, color_discrete_sequence=[ROYAL])
        f.update_layout(title="Leverage distribution (1.0 = average ball)", xaxis_title="leverage index", yaxis_title="deliveries (log)")
        card_chart(f, legend=False)
    defn = st.radio("High-pressure definition for the next charts", ["Composite index (selected)", "Leverage top 25%"], horizontal=True)
    sc = "hp_comp_share" if defn.startswith("Composite") else "hp_share"
    c = st.columns(2)
    with c[0]:
        g = sm.groupby(["innings", "phase"]).apply(lambda x: np.average(x[sc], weights=x.deliveries), include_groups=False).reset_index(name="hp")
        g["innings"] = g.innings.map({1: "1st innings", 2: "2nd innings"}); g["phase"] = pd.Categorical(g.phase, ["powerplay", "middle", "death"])
        f = px.bar(g.sort_values("phase"), x="phase", y="hp", color="innings", barmode="group", color_discrete_sequence=[ROYAL, AMBER])
        f.update_layout(title="Share of balls that are high pressure, by phase", yaxis_tickformat=".0%", xaxis_title="", yaxis_title="")
        card_chart(f)
    with c[1]:
        g = sm.groupby(["comp_type", "season"]).apply(lambda x: np.average(x[sc], weights=x.deliveries), include_groups=False).reset_index(name="hp")
        f = px.line(g, x="season", y="hp", color="comp_type", markers=True, color_discrete_sequence=[ROYAL, AMBER])
        f.update_traces(line_width=2, marker_size=8)
        f.update_layout(title="High-pressure share by competition and season", yaxis_tickformat=".0%", yaxis_title="", xaxis_title="")
        card_chart(f)
    c = st.columns(2)
    with c[0]:
        b = pd.cut(ps.pressure, np.linspace(0, 1, 11), include_lowest=True)
        g = ps.groupby(b, observed=True)[["vaa_bat_state", "vaa_bowl_state"]].mean() * 100
        g.index = [iv.mid for iv in g.index]
        f = go.Figure()
        f.add_trace(go.Scatter(x=g.index, y=g.vaa_bat_state, mode="lines+markers", name="Batters", line=dict(color=ROYAL, width=2)))
        f.add_trace(go.Scatter(x=g.index, y=g.vaa_bowl_state, mode="lines+markers", name="Bowlers", line=dict(color=AMBER, width=2)))
        f.add_hline(y=0, line_color="#C9D0DE")
        f.update_layout(title="Pressure vs performance (league average VAA, centred by design)", xaxis_title="pressure percentile", yaxis_title="VAA /100")
        card_chart(f)
    with c[1]:
        i2 = ps[ps.innings == 2]
        f = px.density_heatmap(i2, x="wp", y="pressure", nbinsx=25, nbinsy=25, color_continuous_scale=["#F3F6FD", "#AFC2F2", ROYAL, NAVY])
        f.update_layout(title="Chases: leverage peaks when the result is in the balance", xaxis_title="chasing side win probability", yaxis_title="leverage percentile")
        card_chart(f, legend=False)

# =================================================================== MODEL COMPARISON
elif SEC == "Model Comparison":
    topbar("Model Comparison", "Model Comparison", "Five candidate models form an ablation ladder; selection used only the 2025 out-of-time test.")
    mc = D["model_comparison"]
    st.markdown(f"<span class='chip pink'>FINAL SELECTED MODEL: {S['selected_model']} — {S['selected_model_name']}</span>", unsafe_allow_html=True)
    st.write("")
    t = mc[["model", "name", "pressure", "contextual_vaa", "pressure_response", "wicket_value", "valid_mean_spearman", "test_mean_spearman",
            "valid_bat_top20_overlap", "valid_bowl_top20_overlap", "complexity", "selected"]].copy()
    t.columns = ["Model", "Name", "Pressure", "Context", "Response", "Wicket value", "Validation ρ 2025", "Test ρ 2026",
                 "Top-20 overlap bat", "Top-20 overlap bowl", "Complexity", "Selected?"]
    with st.container(border=True):
        st.dataframe(t, hide_index=True, use_container_width=True,
                     column_config={"Validation ρ 2025": st.column_config.NumberColumn(format="%.3f"), "Test ρ 2026": st.column_config.NumberColumn(format="%.3f")})
    c = st.columns(2)
    for col, (d, ttl) in zip(c, ((D["validation_results"], "2025 validation"), (D["test_results"], "2026 final test (after selection)"))):
        d = d.assign(series=d.role.map({"bat": "Batters", "bowl": "Bowlers"}) + " · " + d.target.map({"T_lev": "context-adjusted target", "T_raw": "raw target"}))
        f = px.bar(d, x="model", y="spearman", color="series", barmode="group",
                   color_discrete_map={"Batters · context-adjusted target": ROYAL, "Batters · raw target": ROYAL_T,
                                       "Bowlers · context-adjusted target": AMBER, "Bowlers · raw target": AMBER_T})
        f.update_layout(title=f"Spearman: train-period score vs holdout pressure performance — {ttl}", yaxis_title="Spearman ρ", xaxis_title="")
        with col: card_chart(f, 380)
    st.markdown('<div class="sect">Why the final model was selected</div>', unsafe_allow_html=True)
    vm = S["validation_mean_spearman"]
    st.markdown(f"""<div class="card"><p>Pre-registered rule: a richer model replaces the current choice only if its mean Spearman improvement has a 90%
    bootstrap interval (1,000 resamples of 2025 matches) entirely above zero. <b>No candidate cleared that bar</b>, so the simplest model,
    <b>{S['selected_model']}</b>, produces the final ranking. Mean validation ρ: {", ".join(f"{k} {v:.3f}" for k, v in vm.items())}.
    The richer models are better on the context-adjusted bowling target and worse on raw batting outcomes; on the 2026 test they are
    slightly better overall but within noise. We report this honestly rather than promote the more sophisticated model.</p></div>""",
                unsafe_allow_html=True)
    with st.container(border=True):
        st.dataframe(D["selection_log"].round(3), hide_index=True, use_container_width=True)
    c = st.columns(2)
    for i, role in enumerate(("bat", "bowl")):
        with c[i]:
            st.markdown(f'<div class="sect">Top 10 under each model — {ROLE_W[role].lower()}s</div>', unsafe_allow_html=True)
            st.dataframe(D[f"t10_{role}"], hide_index=True, use_container_width=True)

# =================================================================== VALIDATION
elif SEC == "Validation":
    topbar("Validation", "Validation", "Forward-only temporal validation plus 1,000 match-level bootstraps.")
    st.markdown(f"""<div class="tl"><div style="background:{NAVY}">TRAIN · 2022–2024<small>all model fitting, pressure calibration, priors</small></div>
    <div style="background:{ROYAL}">VALIDATE · 2025<small>model selection only</small></div>
    <div style="background:{PINK}">TEST · 2026<small>run once, after selection</small></div></div>""", unsafe_allow_html=True)
    st.write("")
    sel = S["selected_model"]
    vr, tr = D["validation_results"], D["test_results"]
    k = st.columns(4)
    for c_, (lab, d, role) in zip(k, (("Validation ρ · batters", vr, "bat"), ("Validation ρ · bowlers", vr, "bowl"),
                                      ("Test ρ · batters", tr, "bat"), ("Test ρ · bowlers", tr, "bowl"))):
        q = d[(d.model == sel) & (d.role == role)]
        with c_: kpi(lab, f"{q.spearman.mean():.3f}", f"mean of 2 targets · {int(q.n_players.mean())} players")
    c = st.columns(2)
    with c[0]:
        cal = D["wp_calibration_2025"]; f = go.Figure()
        for inn, col in ((1, ROYAL), (2, AMBER)):
            q = cal[cal.innings == inn]
            f.add_trace(go.Scatter(x=q.pred, y=q.obs, mode="markers+lines", name=f"Innings {inn}", line=dict(color=col, width=2),
                                   marker=dict(size=np.clip(q.n / 400, 7, 20), line=dict(color="white", width=2))))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#C3CBDB", dash="dot"), showlegend=False))
        f.update_layout(title="Win-probability calibration on 2025 (fitted on 2022–24)", xaxis_title="predicted", yaxis_title="observed")
        card_chart(f, 360)
    with c[1]:
        rows = []
        for role in ("bat", "bowl"):
            fr = P[P.role == role].head(20)
            rows.append(fr.assign(r=ROLE_W[role]))
        q = pd.concat(rows)
        f = go.Figure()
        for role, col in (("Batter", ROYAL), ("Bowler", AMBER)):
            z = q[q.r == role]
            f.add_trace(go.Scatter(x=z.boot_median_rank, y=z.player, mode="markers", name=role + "s", marker=dict(color=col, size=9),
                                   error_x=dict(type="data", symmetric=False, array=z.boot_rank_hi - z.boot_median_rank,
                                                arrayminus=z.boot_median_rank - z.boot_rank_lo, color=col, thickness=1.5)))
        f.add_vline(x=10.5, line_dash="dot", line_color=MUTED)
        f.update_layout(title="Rank intervals (median, 10th–90th percentile), top 20 per role", xaxis_title="bootstrap rank",
                        yaxis=dict(autorange="reversed", tickfont=dict(size=9)))
        card_chart(f, 360)
    c = st.columns(2)
    for col, role in zip(c, ("bat", "bowl")):
        fr = FINAL[role]; f = go.Figure()
        for k_, colr, nm in (("p_top10", ROLE_C[role], "Top 10"), ("p_top5", ROYAL_T if role == "bat" else AMBER_T, "Top 5"), ("p_top3", NAVY, "Top 3")):
            f.add_trace(go.Bar(x=fr.player, y=fr[k_], name=nm, marker_color=colr))
        f.update_layout(barmode="group", title=f"{ROLE_W[role]}s: inclusion probabilities", yaxis_tickformat=".0%")
        with col: card_chart(f, 330)
    with st.expander("Delivery-level model diagnostics (out-of-time)"):
        st.dataframe(D["delivery_model_diagnostics"].round(4), hide_index=True, use_container_width=True)

# =================================================================== SENSITIVITY
elif SEC == "Sensitivity & Fairness":
    topbar("Sensitivity & Fairness", "Sensitivity & Fairness", "Does the Top 10 survive reasonable changes to eligibility, competition weighting, pressure definition and wicket value?")
    sr = D["sensitivity_results"]
    grp = sr[sr.group != "Model"].groupby("group").top10_overlap.mean().sort_values()
    mod = sr[sr.group == "Model"].top10_overlap.mean()
    st.markdown(f"""<div class="card"><b>Which assumptions matter most?</b> Mean Top-10 overlap with the final lists (10 = identical):
    {" · ".join(f"<b>{g}</b> {v:.1f}" for g, v in grp.items())} (switching to another candidate model: {mod:.1f}). The pressure definition moves the lists most;
    eligibility thresholds and equal IPL/T20I weighting barely move them. The T20I-only batting list differs strongly because the batting
    evidence is dominated by IPL deliveries.</div>""", unsafe_allow_html=True)
    role = st.radio("Role", ["bat", "bowl"], format_func=lambda r: ROLE_W[r] + "s", horizontal=True)
    s = sr[sr.role == role]
    fr = FINAL[role]
    mat = pd.DataFrame([[1 if n in str(t).split(" | ") else 0 for t in s.top10] for n in fr.player], index=fr.player,
                       columns=s.group + ": " + s.variant)
    f = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, colorscale=[[0, "#F1F3F8"], [1, ROLE_C[role]]], showscale=False,
                             xgap=2, ygap=2, hovertemplate="%{y}<br>%{x}<br>in Top 10: %{z}<extra></extra>"))
    f.update_layout(title="Does each final Top-10 player stay in the Top 10 under each alternative?", xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                    yaxis=dict(autorange="reversed"))
    card_chart(f, 520, False)
    f = px.bar(s, x="top10_overlap", y="variant", color="group", orientation="h", range_x=[0, 10],
               color_discrete_sequence=[ROYAL, AMBER, PINK, NAVY, ROYAL_T, AMBER_T, "#9AA4B8"])
    f.update_layout(title="Top-10 overlap by variant", xaxis_title="players in common with final Top 10", yaxis_title="")
    card_chart(f, max(380, 24 * len(s)))
    pm = P[P.role == role].head(40)
    f = px.scatter(pm, x="ipl_balls", y="t20i_balls", size="hp_balls", hover_name="player",
                   color=np.where(pm["rank"] <= 10, "Final Top 10", "Rank 11–40"), color_discrete_sequence=[ROLE_C[role], "#C3CBDB"])
    f.update_layout(title="IPL vs T20I evidence mix (top 40)", xaxis_title="IPL balls", yaxis_title="T20I balls")
    card_chart(f, 380)

# =================================================================== MATCH EVIDENCE
elif SEC == "Match Evidence":
    topbar("Match Evidence", "Match & Video Evidence", "Qualitative supporting evidence only — nothing on this page is a ranking input.")
    ev = D["evidence"]
    role = st.radio("Role", ["bat", "bowl"], format_func=lambda r: ROLE_W[r] + "s", horizontal=True)
    e = ev[ev.role == role]
    who = st.selectbox("Player", list(FINAL[role].player))
    with st.container(border=True):
        st.markdown("**Highest pressure-weighted matches (from the Data Pack)**")
        st.dataframe(e[e.player == who][["date", "competition", "stage", "team", "opponent", "match_state", "peak_pressure", "action", "vaa",
                                          "pressure_weighted_vaa", "team_result"]], hide_index=True, use_container_width=True,
                     column_config={"vaa": st.column_config.NumberColumn("VAA", format="%+.1f"),
                                    "pressure_weighted_vaa": st.column_config.NumberColumn("Pressure-weighted VAA", format="%+.1f"),
                                    "peak_pressure": st.column_config.NumberColumn("Peak leverage pct.", format="%.2f")})
    vv = D["video"][D["video"].player == who]
    if len(vv):
        for _, r in vv.iterrows():
            st.markdown(f"""<div class="card"><span class="lbl i">QUALITATIVE</span><b>{r.match}</b><p style="font-size:.85rem;margin:6px 0">{r.observation}</p>
            <a href="{r.url}" target="_blank">{r.source_title}</a> <span class="sm">· {r.source_type} · retrieved {r.retrieved}</span></div>""",
                        unsafe_allow_html=True)
    else:
        st.caption("No public video/report link was recorded for this player; the Data Pack evidence above stands on its own.")

# =================================================================== PREVIOUS WINNERS
elif SEC == "Previous Winners":
    pv = D["prev"]
    topbar("Previous Winners", "Season 1 — Previous Winners", "What was publicly discoverable, clearly labelled as verified or inferred.")
    st.markdown(f"<div class='card'><span class='lbl i'>METHOD</span>{pv['method_note']}</div>", unsafe_allow_html=True)
    st.markdown('<div class="sect">Verified (public sources)</div>', unsafe_allow_html=True)
    for f_ in pv["verified_facts"]:
        st.markdown(f"<div class='card' style='margin-bottom:8px'><span class='lbl v'>VERIFIED</span>{f_['fact']}<br>"
                    f"<a class='sm' href='{f_['source']}' target='_blank'>{f_['source']}</a></div>", unsafe_allow_html=True)
    st.markdown('<div class="sect">Claimed · implemented · inferred</div>', unsafe_allow_html=True)
    cii = pv["claimed_vs_implemented_vs_inferred"]
    st.markdown(f"""<div class="card"><p><span class='lbl v'>CLAIMED</span>{cii['claimed']}</p><p><span class='lbl v'>IMPLEMENTED</span>{cii['implemented']}</p>
    <p><span class='lbl i'>INFERENCE</span>{cii['inferred']}</p></div>""", unsafe_allow_html=True)
    st.markdown('<div class="sect">Public approaches vs our solution</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(pv["comparison"]).rename(columns={"aspect": "Aspect", "previous": "Previous / public approaches", "ours": "Our solution"}),
                 hide_index=True, use_container_width=True)
    st.markdown('<div class="sect">Lessons learned <span class="lbl i">INFERENCE</span></div>', unsafe_allow_html=True)
    for l in pv["lessons"]:
        st.markdown(f"- {l}")

# =================================================================== METHODOLOGY
elif SEC == "Methodology":
    topbar("Methodology", "Methodology", "From raw deliveries to the final Top 10 — every formula here is the one implemented in src/.")
    a, b = st.columns([1, 1.5])
    steps = [("Raw Data", "Official Data Pack: 6 CSVs, IPL + T20I, 2022–2026"),
             ("Match State Reconstruction", "score, wickets, balls left, target, RRR, phase, collapse — before every ball"),
             ("Pressure Modelling", "composite index (selected) + win-probability leverage (compared)"),
             ("Expected Performance", "gradient-boosted xR, xW for an average player in the same state"),
             ("Contextual Value", "VAA with a state-dependent wicket value"),
             ("Empirical-Bayes Shrinkage", "small samples pulled to the average; match-clustered SEs"),
             ("Model Comparison", "M0–M4 ablation ladder"),
             ("Temporal Validation", "train 2022–24 · validate 2025 · test 2026"),
             ("1,000 Match Bootstrap", "rank intervals and Top-10 probabilities"),
             ("Final Top 10", "batters and bowlers")]
    with a:
        st.markdown("<div class='flow'>" + "<div class='arrow'>↓</div>".join(
            f"<div class='fbox'><b>{t}</b><br><span>{d}</span></div>" for t, d in steps) + "</div>", unsafe_allow_html=True)
    with b:
        wv = S["phase_wv"]
        eqs = [
            ("Pressure — composite index (2nd innings)", r"PI=0.50\,\sigma\!\left(\tfrac{RRR-8.5}{1.5}\right)+0.25\,D+0.15\,\tfrac{W}{10}+0.10\,\sigma(\cdot)\,D",
             "σ = logistic function; RRR = required run rate; D = 1 in overs 17–20; W = wickets lost. 1st innings: 0.25·D + 0.40·W/10 + 0.35·collapse "
             "(wickets in the previous 18 balls ÷ 3, capped at 1). High pressure = PI above the 2022–24 75th percentile."),
            ("Pressure — leverage", r"LI_t=\sum_{o}p(o\mid S_t)\,\big|WP(S_t\oplus o)-WP(S_t)\big|",
             "S_t = match state before ball t; o = one of 9 outcomes (0,1,2,3,4,6,W,wide,no-ball); p = outcome probability; WP = win probability; "
             "S⊕o = state after outcome o. Percentile taken within 2022–24."),
            ("Expected performance", r"xR_t=\mathbb{E}[\text{Runs}_t\mid S_t],\qquad xW_t=P(\text{Wicket}_t=1\mid S_t)",
             "What an average batter/bowler produces in the identical state (fitted without the evaluated season/match)."),
            ("Wicket value", r"WV_t=\mathbb{E}[\text{FutureRuns}\mid S_t,\text{no wkt}]-\mathbb{E}[\text{FutureRuns}\mid S_t,\text{wkt}]",
             f"Training means: fixed {S['fixed_wv']:.2f} runs; powerplay {wv['powerplay']:.2f}, middle {wv['middle']:.2f}, death {wv['death']:.2f}."),
            ("Contextual value", r"VAA^{bat}_t=R_t-xR_t-WV_t(O_t-xW_t)\qquad VAA^{bowl}_t=(xR_t-R_t)+WV_t(K_t-xW_t)",
             "R = runs (off the bat / charged to bowler); O = batter dismissed; K = bowler-credited wicket. Higher = better for both."),
            ("Pressure response", r"VAA_{it}=\alpha_i+\beta_i\,(P_t-0.5)+\varepsilon_{it}",
             "α_i = player i at median pressure; β_i = pressure elasticity; lift = β_i × 0.6 (P 0.3 → 0.9)."),
            ("Empirical Bayes", r"\hat\theta_i^{EB}=\frac{\hat\theta_i/V_i+\mu/\tau^2}{1/V_i+1/\tau^2}",
             "θ̂_i = raw estimate; V_i = its match-clustered variance; μ, τ² = mean and spread of true values across eligible players (method of moments)."),
            (f"Final score (selected {S['selected_model']})", r"\text{Score}_i=\text{EB}\Big[\tfrac{1}{n_i}\textstyle\sum_{t\in HP_i}y_{it}\Big]-\mu",
             f"{S['metric_desc']['bat']} (batters); {S['metric_desc']['bowl']} (bowlers). y = runs − WV·out (bat) or WV·wicket − runs conceded (bowl) on high-pressure balls HP_i; per 100 balls."),
        ]
        for t, e, n in eqs:
            with st.container(border=True):
                st.markdown(f"**{t}**"); st.latex(e); st.caption(n)

# =================================================================== ABOUT
elif SEC == "About / Reproducibility":
    topbar("About / Reproducibility", "About / Reproducibility", "One command reproduces every number on this dashboard.")
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><b>Commands</b>", unsafe_allow_html=True)
        st.code("pip install -r requirements.txt\npython run_pipeline.py --rebuild\npython -m src.visualization.figures\n"
                "python report/build_report.py\nstreamlit run dashboard/app.py\npython -m pytest -q tests", language="bash")
        st.markdown(f"""<div class='card'><b>Run fingerprint</b><br><span class='sm'>SHA-256 of both final ranking files</span><br>
        <code>{D['manifest']['rankings_sha256']}</code><br><span class='sm'>Seed {S['seed']} · {S['n_boot']:,} bootstraps ·
        split 2022–24 / 2025 / 2026 · runtime {S['runtime_sec']:.0f}s</span></div>""", unsafe_allow_html=True)
    with b:
        el = S["elig"]
        st.markdown(f"""<div class='card'><b>Data rule</b><p style='font-size:.85rem'>All model training and player rankings use exclusively
        the competition-provided Data Pack ({S['matches_in_pack']} matches in the pack; {S['matches_analysed']} IPL + T20I matches analysed;
        {S['excluded_other_matches']} emerging/A-team matches excluded as outside the brief). External sources are used only for qualitative
        research, previous-winner analysis and supporting video evidence.</p>
        <b>Eligibility</b><p style='font-size:.85rem'>Batters ≥{el['bat']['min_balls']} balls, ≥{el['bat']['min_hp_balls']} high-pressure balls,
        ≥{el['bat']['min_matches']} matches. Bowlers ≥{el['bowl']['min_balls']} balls, ≥{el['bowl']['min_hp_balls']} high-pressure balls,
        ≥{el['bowl']['min_matches']} matches.</p>
        <b>Scope note</b><p style='font-size:.85rem'>Per the organisers' clarification, these lists rank pressure performance; they are not
        retention or auction recommendations.</p></div>""", unsafe_allow_html=True)
