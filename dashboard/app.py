import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    proportions_ztest,
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Steam Player Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# THEME / CSS
# ==================================================
CONTROL_COLOR = "#94A3B8"
TREAT_COLOR = "#6366F1"
ACCENT = "#22C55E"
NEG = "#EF4444"
CLUSTER_PALETTE = px.colors.qualitative.Bold

st.markdown(
    """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px;}

    .hero {
        background: linear-gradient(120deg, #1E1B4B 0%, #4338CA 55%, #6366F1 100%);
        padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(67,56,202,.25);
    }
    .hero h1 {color: white; margin: 0; font-size: 2rem;}
    .hero p {color: #E0E7FF; margin: 6px 0 0 0; font-size: 1.02rem;}

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.18);
        padding: 16px 18px; border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,.04);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stMetric"]:hover {transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,.10);}

    .note {
        background: rgba(99,102,241,.10); padding: 12px 16px; border-radius: 12px;
        border-left: 4px solid #6366F1; font-size: .92rem;
    }
    .verdict {
        padding: 16px 20px; border-radius: 14px; font-size: 1.05rem; font-weight: 500;
    }
    .verdict.good {background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.45);}
    .verdict.bad  {background: rgba(239,68,68,.10); border: 1px solid rgba(239,68,68,.40);}
    .verdict.mid  {background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.45);}

    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 16px;}
</style>
""",
    unsafe_allow_html=True,
)


# ==================================================
# DATA LOADING
# ==================================================
def find_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "data", here / "data"):
        if (candidate / "simulated_ab_test_data.csv").exists():
            return candidate
    return here.parent / "data"


DATA_DIR = find_data_dir()


@st.cache_data
def load_data():
    experiment = pd.read_csv(DATA_DIR / "simulated_ab_test_data.csv")
    players = pd.read_csv(DATA_DIR / "player_features_with_clusters.csv")
    steam = pd.read_csv(DATA_DIR / "cleaned_steam_data.csv")

    for df in (experiment, players, steam):
        df.drop(
            columns=[c for c in df.columns if c.startswith("Unnamed:")],
            errors="ignore",
            inplace=True,
        )

    if "user_id" not in players.columns:
        raise ValueError("user_id missing from player_features_with_clusters.csv.")

    # Make sure the experiment table knows each player's cluster
    if "cluster" not in experiment.columns:
        experiment = experiment.merge(
            players[["user_id", "cluster"]], on="user_id", how="left"
        )

    experiment["group"] = experiment["group"].astype(str)
    return experiment, players, steam


try:
    experiment_df, player_df, steam_df = load_data()
except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.info(f"Looked for CSV files in: `{DATA_DIR}`")
    st.stop()


# ==================================================
# HELPERS
# ==================================================
def split_groups(df: pd.DataFrame):
    return df[df["group"] == "A"], df[df["group"] == "B"]


def safe_div(a, b):
    return a / b if b else np.nan


def ab_stats(df: pd.DataFrame, alpha: float = 0.05):
    """Core A/B statistics for a (possibly filtered) experiment frame."""
    c, t = split_groups(df)
    n_c, n_t = len(c), len(t)
    x_c, x_t = int(c["converted"].sum()), int(t["converted"].sum())
    p_c, p_t = safe_div(x_c, n_c), safe_div(x_t, n_t)

    out = dict(n_c=n_c, n_t=n_t, x_c=x_c, x_t=x_t, p_c=p_c, p_t=p_t)
    out["abs_lift"] = (p_t - p_c) * 100
    out["rel_lift"] = safe_div(p_t - p_c, p_c) * 100 if p_c else np.nan
    out["ctr_c"] = c["clicked"].mean() * 100 if "clicked" in c and n_c else np.nan
    out["ctr_t"] = t["clicked"].mean() * 100 if "clicked" in t and n_t else np.nan

    try:
        z, p = proportions_ztest([x_t, x_c], [n_t, n_c])
        lo, hi = confint_proportions_2indep(
            x_t, n_t, x_c, n_c, method="newcomb", compare="diff", alpha=alpha
        )
    except Exception:
        z, p, lo, hi = np.nan, np.nan, np.nan, np.nan
    out.update(z=z, p=p, ci_lo=lo * 100, ci_hi=hi * 100)
    return out


@st.cache_data
def bayesian_sim(x_c, n_c, x_t, n_t, draws=20000, seed=42):
    rng = np.random.default_rng(seed)
    post_c = rng.beta(1 + x_c, 1 + n_c - x_c, draws)
    post_t = rng.beta(1 + x_t, 1 + n_t - x_t, draws)
    return post_c, post_t


@st.cache_data
def bootstrap_lift(c_arr, t_arr, reps=3000, seed=7):
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for i in range(reps):
        diffs[i] = (
            rng.choice(t_arr, len(t_arr)).mean() - rng.choice(c_arr, len(c_arr)).mean()
        )
    return diffs * 100


def style_fig(fig, height=400):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", y=1.12, x=0),
        hoverlabel=dict(font_size=13),
    )
    return fig


# ==================================================
# SIDEBAR: NAVIGATION + GLOBAL FILTERS
# ==================================================
st.sidebar.title("🎮 Steam Analytics")
st.sidebar.caption("Player Behaviour & Recommendation Experiment")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "👥 Player Analytics",
        "🧪 A/B Testing",
        "🎯 Segment Analysis",
        "💰 Business Impact",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

all_clusters = sorted(experiment_df["cluster"].dropna().unique().tolist())
selected_clusters = st.sidebar.multiselect(
    "Player clusters",
    all_clusters,
    default=all_clusters,
    help="Filters the experiment and player data on every page.",
)
alpha = 1 - st.sidebar.select_slider(
    "Confidence level",
    options=[0.80, 0.90, 0.95, 0.99],
    value=0.95,
    format_func=lambda v: f"{int(v * 100)}%",
)

if not selected_clusters:
    st.warning("Select at least one cluster in the sidebar.")
    st.stop()

exp = experiment_df[experiment_df["cluster"].isin(selected_clusters)]
players = player_df[player_df["cluster"].isin(selected_clusters)]
S = ab_stats(exp, alpha)

st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(exp):,}** of {len(experiment_df):,} experiment users")
st.sidebar.caption("Dataset: Steam Video Games (Kaggle)")
st.sidebar.caption("Experiment outcomes: Simulated")

# ==================================================
# HEADER
# ==================================================
st.markdown(
    """
<div class="hero">
  <h1>Steam Player Analytics</h1>
  <p>Understanding player behaviour and evaluating personalized game recommendation strategies.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="note">ℹ️ The original Steam data is observational. A/B assignments and '
    "outcomes in this project are <b>simulated</b> for analytical demonstration.</div>",
    unsafe_allow_html=True,
)
st.write("")

# ==================================================
# PAGE 1: OVERVIEW
# ==================================================
if page == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Players", f"{len(players):,}")
    c2.metric(
        "Unique Games",
        f"{steam_df['game_title'].nunique():,}" if "game_title" in steam_df.columns else "N/A",
    )
    c3.metric("Control Conversion", f"{S['p_c'] * 100:.2f}%")
    c4.metric("Treatment Conversion", f"{S['p_t'] * 100:.2f}%", delta=f"{S['abs_lift']:+.2f} pp")

    st.write("")
    left, right = st.columns(2)

    metric_df = pd.DataFrame(
        {
            "Group": ["Control A", "Treatment B"],
            "Conversion": [S["p_c"] * 100, S["p_t"] * 100],
            "CTR": [S["ctr_c"], S["ctr_t"]],
        }
    )
    with left:
        fig = px.bar(
            metric_df, x="Group", y="Conversion", color="Group", text_auto=".2f",
            title="Conversion Rate (%)",
            color_discrete_map={"Control A": CONTROL_COLOR, "Treatment B": TREAT_COLOR},
        )
        fig.update_layout(showlegend=False)
        fig.update_traces(marker_cornerradius=8, textposition="outside")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with right:
        fig = px.bar(
            metric_df, x="Group", y="CTR", color="Group", text_auto=".2f",
            title="Click-Through Rate (%)",
            color_discrete_map={"Control A": CONTROL_COLOR, "Treatment B": TREAT_COLOR},
        )
        fig.update_layout(showlegend=False)
        fig.update_traces(marker_cornerradius=8, textposition="outside")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    sig = S["p"] < alpha
    css = "good" if sig and S["abs_lift"] > 0 else ("bad" if sig else "mid")
    icon = "✅" if css == "good" else ("⚠️" if css == "bad" else "🟡")
    verdict = (
        "statistically significant" if sig else "not statistically significant"
    )
    st.markdown(
        f'<div class="verdict {css}">{icon} Personalized recommendations changed conversion by '
        f"<b>{S['abs_lift']:+.2f} pp</b> (<b>{S['rel_lift']:+.2f}%</b> relative). "
        f"The result is <b>{verdict}</b> at the {int((1 - alpha) * 100)}% confidence level "
        f"(p = {S['p']:.4g}).</div>",
        unsafe_allow_html=True,
    )

# ==================================================
# PAGE 2: PLAYER ANALYTICS
# ==================================================
elif page == "👥 Player Analytics":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(players):,}")
    c2.metric("Avg Playtime", f"{players['total_hours'].mean():.1f} hrs")
    c3.metric("Median Playtime", f"{players['total_hours'].median():.1f} hrs")
    c4.metric("Avg Games Owned", f"{players['total_purchases'].mean():.1f}")

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🧩 Clusters", "🕹️ Games"])

    with tab1:
        l, r = st.columns(2)
        with l:
            use_log = st.toggle("Log scale (1 + hours)", value=True)
            xcol = "log_total_hours" if use_log and "log_total_hours" in players else "total_hours"
            fig = px.histogram(
                players, x=xcol, nbins=45, color="cluster", marginal="box",
                title="Playtime Distribution by Cluster",
                color_discrete_sequence=CLUSTER_PALETTE, barmode="overlay", opacity=0.7,
            )
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)
        with r:
            sample_n = st.slider("Scatter sample size", 500, min(20000, max(len(players), 501)),
                                 min(3000, len(players)), step=500)
            sample = players.sample(min(sample_n, len(players)), random_state=1)
            fig = px.scatter(
                sample, x="total_purchases", y="total_hours", color="cluster",
                log_y=True, opacity=0.65, title="Games Owned vs Playtime",
                color_discrete_sequence=CLUSTER_PALETTE,
                hover_data=["user_id"],
            )
            st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    with tab2:
        l, r = st.columns([1, 1.3])
        counts = players["cluster"].value_counts().sort_index().reset_index()
        counts.columns = ["Cluster", "Players"]
        counts["Cluster"] = counts["Cluster"].astype(str)
        with l:
            fig = px.pie(counts, names="Cluster", values="Players", hole=0.55,
                         title="Cluster Share", color_discrete_sequence=CLUSTER_PALETTE)
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        with r:
            summary = players.groupby("cluster").agg(
                Players=("user_id", "count"),
                Median_Playtime=("total_hours", "median"),
                Mean_Playtime=("total_hours", "mean"),
                Median_Games_Owned=("total_purchases", "median"),
            ).reset_index().round(2)
            st.markdown("**Segment summary**")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download summary CSV", summary.to_csv(index=False),
                               "cluster_summary.csv", "text/csv")

        fig = px.box(players, x="cluster", y="total_hours", color="cluster", log_y=True,
                     title="Playtime Spread by Cluster (log axis)",
                     color_discrete_sequence=CLUSTER_PALETTE)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    with tab3:
        if {"game_title", "behavior"}.issubset(steam_df.columns):
            top_n = st.slider("Number of games", 5, 30, 10)
            behavior = st.radio("Behaviour", sorted(steam_df["behavior"].unique()),
                                horizontal=True,
                                index=sorted(steam_df["behavior"].unique()).index("purchase")
                                if "purchase" in steam_df["behavior"].unique() else 0)
            sub = steam_df[steam_df["behavior"] == behavior]
            if "user_id" in sub.columns and len(selected_clusters) < len(all_clusters):
                sub = sub[sub["user_id"].isin(players["user_id"])]
            top = sub["game_title"].value_counts().head(top_n).reset_index()
            top.columns = ["Game", "Records"]
            fig = px.bar(top.sort_values("Records"), x="Records", y="Game", orientation="h",
                         title=f"Top {top_n} Games by '{behavior}' Records",
                         color="Records", color_continuous_scale="Purples")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, 150 + 28 * top_n), use_container_width=True)

            query = st.text_input("🔍 Search a game")
            if query:
                hits = steam_df[steam_df["game_title"].str.contains(query, case=False, na=False)]
                st.dataframe(
                    hits.groupby(["game_title", "behavior"]).size()
                    .reset_index(name="Records").sort_values("Records", ascending=False).head(50),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("`game_title` / `behavior` columns not found in the Steam data.")

# ==================================================
# PAGE 3: A/B TESTING
# ==================================================
elif page == "🧪 A/B Testing":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control Conversion", f"{S['p_c'] * 100:.2f}%", help=f"n = {S['n_c']:,}")
    c2.metric("Treatment Conversion", f"{S['p_t'] * 100:.2f}%", help=f"n = {S['n_t']:,}")
    c3.metric("Absolute Lift", f"{S['abs_lift']:+.2f} pp")
    c4.metric("Relative Lift", f"{S['rel_lift']:+.2f}%")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📐 Frequentist", "🎲 Bayesian", "🔁 Bootstrap", "⚡ Power & Sample Size"]
    )

    # ---------- Frequentist ----------
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Z-statistic", f"{S['z']:.4f}")
        c2.metric("P-value", f"{S['p']:.6f}")
        c3.metric(f"{int((1 - alpha) * 100)}% CI (diff)", f"{S['ci_lo']:.2f} to {S['ci_hi']:.2f} pp")

        if S["p"] < alpha:
            st.success(f"Significant at α = {alpha:.2f}: reject the null of equal conversion.")
        else:
            st.warning(f"Not significant at α = {alpha:.2f}: cannot reject the null.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[S["abs_lift"]], y=["Treatment − Control"], mode="markers",
            marker=dict(size=14, color=TREAT_COLOR),
            error_x=dict(type="data", symmetric=False,
                         array=[S["ci_hi"] - S["abs_lift"]],
                         arrayminus=[S["abs_lift"] - S["ci_lo"]], thickness=3),
            name="Lift",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=NEG)
        fig.update_layout(title="Lift with Confidence Interval (pp)", showlegend=False,
                          xaxis_title="Absolute difference (pp)")
        st.plotly_chart(style_fig(fig, 240), use_container_width=True)

        st.subheader("Click-Through Rate")
        a, b = st.columns(2)
        a.metric("Control CTR", f"{S['ctr_c']:.2f}%")
        b.metric("Treatment CTR", f"{S['ctr_t']:.2f}%", delta=f"{S['ctr_t'] - S['ctr_c']:+.2f} pp")
        st.caption("Clicks and conversions were independently simulated; CTR should not be "
                   "interpreted as a measured step in a purchase funnel.")

    # ---------- Bayesian ----------
    with tab2:
        post_c, post_t = bayesian_sim(S["x_c"], S["n_c"], S["x_t"], S["n_t"])
        prob_better = (post_t > post_c).mean()
        exp_lift = ((post_t - post_c) * 100).mean()
        c1, c2 = st.columns(2)
        c1.metric("P(Treatment > Control)", f"{prob_better * 100:.1f}%")
        c2.metric("Expected lift", f"{exp_lift:+.3f} pp")

        fig = go.Figure()
        fig.add_histogram(x=post_c * 100, name="Control", opacity=0.6,
                          marker_color=CONTROL_COLOR, histnorm="probability density", nbinsx=80)
        fig.add_histogram(x=post_t * 100, name="Treatment", opacity=0.6,
                          marker_color=TREAT_COLOR, histnorm="probability density", nbinsx=80)
        fig.update_layout(barmode="overlay", title="Posterior Distributions of Conversion Rate (Beta-Binomial, flat prior)",
                          xaxis_title="Conversion rate (%)")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
        st.caption("Uses a Beta(1,1) prior and 20,000 posterior draws per group.")

    # ---------- Bootstrap ----------
    with tab3:
        c_arr = exp.loc[exp["group"] == "A", "converted"].to_numpy()
        t_arr = exp.loc[exp["group"] == "B", "converted"].to_numpy()
        reps = st.select_slider("Bootstrap resamples", [500, 1000, 3000, 5000], value=1000)
        diffs = bootstrap_lift(c_arr, t_arr, reps=reps)
        lo, hi = np.percentile(diffs, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        fig = px.histogram(diffs, nbins=60, title="Bootstrap Distribution of Lift (pp)",
                           color_discrete_sequence=[TREAT_COLOR])
        fig.add_vline(x=0, line_dash="dash", line_color=NEG)
        fig.add_vrect(x0=lo, x1=hi, fillcolor=ACCENT, opacity=0.12, line_width=0)
        fig.update_layout(showlegend=False, xaxis_title="Lift (pp)")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        st.info(f"Bootstrap {int((1 - alpha) * 100)}% interval: **{lo:.2f} to {hi:.2f} pp**")

    # ---------- Power ----------
    with tab4:
        st.markdown("How many users per group are needed to detect a given lift?")
        a, b, c = st.columns(3)
        base = a.number_input("Baseline conversion (%)", 0.1, 99.0,
                              float(round(S["p_c"] * 100, 2)) if S["p_c"] == S["p_c"] else 5.0, 0.1)
        mde = b.slider("Minimum detectable lift (pp)", 0.1, 10.0,
                       float(max(0.1, round(abs(S["abs_lift"]), 1))) or 1.0, 0.1)
        power = c.slider("Statistical power", 0.5, 0.99, 0.80, 0.01)
        p1, p2 = base / 100, min(base / 100 + mde / 100, 0.999)
        es = proportion_effectsize(p2, p1)
        n_needed = NormalIndPower().solve_power(es, alpha=alpha, power=power, ratio=1.0)
        st.metric("Required users per group", f"{int(np.ceil(n_needed)):,}")

        grid = np.arange(0.1, 5.01, 0.1)
        ns = [NormalIndPower().solve_power(proportion_effectsize(min(p1 + g / 100, 0.999), p1),
                                           alpha=alpha, power=power) for g in grid]
        fig = px.line(x=grid, y=ns, log_y=True, title="Sample Size vs Minimum Detectable Lift",
                      labels={"x": "MDE (pp)", "y": "Users per group"})
        fig.update_traces(line_color=TREAT_COLOR, line_width=3)
        fig.add_scatter(x=[mde], y=[n_needed], mode="markers", marker=dict(size=13, color=NEG),
                        name="Your choice")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

# ==================================================
# PAGE 4: SEGMENT ANALYSIS
# ==================================================
elif page == "🎯 Segment Analysis":
    rows = []
    for cl in sorted(exp["cluster"].dropna().unique()):
        s = ab_stats(exp[exp["cluster"] == cl], alpha)
        rows.append({
            "Cluster": str(cl),
            "Control_n": s["n_c"], "Treatment_n": s["n_t"],
            "Control_%": s["p_c"] * 100, "Treatment_%": s["p_t"] * 100,
            "Abs_Lift_pp": s["abs_lift"], "Rel_Lift_%": s["rel_lift"],
            "CI_low": s["ci_lo"], "CI_high": s["ci_hi"], "P_value": s["p"],
        })
    seg = pd.DataFrame(rows)

    metric_choice = st.radio("Chart metric", ["Conversion rate", "Absolute lift (pp)", "Relative lift (%)"],
                             horizontal=True)

    if metric_choice == "Conversion rate":
        fig = go.Figure()
        fig.add_bar(x=seg["Cluster"], y=seg["Control_%"], name="Control A", marker_color=CONTROL_COLOR)
        fig.add_bar(x=seg["Cluster"], y=seg["Treatment_%"], name="Treatment B", marker_color=TREAT_COLOR)
        fig.update_layout(barmode="group", title="Conversion Rate by Player Cluster",
                          xaxis_title="Cluster", yaxis_title="Conversion (%)")
    else:
        col = "Abs_Lift_pp" if "Absolute" in metric_choice else "Rel_Lift_%"
        colors = [ACCENT if v >= 0 else NEG for v in seg[col]]
        fig = go.Figure(go.Bar(x=seg["Cluster"], y=seg[col], marker_color=colors,
                               text=seg[col].round(2), textposition="outside"))
        fig.update_layout(title=f"{metric_choice} by Cluster", xaxis_title="Cluster", yaxis_title=metric_choice)
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.subheader("Forest Plot: Lift per Segment")
    fp = go.Figure()
    fp.add_trace(go.Scatter(
        x=seg["Abs_Lift_pp"], y=seg["Cluster"], mode="markers",
        marker=dict(size=12, color=TREAT_COLOR),
        error_x=dict(type="data", symmetric=False,
                     array=seg["CI_high"] - seg["Abs_Lift_pp"],
                     arrayminus=seg["Abs_Lift_pp"] - seg["CI_low"], thickness=2.5),
    ))
    fp.add_vline(x=0, line_dash="dash", line_color=NEG)
    fp.update_layout(xaxis_title="Absolute lift (pp)", yaxis_title="Cluster", showlegend=False)
    st.plotly_chart(style_fig(fp, 120 + 70 * len(seg)), use_container_width=True)

    st.subheader("Segment-Level Treatment Lift")
    st.dataframe(
        seg.round(3), use_container_width=True, hide_index=True,
        column_config={
            "Abs_Lift_pp": st.column_config.NumberColumn("Abs lift (pp)", format="%.2f"),
            "P_value": st.column_config.NumberColumn("p-value", format="%.4f"),
        },
    )
    st.download_button("⬇️ Download segment table", seg.to_csv(index=False),
                       "segment_lift.csv", "text/csv")

    best = seg.loc[seg["Abs_Lift_pp"].idxmax()]
    st.info(
        f"Cluster **{best['Cluster']}** shows the largest observed lift ({best['Abs_Lift_pp']:+.2f} pp). "
        "Per-segment p-values are not multiple-comparison corrected, and the interaction analysis "
        "did not establish statistically significant treatment heterogeneity."
    )

# ==================================================
# PAGE 5: BUSINESS IMPACT
# ==================================================
elif page == "💰 Business Impact":
    st.write("Explore potential outcomes under different platform and revenue assumptions.")

    c1, c2 = st.columns(2)
    with c1:
        platform_users = st.slider("Eligible platform users", 10_000, 1_000_000, 100_000, 10_000)
        revenue = st.slider("Average revenue per conversion (₹)", 100, 5000, 500, 100)
    with c2:
        rollout = st.slider("Rollout coverage (%)", 10, 100, 100, 10)
        cost = st.number_input("Fixed cost of running personalization (₹)", 0, 100_000_000, 500_000, 50_000)

    scenario = st.radio("Lift scenario", ["Conservative (CI lower bound)", "Expected", "Optimistic (CI upper bound)"],
                        index=1, horizontal=True)
    lift_pp = {"Conservative (CI lower bound)": S["ci_lo"], "Expected": S["abs_lift"],
               "Optimistic (CI upper bound)": S["ci_hi"]}[scenario]

    reached = platform_users * rollout / 100
    base_conv = reached * S["p_c"]
    extra_conv = reached * lift_pp / 100
    extra_rev = extra_conv * revenue
    net = extra_rev - cost
    roi = net / cost * 100 if cost else np.nan
    breakeven = cost / revenue if revenue else np.nan

    a, b, c, d = st.columns(4)
    a.metric("Baseline conversions", f"{base_conv:,.0f}")
    b.metric("Additional conversions", f"{extra_conv:,.0f}")
    c.metric("Additional revenue", f"₹{extra_rev:,.0f}")
    d.metric("Net impact", f"₹{net:,.0f}", delta=f"ROI {roi:,.0f}%" if cost else None)

    l, r = st.columns(2)
    with l:
        fig = go.Figure(go.Waterfall(
            x=["Baseline revenue", "Incremental revenue", "Cost", "Net total"],
            measure=["absolute", "relative", "relative", "total"],
            y=[base_conv * revenue, extra_rev, -cost, 0],
            connector={"line": {"color": "#94A3B8"}},
            increasing={"marker": {"color": ACCENT}}, decreasing={"marker": {"color": NEG}},
            totals={"marker": {"color": TREAT_COLOR}},
        ))
        fig.update_layout(title="Revenue Bridge (₹)")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    with r:
        users_axis = np.linspace(10_000, 1_000_000, 40)
        fig = go.Figure()
        for name, lift, color in [("Conservative", S["ci_lo"], CONTROL_COLOR),
                                  ("Expected", S["abs_lift"], TREAT_COLOR),
                                  ("Optimistic", S["ci_hi"], ACCENT)]:
            fig.add_scatter(x=users_axis, y=users_axis * rollout / 100 * lift / 100 * revenue - cost,
                            mode="lines", name=name, line=dict(color=color, width=3))
        fig.add_hline(y=0, line_dash="dash", line_color=NEG)
        fig.update_layout(title="Net Impact vs Platform Size (₹)", xaxis_title="Eligible users")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    st.subheader("Sensitivity: Net Impact (₹) by Users × Revenue per Conversion")
    users_grid = np.linspace(50_000, 1_000_000, 8).astype(int)
    rev_grid = np.arange(100, 5001, 500)
    z = [[u * rollout / 100 * lift_pp / 100 * rv - cost for u in users_grid] for rv in rev_grid]
    fig = px.imshow(z, x=[f"{u // 1000}k" for u in users_grid], y=rev_grid, aspect="auto",
                    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    labels=dict(x="Eligible users", y="Revenue / conversion (₹)", color="Net ₹"))
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    if cost:
        st.info(f"Break-even requires about **{breakeven:,.0f}** additional conversions at ₹{revenue}/conversion.")
    st.warning("Illustrative scenario only. This projection assumes the simulated conversion rates "
               "generalize to the selected population. It is not actual Steam revenue or a validated "
               "business forecast.")

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption("Steam Player Analytics | Data Science Portfolio Project")
st.caption("Historical player data: Observational | A/B experiment: Simulated")