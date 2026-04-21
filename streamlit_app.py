from pathlib import Path
import pandas as pd
import streamlit as st

from recommender_core import build_assets, get_top_n, recommendation_df


st.set_page_config(
    page_title="Robinson Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 1100px;
}

/* ── Hero banner ── */
.hero-block {
    background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 60%, #16213e 100%);
    border-radius: 16px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,179,237,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #f0f4f8;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
    letter-spacing: -0.5px;
}
.hero-title em {
    font-style: italic;
    color: #63b3ed;
}
.hero-sub {
    font-size: 1.05rem;
    color: #a0aec0;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.6;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.35);
    color: #63b3ed;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: #1a202c;
    margin: 2.5rem 0 0.3rem 0;
    letter-spacing: -0.3px;
}
.section-divider {
    height: 3px;
    width: 40px;
    background: #63b3ed;
    border-radius: 2px;
    margin-bottom: 1.5rem;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #718096 !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.8rem !important;
    color: #1a202c !important;
}

/* ── Info / callout boxes ── */
.callout-box {
    border-left: 4px solid #63b3ed;
    background: #ebf8ff;
    border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    color: #2c5282;
    font-size: 0.93rem;
    line-height: 1.65;
}
.callout-box strong {
    color: #1a365d;
}
.callout-box-green {
    border-left: 4px solid #48bb78;
    background: #f0fff4;
    border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    color: #276749;
    font-size: 0.95rem;
    line-height: 1.65;
}

/* ── Dataframes ── */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}

/* ── Expander ── */
details summary {
    font-weight: 600;
    font-size: 0.9rem;
    color: #4a5568;
    letter-spacing: 0.02em;
}

/* ── Horizontal rule ── */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 2.5rem 0;
}

/* ── Selectbox / slider labels ── */
label[data-testid="stWidgetLabel"] {
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #4a5568;
}

/* ── Comparison column headers ── */
.compare-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    margin-bottom: 0.25rem;
}
.compare-caption {
    font-size: 0.8rem;
    color: #718096;
    margin-bottom: 0.75rem;
}

/* ── Insight pill ── */
.insight-pill {
    display: inline-block;
    background: #edf2f7;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #4a5568;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ── Image caption ── */
.img-caption {
    text-align: center;
    font-size: 0.78rem;
    color: #a0aec0;
    margin-top: 0.4rem;
    font-style: italic;
}

/* ── Intro / explainer cards ── */
.intro-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    height: 100%;
}
.intro-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #1a202c;
    margin: 0 0 0.6rem 0;
}
.intro-card p, .intro-card li {
    font-size: 0.9rem;
    color: #4a5568;
    line-height: 1.65;
    margin: 0.2rem 0;
}
.intro-card ul { padding-left: 1.2rem; margin: 0.4rem 0 0 0; }

/* ── Score table ── */
.score-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    font-family: 'DM Sans', sans-serif;
}
.score-table th {
    background: #edf2f7;
    color: #4a5568;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.55rem 0.9rem;
    text-align: left;
}
.score-table td {
    padding: 0.5rem 0.9rem;
    border-bottom: 1px solid #e2e8f0;
    color: #2d3748;
}
.score-table tr:last-child td { border-bottom: none; }
.score-table tr:hover td { background: #f7fafc; }
.score-pos { color: #276749; font-weight: 700; }
.score-neg { color: #9b2c2c; font-weight: 700; }
.score-zero { color: #2c5282; font-weight: 700; }

/* ── Distribution bar ── */
.dist-bar-wrap { margin: 0.3rem 0; }
.dist-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.35rem;
}
.dist-label {
    font-family: 'DM Serif Display', serif;
    font-size: 0.95rem;
    width: 28px;
    text-align: right;
    flex-shrink: 0;
}
.dist-bar-bg {
    flex: 1;
    background: #edf2f7;
    border-radius: 4px;
    height: 18px;
    overflow: hidden;
}
.dist-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}
.dist-count {
    font-size: 0.78rem;
    color: #718096;
    width: 40px;
    flex-shrink: 0;
}

/* ── Stat foundation pills ── */
.stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.8rem;
}
.stat-chip {
    background: #1a1a2e;
    color: #63b3ed;
    border-radius: 8px;
    padding: 0.4rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
}
.stat-chip span {
    color: #a0aec0;
    font-weight: 400;
    margin-left: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    return build_assets()


@st.cache_data
def load_metric_files():
    model_path = Path("model_comparison_results.csv")
    system_path = Path("system_comparison_metrics.csv")
    holdout_path = Path("holdout_experiment_results.csv")

    model_df = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
    system_df = pd.read_csv(system_path) if system_path.exists() else pd.DataFrame()
    holdout_df = pd.read_csv(holdout_path) if holdout_path.exists() else pd.DataFrame()

    return model_df, system_df, holdout_df


def get_metric_row(model_df: pd.DataFrame, column_name: str, value: str) -> pd.Series | None:
    if model_df.empty:
        return None
    rows = model_df[model_df[column_name] == value]
    if rows.empty:
        return None
    return rows.iloc[0]


assets = load_assets()
model_metrics_df, system_metrics_df, holdout_metrics_df = load_metric_files()
user_ids = sorted(assets["df"]["user_id"].unique().tolist())


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-badge">Research Demo · MovieLens 100K</div>
    <h1 class="hero-title">Robinson <em>Recommender</em></h1>
    <p class="hero-sub">
        Comparing two recommendation systems — one trained on standard 1–5 ratings,
        one on Robinson-transformed ratings — to explore how rating structure shapes discovery.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Robinson Scale Intro ──────────────────────────────────────────────────────
st.markdown("""
<p class="section-header">The Robinson Scale</p>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

intro_c1, intro_c2, intro_c3 = st.columns(3, gap="large")

with intro_c1:
    st.markdown("""
<div class="intro-card">
    <p class="intro-card-title">The Problem</p>
    <p>Most rating systems are broken by <strong>score inflation</strong>. People rate things 7, 8, or 9 out of 10 even when they're genuinely average — compressing everything into the top of the scale and making it nearly impossible to tell good from great.</p>
    <p style="margin-top:0.8rem;">Traditional scales also have no true neutral. A 5 out of 10 <em>sounds</em> like the middle, but in practice nobody uses it that way.</p>
</div>
""", unsafe_allow_html=True)

with intro_c2:
    st.markdown("""
<div class="intro-card">
    <p class="intro-card-title">The Key Idea</p>
    <p><strong>Most things are average.</strong> The Robinson Scale is built around this truth.</p>
    <ul>
        <li><strong>0</strong> = expected, unremarkable, baseline</li>
        <li><strong>Positive values</strong> = above average</li>
        <li><strong>Negative values</strong> = below average</li>
        <li><strong>±5</strong> = truly extreme — exceptional or catastrophic</li>
    </ul>
    <p style="margin-top:0.8rem;">By centering the scale at zero, neutrality becomes meaningful rather than something to avoid.</p>
</div>
""", unsafe_allow_html=True)

with intro_c3:
    st.markdown("""
<div class="intro-card">
    <p class="intro-card-title">Statistical Foundation</p>
    <p>The scale assumes a <strong>normal distribution</strong> of human evaluations:</p>
    <div class="stat-row">
        <div class="stat-chip">μ = 0 <span>mean</span></div>
        <div class="stat-chip">σ ≈ 1.67 <span>std dev</span></div>
        <div class="stat-chip">±3σ <span>full range</span></div>
    </div>
    <p style="margin-top:0.9rem; font-size:0.85rem; color:#718096;">This means ~46% of ratings fall between -1 and +1, ~76% between -2 and +2, and extreme values (±5) occur only ~0.3% of the time.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Score interpretation + distribution side by side
score_col, dist_col = st.columns([1, 1], gap="large")

with score_col:
    st.markdown("""
<p style="font-family:'DM Serif Display',serif; font-size:1.05rem; color:#1a202c; margin-bottom:0.6rem;">Score Interpretation</p>
<table class="score-table">
    <thead><tr><th>Score</th><th>What it means</th></tr></thead>
    <tbody>
        <tr><td><span class="score-neg">−5</span></td><td>Extreme — catastrophic, life-altering</td></tr>
        <tr><td><span class="score-neg">−4</span></td><td>Rare — deeply negative, exceptional</td></tr>
        <tr><td><span class="score-neg">−3</span></td><td>Strong — memorable in a bad way</td></tr>
        <tr><td><span class="score-neg">−2</span></td><td>Clearly noticeable, below average</td></tr>
        <tr><td><span class="score-neg">−1</span></td><td>Slightly below average</td></tr>
        <tr><td><span class="score-zero">0</span></td><td>Neutral — expected, unremarkable</td></tr>
        <tr><td><span class="score-pos">+1</span></td><td>Slightly above average</td></tr>
        <tr><td><span class="score-pos">+2</span></td><td>Clearly noticeable, above average</td></tr>
        <tr><td><span class="score-pos">+3</span></td><td>Strong — memorable in a good way</td></tr>
        <tr><td><span class="score-pos">+4</span></td><td>Rare — deeply positive, exceptional</td></tr>
        <tr><td><span class="score-pos">+5</span></td><td>Extreme — life-altering, unforgettable</td></tr>
    </tbody>
</table>
""", unsafe_allow_html=True)

with dist_col:
    st.markdown("""
<p style="font-family:'DM Serif Display',serif; font-size:1.05rem; color:#1a202c; margin-bottom:0.6rem;">Distribution in Practice <span style="font-family:'DM Sans',sans-serif; font-size:0.78rem; color:#a0aec0; font-weight:400;">(per 100 items)</span></p>
""", unsafe_allow_html=True)

    dist_data = [
        ("-5", 1,  "#9b2c2c", "#fc8181"),
        ("-4", 2,  "#9b2c2c", "#fc8181"),
        ("-3", 8,  "#c05621", "#fbd38d"),
        ("-2", 16, "#c05621", "#fbd38d"),
        ("-1", 23, "#744210", "#fefcbf"),
        ("0",  24, "#1a365d", "#63b3ed"),
        ("+1", 23, "#276749", "#9ae6b4"),
        ("+2", 16, "#276749", "#9ae6b4"),
        ("+3", 8,  "#1a4731", "#68d391"),
        ("+4", 2,  "#1a4731", "#68d391"),
        ("+5", 1,  "#1a4731", "#68d391"),
    ]

    bars_html = '<div class="dist-bar-wrap">'
    for label, count, _, color in dist_data:
        pct = count / 24 * 100  # normalize to max (0 = 24)
        label_class = "score-zero" if label == "0" else ("score-pos" if label.startswith("+") else "score-neg")
        bars_html += f"""
        <div class="dist-row">
            <span class="dist-label {label_class}">{label}</span>
            <div class="dist-bar-bg">
                <div class="dist-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="dist-count">{count} / 100</span>
        </div>"""
    bars_html += "</div>"
    bars_html += """<p style="font-size:0.78rem; color:#a0aec0; margin-top:0.8rem; font-style:italic;">
        ~24% of items score exactly 0 &nbsp;·&nbsp; ~70% fall between −1 and +1 &nbsp;·&nbsp; ±5 occurs ~1% of the time
    </p>"""

    st.markdown(bars_html, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── What this app does ────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown("""
<p class="section-header">What this app does</p>
<div class="section-divider"></div>
""", unsafe_allow_html=True)
    st.markdown("""
This app compares two recommendation systems built on the **MovieLens 100K** dataset:

- **Standard** — trained on the original 1–5 ratings
- **Robinson** — trained on Robinson-transformed ratings that re-center users around a true neutral point

The goal is not just to predict exact ratings. It is to test whether the Robinson Scale can produce recommendation lists that are more useful for **discovery**.
""")

with right_col:
    scale_img = Path("robinson_scale.png")
    if scale_img.exists():
        st.image(str(scale_img), use_container_width=True)
        st.markdown('<p class="img-caption">The Robinson Scale — normal distribution centered at 0</p>', unsafe_allow_html=True)

# ── Reload button (small, right-aligned) ──────────────────────────────────────
_, btn_col = st.columns([6, 1])
with btn_col:
    if st.button("↺ Reload", help="Clear cached metrics and reload"):
        st.cache_data.clear()
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ── Model Findings ────────────────────────────────────────────────────────────
st.markdown("""
<p class="section-header">Model Findings</p>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

standard_row = get_metric_row(model_metrics_df, "rating_column", "rating")
robinson_row = get_metric_row(model_metrics_df, "rating_column", "robinson_rating")

if standard_row is not None and robinson_row is not None:
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Standard Precision@10", f"{standard_row['precision_at_10']:.3f}")
        st.metric("Standard Recall@10", f"{standard_row['recall_at_10']:.3f}")

    with m2:
        st.metric("Robinson Precision@10", f"{robinson_row['precision_at_10']:.3f}")
        st.metric("Robinson Recall@10", f"{robinson_row['recall_at_10']:.3f}")

    with m3:
        st.metric("Standard RMSE", f"{standard_row['rmse']:.3f}")
        st.metric("Robinson RMSE", f"{robinson_row['rmse']:.3f}")

    with m4:
        if not system_metrics_df.empty:
            system_row = system_metrics_df.iloc[0]
            st.metric("Average Overlap", f"{system_row['avg_overlap']:.3f}")
            st.metric(
                "Catalog Diversity Gain",
                f"{int(system_row['robinson_catalog_diversity']) - int(system_row['standard_catalog_diversity'])}"
            )

    st.markdown("""
**Interpretation**

- The **standard model** is better at predicting exact rating values.
- The **Robinson model** has **higher recall**, meaning it surfaces more relevant items.
- The **Robinson model** also shows **greater catalog diversity**, meaning it spreads recommendations across more distinct movies.

That makes Robinson especially interesting for **discovery-oriented recommendation systems**.
""")

    st.markdown("""
<div class="callout-box">
<strong>Why Recall Matters Here</strong><br><br>
In many recommendation systems, <strong>precision</strong> is treated as the main goal. Precision asks whether the recommended items are highly accurate.
That is useful, but it often leads to repetitive recommendations, over-reliance on the safest popular choices, and less exploration of the catalog.<br><br>
For this project, the Robinson model is meant to support <strong>discovery</strong>, not just exact prediction. That makes <strong>recall</strong> especially important here — higher recall means the system is surfacing <strong>more items a user would actually like</strong>, even if they are not all the most obvious or conservative choices.<br><br>
In other words: <strong>precision</strong> favors safer recommendations — <strong>recall</strong> favors broader discovery.<br><br>
The Robinson model should therefore be understood as a system that may trade some precision for a better ability to uncover relevant content.
</div>
""", unsafe_allow_html=True)

else:
    st.warning(
        "Metric files were not found. Run `robinson_recommender.py` first to generate "
        "`model_comparison_results.csv` and `system_comparison_metrics.csv`."
    )

with st.expander("Show full metric tables"):
    if not model_metrics_df.empty:
        st.subheader("Model Comparison Results")
        st.dataframe(model_metrics_df, use_container_width=True)

    if not system_metrics_df.empty:
        st.subheader("System-Wide Comparison Results")
        st.dataframe(system_metrics_df, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── User-Level Comparison ─────────────────────────────────────────────────────
st.markdown("""
<p class="section-header">User-Level Recommendation Comparison</p>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    user_id = st.selectbox("Pick a user ID", user_ids, index=0)

with c2:
    mode = st.radio(
        "Single-view mode",
        ["standard", "robinson"],
        horizontal=True
    )

with c3:
    top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if mode == "standard":
    single_recs = get_top_n(
        assets["original_algo"],
        assets["original_trainset"],
        user_id,
        n=top_n,
    )
else:
    single_recs = get_top_n(
        assets["robinson_algo"],
        assets["robinson_trainset"],
        user_id,
        n=top_n,
    )

single_df = recommendation_df(single_recs, assets["movie_titles"])

st.markdown(f"**Top {top_n} recommendations for user {user_id} ({mode})**")
st.dataframe(single_df, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<p class="section-header" style="font-size:1.2rem; margin-bottom:0.2rem;">Side-by-Side Comparison</p>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

standard_recs = get_top_n(
    assets["original_algo"],
    assets["original_trainset"],
    user_id,
    n=top_n,
)
robinson_recs = get_top_n(
    assets["robinson_algo"],
    assets["robinson_trainset"],
    user_id,
    n=top_n,
)

standard_df = recommendation_df(standard_recs, assets["movie_titles"])
robinson_df = recommendation_df(robinson_recs, assets["movie_titles"])

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<p class="compare-label">Standard</p><p class="compare-caption">Built from original 1–5 ratings.</p>', unsafe_allow_html=True)
    st.dataframe(standard_df, use_container_width=True)

with right:
    st.markdown('<p class="compare-label">Robinson</p><p class="compare-caption">Built from Robinson-transformed ratings.</p>', unsafe_allow_html=True)
    st.dataframe(robinson_df, use_container_width=True)

std_items = set(standard_df["item_id"].tolist())
rob_items = set(robinson_df["item_id"].tolist())
shared_items = std_items & rob_items
user_overlap = len(shared_items) / len(std_items) if std_items else 0.0

b1, b2, b3 = st.columns(3)

with b1:
    st.metric("User-Level Overlap", f"{user_overlap:.2f}")

with b2:
    st.metric("Shared Recommendations", len(shared_items))

with b3:
    st.metric("Different Recommendations", top_n - len(shared_items))

st.markdown("""
**What this means**

- A **high overlap** means Robinson is mostly re-ranking the same strong candidates.
- A **lower overlap** means Robinson is introducing more different items.
- In the full experiment, the average overlap across users was about **0.73**, so Robinson usually preserves the core recommendation set while still changing part of the list.
""")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Holdout Experiment ────────────────────────────────────────────────────────
st.markdown("""
<p class="section-header">Advanced Experiment: User-Level Holdout</p>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

holdout_standard = get_metric_row(holdout_metrics_df, "model", "standard")
holdout_robinson = get_metric_row(holdout_metrics_df, "model", "robinson")

if holdout_standard is not None and holdout_robinson is not None:
    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric("Standard Holdout Precision@10", f"{holdout_standard['precision_at_10']:.3f}")
        st.metric("Standard Holdout Recall@10", f"{holdout_standard['recall_at_10']:.3f}")

    with h2:
        st.metric("Robinson Holdout Precision@10", f"{holdout_robinson['precision_at_10']:.3f}")
        st.metric("Robinson Holdout Recall@10", f"{holdout_robinson['recall_at_10']:.3f}")

    with h3:
        st.metric("Standard Holdout RMSE", f"{holdout_standard['rmse']:.3f}")
        st.metric("Robinson Holdout RMSE", f"{holdout_robinson['rmse']:.3f}")

    with h4:
        recall_gain = (
            (holdout_robinson["recall_at_10"] - holdout_standard["recall_at_10"])
            / holdout_standard["recall_at_10"]
        ) * 100

        precision_change = (
            (holdout_robinson["precision_at_10"] - holdout_standard["precision_at_10"])
            / holdout_standard["precision_at_10"]
        ) * 100

        st.metric("Recall Change", f"{recall_gain:.1f}%")
        st.metric("Precision Change", f"{precision_change:.1f}%")

    exp_col1, exp_col2 = st.columns(2, gap="large")

    with exp_col1:
        st.markdown("**What we did**")
        st.markdown("""
We ran a stronger validation test called a **user-level holdout experiment**.

Instead of using one global split across the whole dataset, we split **each user's ratings in half**:

- **50% for training**
- **50% for testing**

That means each system had to learn from only part of a user's history and then try to recover the rest.

This is a stronger test because it evaluates how well the recommender can recover **hidden user preferences**, not just fit the overall dataset.
""")

    with exp_col2:
        st.markdown("**How we did it**")
        st.markdown("""
We built two versions of the same recommendation pipeline:

- **Standard pipeline**: trained on original 1–5 ratings
- **Robinson pipeline**: trained on Robinson-transformed ratings

Both pipelines used the same recommendation algorithm: **SVD collaborative filtering**.

So the only thing that changed was the **rating structure**. That makes this a clean experiment — same users, same items, same split, same model, different rating scale.
""")

    st.markdown("""
**What we found**

In the holdout experiment, the **standard model** remained better at exact prediction.
The **Robinson model** achieved **higher recall**, surfacing more relevant hidden items — meaning it was better at discovering items the user actually liked in the held-out set, even though it was less precise at predicting the exact rating number.
""")

    st.markdown("""
<div class="callout-box-green">
<strong>Main takeaway:</strong> In the user-level holdout test, the Robinson model improved recall by about <strong>25%</strong>,
suggesting that it may be more effective for discovery-oriented recommendation systems.
</div>
""", unsafe_allow_html=True)

    with st.expander("Show holdout results table"):
        st.dataframe(holdout_metrics_df, use_container_width=True)

else:
    st.info(
        "Holdout experiment results were not found. Run `robinson_holdout_experiment.py` "
        "to generate `holdout_experiment_results.csv`."
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ── Why it matters ────────────────────────────────────────────────────────────
st.markdown("""
<p class="section-header">Why the Robinson Scale matters</p>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

st.markdown("""
Traditional rating systems often compress everything into the positive end of the scale. That makes it harder for recommendation systems to tell the difference between:

- something a user thought was merely fine
- something they clearly liked
- something they genuinely loved

The Robinson Scale re-centers ratings around **true neutrality**, which can make recommendation signals more meaningful for exploration and discovery.

This app uses transformed data from a traditional rating system, not data that was originally collected on a true Robinson scale. That means this project is a proof of concept: it shows what happens when existing ratings are restructured around Robinson principles.

A recommendation system built on data that was **originally collected from -5 to 5**, with 0 meaning truly neutral, would likely produce even stronger results because the signal would be cleaner from the start rather than reconstructed afterward.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<p style="text-align:center; color:#a0aec0; font-size:0.8rem; font-family:'DM Sans',sans-serif;">
    Robinson Scale &nbsp;·&nbsp; Owen Robinson &nbsp;·&nbsp; MPA Candidate, Syracuse University
</p>
""", unsafe_allow_html=True)