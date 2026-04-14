from pathlib import Path
import pandas as pd
import streamlit as st

from recommender_core import build_assets, get_top_n, recommendation_df


st.set_page_config(
    page_title="Robinson Recommender Demo",
    layout="wide"
)


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

st.title("Robinson Recommender Demo")

st.markdown("""
## What this app does

This app compares two recommendation systems built on the MovieLens 100K dataset:

- **Standard**: trained on the original 1–5 ratings
- **Robinson**: trained on Robinson-transformed ratings that re-center users around a true neutral point

The goal is not just to predict exact ratings. It is to test whether the Robinson Scale can produce recommendation lists that are more useful for discovery.
""")

col_a, col_b = st.columns([3, 1])

with col_b:
    if st.button("Reload metrics"):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

st.header("Model Findings")

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
### Interpretation

- The **standard model** is better at predicting exact rating values.
- The **Robinson model** has **higher recall**, meaning it surfaces more relevant items.
- The **Robinson model** also shows **greater catalog diversity**, meaning it spreads recommendations across more distinct movies.

That makes Robinson especially interesting for **discovery-oriented recommendation systems**.
""")

    st.info("""
**Why Recall Matters Here**

In many recommendation systems, **precision** is treated as the main goal. Precision asks whether the recommended items are highly accurate.

That is useful, but it often leads to:

- repetitive recommendations
- over-reliance on the safest popular choices
- less exploration of the catalog

For this project, the Robinson model is meant to support **discovery**, not just exact prediction.

That makes **recall** especially important here. Higher recall means the system is surfacing **more items a user would actually like**, even if they are not all the most obvious or conservative choices.

In other words:

- **precision** favors safer recommendations
- **recall** favors broader discovery

The Robinson model should therefore be understood as a system that may trade some precision for a better ability to uncover relevant content.
""")
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

st.markdown("---")

st.header("Advanced Experiment: User-Level Holdout")

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

    st.markdown("""
### What we did

We ran a stronger validation test called a **user-level holdout experiment**.

Instead of using one global split across the whole dataset, we split **each user’s ratings in half**:

- **50% for training**
- **50% for testing**

That means each system had to learn from only part of a user’s history and then try to recover the rest.

This is a stronger test because it evaluates how well the recommender can recover **hidden user preferences**, not just fit the overall dataset.
""")

    st.markdown("""
### How we did it

We built two versions of the same recommendation pipeline:

- **Standard pipeline**: trained on original 1–5 ratings
- **Robinson pipeline**: trained on Robinson-transformed ratings

Both pipelines used the same recommendation algorithm:

- **SVD collaborative filtering**

So the only thing that changed was the **rating structure**.

That makes this a clean experiment:
- same users
- same items
- same split
- same model
- different rating scale
""")

    st.markdown("""
### What we found

In the holdout experiment:

- the **standard model** remained better at exact prediction
- the **Robinson model** achieved **higher recall**
- the Robinson model surfaced **more relevant hidden items**

This means the Robinson model was better at discovering items the user actually liked in the held-out set, even though it was less precise and less accurate in the narrow sense of predicting the exact rating number.
""")

    st.success("""
**Main takeaway:**  
In the user-level holdout test, the Robinson model improved recall by about **25%**, suggesting that it may be more effective for discovery-oriented recommendation systems.
""")

    with st.expander("Show holdout results table"):
        st.dataframe(holdout_metrics_df, use_container_width=True)
else:
    st.info(
        "Holdout experiment results were not found. Run `robinson_holdout_experiment.py` "
        "to generate `holdout_experiment_results.csv`."
    )

st.markdown("---")

st.header("User-Level Recommendation Comparison")

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

st.subheader(f"Top {top_n} recommendations for user {user_id} ({mode})")
st.dataframe(single_df, use_container_width=True)

st.markdown("---")

st.subheader("Side-by-Side Comparison")

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

left, right = st.columns(2)

with left:
    st.markdown("### Standard")
    st.caption("Built from original 1–5 ratings.")
    st.dataframe(standard_df, use_container_width=True)

with right:
    st.markdown("### Robinson")
    st.caption("Built from Robinson-transformed ratings.")
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
### What this means

- A **high overlap** means Robinson is mostly re-ranking the same strong candidates.
- A **lower overlap** means Robinson is introducing more different items.
- In the full experiment, the average overlap across users was about **0.73**, so Robinson usually preserves the core recommendation set while still changing part of the list.
""")

st.markdown("---")

st.header("Why the Robinson Scale matters")

st.markdown("""
Traditional rating systems often compress everything into the positive end of the scale. That makes it harder for recommendation systems to tell the difference between:

- something a user thought was merely fine
- something they clearly liked
- something they genuinely loved

The Robinson Scale re-centers ratings around **true neutrality**, which can make recommendation signals more meaningful for exploration and discovery.

This app uses transformed data from a traditional rating system, not data that was originally collected on a true Robinson scale. That means this project is a proof of concept: it shows what happens when existing ratings are restructured around Robinson principles.

A recommendation system built on data that was **originally collected from -5 to 5**, with 0 meaning truly neutral, would likely produce even stronger results because the signal would be cleaner from the start rather than reconstructed afterward.
""")