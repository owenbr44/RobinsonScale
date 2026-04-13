import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "ml-100k.zip"
EXTRACT_DIR = DATA_DIR / "ml-100k"


def download_movielens_100k() -> Path:
    DATA_DIR.mkdir(exist_ok=True)

    ratings_path = EXTRACT_DIR / "u.data"
    if ratings_path.exists():
        print(f"Found existing dataset at: {ratings_path}")
        return ratings_path

    if not ZIP_PATH.exists():
        print("Downloading MovieLens 100K...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print(f"Downloaded to: {ZIP_PATH}")

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    if not ratings_path.exists():
        raise FileNotFoundError("Could not find extracted ratings file.")

    print(f"Extracted to: {ratings_path}")
    return ratings_path


def load_ratings(ratings_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python"
    )
    return df


def percentile_to_robinson(user_ratings: pd.Series) -> pd.Series:
    n = len(user_ratings)
    if n < 2:
        return pd.Series(np.zeros(n), index=user_ratings.index)

    ranks = user_ratings.rank(method="average")
    percentiles = (ranks - 0.5) / n
    percentiles = percentiles.clip(0.001, 0.999)

    z_scores = norm.ppf(percentiles)
    robinson_scores = np.clip(z_scores * (5 / 3), -5, 5)

    return pd.Series(robinson_scores, index=user_ratings.index)


def add_robinson_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["robinson_rating"] = (
        out.groupby("user_id")["rating"]
        .transform(percentile_to_robinson)
    )
    return out


def load_movie_titles() -> pd.DataFrame:
    """
    Load movie titles from MovieLens 100K u.item file.
    """
    item_path = EXTRACT_DIR / "u.item"
    items = pd.read_csv(
        item_path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"],
        engine="python"
    )
    return items


def plot_distributions(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5), dpi=160)
    plt.hist(df["rating"], bins=np.arange(0.75, 5.76, 0.5), edgecolor="black")
    plt.title("Original MovieLens Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("original_ratings_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5), dpi=160)
    plt.hist(df["robinson_rating"], bins=30, edgecolor="black")
    plt.title("Robinson-Transformed Ratings")
    plt.xlabel("Robinson Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("robinson_ratings_distribution.png", dpi=300)
    plt.close()

    print("Saved:")
    print("- original_ratings_distribution.png")
    print("- robinson_ratings_distribution.png")


def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_est_true = {}

    for pred in predictions:
        uid = pred.uid
        est = pred.est
        true_r = pred.r_ui
        user_est_true.setdefault(uid, []).append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum(true_r >= threshold for (_, true_r) in user_ratings)
        n_rec_k = sum(est >= threshold for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(
            (true_r >= threshold) and (est >= threshold)
            for (est, true_r) in user_ratings[:k]
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k > 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel > 0 else 0

    avg_precision = np.mean(list(precisions.values()))
    avg_recall = np.mean(list(recalls.values()))

    return avg_precision, avg_recall


def evaluate_with_surprise(
    df: pd.DataFrame,
    rating_col: str,
    rating_scale: tuple[float, float],
    relevance_threshold: float,
    random_state: int = 42
) -> dict:
    data_for_surprise = df[["user_id", "item_id", rating_col]].copy()

    reader = Reader(rating_scale=rating_scale)
    dataset = Dataset.load_from_df(data_for_surprise, reader)

    trainset, testset = train_test_split(
        dataset,
        test_size=0.2,
        random_state=random_state
    )

    algo = SVD(random_state=random_state)
    algo.fit(trainset)
    predictions = algo.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    precision_10, recall_10 = precision_recall_at_k(
        predictions,
        k=10,
        threshold=relevance_threshold
    )

    return {
        "rating_column": rating_col,
        "rmse": rmse,
        "mae": mae,
        "precision_at_10": precision_10,
        "recall_at_10": recall_10,
        "algo": algo,
        "trainset": trainset,
        "predictions": predictions
    }


def get_top_n(algo, trainset, user_id, n=10):
    """
    Get top-N unrated item recommendations for a user.
    """
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        return []

    all_inner_items = set(trainset.all_items())
    rated_inner_items = {iid for (iid, _) in trainset.ur[inner_uid]}
    unrated_inner_items = all_inner_items - rated_inner_items

    recommendations = []
    for inner_iid in unrated_inner_items:
        raw_iid = trainset.to_raw_iid(inner_iid)
        pred = algo.predict(user_id, raw_iid)
        recommendations.append((int(raw_iid), pred.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]


def add_titles_to_recommendations(recommendations, movie_titles_df):
    """
    Join movie titles onto recommendation tuples.
    """
    title_map = dict(zip(movie_titles_df["item_id"], movie_titles_df["title"]))
    output = []

    for item_id, score in recommendations:
        title = title_map.get(item_id, f"Movie {item_id}")
        output.append({
            "item_id": item_id,
            "title": title,
            "predicted_score": score
        })

    return pd.DataFrame(output)


def get_item_popularity(df: pd.DataFrame) -> dict:
    """
    Count how many ratings each movie has in the dataset.
    Higher count = more popular.
    """
    return df.groupby("item_id").size().to_dict()


def recommendation_overlap(recs_a, recs_b) -> float:
    """
    Share of overlapping items between two recommendation lists.
    """
    set_a = {item_id for item_id, _ in recs_a}
    set_b = {item_id for item_id, _ in recs_b}

    if len(set_a) == 0:
        return 0.0

    return len(set_a & set_b) / len(set_a)


def average_popularity(recs, popularity_dict: dict) -> float:
    """
    Average popularity of recommended items.
    Lower means recommendations are less dominated by blockbuster/popular movies.
    """
    if not recs:
        return 0.0

    pops = [popularity_dict.get(item_id, 0) for item_id, _ in recs]
    return float(np.mean(pops))


def catalog_diversity(all_recs) -> int:
    """
    Number of unique items recommended across many users.
    Higher means broader recommendation diversity.
    """
    unique_items = set()
    for recs in all_recs:
        for item_id, _ in recs:
            unique_items.add(item_id)
    return len(unique_items)


def compare_systems_across_users(assets: dict, df: pd.DataFrame, user_ids, n=10) -> dict:
    popularity = get_item_popularity(df)

    overlaps = []
    std_popularities = []
    rob_popularities = []

    all_std_recs = []
    all_rob_recs = []

    for user_id in user_ids:
        std_recs = get_top_n(
            assets["original_algo"],
            assets["original_trainset"],
            user_id,
            n=n
        )

        rob_recs = get_top_n(
            assets["robinson_algo"],
            assets["robinson_trainset"],
            user_id,
            n=n
        )

        if len(std_recs) == 0 or len(rob_recs) == 0:
            continue

        overlaps.append(recommendation_overlap(std_recs, rob_recs))
        std_popularities.append(average_popularity(std_recs, popularity))
        rob_popularities.append(average_popularity(rob_recs, popularity))

        all_std_recs.append(std_recs)
        all_rob_recs.append(rob_recs)

    results = {
        "avg_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "avg_standard_popularity": float(np.mean(std_popularities)) if std_popularities else 0.0,
        "avg_robinson_popularity": float(np.mean(rob_popularities)) if rob_popularities else 0.0,
        "standard_catalog_diversity": catalog_diversity(all_std_recs),
        "robinson_catalog_diversity": catalog_diversity(all_rob_recs),
    }

    return results


def main():
    ratings_path = download_movielens_100k()
    df = load_ratings(ratings_path)
    movie_titles = load_movie_titles()

    print("\nLoaded ratings:")
    print(df.head())
    print(f"\nRows: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Items: {df['item_id'].nunique():,}")

    df = add_robinson_scores(df)

    print("\nSample with Robinson scores:")
    print(df[["user_id", "item_id", "rating", "robinson_rating"]].head(10))

    plot_distributions(df)

    print("\nEvaluating original 1-5 ratings...")
    original_results = evaluate_with_surprise(
        df=df,
        rating_col="rating",
        rating_scale=(1, 5),
        relevance_threshold=4.0
    )

    print("Evaluating Robinson ratings...")
    robinson_results = evaluate_with_surprise(
        df=df,
        rating_col="robinson_rating",
        rating_scale=(-5, 5),
        relevance_threshold=0.5
    )

    results_df = pd.DataFrame([
        {
            "rating_column": original_results["rating_column"],
            "rmse": original_results["rmse"],
            "mae": original_results["mae"],
            "precision_at_10": original_results["precision_at_10"],
            "recall_at_10": original_results["recall_at_10"],
        },
        {
            "rating_column": robinson_results["rating_column"],
            "rmse": robinson_results["rmse"],
            "mae": robinson_results["mae"],
            "precision_at_10": robinson_results["precision_at_10"],
            "recall_at_10": robinson_results["recall_at_10"],
        }
    ])

    print("\nResults:")
    print(results_df)

    results_df.to_csv("model_comparison_results.csv", index=False)
    df.to_csv("movielens_with_robinson.csv", index=False)

    print("\nSaved:")
    print("- movielens_with_robinson.csv")
    print("- model_comparison_results.csv")

    # Sample recommendation comparison
    sample_user_id = 196

    print("\n=== SAMPLE RECOMMENDATIONS ===")
    print(f"User ID: {sample_user_id}")

    original_top = get_top_n(
        algo=original_results["algo"],
        trainset=original_results["trainset"],
        user_id=sample_user_id,
        n=10
    )
    robinson_top = get_top_n(
        algo=robinson_results["algo"],
        trainset=robinson_results["trainset"],
        user_id=sample_user_id,
        n=10
    )

    original_top_df = add_titles_to_recommendations(original_top, movie_titles)
    robinson_top_df = add_titles_to_recommendations(robinson_top, movie_titles)

    print("\nOriginal model recommendations:")
    print(original_top_df.to_string(index=False))

    print("\nRobinson model recommendations:")
    print(robinson_top_df.to_string(index=False))

    original_top_df.to_csv("original_top_recommendations.csv", index=False)
    robinson_top_df.to_csv("robinson_top_recommendations.csv", index=False)

    print("\nSaved:")
    print("- original_top_recommendations.csv")
    print("- robinson_top_recommendations.csv")

    # System-wide comparison
    print("\nRunning system-wide comparison...")

    sample_users = sorted(df["user_id"].unique())[:100]

    comparison = compare_systems_across_users(
        assets={
            "original_algo": original_results["algo"],
            "original_trainset": original_results["trainset"],
            "robinson_algo": robinson_results["algo"],
            "robinson_trainset": robinson_results["trainset"],
        },
        df=df,
        user_ids=sample_users,
        n=10
    )

    print("\n=== SYSTEM COMPARISON ACROSS USERS ===")
    for key, value in comparison.items():
        print(f"{key}: {value}")

    comparison_df = pd.DataFrame([comparison])
    comparison_df.to_csv("system_comparison_metrics.csv", index=False)

    print("\nSaved:")
    print("- system_comparison_metrics.csv")


if __name__ == "__main__":
    main()