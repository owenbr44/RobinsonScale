import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from surprise import Dataset, Reader, SVD

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "ml-100k.zip"
EXTRACT_DIR = DATA_DIR / "ml-100k"


def download_movielens_100k() -> Path:
    DATA_DIR.mkdir(exist_ok=True)

    ratings_path = EXTRACT_DIR / "u.data"
    if ratings_path.exists():
        return ratings_path

    if not ZIP_PATH.exists():
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    if not ratings_path.exists():
        raise FileNotFoundError("Could not find extracted ratings file.")

    return ratings_path


def load_ratings() -> pd.DataFrame:
    ratings_path = download_movielens_100k()
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return df


def load_movie_titles() -> pd.DataFrame:
    item_path = EXTRACT_DIR / "u.item"
    items = pd.read_csv(
        item_path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"],
        engine="python",
    )
    return items


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


def train_model(
    df: pd.DataFrame,
    rating_col: str,
    rating_scale: tuple[float, float],
    random_state: int = 42,
):
    data_for_surprise = df[["user_id", "item_id", rating_col]].copy()

    reader = Reader(rating_scale=rating_scale)
    dataset = Dataset.load_from_df(data_for_surprise, reader)
    trainset = dataset.build_full_trainset()

    algo = SVD(random_state=random_state)
    algo.fit(trainset)

    return algo, trainset


def get_top_n(algo, trainset, user_id: int, n: int = 10):
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        return []

    all_inner_items = set(trainset.all_items())
    rated_inner_items = {iid for (iid, _) in trainset.ur[inner_uid]}
    unrated_inner_items = all_inner_items - rated_inner_items

    recommendations = []
    for inner_iid in unrated_inner_items:
        raw_iid = int(trainset.to_raw_iid(inner_iid))
        pred = algo.predict(user_id, raw_iid)
        recommendations.append((raw_iid, pred.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]


def recommendation_df(recommendations, movie_titles_df: pd.DataFrame) -> pd.DataFrame:
    title_map = dict(zip(movie_titles_df["item_id"], movie_titles_df["title"]))

    rows = []
    for item_id, score in recommendations:
        rows.append(
            {
                "item_id": int(item_id),
                "title": title_map.get(int(item_id), f"Movie {item_id}"),
                "predicted_score": float(score),
            }
        )

    return pd.DataFrame(rows)


def build_assets():
    df = add_robinson_scores(load_ratings())
    movie_titles = load_movie_titles()

    original_algo, original_trainset = train_model(
        df=df,
        rating_col="rating",
        rating_scale=(1, 5),
    )

    robinson_algo, robinson_trainset = train_model(
        df=df,
        rating_col="robinson_rating",
        rating_scale=(-5, 5),
    )

    return {
        "df": df,
        "movie_titles": movie_titles,
        "original_algo": original_algo,
        "original_trainset": original_trainset,
        "robinson_algo": robinson_algo,
        "robinson_trainset": robinson_trainset,
    }