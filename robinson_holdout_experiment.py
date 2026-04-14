import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


# -----------------------------
# LOAD DATA (reuse your file)
# -----------------------------
DATA_PATH = Path("data/ml-100k/u.data")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
    engine="python"
)

print("Loaded data:")
print(df.head())


# -----------------------------
# ROBINSON TRANSFORMATION
# -----------------------------
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


# -----------------------------
# USER-LEVEL SPLIT (KEY PART)
# -----------------------------
def user_level_split(df, test_size=0.5, random_state=42):
    np.random.seed(random_state)

    train_rows = []
    test_rows = []

    for user_id, user_data in df.groupby("user_id"):
        user_data = user_data.sample(frac=1, random_state=random_state)

        split_idx = int(len(user_data) * (1 - test_size))

        train_rows.append(user_data.iloc[:split_idx])
        test_rows.append(user_data.iloc[split_idx:])

    train_df = pd.concat(train_rows)
    test_df = pd.concat(test_rows)

    return train_df, test_df


train_df, test_df = user_level_split(df)

print(f"\nTrain size: {len(train_df)}")
print(f"Test size: {len(test_df)}")


# -----------------------------
# CREATE ROBINSON DATASETS
# -----------------------------
train_robinson = train_df.copy()
test_robinson = test_df.copy()

train_robinson["rating"] = (
    train_robinson.groupby("user_id")["rating"]
    .transform(percentile_to_robinson)
)

test_robinson["rating"] = (
    test_robinson.groupby("user_id")["rating"]
    .transform(percentile_to_robinson)
)


# -----------------------------
# PRECISION / RECALL FUNCTION
# -----------------------------
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

    return np.mean(list(precisions.values())), np.mean(list(recalls.values()))


# -----------------------------
# TRAIN + EVALUATE FUNCTION
# -----------------------------
def run_model(train_df, test_df, rating_scale, threshold):
    reader = Reader(rating_scale=rating_scale)

    train_data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]],
        reader
    )
    trainset = train_data.build_full_trainset()

    algo = SVD(random_state=42)
    algo.fit(trainset)

    testset = list(zip(
        test_df["user_id"],
        test_df["item_id"],
        test_df["rating"]
    ))

    predictions = algo.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    precision, recall = precision_recall_at_k(
        predictions,
        k=10,
        threshold=threshold
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "precision_at_10": precision,
        "recall_at_10": recall
    }


# -----------------------------
# RUN BOTH MODELS
# -----------------------------
print("\nRunning STANDARD model...")
standard_results = run_model(
    train_df,
    test_df,
    rating_scale=(1, 5),
    threshold=4.0
)

print("Running ROBINSON model...")
robinson_results = run_model(
    train_robinson,
    test_robinson,
    rating_scale=(-5, 5),
    threshold=0.5
)


# -----------------------------
# SAVE RESULTS
# -----------------------------
results_df = pd.DataFrame([
    {"model": "standard", **standard_results},
    {"model": "robinson", **robinson_results}
])

print("\nFinal Results:")
print(results_df)

results_df.to_csv("holdout_experiment_results.csv", index=False)

print("\nSaved: holdout_experiment_results.csv")