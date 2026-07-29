"""
collaborative.py
------------------
Collaborative Filtering module (memory-based item-item CF, with a
matrix-factorization/SVD variant) that complements the content-based
recommender.py.

Two things are provided:
  1. Item-based CF: "users who liked this movie also liked..." — built
     from the user-item ratings matrix's cosine similarity between
     movie rating-vectors.
  2. User-based recommendations: build a low-rank latent embedding
     (TruncatedSVD) of the user-item matrix and predict ratings for
     movies a given user hasn't rated yet.

Run this file directly to train and save:
    - cf_item_similarity.pkl
    - cf_model.pkl (user-item matrix + SVD components, for user recs)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class CollaborativeRecommender:
    def __init__(self):
        self.user_item_matrix = None      # rows=users, cols=movie_id (raw, uncentered ratings)
        self.item_similarity = None        # movie_id x movie_id cosine similarity
        self.movie_ids = None
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.user_means = None             # per-user mean rating (for mean-centering)
        self.global_mean = None
        self.movie_id_to_title = {}
        self.title_to_movie_id = {}

    # ---------- training ----------
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, n_components: int = 20):
        self.movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))
        self.title_to_movie_id = {v: k for k, v in self.movie_id_to_title.items()}

        # build the user-item ratings matrix (sparse-ish, filled with 0 for unrated)
        pivot = ratings_df.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).fillna(0)
        self.user_item_matrix = pivot
        self.movie_ids = pivot.columns.tolist()

        # item-item similarity: transpose so rows = movies, cols = users
        item_matrix = pivot.T.values
        self.item_similarity = cosine_similarity(item_matrix)

        # --- mean-centering ---
        # A raw zero-filled matrix biases naive SVD toward predicting near-zero
        # ratings for anything a user hasn't rated (0 gets treated as "hated it"
        # instead of "unknown"), and the resulting dot products aren't bounded
        # to the 1-5 scale at all. Subtracting each user's mean rating before
        # factorizing -- and adding it back at prediction time -- fixes both:
        # the model now learns deviations from a user's own average, and
        # unrated cells contribute a neutral (mean) signal instead of "hated it".
        self.global_mean = float(ratings_df["rating"].mean())
        rated_mask = pivot.values > 0
        row_sums = (pivot.values * rated_mask).sum(axis=1)
        row_counts = rated_mask.sum(axis=1)
        row_means = np.divide(row_sums, row_counts, out=np.full(len(pivot), self.global_mean), where=row_counts > 0)
        self.user_means = pd.Series(row_means, index=pivot.index)

        centered = pivot.values.copy().astype(float)
        centered[rated_mask] -= row_means[:, None].repeat(pivot.shape[1], axis=1)[rated_mask]
        # unrated cells stay at 0 in the centered matrix -> "neutral / average" signal

        # matrix factorization for personalized user predictions
        n_components = min(n_components, min(pivot.shape) - 1) if min(pivot.shape) > 1 else 1
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(centered)
        self.item_factors = self.svd.components_.T
        return self

    # ---------- item-based: "users who liked X also liked..." ----------
    def similar_movies(self, title: str, top_n: int = 10):
        if title not in self.title_to_movie_id:
            return None, []
        movie_id = self.title_to_movie_id[title]
        if movie_id not in self.movie_ids:
            return title, []

        idx = self.movie_ids.index(movie_id)
        sims = list(enumerate(self.item_similarity[idx]))
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        sims = [s for s in sims if s[0] != idx][:top_n]

        results = []
        for i, score in sims:
            mid = self.movie_ids[i]
            results.append({
                "movie_id": mid,
                "title": self.movie_id_to_title.get(mid, "Unknown"),
                "similarity": round(float(score), 3),
            })
        return title, results

    # ---------- core prediction: mean-centered SVD, clipped to the rating scale ----------
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predicted rating for a (user, movie) pair, clipped to [1, 5]."""
        if user_id not in self.user_item_matrix.index or movie_id not in self.movie_ids:
            return self.global_mean
        user_pos = self.user_item_matrix.index.get_loc(user_id)
        item_pos = self.movie_ids.index(movie_id)
        deviation = self.user_factors[user_pos].dot(self.item_factors[item_pos])
        pred = self.user_means.loc[user_id] + deviation
        return float(np.clip(pred, 1.0, 5.0))

    # ---------- user-based: predicted ratings for unrated movies ----------
    def recommend_for_user(self, user_id: int, top_n: int = 10):
        if user_id not in self.user_item_matrix.index:
            return []

        user_pos = self.user_item_matrix.index.get_loc(user_id)
        deviations = self.user_factors[user_pos].dot(self.item_factors.T)
        predicted_ratings = np.clip(self.user_means.loc[user_id] + deviations, 1.0, 5.0)

        already_rated = set(
            self.user_item_matrix.columns[self.user_item_matrix.loc[user_id] > 0]
        )

        scored = []
        for i, mid in enumerate(self.movie_ids):
            if mid in already_rated:
                continue
            scored.append((mid, predicted_ratings[i]))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_n]

        return [{
            "movie_id": mid,
            "title": self.movie_id_to_title.get(mid, "Unknown"),
            "predicted_rating": round(float(score), 2),
        } for mid, score in scored]

    def get_user_history(self, user_id: int):
        if user_id not in self.user_item_matrix.index:
            return []
        row = self.user_item_matrix.loc[user_id]
        rated = row[row > 0].sort_values(ascending=False)
        return [{
            "title": self.movie_id_to_title.get(mid, "Unknown"),
            "rating": float(r),
        } for mid, r in rated.items()]

    # ---------- persistence ----------
    def save(self, path=None):
        if path is None:
            path = os.path.join(BASE_DIR, "cf_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path=None):
        if path is None:
            path = os.path.join(BASE_DIR, "cf_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


def train_and_save(ratings_csv=None,
                    movies_csv=None,
                    out_path=None):
    if ratings_csv is None:
        ratings_csv = os.path.join(BASE_DIR, "data", "ratings.csv")
    if movies_csv is None:
        movies_csv = os.path.join(BASE_DIR, "movies.csv")
    if out_path is None:
        out_path = os.path.join(BASE_DIR, "cf_model.pkl")
    ratings_df = pd.read_csv(ratings_csv)
    movies_df = pd.read_csv(movies_csv)

    print(f"Training collaborative filtering model on {len(ratings_df)} ratings "
          f"from {ratings_df['user_id'].nunique()} users...")
    model = CollaborativeRecommender().fit(ratings_df, movies_df)
    model.save(out_path)
    print(f"Saved: {out_path}")
    return model


if __name__ == "__main__":
    model = train_and_save()

    # sanity checks
    title, sims = model.similar_movies("Inception", top_n=5)
    print(f"\nItem-based CF — users who rated '{title}' highly also liked:")
    for s in sims:
        print(f"  - {s['title']} | similarity={s['similarity']}")

    recs = model.recommend_for_user(user_id=1, top_n=5)
    print(f"\nPersonalized picks for user 1:")
    for r in recs:
        print(f"  - {r['title']} | predicted_rating={r['predicted_rating']}")

    history = model.get_user_history(1)
    print(f"\nUser 1's rating history ({len(history)} movies):")
    for h in history[:5]:
        print(f"  - {h['title']}: {h['rating']}")
