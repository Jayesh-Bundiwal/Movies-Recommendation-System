"""
generate_ratings.py
--------------------
Generates a synthetic user-item ratings dataset so the project can
demonstrate Collaborative Filtering alongside Content-Based Filtering.

Real deployments would use MovieLens' ratings.csv (userId, movieId, rating,
timestamp) directly. Since our sample catalogue is hand-built, we simulate
1,800 users rating movies with a genre-affinity bias per user (each user
prefers 1-3 genres and rates those movies higher on average), which produces
a realistic-ish clustering structure for the CF model to pick up on.
"""

import os
import pandas as pd
import numpy as np

np.random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

N_USERS = 1800
MIN_RATINGS_PER_USER = 10
MAX_RATINGS_PER_USER = 30


def generate_ratings(movies_csv=None, out_csv=None):
    if movies_csv is None:
        movies_csv = os.path.join(BASE_DIR, "movies.csv")
    if out_csv is None:
        out_csv = os.path.join(BASE_DIR, "data", "ratings.csv")
    movies = pd.read_csv(movies_csv)
    movies["genre_list"] = movies["genres"].apply(lambda g: str(g).split())
    all_genres = sorted(set(g for row in movies["genre_list"] for g in row))

    records = []
    for user_id in range(1, N_USERS + 1):
        # each synthetic user has an affinity for 1-3 genres
        n_pref = np.random.randint(1, 4)
        preferred_genres = set(np.random.choice(all_genres, size=n_pref, replace=False))

        n_ratings = np.random.randint(MIN_RATINGS_PER_USER, MAX_RATINGS_PER_USER + 1)
        rated_movies = movies.sample(n=min(n_ratings, len(movies)), replace=False)

        for _, movie in rated_movies.iterrows():
            overlap = len(preferred_genres.intersection(movie["genre_list"]))
            base = 3.0 + overlap * 1.0  # more genre overlap -> higher base rating
            noise = np.random.normal(0, 0.8)
            rating = np.clip(round((base + noise) * 2) / 2, 1.0, 5.0)  # round to nearest 0.5, clip 1-5
            records.append((user_id, int(movie["movie_id"]), rating))

    ratings_df = pd.DataFrame(records, columns=["user_id", "movie_id", "rating"])
    ratings_df.to_csv(out_csv, index=False)
    print(f"Generated {len(ratings_df)} ratings from {N_USERS} synthetic users -> {out_csv}")
    return ratings_df


if __name__ == "__main__":
    generate_ratings()
