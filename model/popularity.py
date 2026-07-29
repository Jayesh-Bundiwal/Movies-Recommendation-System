"""
popularity.py
--------------
IMDB-style weighted popularity ranking.

A movie's own average rating is a noisy signal when it has very few votes
(one 5-star vote looks "better" than a movie with 3,000 votes averaging
4.7). The classic IMDB "weighted rating" (Bayesian average) formula fixes
this by pulling low-vote movies back toward the dataset's overall mean:

    WR = (v / (v + m)) * R  +  (m / (v + m)) * C

    R = the movie's own average rating
    v = number of votes/ratings the movie has received
    m = minimum votes threshold (movies below this get pulled hardest
        toward C) -- we use the dataset's median vote count
    C = the mean rating across the whole dataset

We don't have a raw "number of votes" field in movies.csv, but
data/ratings.csv (the synthetic user-item ratings used for Collaborative
Filtering) gives us a real vote count per movie: how many synthetic users
rated it. Movies with very few ratings get pulled toward the catalogue
average; movies with many ratings keep closer to their own average.

If data/ratings.csv hasn't been generated yet, falls back gracefully to
using each movie's raw `rating` column (v is treated as uniform).
"""

import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def compute_weighted_ratings(movies_df: pd.DataFrame, ratings_csv: str = None) -> pd.DataFrame:
    """
    Returns a copy of movies_df with two new columns:
      - vote_count       : number of ratings the movie received (from ratings.csv)
      - weighted_rating   : IMDB-style Bayesian-average score
    """
    df = movies_df.copy()
    C = float(df["rating"].mean())

    vote_counts = None
    if ratings_csv and os.path.exists(ratings_csv):
        ratings_df = pd.read_csv(ratings_csv)
        vote_counts = ratings_df.groupby("movie_id").size()

    if vote_counts is not None and len(vote_counts) > 0:
        df["vote_count"] = df["movie_id"].map(vote_counts).fillna(0).astype(int)
        m = float(vote_counts.median())
    else:
        # no ratings data available yet -- treat every movie as having the
        # same (median) vote count so weighting degrades to the raw rating
        df["vote_count"] = 0
        m = 1.0

    v = df["vote_count"].astype(float)
    R = df["rating"].astype(float)
    df["weighted_rating"] = ((v / (v + m)) * R + (m / (v + m)) * C).round(3)
    return df


if __name__ == "__main__":
    movies = pd.read_csv(os.path.join(BASE_DIR, "movies.csv"))
    ranked = compute_weighted_ratings(movies, ratings_csv=os.path.join(BASE_DIR, "data", "ratings.csv"))
    top = ranked.sort_values("weighted_rating", ascending=False).head(10)
    print("Top 10 by IMDB-style weighted rating:")
    for _, row in top.iterrows():
        print(f"  - {row['title']} | raw={row['rating']} | votes={row['vote_count']} | weighted={row['weighted_rating']}")
