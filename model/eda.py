"""
eda.py
------
Exploratory Data Analysis for the movie dataset + synthetic ratings.
Generates charts (saved as PNGs into static/images/eda/) that get displayed
on the Flask app's /analytics page, and are useful directly in a project
report.

Run directly: python model/eda.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt

plt.style.use("dark_background")
ACCENT = "#e63946"
ACCENT_SOFT = "#ff6b6b"
MUTED = "#9aa0ac"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT_DIR = os.path.join(BASE_DIR, "static", "images", "eda")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved {path}")


def genre_distribution(movies: pd.DataFrame):
    genre_counts = {}
    for g in movies["genres"]:
        for tok in str(g).split():
            genre_counts[tok] = genre_counts.get(tok, 0) + 1
    series = pd.Series(genre_counts).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(series.index, series.values, color=ACCENT)
    ax.set_title("Movie Count by Genre", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Number of Movies")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, "genre_distribution.png")


def rating_distribution(movies: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(movies["rating"], bins=15, color=ACCENT_SOFT, edgecolor="none")
    ax.set_title("Distribution of Movie Ratings (IMDb-style)", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, "rating_distribution.png")


def movies_by_decade(movies: pd.DataFrame):
    decades = (movies["release_year"] // 10 * 10).astype(int)
    counts = decades.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"{d}s" for d in counts.index], counts.values, color=ACCENT)
    ax.set_title("Movies by Release Decade", fontsize=14, fontweight="bold", color="white")
    ax.set_ylabel("Number of Movies")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, "movies_by_decade.png")


def top_directors(movies: pd.DataFrame, top_n=10):
    counts = movies["director"].value_counts().head(top_n).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(counts.index, counts.values, color=ACCENT_SOFT)
    ax.set_title(f"Top {top_n} Directors by Movie Count", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Number of Movies in Dataset")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, "top_directors.png")


def ratings_per_user_distribution(ratings: pd.DataFrame):
    counts = ratings.groupby("user_id").size()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=20, color=ACCENT, edgecolor="none")
    ax.set_title("Number of Ratings per User (Synthetic Data)", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Ratings per user")
    ax.set_ylabel("Number of users")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, "ratings_per_user.png")


def user_rating_value_distribution(ratings: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = ratings["rating"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=ACCENT_SOFT, width=0.6)
    ax.set_title("Distribution of User Rating Values (Synthetic Data)", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Rating given")
    ax.set_ylabel("Count")
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, "user_rating_values.png")


def run_all(movies_csv=None, ratings_csv=None):
    if movies_csv is None:
        movies_csv = os.path.join(BASE_DIR, "movies.csv")
    if ratings_csv is None:
        ratings_csv = os.path.join(BASE_DIR, "data", "ratings.csv")
    movies = pd.read_csv(movies_csv)
    ratings = pd.read_csv(ratings_csv)

    genre_distribution(movies)
    rating_distribution(movies)
    movies_by_decade(movies)
    top_directors(movies)
    ratings_per_user_distribution(ratings)
    user_rating_value_distribution(ratings)

    print("\nAll EDA charts generated.")


if __name__ == "__main__":
    run_all()
