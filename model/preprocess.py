"""
preprocess.py
--------------
Data Cleaning + Feature Engineering module for the Movie Recommendation System.

Reads the raw movies.csv dataset and builds a single combined "tags" column
(genres + director + cast + keywords + overview) that will be fed into the
TF-IDF vectorizer. Works with any dataset that has these columns:
movie_id, title, release_year, rating, genres, director, cast, keywords, overview

If you swap in the full MovieLens/TMDB dataset, just make sure the columns
match (or update COLUMN NAMES below) -- no other code needs to change.
"""

import os
import pandas as pd
import re


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/extra spaces from a text field."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_spaces(text: str) -> str:
    """
    Collapse multi-word names/keywords into single tokens
    (e.g. 'Christopher Nolan' -> 'christophernolan') so the vectorizer
    treats a full name/keyword as one distinctive feature instead of
    splitting it into common first/last names shared across many movies.
    """
    return text.replace(" ", "")


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning: drop duplicates/nulls, normalize dtypes."""
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    df = df.dropna(subset=["title", "genres"]).reset_index(drop=True)
    df["overview"] = df["overview"].fillna("")
    df["keywords"] = df["keywords"].fillna("")
    df["cast"] = df["cast"].fillna("")
    df["director"] = df["director"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the combined 'tags' column used for content-based similarity.
    Genre/director/cast/keyword tokens are joined without spaces so each
    name/keyword counts as a single distinctive feature; the overview stays
    as normal free text for broader thematic matching.
    """
    df["genres_clean"] = df["genres"].apply(lambda x: " ".join(
        [remove_spaces(g.lower()) for g in str(x).split()]
    ))
    df["director_clean"] = df["director"].apply(lambda x: remove_spaces(clean_text(x)))
    df["cast_clean"] = df["cast"].apply(lambda x: " ".join(
        [remove_spaces(c.lower()) for c in re.split(r"\s{2,}|,", str(x)) if c.strip()]
    ) if "," in str(x) else " ".join(
        [remove_spaces(w.lower()) for w in _group_names(str(x))]
    ))
    df["keywords_clean"] = df["keywords"].apply(lambda x: " ".join(
        [remove_spaces(k.lower()) for k in str(x).split()]
    ))
    df["overview_clean"] = df["overview"].apply(clean_text)

    # weight genres/director more heavily by repeating them
    df["tags"] = (
        (df["genres_clean"] + " ") * 3
        + (df["director_clean"] + " ") * 2
        + (df["cast_clean"] + " ") * 2
        + df["keywords_clean"] + " "
        + df["overview_clean"]
    ).str.strip()

    return df


def _group_names(text: str):
    """Group a space-separated 'First Last First Last' string into full names.
    Heuristic: pairs of capitalized words. Falls back to raw tokens on odd counts."""
    words = text.split()
    names = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            names.append(words[i] + " " + words[i + 1])
            i += 2
        else:
            names.append(words[i])
            i += 1
    return names


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def preprocess(csv_path: str) -> pd.DataFrame:
    """Full pipeline: load -> clean -> engineer features."""
    df = load_data(csv_path)
    df = clean_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    data = preprocess(os.path.join(BASE_DIR, "movies.csv"))
    print(data[["title", "tags"]].head())
    print(f"\nTotal movies processed: {len(data)}")
