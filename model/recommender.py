"""
recommender.py
---------------
Content-Based Filtering recommendation engine.

Pipeline:
    tags column --> TF-IDF Vectorization --> Cosine Similarity Matrix --> Top-N lookup

Run this file directly to train the model and save:
    - movie_list.pkl   (processed dataframe)
    - similarity.pkl   (cosine similarity matrix)
to the project root, ready to be loaded by app.py.
"""

import os
import pickle
import difflib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


class MovieRecommender:
    def __init__(self, df: pd.DataFrame = None, similarity=None):
        self.df = df
        self.similarity = similarity
        self.vectorizer = None

    # ---------- training ----------
    def fit(self, df: pd.DataFrame):
        """Build the TF-IDF matrix and cosine similarity matrix from the tags column."""
        self.df = df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = self.vectorizer.fit_transform(self.df["tags"])
        self.similarity = cosine_similarity(tfidf_matrix)
        return self

    # ---------- lookup ----------
    def find_closest_title(self, query: str):
        """Fuzzy-match a user's search string to the closest movie title in the dataset."""
        titles = self.df["title"].tolist()
        matches = difflib.get_close_matches(query, titles, n=1, cutoff=0.4)
        if matches:
            return matches[0]
        # fallback: substring search
        query_lower = query.lower()
        for t in titles:
            if query_lower in t.lower():
                return t
        return None

    def recommend(self, title: str, top_n: int = 10):
        """Return the top_n most similar movies to the given title."""
        matched_title = self.find_closest_title(title)
        if matched_title is None:
            return None, []

        idx = self.df[self.df["title"] == matched_title].index[0]
        sim_scores = list(enumerate(self.similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

        recommendations = []
        for i, score in sim_scores:
            row = self.df.iloc[i]
            recommendations.append({
                "title": row["title"],
                "genres": row["genres"],
                "release_year": int(row["release_year"]),
                "rating": float(row["rating"]),
                "director": row["director"],
                "cast": row["cast"],
                "overview": row["overview"],
                "similarity": round(float(score), 3),
            })
        return matched_title, recommendations

    # ---------- persistence ----------
    def save(self, movie_list_path="movie_list.pkl", similarity_path="similarity.pkl"):
        with open(movie_list_path, "wb") as f:
            pickle.dump(self.df, f)
        with open(similarity_path, "wb") as f:
            pickle.dump(self.similarity, f)

    @classmethod
    def load(cls, movie_list_path="movie_list.pkl", similarity_path="similarity.pkl"):
        with open(movie_list_path, "rb") as f:
            df = pickle.load(f)
        with open(similarity_path, "rb") as f:
            similarity = pickle.load(f)
        return cls(df=df, similarity=similarity)


def train_and_save(csv_path=None,
                    movie_list_path=None,
                    similarity_path=None):
    """End-to-end training script: preprocess data, fit model, persist artifacts."""
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, "movies.csv")
    if movie_list_path is None:
        movie_list_path = os.path.join(BASE_DIR, "movie_list.pkl")
    if similarity_path is None:
        similarity_path = os.path.join(BASE_DIR, "similarity.pkl")

    print("Loading and preprocessing dataset...")
    df = preprocess(csv_path)

    print(f"Training TF-IDF + Cosine Similarity model on {len(df)} movies...")
    model = MovieRecommender().fit(df)

    print("Saving model artifacts...")
    model.save(movie_list_path, similarity_path)
    print(f"Saved: {movie_list_path}, {similarity_path}")
    return model


if __name__ == "__main__":
    model = train_and_save()

    # quick sanity check
    test_title = "Inception"
    matched, recs = model.recommend(test_title, top_n=5)
    print(f"\nTop 5 recommendations for '{matched}':")
    for r in recs:
        print(f"  - {r['title']} ({r['release_year']}) | similarity={r['similarity']}")
