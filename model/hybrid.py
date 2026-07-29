"""
hybrid.py
---------
Hybrid Recommender: blends the Content-Based (TF-IDF + cosine similarity)
score with the item-based Collaborative Filtering score.

final_score = alpha * content_score + (1 - alpha) * collaborative_score

Both component scores are min-max normalized to [0, 1] per query before
blending so neither model dominates just because of its raw score scale.
Falls back gracefully to content-only if a movie has too few ratings for
collaborative filtering to say anything useful about it.
"""


def _normalize(scores: dict) -> dict:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_recommend(content_model, cf_model, title: str, top_n: int = 10, alpha: float = 0.6):
    """
    alpha: weight given to content-based similarity (0-1).
           alpha=1.0 -> pure content-based, alpha=0.0 -> pure collaborative.
    """
    matched_title, content_recs = content_model.recommend(title, top_n=top_n * 3)
    if matched_title is None:
        return None, []

    content_scores = {r["title"]: r["similarity"] for r in content_recs}

    _, cf_recs = cf_model.similar_movies(matched_title, top_n=top_n * 3)
    cf_scores = {r["title"]: r["similarity"] for r in cf_recs}

    content_norm = _normalize(content_scores)
    cf_norm = _normalize(cf_scores)

    all_titles = set(content_norm) | set(cf_norm)
    blended = {}
    for t in all_titles:
        c = content_norm.get(t, 0.0)
        cf = cf_norm.get(t, 0.0)
        # if a movie is missing from one signal entirely, lean fully on the other
        if t not in content_norm:
            blended[t] = cf
        elif t not in cf_norm:
            blended[t] = c
        else:
            blended[t] = alpha * c + (1 - alpha) * cf

    ranked_titles = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # attach full movie metadata from the content model's dataframe
    df = content_model.df
    results = []
    for t, score in ranked_titles:
        row = df[df["title"] == t]
        if row.empty:
            continue
        row = row.iloc[0]
        results.append({
            "title": row["title"],
            "genres": row["genres"],
            "release_year": int(row["release_year"]),
            "rating": float(row["rating"]),
            "director": row["director"],
            "cast": row["cast"],
            "overview": row["overview"],
            "hybrid_score": round(float(score), 3),
            "content_score": round(float(content_norm.get(t, 0.0)), 3),
            "collaborative_score": round(float(cf_norm.get(t, 0.0)), 3),
        })
    return matched_title, results
