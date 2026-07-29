"""
app.py
------
Flask web application for the Movie Recommendation System.

Routes:
    GET  /                -> search/home page
    POST /recommend        -> form submit, redirects to results page
    GET  /recommend/<title> -> results page (also usable directly / via API)
    GET  /movie/<title>     -> movie detail page (full info + recs + watchlist)
    GET  /genre/<genre>      -> browse movies by genre
    GET  /surprise            -> random movie -> redirects to its detail page
    GET  /watchlist             -> view session watchlist
    POST /watchlist/toggle/<title> -> add/remove a movie from the watchlist
    GET  /api/recommend    -> JSON API (query param: title, top_n)
    GET  /api/search       -> JSON autocomplete suggestions (query param: q)
"""

import os
import sys
import csv
import random
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from recommender import MovieRecommender  # noqa: E402
from popularity import compute_weighted_ratings  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIE_LIST_PATH = os.path.join(BASE_DIR, "movie_list.pkl")
SIMILARITY_PATH = os.path.join(BASE_DIR, "similarity.pkl")
CF_MODEL_PATH = os.path.join(BASE_DIR, "cf_model.pkl")
MOVIES_CSV_PATH = os.path.join(BASE_DIR, "movies.csv")
RATINGS_CSV_PATH = os.path.join(BASE_DIR, "data", "ratings.csv")

app = Flask(__name__)
# Needed for session-based features (watchlist). In a real deployment this
# should come from an environment variable, not be hardcoded.
app.secret_key = os.environ.get("SECRET_KEY", "mca-minor-project-dev-key-change-me")

# Load the trained content-based model once at startup
if not (os.path.exists(MOVIE_LIST_PATH) and os.path.exists(SIMILARITY_PATH)):
    raise RuntimeError(
        "Model artifacts not found. Run `python model/recommender.py` first "
        "to train and save movie_list.pkl / similarity.pkl."
    )

model = MovieRecommender.load(MOVIE_LIST_PATH, SIMILARITY_PATH)
ALL_TITLES = sorted(model.df["title"].tolist())
ALL_GENRES = sorted(set(g for row in model.df["genres"] for g in str(row).split()))

# IMDB-style weighted popularity ranking (Bayesian average of rating vs.
# vote count, using data/ratings.csv as the vote-count source).
WEIGHTED_DF = compute_weighted_ratings(model.df, ratings_csv=RATINGS_CSV_PATH)
WEIGHTED_LOOKUP = dict(zip(WEIGHTED_DF["title"], WEIGHTED_DF["weighted_rating"]))
VOTE_COUNT_LOOKUP = dict(zip(WEIGHTED_DF["title"], WEIGHTED_DF["vote_count"]))

# Collaborative filtering model is optional -- app still works content-only
# if it hasn't been trained yet (run model/collaborative.py to enable it).
cf_model = None
if os.path.exists(CF_MODEL_PATH):
    from collaborative import CollaborativeRecommender  # noqa: E402
    cf_model = CollaborativeRecommender.load(CF_MODEL_PATH)
    ALL_USER_IDS = sorted(cf_model.user_item_matrix.index.tolist())
else:
    ALL_USER_IDS = []


def _refresh_weighted_ratings():
    """Recompute weighted ratings after data/ratings.csv changes (new user ratings)."""
    global WEIGHTED_DF, WEIGHTED_LOOKUP, VOTE_COUNT_LOOKUP
    WEIGHTED_DF = compute_weighted_ratings(model.df, ratings_csv=RATINGS_CSV_PATH)
    WEIGHTED_LOOKUP = dict(zip(WEIGHTED_DF["title"], WEIGHTED_DF["weighted_rating"]))
    VOTE_COUNT_LOOKUP = dict(zip(WEIGHTED_DF["title"], WEIGHTED_DF["vote_count"]))


def _retrain_cf_model():
    """Reload data/ratings.csv from disk and retrain the collaborative model.
    Called after a visitor submits a new star rating, so 'For You' and
    Hybrid recommendations reflect their feedback immediately."""
    global cf_model, ALL_USER_IDS
    from collaborative import CollaborativeRecommender  # noqa: E402
    ratings_df = pd.read_csv(RATINGS_CSV_PATH)
    movies_df = pd.read_csv(MOVIES_CSV_PATH)
    cf_model = CollaborativeRecommender().fit(ratings_df, movies_df)
    cf_model.save(CF_MODEL_PATH)
    ALL_USER_IDS = sorted(cf_model.user_item_matrix.index.tolist())
    _refresh_weighted_ratings()


def _get_or_create_guest_user_id():
    """Each browser session gets its own persistent synthetic user_id so
    star ratings submitted across the visit all accumulate onto the same
    'user', letting the CF model build up a real preference profile for
    them (used to power their personalized 'For You' picks)."""
    if "cf_user_id" not in session:
        existing = ALL_USER_IDS if ALL_USER_IDS else [0]
        session["cf_user_id"] = int(max(existing)) + 1
    return session["cf_user_id"]


def _get_watchlist():
    return session.get("watchlist", [])


def _movie_row_to_dict(row):
    return {
        "movie_id": int(row["movie_id"]),
        "title": row["title"],
        "genres": row["genres"],
        "release_year": int(row["release_year"]),
        "rating": float(row["rating"]),
        "director": row["director"],
        "cast": row["cast"],
        "overview": row["overview"],
        "weighted_rating": WEIGHTED_LOOKUP.get(row["title"]),
        "vote_count": VOTE_COUNT_LOOKUP.get(row["title"], 0),
    }


@app.route("/")
def index():
    return render_template("index.html", titles=ALL_TITLES)


@app.route("/recommend", methods=["POST"])
def recommend_form():
    title = request.form.get("title", "").strip()
    if not title:
        return redirect(url_for("index"))
    return redirect(url_for("recommend_results", title=title))


@app.route("/recommend/<title>")
def recommend_results(title):
    top_n = request.args.get("top_n", default=10, type=int)
    method = request.args.get("method", default="content")
    if method != "content" or cf_model is None:
        method = "content"  # fall back gracefully if CF model isn't trained

    matched_title, recommendations = model.recommend(title, top_n=top_n)

    if matched_title is None:
        return render_template(
            "result.html",
            found=False,
            query=title,
            titles=ALL_TITLES,
            cf_available=cf_model is not None,
            method=method,
        )

    source_row = model.df[model.df["title"] == matched_title].iloc[0]
    source_movie = {
        "title": source_row["title"],
        "genres": source_row["genres"],
        "release_year": int(source_row["release_year"]),
        "rating": float(source_row["rating"]),
        "director": source_row["director"],
        "cast": source_row["cast"],
        "overview": source_row["overview"],
    }

    return render_template(
        "result.html",
        found=True,
        query=title,
        source_movie=source_movie,
        recommendations=recommendations,
        titles=ALL_TITLES,
        cf_available=cf_model is not None,
        method=method,
    )


@app.route("/movie/<title>")
def movie_detail(title):
    row = model.df[model.df["title"] == title]
    if row.empty:
        matched = model.find_closest_title(title)
        if matched is None:
            return render_template("result.html", found=False, query=title, titles=ALL_TITLES,
                                    cf_available=cf_model is not None, method="content")
        return redirect(url_for("movie_detail", title=matched))

    movie = _movie_row_to_dict(row.iloc[0])
    _, recommendations = model.recommend(title, top_n=6)
    watchlist = _get_watchlist()

    my_rating = None
    if cf_model is not None and "cf_user_id" in session:
        user_id = session["cf_user_id"]
        if user_id in cf_model.user_item_matrix.index and movie["movie_id"] in cf_model.movie_ids:
            val = cf_model.user_item_matrix.loc[user_id, movie["movie_id"]]
            my_rating = float(val) if val and val > 0 else None

    return render_template(
        "movie_detail.html",
        movie=movie,
        recommendations=recommendations,
        titles=ALL_TITLES,
        in_watchlist=movie["title"] in watchlist,
        watchlist_count=len(watchlist),
        my_rating=my_rating,
    )


@app.route("/genres")
def genres_index():
    counts = {}
    for row in model.df["genres"]:
        for g in str(row).split():
            counts[g] = counts.get(g, 0) + 1
    genre_counts = sorted(counts.items(), key=lambda x: -x[1])
    return render_template("genres_index.html", genre_counts=genre_counts, titles=ALL_TITLES)


@app.route("/genre/<genre>")
def genre_browse(genre):
    sort_by = request.args.get("sort", default="rating")
    matches = model.df[model.df["genres"].str.contains(genre, case=False, na=False, regex=False)]
    if sort_by == "year":
        matches = matches.sort_values("release_year", ascending=False)
    elif sort_by == "weighted":
        matches = matches.assign(_wr=matches["title"].map(WEIGHTED_LOOKUP)).sort_values("_wr", ascending=False)
    else:
        matches = matches.sort_values("rating", ascending=False)

    movies = [_movie_row_to_dict(row) for _, row in matches.iterrows()]
    return render_template(
        "genre_browse.html",
        genre=genre,
        movies=movies,
        all_genres=ALL_GENRES,
        sort_by=sort_by,
        titles=ALL_TITLES,
    )


@app.route("/popular")
def popular():
    """IMDB-style weighted popularity ranking -- pulls low-vote-count movies
    back toward the catalogue average so a single 5-star vote can't outrank
    a movie with hundreds of consistently-good ratings."""
    ranked = WEIGHTED_DF.sort_values("weighted_rating", ascending=False)
    movies = []
    for _, row in ranked.iterrows():
        d = _movie_row_to_dict(row)
        movies.append(d)
    return render_template("popular.html", movies=movies, titles=ALL_TITLES)


@app.route("/discover")
def discover():
    """Advanced filter/browse page: genre + release-year range + minimum
    rating, all combinable, with sortable results."""
    selected_genres = request.args.getlist("genre")
    year_min = request.args.get("year_min", type=int)
    year_max = request.args.get("year_max", type=int)
    rating_min = request.args.get("rating_min", type=float)
    sort_by = request.args.get("sort", default="rating")

    df_year_min = int(model.df["release_year"].min())
    df_year_max = int(model.df["release_year"].max())

    matches = model.df.copy()
    # require ALL selected genres to be present, not just any one
    for g in selected_genres:
        matches = matches[matches["genres"].str.contains(rf"\b{g}\b", case=False, na=False, regex=True)]
    if year_min is not None:
        matches = matches[matches["release_year"] >= year_min]
    if year_max is not None:
        matches = matches[matches["release_year"] <= year_max]
    if rating_min is not None:
        matches = matches[matches["rating"] >= rating_min]

    if sort_by == "year":
        matches = matches.sort_values("release_year", ascending=False)
    elif sort_by == "weighted":
        matches = matches.assign(_wr=matches["title"].map(WEIGHTED_LOOKUP)).sort_values("_wr", ascending=False)
    else:
        matches = matches.sort_values("rating", ascending=False)

    movies = [_movie_row_to_dict(row) for _, row in matches.iterrows()]

    return render_template(
        "discover.html",
        movies=movies,
        titles=ALL_TITLES,
        all_genres=ALL_GENRES,
        selected_genres=selected_genres,
        year_min=year_min if year_min is not None else df_year_min,
        year_max=year_max if year_max is not None else df_year_max,
        df_year_min=df_year_min,
        df_year_max=df_year_max,
        rating_min=rating_min if rating_min is not None else 0,
        sort_by=sort_by,
    )


@app.route("/surprise")
def surprise_me():
    title = random.choice(ALL_TITLES)
    return redirect(url_for("movie_detail", title=title))


@app.route("/watchlist")
def watchlist_page():
    watchlist_titles = _get_watchlist()
    movies = []
    for t in watchlist_titles:
        row = model.df[model.df["title"] == t]
        if not row.empty:
            movies.append(_movie_row_to_dict(row.iloc[0]))
    return render_template("watchlist.html", movies=movies, titles=ALL_TITLES)


@app.route("/watchlist/toggle/<title>", methods=["POST"])
def watchlist_toggle(title):
    watchlist = _get_watchlist()
    if title in watchlist:
        watchlist.remove(title)
    else:
        watchlist.append(title)
    session["watchlist"] = watchlist
    session.modified = True

    next_url = request.form.get("next") or url_for("index")
    return redirect(next_url)


@app.route("/rate/<title>", methods=["POST"])
def rate_movie(title):
    """Visitor submits a 1-5 star rating for a movie. It's appended to
    data/ratings.csv under this browser session's synthetic user_id and
    the collaborative filtering model is retrained on the spot, so the
    rating immediately feeds into Hybrid scores and the 'For You' page."""
    rating_value = request.form.get("rating", type=float)
    next_url = request.form.get("next") or url_for("movie_detail", title=title)

    row = model.df[model.df["title"] == title]
    if row.empty or rating_value is None or not (1.0 <= rating_value <= 5.0):
        return redirect(next_url)

    movie_id = int(row.iloc[0]["movie_id"])
    user_id = _get_or_create_guest_user_id()

    file_exists = os.path.exists(RATINGS_CSV_PATH)
    with open(RATINGS_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user_id", "movie_id", "rating"])
        writer.writerow([user_id, movie_id, rating_value])

    _retrain_cf_model()
    return redirect(next_url)


@app.route("/analytics")
def analytics():
    chart_dir = os.path.join(BASE_DIR, "static", "images", "eda")
    charts = sorted(os.listdir(chart_dir)) if os.path.isdir(chart_dir) else []
    stats = {
        "total_movies": len(model.df),
        "total_genres": len(set(g for row in model.df["genres"] for g in str(row).split())),
        "avg_rating": round(float(model.df["rating"].mean()), 2),
        "year_range": f"{int(model.df['release_year'].min())}-{int(model.df['release_year'].max())}",
    }
    if cf_model is not None:
        stats["total_users"] = len(ALL_USER_IDS)
        stats["total_ratings"] = int((cf_model.user_item_matrix > 0).sum().sum())
    return render_template("analytics.html", charts=charts, stats=stats)




@app.route("/for-you")
def for_you():
    if cf_model is None:
        return redirect(url_for("index"))
    user_id = request.args.get("user_id", type=int)
    is_me = False
    if user_id is None and "cf_user_id" in session and session["cf_user_id"] in ALL_USER_IDS:
        user_id = session["cf_user_id"]
        is_me = True
    recommendations, history = [], []
    if user_id is not None:
        recommendations = cf_model.recommend_for_user(user_id, top_n=10)
        history = cf_model.get_user_history(user_id)
    return render_template(
        "for_you.html",
        user_ids=ALL_USER_IDS,
        selected_user=user_id,
        recommendations=recommendations,
        history=history,
        my_user_id=session.get("cf_user_id"),
        is_me=is_me,
    )


@app.route("/api/recommend")
def api_recommend():
    title = request.args.get("title", "").strip()
    top_n = request.args.get("top_n", default=10, type=int)
    if not title:
        return jsonify({"error": "Missing 'title' query parameter"}), 400

    matched_title, recommendations = model.recommend(title, top_n=top_n)
    if matched_title is None:
        return jsonify({"error": f"No close match found for '{title}'"}), 404

    return jsonify({
        "query": title,
        "matched_title": matched_title,
        "count": len(recommendations),
        "recommendations": recommendations,
    })




@app.route("/api/recommend/for-user")
def api_recommend_for_user():
    if cf_model is None:
        return jsonify({"error": "Collaborative model not trained. Run model/collaborative.py first."}), 503
    user_id = request.args.get("user_id", type=int)
    top_n = request.args.get("top_n", default=10, type=int)
    if user_id is None:
        return jsonify({"error": "Missing 'user_id' query parameter"}), 400
    if user_id not in ALL_USER_IDS:
        return jsonify({"error": f"Unknown user_id {user_id}"}), 404

    recommendations = cf_model.recommend_for_user(user_id, top_n=top_n)
    return jsonify({
        "user_id": user_id,
        "count": len(recommendations),
        "recommendations": recommendations,
    })


@app.route("/api/search")
def api_search():
    """Simple autocomplete: substring match against known titles."""
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    matches = [t for t in ALL_TITLES if q in t.lower()][:10]
    return jsonify(matches)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
