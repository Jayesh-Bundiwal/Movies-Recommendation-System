# Movie Recommendation System

A compact Flask app that recommends movies using content-based techniques (TF-IDF + cosine similarity) and an optional collaborative filtering component.

Quick start

1. Install dependencies:

   pip install -r requirements.txt

2. Train content-based model (creates similarity and movie files):

   python model/recommender.py

3. (Optional) Generate synthetic ratings and train collaborative model:

   python model/generate_ratings.py
   python model/collaborative.py

4. Run the app:

   python app.py

5. Open in your browser:

   http://127.0.0.1:5000

What it does

- Search movies and get similar recommendations
- Optional collaborative mode: personalized recommendations using collaborative filtering
- Simple session-based watchlist and movie detail pages

Where to look

- model/ — training and recommendation scripts
- templates/ and static/ — frontend views and assets
- app.py — Flask routes and API

Tech

Python, Flask, scikit-learn, Pandas, Bootstrap

Project Details

Purpose

This project provides a small, easy-to-run movie recommendation system demonstrating content-based filtering (TF-IDF + cosine similarity) and an optional collaborative filtering pipeline. It's intended as a learning/demo application and a starting point for experimentation.

Architecture

- model/: data preprocessing, model training, and recommendation logic
- app.py: Flask app with web UI and JSON API endpoints
- templates/ and static/: frontend views and assets

Data

- Movie metadata is used to build content vectors (title, genres, description). The repo includes scripts to generate synthetic ratings for the collaborative component.

API Endpoints

- /api/recommend — content-based recommendations
- /api/recommend/for-user — user-specific recommendations (requires ratings)

Running & Development

Follow the Quick start above to install dependencies, train models, and run the app. For experiments, adjust scripts in model/ and re-run training steps.

Notes

The app runs fine in content-based-only mode if you skip the collaborative steps. See model/ for training scripts.
