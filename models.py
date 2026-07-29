"""
Database models for the movie recommendation system.

This project no longer includes user authentication. The User and MovieRating
models have been removed so the app runs without a login system.
"""

from flask_sqlalchemy import SQLAlchemy

# Keep a SQLAlchemy instance for any remaining models (non-auth related)
db = SQLAlchemy()

