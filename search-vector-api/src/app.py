"""Application initialization module.

This module handles the initialization and configuration of the Flask application
for the Vector Search API. It's responsible for registering blueprints,
loading configuration settings, and initializing the structured configuration
objects that will be used throughout the application.
"""

import logging
import os

from flask import Flask
from psycopg_pool import ConnectionPool

from utils.config import get_named_config, VectorSettings, SearchSettings, ModelSettings
from utils.version import get_version

LOGGER = logging.getLogger(__name__)


def create_app(run_mode=os.getenv("FLASK_ENV", "development")):
    """Create and configure the Flask application.

    Initializes a Flask application instance with the appropriate configuration
    based on the specified run mode. Registers all necessary blueprints and
    sets up configuration objects that provide structured access to settings.

    Args:
        run_mode (str): The environment to run the application in.
                       Options: 'development', 'testing', 'production', 'docker'
                       Defaults to the FLASK_ENV environment variable or 'development'.

    Returns:
        Flask: A configured Flask application instance ready to run.
    """
    # pylint: disable=import-outside-toplevel
    from resources import (
        API_BLUEPRINT,
        HEALTH_BLUEPRINT
    )

    # Flask app initialize
    app = Flask(__name__)

    version = get_version()
    LOGGER.info("Starting Vector Search API - version %s (mode=%s)", version, run_mode)
    print(f"Starting Vector Search API - version {version} (mode={run_mode})")

    # Register blueprints
    app.register_blueprint(API_BLUEPRINT)
    app.register_blueprint(HEALTH_BLUEPRINT)

    # All configuration are in config file
    app.config.from_object(get_named_config(run_mode))

    # Initialize structured configuration objects
    app.vector_settings = VectorSettings(app.config)
    app.search_settings = SearchSettings(app.config)
    app.model_settings = ModelSettings(app.config)

    # Initialize PostgreSQL connection pool (reused across requests, eliminating per-query TCP overhead)
    app.db_pool = ConnectionPool(
        conninfo=app.vector_settings.database_url,
        min_size=2,
        max_size=10,
        open=True,
    )
    LOGGER.info("PostgreSQL connection pool initialized (min=2, max=10)")

    # Pre-load ML models so the first request does not pay the cold-start cost (~500-800ms)
    with app.app_context():
        _preload_models()

    return app


def _preload_models():
    """Warm up embedding and cross-encoder models inside an app context."""
    from services.embedding import preload_embedding_model
    from services.re_ranker import get_cross_encoder

    try:
        preload_embedding_model()
        LOGGER.info("Embedding model pre-loaded")
    except Exception as exc:
        LOGGER.warning("Failed to pre-load embedding model: %s", exc)

    try:
        get_cross_encoder()
        LOGGER.info("Cross-encoder model pre-loaded")
    except Exception as exc:
        LOGGER.warning("Failed to pre-load cross-encoder model: %s", exc)

    # Project embeddings are built lazily on first /match-projects call.
    # Each worker loads from disk if a sibling already computed them (~0.2s),
    # otherwise computes fresh (~10-20s for 358 short descriptions on CPU).
