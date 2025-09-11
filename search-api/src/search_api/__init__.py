"""The App Initiation file.

This module is for the initiation of the flask app.
"""

import os

from http import HTTPStatus
import secure
from flask import Flask, current_app, g, request
from flask_cors import CORS
from werkzeug.exceptions import NotFound

from search_api.auth import jwt
from search_api.config import get_named_config
from search_api.utils.cache import cache
from search_api.utils.util import allowedorigins

# Security Response headers
csp = (
    secure.ContentSecurityPolicy()
    .default_src("'self'")
    .script_src("'self' 'unsafe-inline'")
    .style_src("'self' 'unsafe-inline'")
    .img_src("'self' data:")
    .object_src("'self'")
    .connect_src("'self'")
)

hsts = secure.StrictTransportSecurity().include_subdomains().preload().max_age(31536000)
referrer = secure.ReferrerPolicy().no_referrer()
cache_value = secure.CacheControl().no_store().max_age(0)
xfo_value = secure.XFrameOptions().deny()
secure_headers = secure.Secure(
    csp=csp, hsts=hsts, referrer=referrer, cache=cache_value, xfo=xfo_value
)


def create_app(run_mode=os.getenv("FLASK_ENV", "development")):
    """Create flask app."""
    # pylint: disable=import-outside-toplevel
    from search_api.resources import (
        API_BLUEPRINT,
        HEALTH_BLUEPRINT        
    ) 

    # Flask app initialize
    app = Flask(__name__)

    # All configuration are in config file
    app.config.from_object(get_named_config(run_mode))

    CORS(app, resources={r"/*": {"origins": allowedorigins()}}, supports_credentials=True)

    # Register blueprints
    app.register_blueprint(API_BLUEPRINT)

    app.register_blueprint(HEALTH_BLUEPRINT)

    # Setup jwt for keycloak
    #if os.getenv("FLASK_ENV", "production") != "testing":
    #    setup_jwt_manager(app, jwt)

    @app.before_request
    def log_request_info():
        """Log request information for debugging."""
        current_app.logger.info("=== Incoming Request ===")
        current_app.logger.info(f"Method: {request.method}")
        current_app.logger.info(f"URL: {request.url}")
        current_app.logger.info(f"Path: {request.path}")
        current_app.logger.info(f"Headers: {dict(request.headers)}")
        if request.args:
            current_app.logger.info(f"Query params: {dict(request.args)}")
        current_app.logger.info("=== End Request Info ===")

    @app.before_request
    def set_origin():
        g.origin_url = request.environ.get("HTTP_ORIGIN", "localhost")

    build_cache(app)

    @app.after_request
    def log_response_info(response):
        """Log response information for debugging."""
        current_app.logger.info("=== Outgoing Response ===")
        current_app.logger.info(f"Status: {response.status}")
        current_app.logger.info(f"Headers: {dict(response.headers)}")
        if hasattr(response, 'content_length') and response.content_length:
            current_app.logger.info(f"Content Length: {response.content_length}")
        current_app.logger.info("=== End Response Info ===")
        return response

    @app.after_request
    def set_secure_headers(response):
        """Set CORS headers for security."""
       # secure_headers.framework.flask(response)
        response.headers.add("Cross-Origin-Resource-Policy", "*")
        response.headers["Cross-Origin-Opener-Policy"] = "*"
        response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"
        return response

    @app.errorhandler(Exception)
    def handle_error(err):
        # Handle 404 errors gracefully in all environments
        if isinstance(err, NotFound):
            current_app.logger.warning(f"404 Not Found: {request.url}")
            return {"error": "Not Found", "message": f"The requested URL {request.path} was not found on the server."}, HTTPStatus.NOT_FOUND
        
        if run_mode != "production":
            # To get stacktrace in local development for internal server errors
            raise err
        current_app.logger.error(str(err))
        return "Internal server error", HTTPStatus.INTERNAL_SERVER_ERROR

    # Return App for run in run.py file
    return app


def build_cache(app):
    """Build cache."""
    cache.init_app(app)


def setup_jwt_manager(app_context, jwt_manager):
    """Use flask app to configure the JWTManager to work for a particular Realm."""

    def get_roles(a_dict):
        return a_dict["realm_access"]["roles"]  # pragma: no cover

    app_context.config["JWT_ROLE_CALLBACK"] = get_roles
    jwt_manager.init_app(app_context)
