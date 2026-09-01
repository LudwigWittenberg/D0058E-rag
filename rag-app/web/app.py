"""
Flask application factory for the RAG app.

Creates and configures the Flask application instance,
registers blueprints, and sets template/static/upload folder paths.
"""

import os
import sys
from pathlib import Path
from flask import Flask

# Ensure shared module is importable
_app_root = Path(__file__).resolve().parent.parent
_project_root = _app_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask application instance.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.path.join(base_dir, "templates")
    static_folder = os.path.join(base_dir, "static")

    app_root = os.path.dirname(base_dir)
    upload_folder = os.path.join(app_root, "data")

    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder,
    )
    app.secret_key = "rag-app-dev-key"
    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

    os.makedirs(upload_folder, exist_ok=True)

    # Register the main routes blueprint
    from web.routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
