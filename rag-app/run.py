"""
Entry point for the RAG application.

Starts the Flask development server on port 8002.
"""

import sys
from pathlib import Path

# Add parent directory to sys.path so that 'shared' is importable
APP_ROOT = Path(__file__).resolve().parent
PARENT_DIR = APP_ROOT.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Add app root so that 'web' and 'ai' packages are importable
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from web.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)
