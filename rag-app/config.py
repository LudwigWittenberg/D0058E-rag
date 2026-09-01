"""
Configuration for the RAG app.

Instantiates a ConfigManager pointed at this application's root directory.
"""

import sys
from pathlib import Path

# Ensure shared module is importable
APP_ROOT = Path(__file__).resolve().parent
SHARED_DIR = APP_ROOT.parent / "shared"
if str(SHARED_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR.parent))

from shared.config_manager import ConfigManager

# Application configuration instance
config = ConfigManager(APP_ROOT)
