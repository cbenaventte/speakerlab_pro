import sys
import os
from pathlib import Path

# Add the project root to sys.path so 'backend' is importable
repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Import the FastAPI app from the backend module
from backend.main import app
