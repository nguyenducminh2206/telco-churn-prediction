"""
Pytest config: add the project root to sys.path so `from src.x.y import z` works
without installing the project as a package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
