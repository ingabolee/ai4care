"""
conftest.py -- pytest configuration for AI4CARE test suite.

Ensures the repository root is on sys.path so all modules are importable
without installing the package.
"""

import sys
from pathlib import Path

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
