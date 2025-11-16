"""Pytest configuration file.

This file configures pytest to properly find the source modules.
"""

import sys
from pathlib import Path

# Add src directory to path for pytest
src_path = (Path(__file__).parent.parent / "src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
