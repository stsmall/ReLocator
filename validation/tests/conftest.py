"""Conftest for validation tests — adds project root to sys.path so
`from validation import …` works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
