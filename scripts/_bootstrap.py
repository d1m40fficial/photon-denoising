"""Shared paths for the scripts in this folder.

The repository is not installed as a package, so scripts add ``src`` to
``sys.path`` themselves.  Import this module first::

    from _bootstrap import ASSETS, DATA, RESULTS  # noqa: F401
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data" / "samples"
LEGACY = ROOT / "data" / "legacy"
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
