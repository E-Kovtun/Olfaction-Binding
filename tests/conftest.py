"""Put the repo root on sys.path, the same way every script and notebook does.

`pyproject.toml` sets `package = false` -- this is an application, not an
installed distribution -- so there is no `pip install -e .` to lean on.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
