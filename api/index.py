"""Vercel entrypoint.

Vercel treats a top-level `app` in this file as the ASGI application and routes
every request to it, so FastAPI does its own routing from `/` down. `src/` is not
installed as a package in the build, so it goes on the path here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sepsis_icu.serve import app  # noqa: E402

__all__ = ["app"]
