"""
Expose the FastAPI instance for uvicorn app.main:app.
"""

from main_api import app  # noqa: F401

__all__ = ["app"]

