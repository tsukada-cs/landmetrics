"""Public exception types."""

from __future__ import annotations

__all__ = ["GridNotFoundError", "GridFormatError"]


class GridNotFoundError(FileNotFoundError):
    """Raised when a requested grid file cannot be resolved locally and
    either downloading is disabled or no download source is configured."""


class GridFormatError(ValueError):
    """Raised when a grid file does not match the schema documented in
    ``docs/grid_format.md`` (missing variable, non-ascending axis, etc.)."""
