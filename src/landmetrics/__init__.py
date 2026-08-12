"""Distance to land, land fraction within radius, and land/ocean tests
from precomputed GSHHG-coastline grids.

Every query reads only the handful of cells its interpolation needs
directly out of the backing netCDF file -- never the whole grid -- so
memory use stays flat regardless of grid resolution. See
:class:`DistanceToLand` and :class:`LandFraction` for the thread-safety
note (each holds an open netCDF handle and is not shareable across
threads).
"""

from __future__ import annotations

from ._constants import DEFAULT_MIN_ISLAND_AREA_KM2, DEFAULT_RADII_KM, DEFAULT_RESOLUTION_DEG, EARTH_RADIUS_KM
from .data import GridSpec, available_grids, cache_dir, grid_filename, grid_path
from .exceptions import GridFormatError, GridNotFoundError
from .query import (
    DistanceToLand,
    LandFraction,
    distance_to_land,
    is_land,
    is_ocean,
    land_fraction,
    open_distance_to_land,
    open_land_fraction,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DistanceToLand",
    "LandFraction",
    "distance_to_land",
    "land_fraction",
    "is_land",
    "is_ocean",
    "open_distance_to_land",
    "open_land_fraction",
    "available_grids",
    "grid_path",
    "grid_filename",
    "cache_dir",
    "GridSpec",
    "GridNotFoundError",
    "GridFormatError",
    "DEFAULT_RADII_KM",
    "DEFAULT_RESOLUTION_DEG",
    "DEFAULT_MIN_ISLAND_AREA_KM2",
    "EARTH_RADIUS_KM",
]
