"""Shared constants: earth radius, default grid parameters, license text."""

from __future__ import annotations

EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius

DEFAULT_RADII_KM: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0)

# The bundled grid's own parameters -- this package's query-side defaults,
# deliberately coarse (small enough to ship in a wheel). A caller wanting a
# finer grid passes resolution_deg/min_island_area_km2 explicitly and lets
# it be fetched, or bundles their own via LANDMETRICS_DATA_DIR.
DEFAULT_RESOLUTION_DEG = 0.1
DEFAULT_MIN_ISLAND_AREA_KM2 = 0.0

DATA_LICENSE = (
    "Derived from GSHHG (Wessel & Smith 1996), which is released under the "
    "GNU Lesser General Public License. This derived dataset is redistributed "
    "under LGPL-3.0-or-later. See DATA_LICENSE and docs/grid_format.md for the "
    "full derivation."
)

GSHHG_REFERENCE = (
    "Wessel, P., and W. H. F. Smith (1996), A global, self-consistent, "
    "hierarchical, high-resolution shoreline database, J. Geophys. Res., "
    "101(B4), 8741-8743, doi:10.1029/96JB00104. "
    "https://www.soest.hawaii.edu/pwessel/gshhg/"
)
