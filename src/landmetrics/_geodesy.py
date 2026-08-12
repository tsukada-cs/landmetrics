"""Great-circle geometry on unit-sphere ECEF coordinates.

Nearest-neighbor search for great-circle distance is exact when done in
ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates on the unit
sphere, because chord length is monotonic with central angle -- unlike a
naive lat/lon-degree distance, which is wrong away from the equator.
"""

from __future__ import annotations

import numpy as np

from ._constants import EARTH_RADIUS_KM

__all__ = ["lonlat_to_ecef", "chord_to_km", "km_to_chord"]


def lonlat_to_ecef(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """(..., 3) unit-sphere ECEF Cartesian coordinates for (lon_deg, lat_deg)."""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    """Great-circle distance (km) from a unit-sphere chord length -- exact
    for points on a unit sphere (chord = 2*sin(central_angle/2))."""
    chord = np.clip(chord, 0.0, 2.0)
    central_angle = 2.0 * np.arcsin(chord / 2.0)
    return EARTH_RADIUS_KM * central_angle


def km_to_chord(km: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`chord_to_km` -- unit-sphere chord length for a
    given great-circle distance in km, for use as a nearest-neighbor query
    radius (chord = 2*sin(central_angle/2))."""
    central_angle = np.asarray(km, dtype="float64") / EARTH_RADIUS_KM
    return 2.0 * np.sin(central_angle / 2.0)
