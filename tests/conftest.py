"""Synthetic grid fixtures written as real netCDF files, matching the
production grid schema (see ``docs/grid_format.md``): ``lat`` ascending
-90..90 inclusive, ``lon`` ascending half-open [-180, 180) with no
duplicated seam column, packed int16 with ``scale_factor``/``add_offset``
for the distance/fraction grids.

Tests read these files back independently with plain ``netCDF4`` and
hand-compute expected values, rather than re-exercising landmetrics' own
bracketing/reader code -- so a bug shared between the production code and
a test wouldn't silently cancel out.
"""

from __future__ import annotations

import netCDF4 as nc
import numpy as np
import pytest

KM_PER_DEG = 111.32


def _distance_axes(resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    n_lat = int(round(180.0 / resolution_deg)) + 1
    n_lon = int(round(360.0 / resolution_deg))
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon = -180.0 + resolution_deg * np.arange(n_lon)
    return lat.astype("float64"), lon.astype("float64")


def _write_packed_grid(path, lat, lon, var_name, values, scale_factor, extra_dims=()):
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        dims = ["lat", "lon"]
        for name, coord in extra_dims:
            ds.createDimension(name, coord.size)
            dims.append(name)
            coord_var = ds.createVariable(name, "f4", (name,))
            coord_var.set_auto_maskandscale(False)
            coord_var[:] = coord.astype("float32")

        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lat_var.set_auto_maskandscale(False)
        lon_var.set_auto_maskandscale(False)
        lat_var[:] = lat.astype("float32")
        lon_var[:] = lon.astype("float32")

        var = ds.createVariable(var_name, "i2", tuple(dims), fill_value=None)
        var.scale_factor = scale_factor
        var.add_offset = 0.0
        var.set_auto_maskandscale(False)
        raw = np.round(values / scale_factor).astype("int16")
        var[...] = raw


def _write_float_grid(path, lat, lon, var_name, values):
    """Unpacked float storage (no scale_factor/add_offset) -- used only
    for index_grid, where exact index recovery matters more than
    exercising the packed-integer decode path (covered elsewhere)."""
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("lat", lat.size)
        ds.createDimension("lon", lon.size)
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lat_var.set_auto_maskandscale(False)
        lon_var.set_auto_maskandscale(False)
        lat_var[:] = lat.astype("float32")
        lon_var[:] = lon.astype("float32")
        var = ds.createVariable(var_name, "f8", ("lat", "lon"))
        var[:, :] = values


# Rectangle bounds deliberately inset 0.1 degree from the round grid lines
# (10/20), not aligned to them: this makes the exact-grid-point rows 10
# and 20 sit *comfortably inside* the island with a small inset distance,
# while the neighboring rows 9 and 21 sit clearly outside with a much
# larger distance. That size asymmetry is what makes it possible for a
# query point whose *nearest* cell is land to bilinearly interpolate to a
# positive (ocean) value -- the scenario is_land is specifically designed
# to avoid (see test_is_land.py).
_ISLAND_LAT_MIN, _ISLAND_LAT_MAX = 9.9, 20.1
_ISLAND_LON_MIN, _ISLAND_LON_MAX = 9.9, 20.1


def _rectangle_signed_distance_km(lat2d, lon2d, lat_min, lat_max, lon_min, lon_max):
    cos_lat = np.cos(np.radians(np.clip(lat2d, -89.9, 89.9)))
    dy_out = np.maximum(0.0, np.maximum(lat_min - lat2d, lat2d - lat_max))
    dx_out = np.maximum(0.0, np.maximum(lon_min - lon2d, lon2d - lon_max))
    outside_dist = np.sqrt((dy_out * KM_PER_DEG) ** 2 + (dx_out * KM_PER_DEG * cos_lat) ** 2)

    inside = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    in_dy = np.minimum(lat2d - lat_min, lat_max - lat2d) * KM_PER_DEG
    in_dx = np.minimum(lon2d - lon_min, lon_max - lon2d) * KM_PER_DEG * cos_lat
    inside_dist = -np.minimum(in_dy, in_dx)

    return np.where(inside, inside_dist, outside_dist)


@pytest.fixture
def tiny_distance_grid(tmp_path):
    """1.0 degree global distance_to_land grid (181 x 360) with a single
    rectangular island at lat/lon 9.9-20.1 (see the inset-bounds comment
    above), int16-packed with scale_factor=0.1."""
    lat, lon = _distance_axes(1.0)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    values = _rectangle_signed_distance_km(
        lat2d,
        lon2d,
        _ISLAND_LAT_MIN,
        _ISLAND_LAT_MAX,
        _ISLAND_LON_MIN,
        _ISLAND_LON_MAX,
    )
    path = tmp_path / "tiny_distance.nc"
    _write_packed_grid(path, lat, lon, "distance_to_land", values, scale_factor=0.1)
    return path


@pytest.fixture
def dateline_distance_grid(tmp_path):
    """1.0 degree global distance_to_land grid with a rectangular island
    straddling the antimeridian: lat in [-5, 5], within 5 degrees of the
    seam (lon = +/-180) on either side. ``d_seam = 180 - abs(lon)`` is 0
    exactly at the seam and grows moving away from it on both sides --
    computed with plain abs(), no wraparound arithmetic needed, since the
    island is deliberately centered on the seam itself."""
    lat, lon = _distance_axes(1.0)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    d_seam = 180.0 - np.abs(lon2d)
    cos_lat = np.cos(np.radians(np.clip(lat2d, -89.9, 89.9)))

    dy_out = np.maximum(0.0, np.abs(lat2d) - 5.0)
    dx_out = np.maximum(0.0, d_seam - 5.0)
    outside_dist = np.sqrt((dy_out * KM_PER_DEG) ** 2 + (dx_out * KM_PER_DEG * cos_lat) ** 2)

    inside = (np.abs(lat2d) <= 5.0) & (d_seam <= 5.0)
    in_dy = (5.0 - np.abs(lat2d)) * KM_PER_DEG
    in_dx = (5.0 - d_seam) * KM_PER_DEG * cos_lat
    inside_dist = -np.minimum(in_dy, in_dx)

    values = np.where(inside, inside_dist, outside_dist)
    path = tmp_path / "dateline_distance.nc"
    _write_packed_grid(path, lat, lon, "distance_to_land", values, scale_factor=0.1)
    return path


@pytest.fixture
def index_grid(tmp_path):
    """1.0 degree global distance_to_land grid whose stored value at row
    *i*, column *j* is exactly ``i * 10000 + j`` (unpacked float64, no
    quantization) -- for unambiguous index/bracketing assertions,
    including across the antimeridian seam (where j0 = n_lon - 1,
    j1 = 0)."""
    lat, lon = _distance_axes(1.0)
    i2d, j2d = np.meshgrid(np.arange(lat.size), np.arange(lon.size), indexing="ij")
    values = (i2d * 10000 + j2d).astype("float64")
    path = tmp_path / "index_grid.nc"
    _write_float_grid(path, lat, lon, "distance_to_land", values)
    return path


@pytest.fixture
def tiny_fraction_grid(tmp_path):
    """1.0 degree global land_fraction grid, radius axis (100, 300, 600)
    km. Value is deliberately independent of longitude and linear in both
    latitude and radius (``f(lat) * g(radius)``), so trilinear
    interpolation reduces to plain arithmetic a test can reproduce by
    hand: ``f(lat) = 0.3 + 0.5 * (lat + 90) / 180`` (range 0.3-0.8),
    ``g(radius) = radius / 600`` (exact passthrough at every stored
    radius, since g is linear)."""
    lat, lon = _distance_axes(1.0)
    radius = np.array([100.0, 300.0, 600.0])
    f_lat = 0.3 + 0.5 * (lat + 90.0) / 180.0
    g_radius = radius / 600.0
    values = f_lat[:, None, None] * g_radius[None, None, :] * np.ones((lat.size, lon.size, radius.size))
    path = tmp_path / "tiny_fraction.nc"
    _write_packed_grid(path, lat, lon, "land_fraction", values, scale_factor=0.001, extra_dims=[("radius", radius)])
    return path
