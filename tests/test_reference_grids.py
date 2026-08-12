"""Regression tests against the real, full-size reference grids (not the
small synthetic fixtures the rest of the suite uses). These are large,
machine-local files, not checked into this repo -- point
$LANDMETRICS_DATA_DIR at a directory containing them (e.g. the NAS
``reference/`` directory this package's grids were originally generated
into) to run this file; it skips entirely otherwise.

Each test independently re-reads the raw netCDF grid and hand-computes the
expected bilinear/trilinear interpolation, rather than re-exercising this
package's own bracket-finding code -- so a bug shared between the
production code and the test wouldn't silently cancel out.
"""

import netCDF4 as nc
import numpy as np
import pytest

from landmetrics import data
from landmetrics._constants import DEFAULT_RADII_KM
from landmetrics.query import DistanceToLand, LandFraction

try:
    DISTANCE_PATH = data.grid_path("distance_to_land", 0.05, 1400.0, download=False)
    LAND_FRACTION_PATH = data.grid_path("land_fraction", 0.01, 0.0, download=False)
    _GRIDS_AVAILABLE = True
except Exception:
    DISTANCE_PATH = LAND_FRACTION_PATH = None
    _GRIDS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _GRIDS_AVAILABLE,
    reason="reference grids not found; set LANDMETRICS_DATA_DIR to a directory containing them",
)


def _manual_bilinear(path, varname, lat, lon):
    """Independent (not landmetrics-derived) bilinear interpolation
    straight off the netCDF file, for a 2-D (lat, lon) grid."""
    ds = nc.Dataset(path, "r")
    try:
        grid_lat = np.asarray(ds.variables["lat"][:], dtype="float64")
        grid_lon = np.asarray(ds.variables["lon"][:], dtype="float64")
        var = ds.variables[varname]
        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))

        lon_norm = ((lon + 180.0) % 360.0) - 180.0
        i = int(np.searchsorted(grid_lat, lat)) - 1
        i = max(0, min(i, grid_lat.size - 2))
        wl = (lat - grid_lat[i]) / (grid_lat[i + 1] - grid_lat[i])

        n = grid_lon.size
        k = int(np.searchsorted(grid_lon, lon_norm)) - 1
        if k < 0:
            j0, j1 = n - 1, 0
            left, right = grid_lon[-1] - 360.0, grid_lon[0]
        elif k >= n - 1:
            j0, j1 = n - 1, 0
            left, right = grid_lon[-1], grid_lon[0] + 360.0
        else:
            j0, j1 = k, k + 1
            left, right = grid_lon[k], grid_lon[k + 1]
        wo = (lon_norm - left) / (right - left)

        var.set_auto_maskandscale(False)
        v00 = float(var[i, j0]) * scale + offset
        v01 = float(var[i, j1]) * scale + offset
        v10 = float(var[i + 1, j0]) * scale + offset
        v11 = float(var[i + 1, j1]) * scale + offset
        top = v00 * (1 - wo) + v01 * wo
        bot = v10 * (1 - wo) + v11 * wo
        return top * (1 - wl) + bot * wl
    finally:
        ds.close()


TEST_POINTS = [
    (35.0, 139.7),  # Tokyo -- coastline
    (25.8, -80.2),  # Miami -- coastline
    (-33.9, 18.4),  # Cape Town -- coastline
    (21.3, -157.9),  # Oahu -- island
    (0.0, 0.0),  # open ocean
    (64.1, -21.9),  # Reykjavik
]

ANTIMERIDIAN_POINTS = [
    (12.0, 179.99),
    (12.0, -179.99),
    (0.0, 180.0),
    (-5.0, 179.995),
    (5.0, -179.995),
]


@pytest.mark.parametrize("lat,lon", TEST_POINTS + ANTIMERIDIAN_POINTS)
def test_distance_to_land_matches_manual_bilinear(lat, lon):
    expected = _manual_bilinear(DISTANCE_PATH, "distance_to_land", lat, lon)
    with DistanceToLand(DISTANCE_PATH) as dtl:
        actual = dtl.query(lat, lon)
    assert actual == pytest.approx(expected, abs=1e-9)


def _manual_bilinear_3d(path, lat, lon, radius_idx):
    """Same as _manual_bilinear but for the (lat, lon, radius) grid,
    reading a single fixed radius index (exact passthrough, no radius
    interpolation needed -- DEFAULT_RADII_KM matches the file's own
    stored radii exactly, asserted in the test below)."""
    ds = nc.Dataset(path, "r")
    try:
        grid_lat = np.asarray(ds.variables["lat"][:], dtype="float64")
        grid_lon = np.asarray(ds.variables["lon"][:], dtype="float64")
        var = ds.variables["land_fraction"]
        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))

        lon_norm = ((lon + 180.0) % 360.0) - 180.0
        i = int(np.searchsorted(grid_lat, lat)) - 1
        i = max(0, min(i, grid_lat.size - 2))
        wl = (lat - grid_lat[i]) / (grid_lat[i + 1] - grid_lat[i])

        n = grid_lon.size
        k = int(np.searchsorted(grid_lon, lon_norm)) - 1
        if k < 0:
            j0, j1 = n - 1, 0
            left, right = grid_lon[-1] - 360.0, grid_lon[0]
        elif k >= n - 1:
            j0, j1 = n - 1, 0
            left, right = grid_lon[-1], grid_lon[0] + 360.0
        else:
            j0, j1 = k, k + 1
            left, right = grid_lon[k], grid_lon[k + 1]
        wo = (lon_norm - left) / (right - left)

        var.set_auto_maskandscale(False)
        v00 = float(var[i, j0, radius_idx]) * scale + offset
        v01 = float(var[i, j1, radius_idx]) * scale + offset
        v10 = float(var[i + 1, j0, radius_idx]) * scale + offset
        v11 = float(var[i + 1, j1, radius_idx]) * scale + offset
        top = v00 * (1 - wo) + v01 * wo
        bot = v10 * (1 - wo) + v11 * wo
        return top * (1 - wl) + bot * wl
    finally:
        ds.close()


@pytest.mark.parametrize("lat,lon", TEST_POINTS + ANTIMERIDIAN_POINTS)
def test_land_fraction_matches_manual_bilinear_per_radius(lat, lon):
    with LandFraction(LAND_FRACTION_PATH) as lf:
        actual = lf.query(lat, lon, np.array(DEFAULT_RADII_KM))
    with nc.Dataset(LAND_FRACTION_PATH, "r") as ds:
        stored_radii = np.asarray(ds.variables["radius"][:], dtype="float64")
    assert np.array_equal(stored_radii, np.asarray(DEFAULT_RADII_KM))
    for i, _radius_km in enumerate(DEFAULT_RADII_KM):
        expected_i = _manual_bilinear_3d(LAND_FRACTION_PATH, lat, lon, i)
        assert actual[i] == pytest.approx(expected_i, abs=1e-9)
    assert np.all((actual >= 0.0) & (actual <= 1.0))


def test_land_fraction_off_grid_radius_interpolates():
    lat, lon = 35.0, 139.7  # Tokyo -- real, non-trivial land fraction
    with LandFraction(LAND_FRACTION_PATH) as lf:
        at_100 = float(lf.query(lat, lon, 100.0))
        at_200 = float(lf.query(lat, lon, 200.0))
        at_150 = float(lf.query(lat, lon, 150.0))
    lo, hi = sorted([at_100, at_200])
    assert lo - 1e-9 <= at_150 <= hi + 1e-9


def test_land_fraction_radius_outside_range_is_nan():
    with LandFraction(LAND_FRACTION_PATH) as lf:
        result = lf.query(35.0, 139.7, 50.0)  # below the stored 100-600 km range
    assert np.isnan(result)


def test_distance_to_land_lat_outside_range_is_nan():
    with DistanceToLand(DISTANCE_PATH) as dtl:
        result = dtl.query(120.0, 0.0)  # not a real latitude
    assert np.isnan(result)


def test_land_fraction_array_radius_matches_scalar_calls():
    lat, lon = 35.0, 139.7
    with LandFraction(LAND_FRACTION_PATH) as lf:
        radii = np.array(DEFAULT_RADII_KM)
        vectorized = lf.query(lat, lon, radii)
        one_at_a_time = np.array([float(lf.query(lat, lon, r)) for r in radii])
    np.testing.assert_array_equal(vectorized, one_at_a_time)


def test_batched_array_matches_scalar_calls_on_real_grid():
    """The batched multi-point path, exercised against the real
    full-resolution grid rather than a small synthetic fixture."""
    rng = np.random.default_rng(50)
    lat = rng.uniform(-89.0, 89.0, 500)
    lon = rng.uniform(-180.0, 180.0, 500)
    with DistanceToLand(DISTANCE_PATH) as dtl:
        batched = dtl.query(lat, lon)
        per_point = np.array([dtl.query(float(a), float(o)) for a, o in zip(lat, lon, strict=True)])
    np.testing.assert_array_equal(batched, per_point)


def test_memory_stays_small():
    """Opening the table must not pull the full grid into the process
    (previously confirmed live at ~31 GB for the land_fraction table
    alone, before the windowed-read rewrite this package carries forward)."""
    import resource

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with LandFraction(LAND_FRACTION_PATH) as lf:
        for lat, lon in TEST_POINTS:
            lf.query(lat, lon, np.array(DEFAULT_RADII_KM))
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grown_mb = (after - before) / 1e3  # ru_maxrss is KB on Linux
    assert grown_mb < 500, f"opening LandFraction grew RSS by {grown_mb:.0f} MB, expected well under 500"
