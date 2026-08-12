import numpy as np

from landmetrics.query import DistanceToLand


def test_island_interior_is_land(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        assert dtl.is_land(15.0, 15.0) is True
        assert not dtl.is_ocean(15.0, 15.0)


def test_open_ocean_is_ocean(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        assert dtl.is_ocean(-60.0, 100.0) is True
        assert not dtl.is_land(-60.0, 100.0)


def test_out_of_range_both_false(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        assert dtl.is_land(95.0, 10.0) is False
        assert dtl.is_ocean(95.0, 10.0) is False
        assert dtl.is_land(-95.0, 10.0) is False
        assert dtl.is_ocean(-95.0, 10.0) is False


def test_nan_both_false(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        assert dtl.is_land(float("nan"), 10.0) is False
        assert dtl.is_ocean(float("nan"), 10.0) is False
        assert dtl.is_land(10.0, float("nan")) is False
        assert dtl.is_ocean(10.0, float("nan")) is False


def test_nearest_cell_land_but_interpolation_positive(tiny_distance_grid):
    """The motivating case for is_land's nearest-cell design: at
    lat=20.49, lon=15.0 the *nearest* grid cell (lat=20.0, the island's
    inset boundary row) is land, but bilinearly interpolating toward the
    next row out (lat=21.0, clearly ocean and further from the boundary
    than the land row is inside it) crosses zero to a positive value.
    is_land must say True here even though query() says positive."""
    with DistanceToLand(tiny_distance_grid) as dtl:
        interpolated = dtl.query(20.49, 15.0)
        assert interpolated > 0, "fixture assumption violated: expected a positive interpolated value here"
        assert dtl.is_land(20.49, 15.0) is True


def test_dtype_scalar_and_array(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        scalar = dtl.is_land(15.0, 15.0)
        assert isinstance(scalar, bool)

        lat = np.array([15.0, -60.0, 95.0])
        lon = np.array([15.0, 100.0, 10.0])
        arr = dtl.is_land(lat, lon)
        assert arr.dtype == bool
        np.testing.assert_array_equal(arr, [True, False, False])

        arr_ocean = dtl.is_ocean(lat, lon)
        assert arr_ocean.dtype == bool
        np.testing.assert_array_equal(arr_ocean, [False, True, False])


def test_batched_is_land_matches_per_point(tiny_distance_grid):
    rng = np.random.default_rng(11)
    lat = rng.uniform(-89.0, 89.0, 300)
    lon = rng.uniform(-180.0, 180.0, 300)
    with DistanceToLand(tiny_distance_grid) as dtl:
        batched = dtl.is_land(lat, lon)
        per_point = np.array([dtl.is_land(float(a), float(o)) for a, o in zip(lat, lon, strict=True)])
    np.testing.assert_array_equal(batched, per_point)
