import netCDF4 as nc
import numpy as np

from landmetrics._axes import normalize_lon
from landmetrics.query import DistanceToLand, LandFraction


def _read_lon_axis(path):
    with nc.Dataset(str(path), "r") as ds:
        return np.asarray(ds.variables["lon"][:], dtype="float64")


def test_query_agrees_across_lon_conventions(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        a = dtl.query(10.0, 200.0)
        b = dtl.query(10.0, -160.0)
        assert a == b


def test_seam_blend_weight(index_grid):
    """index_grid's stored value at (i, j) is i*10000 + j, so the seam
    blend (columns n_lon-1 and 0) at exactly the midpoint longitude is
    hand-computable: with weight 0.5 the interpolated value is the
    average of the two column indices' contributions."""
    lon_axis = _read_lon_axis(index_grid)
    n_lon = lon_axis.size
    assert lon_axis[-1] == 179.0
    # midpoint between lon[-1]=179.0 and lon[0]+360=180.0
    with DistanceToLand(index_grid) as dtl:
        got = dtl.query(0.0, 179.5)
        # lat=0.0 is exactly a grid row (index 90 on a 181-point -90..90 axis)
        i = 90
        expected = i * 10000 + ((n_lon - 1) * 0.5 + 0 * 0.5)
        assert abs(got - expected) < 1e-9


def test_continuity_sweep_across_seam(dateline_distance_grid):
    with DistanceToLand(dateline_distance_grid) as dtl:
        lons_pos = np.arange(179.0, 181.0, 0.001)
        values_pos = dtl.query(np.full(lons_pos.shape, 0.0), lons_pos)

        lons_neg = np.arange(-181.0, -179.0, 0.001)
        values_neg = dtl.query(np.full(lons_neg.shape, 0.0), lons_neg)

        # lons_pos[k] and lons_neg[k] are the same physical longitude
        # (lons_pos[k] - 360 == lons_neg[k], e.g. 179.0 and -181.0 both
        # normalize to 179.0), so the two sweeps must match directly, not
        # reversed.
        np.testing.assert_allclose(values_pos, values_neg, atol=1e-6)

        # No jump larger than a small multiple of the per-step change
        # scale anywhere in the sweep -- a coordinate bug at the seam
        # would show up as a discontinuity far larger than neighboring
        # steps.
        diffs = np.abs(np.diff(values_pos))
        step_scale = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0.0
        assert np.all(diffs < max(step_scale * 50, 1.0))


def test_island_straddling_dateline_is_land_both_sides(dateline_distance_grid):
    with DistanceToLand(dateline_distance_grid) as dtl:
        assert dtl.is_land(0.0, 178.0) is True  # west side of the seam
        assert dtl.is_land(0.0, -178.0) is True  # east side of the seam
        assert dtl.is_ocean(0.0, 150.0) is True  # well away from the island
        assert dtl.is_ocean(0.0, -150.0) is True


def test_batched_array_straddling_seam_matches_per_point(dateline_distance_grid):
    rng = np.random.default_rng(21)
    # points clustered right around the antimeridian, > _SCALAR_THRESHOLD
    # so the batched block-read path is exercised.
    lon = np.concatenate(
        [
            rng.uniform(175.0, 180.0, 60),
            rng.uniform(-180.0, -175.0, 60),
        ]
    )
    lat = rng.uniform(-10.0, 10.0, lon.size)
    with DistanceToLand(dateline_distance_grid) as dtl:
        batched = dtl.query(lat, lon)
        per_point = np.array([dtl.query(float(a), float(o)) for a, o in zip(lat, lon, strict=True)])
    np.testing.assert_array_equal(batched, per_point)


def test_batched_seam_read_is_bounded(dateline_distance_grid):
    """A batched read straddling the seam must use the short wrapped arc,
    not the whole longitude width -- see circular_column_range."""
    rng = np.random.default_rng(22)
    lon = np.concatenate(
        [
            rng.uniform(175.0, 180.0, 100),
            rng.uniform(-180.0, -175.0, 100),
        ]
    )
    lat = rng.uniform(-10.0, 10.0, lon.size)
    with DistanceToLand(dateline_distance_grid) as dtl:
        dtl.query(lat, lon)
        n_lon = dtl._lon.size
        # A single compact block read should need far fewer cells than
        # the grid's full longitude width would require.
        assert dtl._reader.cells_read < dtl._lat.size * n_lon // 4


def test_land_fraction_batched_seam_matches_per_point(tiny_fraction_grid):
    rng = np.random.default_rng(23)
    lon = np.concatenate(
        [
            rng.uniform(175.0, 180.0, 60),
            rng.uniform(-180.0, -175.0, 60),
        ]
    )
    lat = rng.uniform(-10.0, 10.0, lon.size)
    radius = rng.uniform(100.0, 600.0, lon.size)
    with LandFraction(tiny_fraction_grid) as lf:
        batched = lf.query(lat, lon, radius)
        per_point = np.array(
            [float(lf.query(float(a), float(o), float(r))) for a, o, r in zip(lat, lon, radius, strict=True)]
        )
    np.testing.assert_array_equal(batched, per_point)


def test_normalize_lon_used_consistently_at_query_level(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        for raw in (540.0, -540.0, 720.0, -1.0e-9):
            expected_lon = normalize_lon(raw)
            direct = dtl.query(10.0, expected_lon)
            via_raw = dtl.query(10.0, raw)
            assert direct == via_raw
