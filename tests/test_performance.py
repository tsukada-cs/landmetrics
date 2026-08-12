"""Behavioral bounds on how much a query reads -- not wall-clock timing
(flaky in CI), but cell counts and read counts, which are deterministic.
"""

from __future__ import annotations

import netCDF4 as nc
import numpy as np
import pytest

from landmetrics import data
from landmetrics.query import _MAX_WINDOW_CELLS, DistanceToLand, LandFraction


def test_scalar_distance_reads_at_most_8_cells(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        dtl.query(15.3, 15.3)
        assert dtl._reader.cells_read <= 8


def test_scalar_fraction_reads_bounded(tiny_fraction_grid):
    with LandFraction(tiny_fraction_grid) as lf:
        lf.query(15.3, 15.3, 300.0)
        n_radius = lf._radius.size
        assert lf._reader.cells_read <= 4 * n_radius


def test_scattered_global_points_read_far_below_grid_size(tiny_distance_grid):
    rng = np.random.default_rng(30)
    lat = rng.uniform(-89.0, 89.0, 10_000)
    lon = rng.uniform(-180.0, 180.0, 10_000)
    with DistanceToLand(tiny_distance_grid) as dtl:
        dtl.query(lat, lon)
        n_lat, n_lon = dtl._lat.size, dtl._lon.size
        assert dtl._reader.cells_read < n_lat * n_lon
        assert dtl._reader.reads < 10_000


def test_clustered_points_read_one_block(tiny_distance_grid):
    rng = np.random.default_rng(31)
    lat = rng.uniform(10.0, 15.0, 10_000)
    lon = rng.uniform(10.0, 15.0, 10_000)
    with DistanceToLand(tiny_distance_grid) as dtl:
        dtl.query(lat, lon)
        assert dtl._reader.reads == 1


def test_dateline_straddling_points_block_width_small(dateline_distance_grid):
    rng = np.random.default_rng(32)
    lon = np.concatenate(
        [
            rng.uniform(175.0, 180.0, 5000),
            rng.uniform(-180.0, -175.0, 5000),
        ]
    )
    lat = rng.uniform(-10.0, 10.0, lon.size)
    with DistanceToLand(dateline_distance_grid) as dtl:
        dtl.query(lat, lon)
        n_lon = dtl._lon.size
        # the arc actually needed is ~10 degrees wide, far short of the
        # full 360 degree width.
        assert dtl._reader.cells_read < dtl._lat.size * n_lon // 10


def test_repeated_scalar_query_hits_cache(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        for _ in range(1000):
            dtl.query(15.3, 15.3)
        assert dtl._reader.reads == 1


def test_batched_vs_per_point_bit_identical_distance(tiny_distance_grid):
    rng = np.random.default_rng(33)
    lat = rng.uniform(-89.0, 89.0, 2000)
    lon = rng.uniform(-180.0, 180.0, 2000)
    with DistanceToLand(tiny_distance_grid) as dtl:
        batched = dtl.query(lat, lon)
    with DistanceToLand(tiny_distance_grid) as dtl2:
        per_point = np.array([dtl2.query(float(a), float(o)) for a, o in zip(lat, lon, strict=True)])
    np.testing.assert_array_equal(batched, per_point)


def test_batched_vs_per_point_bit_identical_fraction(tiny_fraction_grid):
    rng = np.random.default_rng(34)
    lat = rng.uniform(-89.0, 89.0, 2000)
    lon = rng.uniform(-180.0, 180.0, 2000)
    radius = rng.uniform(100.0, 600.0, 2000)
    with LandFraction(tiny_fraction_grid) as lf:
        batched = lf.query(lat, lon, radius)
    with LandFraction(tiny_fraction_grid) as lf2:
        per_point = np.array(
            [float(lf2.query(float(a), float(o), float(r))) for a, o, r in zip(lat, lon, radius, strict=True)]
        )
    np.testing.assert_array_equal(batched, per_point)


def test_never_reads_more_than_budget(monkeypatch, tiny_distance_grid, tiny_fraction_grid):
    """Regression guard: no read this package performs, at any call size,
    may slice more than _MAX_WINDOW_CELLS elements out of a netCDF
    variable -- if a future change accidentally reverts to loading a
    whole grid, this fails immediately instead of merely being slow."""
    original_getitem = nc.Variable.__getitem__

    def _bounded_getitem(self, key):
        result = original_getitem(self, key)
        size = np.asarray(result).size
        assert size <= _MAX_WINDOW_CELLS, f"read {size} cells, budget is {_MAX_WINDOW_CELLS}"
        return result

    monkeypatch.setattr(nc.Variable, "__getitem__", _bounded_getitem)

    rng = np.random.default_rng(35)
    with DistanceToLand(tiny_distance_grid) as dtl:
        dtl.query(15.0, 15.0)
        dtl.query(rng.uniform(-89, 89, 30), rng.uniform(-180, 180, 30))
        dtl.query(rng.uniform(-89, 89, 5000), rng.uniform(-180, 180, 5000))
        dtl.is_land(rng.uniform(-89, 89, 5000), rng.uniform(-180, 180, 5000))

    with LandFraction(tiny_fraction_grid) as lf:
        lf.query(15.0, 15.0, np.array([100.0, 300.0, 600.0]))
        lf.query(rng.uniform(-89, 89, 5000), rng.uniform(-180, 180, 5000), rng.uniform(100, 600, 5000))


@pytest.mark.slow
def test_peak_memory_on_real_grid():
    pytest.importorskip("resource")
    import resource

    try:
        path = data.grid_path("land_fraction", 0.01, 0.0, download=False)
    except Exception:
        pytest.skip("real 0.01deg land_fraction grid not available locally")

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rng = np.random.default_rng(36)
    with LandFraction(path) as lf:
        lf.query(rng.uniform(-89, 89, 5000), rng.uniform(-180, 180, 5000), rng.uniform(100, 600, 5000))
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in KB on Linux
    assert (after - before) < 200_000
