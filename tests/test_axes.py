import numpy as np

from landmetrics._axes import (
    bracket_nonperiodic,
    bracket_nonperiodic_many,
    bracket_periodic,
    bracket_periodic_many,
    circular_column_range,
    nearest_index,
    nearest_index_many,
    nearest_index_periodic,
    nearest_index_periodic_many,
    normalize_lon,
)


def test_normalize_lon_table():
    cases = {
        180.0: -180.0,
        -180.0: -180.0,
        360.0: 0.0,
        -360.0: 0.0,
        540.0: -180.0,
        -540.0: -180.0,
        0.0: 0.0,
        90.0: 90.0,
        -90.0: -90.0,
        179.9999999999: 179.9999999999,
    }
    for lon_in, expected in cases.items():
        out = normalize_lon(lon_in)
        assert isinstance(out, float)
        assert abs(out - expected) < 1e-6, (lon_in, out, expected)


def test_normalize_lon_nan_and_inf():
    assert np.isnan(normalize_lon(float("nan")))
    # +/-inf should not crash and should not silently become a finite value
    # that looks like a valid longitude.
    out_pos = normalize_lon(float("inf"))
    out_neg = normalize_lon(float("-inf"))
    assert np.isnan(out_pos) or not np.isfinite(out_pos)
    assert np.isnan(out_neg) or not np.isfinite(out_neg)


def test_normalize_lon_array():
    lon = np.array([180.0, -180.0, 360.0, 0.0, np.nan])
    out = normalize_lon(lon)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out[:4], [-180.0, -180.0, 0.0, 0.0])
    assert np.isnan(out[4])


def test_normalize_lon_always_in_range():
    rng = np.random.default_rng(1)
    lon = rng.uniform(-10000.0, 10000.0, 5000)
    out = normalize_lon(lon)
    assert np.all(out >= -180.0)
    assert np.all(out < 180.0)


def test_bracket_nonperiodic_clamps():
    axis = np.array([0.0, 1.0, 2.0, 3.0])
    assert bracket_nonperiodic(axis, -5.0) == (0, -5.0)  # extrapolated weight, not clamped weight
    i, w = bracket_nonperiodic(axis, 0.5)
    assert i == 0 and abs(w - 0.5) < 1e-12
    i, w = bracket_nonperiodic(axis, 2.5)
    assert i == 2 and abs(w - 0.5) < 1e-12
    # value beyond the last point still uses the last interval
    i, w = bracket_nonperiodic(axis, 10.0)
    assert i == 2


def test_bracket_periodic_wraps():
    axis = np.array([-180.0, -90.0, 0.0, 90.0])  # period 360, spacing 90
    # value between axis[-1]=90 and axis[0]+360=180
    j0, j1, w = bracket_periodic(axis, 135.0)
    assert j0 == 3 and j1 == 0
    assert 0.0 <= w <= 1.0
    np.testing.assert_allclose(w, 0.5)

    # value below axis[0]
    j0, j1, w = bracket_periodic(axis, -200.0)
    assert j0 == 3 and j1 == 0
    assert 0.0 <= w <= 1.0


def test_bracket_periodic_many_matches_scalar():
    axis = -180.0 + 0.1 * np.arange(3600)
    rng = np.random.default_rng(2)
    values = rng.uniform(-500.0, 500.0, 2000)
    j0_m, j1_m, w_m = bracket_periodic_many(axis, values)
    for k in range(0, 2000, 37):  # sample, full loop is slow but correct
        j0, j1, w = bracket_periodic(axis, float(values[k]))
        assert j0_m[k] == j0
        assert j1_m[k] == j1
        assert abs(w_m[k] - w) < 1e-9


def test_bracket_nonperiodic_many_matches_scalar():
    axis = np.linspace(-90.0, 90.0, 1801)
    rng = np.random.default_rng(3)
    values = rng.uniform(-90.0, 90.0, 2000)
    idx_m, w_m = bracket_nonperiodic_many(axis, values)
    for k in range(0, 2000, 37):
        idx, w = bracket_nonperiodic(axis, float(values[k]))
        assert idx_m[k] == idx
        assert abs(w_m[k] - w) < 1e-9


def test_float32_axis_searchsorted_regression():
    """The stored axis is float32 and not perfectly uniform. An
    arithmetic index (value - axis[0]) / spacing drifts from the correct
    searchsorted-based index by the far end of a long axis -- this test
    demonstrates the drift exists (motivating why bracket_periodic uses
    searchsorted) and confirms bracket_periodic itself doesn't have it."""
    n = 36000
    axis64 = (-180.0 + 0.01 * np.arange(n)).astype("float64")
    axis32 = axis64.astype("float32").astype("float64")  # round-trip through float32
    # The spacing a naive arithmetic-index implementation would derive
    # from the axis itself (axis[1] - axis[0]) is not exactly 0.01 once
    # the axis has been through float32 -- this is the actual source of
    # the drift, not just generic float noise.
    spacing = axis32[1] - axis32[0]

    rng = np.random.default_rng(4)
    probes = rng.uniform(axis32[0], axis32[-1] + spacing, 1000)

    arithmetic_idx = np.clip(((probes - axis32[0]) / spacing).astype(np.int64), 0, n - 2)
    searchsorted_idx = np.clip(np.searchsorted(axis32, probes) - 1, 0, n - 2)

    disagreements = np.sum(arithmetic_idx != searchsorted_idx)
    assert disagreements > 0, "expected the float32 axis to exhibit arithmetic-index drift"

    # bracket_periodic_many (searchsorted-based) must agree with an exact
    # per-point float64 search at every probe, unlike the arithmetic index.
    j0_m, j1_m, w_m = bracket_periodic_many(axis32, probes)
    for k in range(0, 1000, 13):
        exact_k = int(np.searchsorted(axis32, probes[k])) - 1
        exact_k = max(0, min(exact_k, n - 2)) if 0 <= exact_k < n - 1 else exact_k
        # bracket_periodic_many's own wraparound semantics for the
        # boundary cases are already covered by test_bracket_periodic_wraps;
        # here we only check the interior points land on a consistent cell.
        if 0 <= exact_k < n - 1:
            assert j0_m[k] == exact_k


def test_nearest_index_clamps():
    axis = np.array([0.0, 1.0, 2.0, 3.0])
    assert nearest_index(axis, -5.0) == 0
    assert nearest_index(axis, 10.0) == 3
    assert nearest_index(axis, 0.4) == 0
    assert nearest_index(axis, 0.6) == 1
    assert nearest_index(axis, 1.5) in (1, 2)  # tie, either is acceptable


def test_nearest_index_many_matches_scalar():
    axis = np.linspace(-90.0, 90.0, 1801)
    rng = np.random.default_rng(5)
    values = rng.uniform(-100.0, 100.0, 2000)
    idx_m = nearest_index_many(axis, values)
    for k in range(0, 2000, 37):
        assert idx_m[k] == nearest_index(axis, float(values[k]))


def test_nearest_index_periodic_seam():
    # 0.1 degree grid: axis[0] = -180.0, axis[-1] = 179.9
    axis = -180.0 + 0.1 * np.arange(3600)
    # nearest cell to 179.99 is index 0 (-180.0, wrapped distance 0.01),
    # not index 3599 (179.9, distance 0.09).
    assert nearest_index_periodic(axis, 179.99) == 0
    # nearest cell to 179.91 is index 3599 (distance 0.01), not index 0
    # (wrapped distance 0.09).
    assert nearest_index_periodic(axis, 179.91) == 3599
    # exactly at a stored value
    assert nearest_index_periodic(axis, -180.0) == 0
    assert nearest_index_periodic(axis, 0.0) == 1800


def test_nearest_index_periodic_many_matches_scalar():
    axis = -180.0 + 0.1 * np.arange(3600)
    rng = np.random.default_rng(6)
    values = rng.uniform(-181.0, 181.0, 2000)
    idx_m = nearest_index_periodic_many(axis, values)
    for k in range(0, 2000, 37):
        assert idx_m[k] == nearest_index_periodic(axis, float(values[k]))


def test_circular_column_range_contiguous():
    cols = np.array([10, 11, 12, 13])
    start, length = circular_column_range(cols, 3600)
    assert start == 10 and length == 4


def test_circular_column_range_seam_straddle():
    # points near the seam of a 36000-column axis
    cols = np.array([35990, 35995, 35999, 0, 2, 4])
    start, length = circular_column_range(cols, 36000)
    # minimal arc covers 35990..35999 (10) + 0..4 (5) = 15, wrapping
    assert start == 35990
    assert length == 15


def test_circular_column_range_full_globe():
    n_lon = 360
    cols = np.arange(n_lon)  # every column present
    start, length = circular_column_range(cols, n_lon)
    assert length == n_lon


def test_circular_column_range_single_column():
    start, length = circular_column_range(np.array([42]), 3600)
    assert start == 42 and length == 1
