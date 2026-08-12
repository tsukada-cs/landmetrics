import netCDF4 as nc
import numpy as np

from landmetrics.query import DistanceToLand


def _read_raw(path):
    with nc.Dataset(str(path), "r") as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype="float64")
        lon = np.asarray(ds.variables["lon"][:], dtype="float64")
        var = ds.variables["distance_to_land"]
        var.set_auto_maskandscale(False)
        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))
        raw = np.asarray(var[:, :])
        decoded = raw.astype("float64") * scale + offset
    return lat, lon, raw, decoded


def _manual_bilinear(lat_axis, lon_axis, decoded, lat, lon):
    lon_norm = ((lon + 180.0) % 360.0) - 180.0
    i = int(np.searchsorted(lat_axis, lat)) - 1
    i = max(0, min(i, lat_axis.size - 2))
    wl = (lat - lat_axis[i]) / (lat_axis[i + 1] - lat_axis[i])

    n = lon_axis.size
    k = int(np.searchsorted(lon_axis, lon_norm)) - 1
    if k < 0:
        j0, j1 = n - 1, 0
        left, right = lon_axis[-1] - 360.0, lon_axis[0]
    elif k >= n - 1:
        j0, j1 = n - 1, 0
        left, right = lon_axis[-1], lon_axis[0] + 360.0
    else:
        j0, j1 = k, k + 1
        left, right = lon_axis[k], lon_axis[k + 1]
    wo = (lon_norm - left) / (right - left)

    v00, v01 = decoded[i, j0], decoded[i, j1]
    v10, v11 = decoded[i + 1, j0], decoded[i + 1, j1]
    top = v00 * (1 - wo) + v01 * wo
    bot = v10 * (1 - wo) + v11 * wo
    return top * (1 - wl) + bot * wl


def test_exact_cell_center(tiny_distance_grid):
    lat_axis, lon_axis, raw, decoded = _read_raw(tiny_distance_grid)
    with DistanceToLand(tiny_distance_grid) as dtl:
        # lat index 105 -> lat=15.0, lon index 195 -> lon=15.0
        assert lat_axis[105] == 15.0
        assert lon_axis[195] == 15.0
        value = dtl.query(15.0, 15.0)
        assert abs(value - decoded[105, 195]) < 1e-9


def test_midpoint_matches_manual_bilinear(tiny_distance_grid):
    lat_axis, lon_axis, raw, decoded = _read_raw(tiny_distance_grid)
    with DistanceToLand(tiny_distance_grid) as dtl:
        for lat, lon in [(15.5, 15.5), (-30.3, 100.7), (5.1, -170.2)]:
            expected = _manual_bilinear(lat_axis, lon_axis, decoded, lat, lon)
            got = dtl.query(lat, lon)
            assert abs(got - expected) < 1e-6, (lat, lon, got, expected)


def test_out_of_range_and_nan_lat(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        assert np.isnan(dtl.query(95.0, 10.0))
        assert np.isnan(dtl.query(-95.0, 10.0))
        assert np.isnan(dtl.query(float("nan"), 10.0))
        assert np.isnan(dtl.query(10.0, float("nan")))


def test_scalar_and_array_shapes(tiny_distance_grid):
    with DistanceToLand(tiny_distance_grid) as dtl:
        scalar = dtl.query(15.0, 15.0)
        assert isinstance(scalar, float)

        lat = np.array([15.0, 20.0, -30.0])
        lon = np.array([15.0, 20.0, 100.0])
        arr = dtl.query(lat, lon)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)

        # scalar lat broadcast against lon array
        broadcast = dtl.query(15.0, np.array([10.0, 20.0, 30.0]))
        assert broadcast.shape == (3,)

        # 2-D shape preserved
        lat2d = np.array([[15.0, 20.0], [-30.0, 40.0]])
        lon2d = np.array([[15.0, 20.0], [100.0, -50.0]])
        arr2d = dtl.query(lat2d, lon2d)
        assert arr2d.shape == (2, 2)


def test_packing_is_decoded_not_raw(tiny_distance_grid):
    """The fixture is packed with scale_factor=0.1 so a real distance of
    a few hundred km is stored as a raw int16 in the tens of thousands
    (e.g. 2000 km -> raw 20000) -- if set_auto_maskandscale/manual
    decoding were broken and raw values leaked through unscaled, this
    assertion would fail because 20000 >= 1e4."""
    lat_axis, lon_axis, raw, decoded = _read_raw(tiny_distance_grid)
    assert np.any(np.abs(raw.astype("int64")) > 1e4), "fixture should exercise large raw packed values"
    with DistanceToLand(tiny_distance_grid) as dtl:
        far_ocean_value = dtl.query(-80.0, -170.0)
        assert abs(far_ocean_value) < 1e4


def test_batched_matches_scalar_random(tiny_distance_grid):
    lat_axis, lon_axis, raw, decoded = _read_raw(tiny_distance_grid)
    rng = np.random.default_rng(42)
    lat = rng.uniform(-89.0, 89.0, 300)
    lon = rng.uniform(-180.0, 180.0, 300)
    with DistanceToLand(tiny_distance_grid) as dtl:
        batched = dtl.query(lat, lon)
        scalar_results = np.array([dtl.query(float(a), float(o)) for a, o in zip(lat, lon, strict=True)])
    np.testing.assert_array_equal(batched, scalar_results)
