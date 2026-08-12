import netCDF4 as nc
import numpy as np

from landmetrics.query import LandFraction


def _read_raw(path):
    with nc.Dataset(str(path), "r") as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype="float64")
        lon = np.asarray(ds.variables["lon"][:], dtype="float64")
        radius = np.asarray(ds.variables["radius"][:], dtype="float64")
        var = ds.variables["land_fraction"]
        var.set_auto_maskandscale(False)
        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))
        raw = np.asarray(var[:, :, :])
        decoded = raw.astype("float64") * scale + offset
    return lat, lon, radius, raw, decoded


def test_exact_passthrough_at_stored_radius(tiny_fraction_grid):
    lat, lon, radius, raw, decoded = _read_raw(tiny_fraction_grid)
    with LandFraction(tiny_fraction_grid) as lf:
        # f(lat=15) = 0.3 + 0.5*(15+90)/180, g(300)=0.5
        f_lat = 0.3 + 0.5 * (15.0 + 90.0) / 180.0
        expected = round(f_lat * 0.5 / 0.001) * 0.001  # matches the fixture's own quantization
        got = lf.query(15.0, 15.0, 300.0)
        assert abs(got - expected) < 1e-6


def test_linear_between_stored_radii(tiny_fraction_grid):
    lat, lon, radius, raw, decoded = _read_raw(tiny_fraction_grid)
    with LandFraction(tiny_fraction_grid) as lf:
        got = lf.query(15.0, 15.0, 200.0)  # between 100 and 300
        # decoded values are linear in radius (g(r) = r/600) up to
        # quantization, so linear interpolation between stored 100/300
        # should closely match the decoded value at 200.
        i = int(np.searchsorted(lat, 15.0)) - 1
        j = int(np.searchsorted(lon, ((15.0 + 180.0) % 360.0) - 180.0)) - 1
        v100, v300 = decoded[i, j, 0], decoded[i, j, 1]
        expected = v100 * 0.5 + v300 * 0.5
        # The fixture quantizes to 0.001 independently at each stored
        # radius, so linear interpolation between the two quantized
        # endpoints can differ from the true analytic value by roughly
        # the quantization step itself.
        assert abs(got - expected) < 2e-3


def test_nan_outside_radius_range(tiny_fraction_grid):
    with LandFraction(tiny_fraction_grid) as lf:
        assert np.isnan(lf.query(15.0, 15.0, 50.0))  # below min stored radius (100)
        assert np.isnan(lf.query(15.0, 15.0, 700.0))  # above max stored radius (600)
        # exactly at the bounds is in-range
        assert not np.isnan(lf.query(15.0, 15.0, 100.0))
        assert not np.isnan(lf.query(15.0, 15.0, 600.0))


def test_scalar_lat_lon_with_radius_array_one_read(tiny_fraction_grid):
    with LandFraction(tiny_fraction_grid) as lf:
        values = lf.query(15.0, 15.0, np.array([100.0, 300.0, 600.0]))
        assert values.shape == (3,)
        assert lf._reader.reads == 1


def test_general_broadcast_matches_scalar_path(tiny_fraction_grid):
    rng = np.random.default_rng(7)
    lat = rng.uniform(-89.0, 89.0, 20)
    lon = rng.uniform(-180.0, 180.0, 20)
    radius = rng.uniform(100.0, 600.0, 20)
    with LandFraction(tiny_fraction_grid) as lf:
        batched = lf.query(lat, lon, radius)
        scalar = np.array(
            [float(lf.query(float(a), float(o), float(r))) for a, o, r in zip(lat, lon, radius, strict=True)]
        )
    np.testing.assert_array_equal(batched, scalar)


def test_out_of_range_lat_and_nan(tiny_fraction_grid):
    with LandFraction(tiny_fraction_grid) as lf:
        result = lf.query(95.0, 10.0, np.array([100.0, 300.0]))
        assert np.all(np.isnan(result))
        assert np.isnan(lf.query(float("nan"), 10.0, 300.0))


def test_batched_path_matches_scalar_large_array(tiny_fraction_grid):
    rng = np.random.default_rng(8)
    lat = rng.uniform(-89.0, 89.0, 200)
    lon = rng.uniform(-180.0, 180.0, 200)
    radius = rng.uniform(100.0, 600.0, 200)
    with LandFraction(tiny_fraction_grid) as lf:
        batched = lf.query(lat, lon, radius)  # n=200 > threshold -> _query_many
    with LandFraction(tiny_fraction_grid) as lf2:
        scalar = np.array(
            [float(lf2.query(float(a), float(o), float(r))) for a, o, r in zip(lat, lon, radius, strict=True)]
        )
    np.testing.assert_array_equal(batched, scalar)
