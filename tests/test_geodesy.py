import numpy as np

from landmetrics._constants import EARTH_RADIUS_KM
from landmetrics._geodesy import chord_to_km, km_to_chord, lonlat_to_ecef


def test_lonlat_to_ecef_unit_norm():
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180.0, 180.0, 1000)
    lat = rng.uniform(-90.0, 90.0, 1000)
    xyz = lonlat_to_ecef(lon, lat)
    norms = np.linalg.norm(xyz, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_chord_km_round_trip():
    km = np.linspace(0.0, 20000.0, 500)
    chord = km_to_chord(km)
    back = chord_to_km(chord)
    np.testing.assert_allclose(back, km, atol=1e-6)


def test_known_great_circle_distances():
    # Equator quarter-circle: (0,0) to (0,90) -> pi/2 * R
    p0 = lonlat_to_ecef(np.array(0.0), np.array(0.0))
    p1 = lonlat_to_ecef(np.array(90.0), np.array(0.0))
    chord = np.linalg.norm(p1 - p0)
    dist = chord_to_km(chord)
    expected = np.pi / 2 * EARTH_RADIUS_KM
    assert abs(float(dist) - expected) / expected < 1e-4

    # Pole to pole: (0,90) to (0,-90) -> pi * R
    p0 = lonlat_to_ecef(np.array(0.0), np.array(90.0))
    p1 = lonlat_to_ecef(np.array(0.0), np.array(-90.0))
    chord = np.linalg.norm(p1 - p0)
    dist = chord_to_km(chord)
    expected = np.pi * EARTH_RADIUS_KM
    assert abs(float(dist) - expected) / expected < 1e-4
