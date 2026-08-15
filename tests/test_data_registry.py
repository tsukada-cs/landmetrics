import os

import pytest

from landmetrics import data
from landmetrics.exceptions import GridFormatError, GridNotFoundError
from landmetrics.query import DistanceToLand


def test_every_spec_filename_matches_grid_filename():
    for spec in data.available_grids():
        assert spec.filename == data.grid_filename(spec.kind, spec.resolution_deg, spec.min_island_area_km2)


def test_bundled_specs_exist_and_checksum_matches():
    bundled = [g for g in data.available_grids() if g.bundled]
    assert len(bundled) == 2
    for spec in bundled:
        resolved = data.grid_path(spec.kind, spec.resolution_deg, spec.min_island_area_km2, download=False)
        assert resolved.is_file()
        assert resolved.stat().st_size == spec.size_bytes
        assert data._sha256(resolved) == spec.sha256


def test_data_dir_env_takes_precedence(tmp_path, monkeypatch, tiny_distance_grid):
    # place a decoy under a directory named like the real bundled grid but
    # verify precedence via a distinct, registered filename instead of
    # colliding with the real bundled file's checksum expectations.
    spec = next(g for g in data.available_grids() if g.bundled and g.kind == "distance_to_land")
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    decoy = override_dir / spec.filename
    decoy.write_bytes(b"not a real grid, just needs to exist for path resolution")

    monkeypatch.setenv("LANDMETRICS_DATA_DIR", str(override_dir))
    resolved = data.grid_path(spec.kind, spec.resolution_deg, spec.min_island_area_km2, download=False)
    assert resolved == decoy


def test_data_dir_env_colon_separated_search_order(tmp_path, monkeypatch):
    spec = next(g for g in data.available_grids() if g.bundled and g.kind == "land_fraction")
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_b / spec.filename).write_bytes(b"only in b")

    monkeypatch.setenv("LANDMETRICS_DATA_DIR", f"{dir_a}{os.pathsep}{dir_b}")
    resolved = data.grid_path(spec.kind, spec.resolution_deg, spec.min_island_area_km2, download=False)
    assert resolved == dir_b / spec.filename

    (dir_a / spec.filename).write_bytes(b"now also in a, should win")
    resolved2 = data.grid_path(spec.kind, spec.resolution_deg, spec.min_island_area_km2, download=False)
    assert resolved2 == dir_a / spec.filename


def test_grid_not_found_raises_with_useful_message(monkeypatch, tmp_path):
    monkeypatch.setenv("LANDMETRICS_DATA_DIR", "")
    monkeypatch.setenv("LANDMETRICS_CACHE_DIR", str(tmp_path / "empty_cache"))
    with pytest.raises(GridNotFoundError) as exc_info:
        data.grid_path("distance_to_land", 0.05, 1400.0, download=False)
    message = str(exc_info.value)
    assert "distance_to_land_0.05deg_gt1400km2.nc" in message


def test_no_network_call_without_zenodo_record(monkeypatch, tmp_path):
    monkeypatch.setenv("LANDMETRICS_DATA_DIR", "")
    monkeypatch.setenv("LANDMETRICS_CACHE_DIR", str(tmp_path / "empty_cache2"))
    monkeypatch.setattr(data, "ZENODO_RECORD_ID", None)

    def _fail_urlopen(*args, **kwargs):
        raise AssertionError("should never attempt a network request")

    monkeypatch.setattr("urllib.request.urlopen", _fail_urlopen)
    with pytest.raises(GridNotFoundError):
        data.grid_path("distance_to_land", 0.05, 1400.0, download=True)


def test_download_url_uses_configured_record_id(monkeypatch, tmp_path):
    monkeypatch.setenv("LANDMETRICS_DATA_DIR", "")
    monkeypatch.setenv("LANDMETRICS_CACHE_DIR", str(tmp_path / "empty_cache3"))
    assert data.ZENODO_RECORD_ID == "21959508"

    spec = next(g for g in data.available_grids() if not g.bundled)
    requested_urls = []

    def _fake_urlopen(url, *args, **kwargs):
        requested_urls.append(url)
        raise AssertionError("stop before an actual network request")

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    with pytest.raises(AssertionError):
        data.grid_path(spec.kind, spec.resolution_deg, spec.min_island_area_km2, download=True)

    assert requested_urls == [f"https://zenodo.org/records/21959508/files/{spec.filename}?download=1"]


def test_malformed_grid_missing_variable_raises(tmp_path):
    import netCDF4 as nc

    path = tmp_path / "bad.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", 3)
        ds.createDimension("lon", 4)
        ds.createVariable("lat", "f4", ("lat",))[:] = [0.0, 1.0, 2.0]
        ds.createVariable("lon", "f4", ("lon",))[:] = [-2.0, -1.0, 0.0, 1.0]
        # no distance_to_land variable at all
    with pytest.raises(GridFormatError):
        DistanceToLand(path)


def test_malformed_grid_descending_lat_raises(tmp_path):
    import netCDF4 as nc

    path = tmp_path / "bad_lat.nc"
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", 3)
        ds.createDimension("lon", 4)
        ds.createVariable("lat", "f4", ("lat",))[:] = [2.0, 1.0, 0.0]  # descending
        ds.createVariable("lon", "f4", ("lon",))[:] = [-180.0, -90.0, 0.0, 90.0]
        var = ds.createVariable("distance_to_land", "i2", ("lat", "lon"))
        var.scale_factor = 1.0
        var.add_offset = 0.0
        var[:, :] = 0
    with pytest.raises(GridFormatError):
        DistanceToLand(path)
