import pytest

from landmetrics.cli import main


def _run(argv):
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    return exc_info.value.code


def test_help_exits_zero(capsys):
    code = _run(["--help"])
    assert code == 0


def test_query_command(tiny_distance_grid, capsys):
    code = _run(["query", "--lat", "15.0", "--lon", "15.0", "--path", str(tiny_distance_grid)])
    assert code == 0
    out = capsys.readouterr().out
    assert "distance_to_land_km=" in out
    assert "is_land=True" in out


def test_fraction_command(tiny_fraction_grid, capsys):
    code = _run(
        [
            "fraction",
            "--lat",
            "15.0",
            "--lon",
            "15.0",
            "--radius-km",
            "100",
            "300",
            "600",
            "--path",
            str(tiny_fraction_grid),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("radius_km=") == 3


def test_list_command(capsys):
    code = _run(["list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "distance_to_land_0.1deg_gt0km2.nc" in out


def test_info_command(tiny_distance_grid, capsys):
    code = _run(["info", str(tiny_distance_grid)])
    assert code == 0
    out = capsys.readouterr().out
    assert "distance_to_land" in out


def test_fetch_without_zenodo_record_fails_cleanly(capsys):
    code = _run(["fetch", "distance_to_land", "--resolution-deg", "0.05", "--min-island-area-km2", "1400"])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
