"""``landmetrics`` console script -- query, fraction, list, info, fetch.

Query-only, matching the library itself: there is no ``build`` subcommand
here. Grid generation is not part of this package.
"""

from __future__ import annotations

import argparse
import sys

import netCDF4 as nc

from . import data
from ._constants import DEFAULT_MIN_ISLAND_AREA_KM2, DEFAULT_RESOLUTION_DEG
from .exceptions import GridFormatError, GridNotFoundError
from .query import DistanceToLand, LandFraction


def _cmd_query(args: argparse.Namespace) -> int:
    path = args.path or data.grid_path(
        "distance_to_land",
        args.resolution_deg,
        args.min_island_area_km2,
        download=not args.no_download,
    )
    with DistanceToLand(path) as dtl:
        distance = dtl.query(args.lat, args.lon)
        land = dtl.is_land(args.lat, args.lon)
    print(f"distance_to_land_km={distance:.3f} is_land={land}")
    return 0


def _cmd_fraction(args: argparse.Namespace) -> int:
    path = args.path or data.grid_path(
        "land_fraction",
        args.resolution_deg,
        args.min_island_area_km2,
        download=not args.no_download,
    )
    with LandFraction(path) as lf:
        values = lf.query(args.lat, args.lon, args.radius_km)
    for radius, value in zip(args.radius_km, values, strict=True):
        print(f"radius_km={radius:g} land_fraction={value:.4f}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    for spec in data.available_grids():
        try:
            resolved = data.grid_path(spec.kind, spec.resolution_deg, spec.min_island_area_km2, download=False)
            location = str(resolved)
        except GridNotFoundError:
            location = "(not available locally; would be downloaded)"
        print(f"{spec.filename:45s} {spec.size_bytes / 1e6:8.1f} MB  {location}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    with nc.Dataset(args.path, "r") as ds:
        print(f"path: {args.path}")
        print(f"dimensions: { {name: len(dim) for name, dim in ds.dimensions.items()} }")
        print(f"variables: {list(ds.variables.keys())}")
        for key in ("title", "summary", "resolution_deg", "min_island_area_km2", "license"):
            if key in ds.ncattrs():
                print(f"{key}: {getattr(ds, key)}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    try:
        path = data.grid_path(args.kind, args.resolution_deg, args.min_island_area_km2, download=True)
    except GridNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(str(path))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="landmetrics", description="Query GSHHG-based land-distance/fraction grids.")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_grid_selection(p: argparse.ArgumentParser) -> None:
        p.add_argument("--path", default=None, help="explicit grid file path, overrides resolution/area selection")
        p.add_argument("--resolution-deg", type=float, default=DEFAULT_RESOLUTION_DEG)
        p.add_argument("--min-island-area-km2", type=float, default=DEFAULT_MIN_ISLAND_AREA_KM2)
        p.add_argument("--no-download", action="store_true", help="never fetch from Zenodo, fail if not local")

    p_query = sub.add_parser("query", help="distance to land + is_land at one point")
    p_query.add_argument("--lat", type=float, required=True)
    p_query.add_argument("--lon", type=float, required=True)
    _add_grid_selection(p_query)
    p_query.set_defaults(func=_cmd_query)

    p_fraction = sub.add_parser("fraction", help="land fraction within radius at one point")
    p_fraction.add_argument("--lat", type=float, required=True)
    p_fraction.add_argument("--lon", type=float, required=True)
    p_fraction.add_argument("--radius-km", type=float, nargs="+", required=True)
    _add_grid_selection(p_fraction)
    p_fraction.set_defaults(func=_cmd_fraction)

    p_list = sub.add_parser("list", help="show the grid registry and where each grid currently resolves")
    p_list.set_defaults(func=_cmd_list)

    p_info = sub.add_parser("info", help="dump a grid file's dimensions/variables/attributes")
    p_info.add_argument("path")
    p_info.set_defaults(func=_cmd_info)

    p_fetch = sub.add_parser("fetch", help="download a registered grid into the local cache")
    p_fetch.add_argument("kind", choices=["distance_to_land", "land_fraction"])
    p_fetch.add_argument("--resolution-deg", type=float, required=True)
    p_fetch.add_argument("--min-island-area-km2", type=float, required=True)
    p_fetch.set_defaults(func=_cmd_fetch)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except GridFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
