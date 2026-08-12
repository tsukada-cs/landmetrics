# Changelog

## 0.1.0

Initial release.

- `distance_to_land`, `land_fraction`, `is_land`, `is_ocean` query
  functions, plus the `DistanceToLand` / `LandFraction` classes they wrap.
- Query-only: no grid generation code is included in this package.
- Every query reads only the cells its interpolation needs directly out of
  the backing netCDF file, bounded by a fixed cell budget regardless of
  grid resolution; batched array queries use a single windowed block read
  or row-compact tiling instead of one read per point.
- Longitude handling (normalization, antimeridian wraparound, minimal
  circular column range for batched reads) is centralized and covered by a
  dedicated antimeridian test suite.
- Ships a coarse (0.1 degree) grid for each query kind inside the wheel;
  finer grids are fetched from Zenodo on first use and cached locally.
- `landmetrics` command-line tool: `query`, `fraction`, `list`, `info`,
  `fetch`.
