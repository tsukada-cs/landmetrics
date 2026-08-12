# Grid file format

This document specifies the netCDF schema `landmetrics` reads, and describes
how the bundled/Zenodo-hosted grids were derived from GSHHG. It exists so the
library is not a black box around an opaque binary file, and so a third party
could in principle produce a compatible grid without needing this project's
(unpublished) generation code.

## `distance_to_land`

- Dimensions: `lat`, `lon`.
- Variable `distance_to_land(lat, lon)`: signed great-circle distance to the
  nearest coastline, in kilometers, **negative over land**, positive over
  ocean. Typically stored as packed `int16` with `scale_factor` /
  `add_offset` attributes (plain unpacked float storage is also accepted --
  the reader falls back to `scale_factor=1.0`, `add_offset=0.0` when those
  attributes are absent).
- `lat`: ascending, `-90.0` to `90.0` inclusive, `n = 180 / resolution_deg + 1`
  points.
- `lon`: ascending, **half-open** `[-180.0, 180.0)`, `n = 360 / resolution_deg`
  points, with **no duplicated seam column** -- a 0.1 degree grid stores
  `-180.0, -179.9, ..., 179.9`, never a trailing `180.0`. This is what makes
  the antimeridian a wraparound between the last and first columns rather
  than an ordinary interior cell.
- No `_FillValue`: every cell has a well-defined value.

## `land_fraction`

- Dimensions: `lat`, `lon`, `radius`.
- Variable `land_fraction(lat, lon, radius)`: fraction of land area within
  the given great-circle radius of the cell center, dimensionless, in
  `[0, 1]`.
- `lat` / `lon`: same convention as `distance_to_land` above.
- `radius`: ascending, kilometers.

## Validation

Opening a file with `DistanceToLand`/`LandFraction` performs a cheap check on
the coordinate arrays only (never scans the data variable): the required
variable is present, `lat` and `lon` are strictly ascending, and `lon`'s
span is consistent with a half-open 360-degree domain. A file that fails
these checks raises `landmetrics.GridFormatError` with a specific message.

## Derivation of the bundled/hosted grids

The shipped grids are derived from
[GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/) (Global
Self-consistent, Hierarchical, High-resolution Geography), scale `f` (full
resolution), levels 1 (mainland and islands) and 5 (Antarctica, ice-shelf-front
definition -- level 1 excludes Antarctica entirely; level 5 rather than level
6, the grounding line, because a floating ice shelf presents the same solid
surface to the atmosphere as bedrock land for the purposes this library was
built for).

- **`distance_to_land`**: nearest-coastline distance is computed with a
  k-d tree over every kept polygon's boundary vertices, converted to
  unit-sphere ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates so
  nearest-neighbor search is exact for great-circle distance (chord length
  is monotonic with central angle). The land/ocean sign is computed by
  rasterizing the same polygon set onto the output grid, testing each cell
  center with a boundary-inclusive containment predicate (a coastline
  vertex itself counts as land).
- **`land_fraction`**: the land/ocean mask is rasterized once at the grid's
  resolution; every land cell's center becomes a point in a k-d tree (same
  ECEF representation). For each radius, every grid point's land-cell count
  within that great-circle radius (via a ball-point query) is divided by the
  expected total cell count within the same radius at that point's latitude
  (a flat-earth approximation of cell geometry -- accurate enough at these
  radii, and what keeps the whole computation tractable at global-grid
  scale) to give a `[0, 1]` fraction. This is a grid-cell-counting estimate,
  not an exact polygon-disc area intersection.
- Islands smaller than a minimum area threshold (encoded in the filename,
  see below) are excluded from both computations, since they are too small
  to meaningfully affect the phenomena this library was built to support.

## Filenames

`{kind}_{resolution_deg:g}deg_gt{min_island_area_km2:g}km2.nc`, e.g.
`distance_to_land_0.05deg_gt1400km2.nc`. Both generation parameters are
encoded in the filename because multiple resolution/threshold variants exist
side by side rather than a single fixed file per kind. "gt" is for
readability even though the underlying filter is inclusive (`>=`); the
distinction is immaterial for real GSHHG polygon areas at these thresholds.

## License

The grid data files are a derivative of GSHHG, which is released under the
LGPL. See `DATA_LICENSE` at the repository root.
