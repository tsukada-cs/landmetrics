# landmetrics

Distance to the nearest coastline, land fraction within a radius, and
land/ocean point tests, backed by precomputed [GSHHG](https://www.soest.hawaii.edu/pwessel/gshhg/)
coastline grids.

```python
import landmetrics as lm

lm.distance_to_land(25.0, -80.0)  # km, negative over land
lm.land_fraction(25.0, -80.0, 300.0)  # fraction in [0, 1] within 300 km
lm.is_land(25.0, -80.0)  # bool
lm.is_ocean(25.0, -80.0)  # bool
```

This is a **query-only** library: it ships precomputed grid files and reads
them efficiently. It does not generate grids from raw GSHHG polygons -- that
requires a heavier geospatial toolchain (shapely, geopandas, cartopy, scipy)
and is not part of this package.

## Install

```bash
pip install landmetrics
```

A coarse (0.1 degree) grid for each query kind ships inside the wheel, so a
query works immediately with no network access. Finer grids are fetched from
Zenodo on first use and cached locally (see [Data](#data) below).

## API

```python
distance_to_land(lat, lon, path=None, *, resolution_deg=None, min_island_area_km2=None)
land_fraction(lat, lon, radius_km, path=None, *, resolution_deg=None, min_island_area_km2=None)
is_land(lat, lon, path=None, *, resolution_deg=None, min_island_area_km2=None)
is_ocean(lat, lon, path=None, *, resolution_deg=None, min_island_area_km2=None)
```

`lat`/`lon`/`radius_km` may be scalars or array-like (broadcast together).
`lon` is normalized into `[-180, 180)` first, so any input convention works,
including values that accumulate past `+/-180` across the antimeridian.

For repeated queries, open a reusable, explicit object instead of the cached
convenience functions:

```python
with lm.DistanceToLand(path) as dtl:
    values = dtl.query(lats, lons)
    land = dtl.is_land(lats, lons)

with lm.LandFraction(path) as lf:
    fractions = lf.query(lats, lons, radius_km=300.0)
```

`open_distance_to_land(...)` / `open_land_fraction(...)` resolve and open a
grid by resolution/island-area threshold rather than an explicit path.

### `is_land` / `is_ocean`

These read the *nearest grid cell's* sign, not an interpolated value:
bilinearly blending the signed distance field across a coastline can carry a
land point within one grid cell of the shore to a positive (ocean) value.
Accuracy is therefore bounded by grid resolution -- roughly half a cell width
at the equator (about 5.5 km at 0.05 degrees, 1.1 km at 0.01 degrees).

### Thread safety

`DistanceToLand` and `LandFraction` hold an open netCDF handle, and
netCDF4/HDF5 access is not safe to share across threads. Use one instance per
thread (including the module-level cached instances behind the convenience
functions above), or serialize access with a lock. Separate *processes* are
fine, each with its own instance -- just don't open one before forking and
then use it from both parent and child.

### Performance

Every query reads only the cells its interpolation needs directly out of the
netCDF file -- never the whole grid. A scalar query reads at most 8 cells. An
array query reads one bounding block when the points are geographically
compact, or splits into row-compact tiles otherwise, bounded by a fixed cell
budget regardless of the underlying grid's resolution. See
`docs/grid_format.md` for the file layout this depends on.

## Data

| grid | resolution | min island area | bundled |
|---|---|---|---|
| `distance_to_land` | 0.1 deg | 0 km^2 | yes |
| `land_fraction` | 0.1 deg | 0 km^2 | yes |
| `distance_to_land` / `land_fraction` | 0.05, 0.01 deg | 0, 1400, 4748 km^2 | fetched from Zenodo |

`grid_path(kind, resolution_deg, min_island_area_km2)` resolves a grid file,
searching in order: an explicit path, `$LANDMETRICS_DATA_DIR` (colon-separated
directory list), the bundled copy, the local cache
(`$LANDMETRICS_CACHE_DIR`, else a platform user-cache directory), and finally
a Zenodo download into the cache. `landmetrics list` shows every known grid
and where it currently resolves; `landmetrics fetch <kind> --resolution-deg
... --min-island-area-km2 ...` downloads one explicitly.

The two bundled grids were generated from GSHHG scale "f" (full resolution),
levels 1 (mainland and islands) and 5 (Antarctica, ice-shelf-front
definition). See `docs/grid_format.md` for the full derivation and the file
schema a third party would need to match to produce a compatible grid.

### License

Code is MIT (see `LICENSE`). The grid data files are derived from GSHHG
(Wessel & Smith 1996), which is released under the LGPL; the derived grids are
redistributed under LGPL-3.0-or-later -- see `DATA_LICENSE`.

## Command line

```bash
landmetrics query --lat 25.0 --lon -80.0
landmetrics fraction --lat 25.0 --lon -80.0 --radius-km 100 300 600
landmetrics list
landmetrics info <path-to-grid.nc>
landmetrics fetch land_fraction --resolution-deg 0.01 --min-island-area-km2 0
```

## Citing

See `CITATION.cff`. A software DOI is minted by Zenodo on release; see the
project's GitHub releases page for the current record.
