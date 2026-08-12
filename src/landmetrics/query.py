"""Query wrappers over precomputed distance-to-land / land-fraction grid
files.

Both classes read only the handful of cells a single interpolation needs,
directly out of the netCDF file (see :class:`~landmetrics._reader.WindowReader`)
-- never the whole grid. A scalar query reads a 2x2 window. An array query
reads one bounding block when the points are compact enough to fit a fixed
cell budget, and otherwise splits the points (sorted by row) into
row-compact tiles, each read as its own block -- see :data:`_MAX_WINDOW_CELLS`.

All longitude handling (normalization, seam wraparound, the minimal
circular column range for a batched read) goes through
:mod:`landmetrics._axes` -- nothing in this module does its own longitude
arithmetic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._axes import (
    bracket_nonperiodic_many,
    bracket_periodic_many,
    circular_column_range,
    nearest_index_many,
    nearest_index_periodic_many,
    normalize_lon,
)
from ._constants import DEFAULT_MIN_ISLAND_AREA_KM2, DEFAULT_RESOLUTION_DEG
from ._reader import WindowReader
from .exceptions import GridFormatError

__all__ = [
    "DistanceToLand",
    "LandFraction",
    "distance_to_land",
    "land_fraction",
    "is_land",
    "is_ocean",
    "open_distance_to_land",
    "open_land_fraction",
]

# Batched-read cell budget -- about 32 MB as float64. No query, scalar or
# array, ever reads more than this many decoded cells in one netCDF call;
# an array whose bounding block would exceed it is split into row-compact
# tiles instead (see DistanceToLand._tiled_interp / LandFraction._tiled_interp).
_MAX_WINDOW_CELLS = 4_000_000

# Below this many points, per-point 2x2 reads through the window cache are
# used instead of a single batched block read -- the batching machinery's
# own bookkeeping (unique-column search, circular-range computation) costs
# more than it saves at this size, and repeated nearby scalar queries
# benefit from the cache that only the per-point path uses.
_SCALAR_THRESHOLD = 64


def _validate_lat_lon(lat: np.ndarray, lon: np.ndarray, source: str) -> None:
    if lat.ndim != 1 or lon.ndim != 1:
        raise GridFormatError(f"{source}: 'lat'/'lon' must be 1-D")
    if lat.size < 2 or lon.size < 2:
        raise GridFormatError(f"{source}: 'lat'/'lon' must have at least 2 points")
    if not np.all(np.diff(lat) > 0):
        raise GridFormatError(f"{source}: 'lat' must be strictly ascending")
    if not np.all(np.diff(lon) > 0):
        raise GridFormatError(f"{source}: 'lon' must be strictly ascending")
    lon_span = (lon[-1] - lon[0]) + (lon[1] - lon[0])
    if not (300.0 <= lon_span <= 360.0001):
        raise GridFormatError(
            f"{source}: 'lon' does not look like a half-open [-180, 180) style "
            f"360-degree axis (implied span {lon_span:.4f} degrees)",
        )


class DistanceToLand:
    """Query wrapper around a distance-to-land grid file (signed
    great-circle distance to the nearest coastline, km, negative over
    land). See ``docs/grid_format.md`` for the required file schema.

    Not thread-safe (holds an open netCDF handle) -- use one instance per
    thread, or serialize access with a lock.
    """

    def __init__(self, path: str | Path):
        self._reader = WindowReader(path, "distance_to_land")
        ds = self._reader._ds
        if "lat" not in ds.variables or "lon" not in ds.variables:
            raise GridFormatError(f"{path}: missing 'lat'/'lon' coordinate variables")
        self._lat = np.asarray(ds.variables["lat"][:], dtype="float64")
        self._lon = np.asarray(ds.variables["lon"][:], dtype="float64")
        _validate_lat_lon(self._lat, self._lon, str(path))

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> DistanceToLand:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- distance query ---------------------------------------------------

    def _query_one(self, lat: float, lon_norm: float) -> float:
        if not (self._lat[0] <= lat <= self._lat[-1]):
            return float("nan")
        i = int(np.searchsorted(self._lat, lat)) - 1
        i = max(0, min(i, self._lat.size - 2))
        wl = (lat - self._lat[i]) / (self._lat[i + 1] - self._lat[i])
        j0, j1, wo = self._bracket_lon_one(lon_norm)
        blk = self._reader.read_2x2(i, j0, j1)
        top = blk[0, 0] * (1 - wo) + blk[0, 1] * wo
        bot = blk[1, 0] * (1 - wo) + blk[1, 1] * wo
        return float(top * (1 - wl) + bot * wl)

    def _bracket_lon_one(self, lon_norm: float) -> tuple[int, int, float]:
        n = self._lon.size
        k = int(np.searchsorted(self._lon, lon_norm)) - 1
        if k < 0:
            left, right = self._lon[-1] - 360.0, self._lon[0]
            return n - 1, 0, (lon_norm - left) / (right - left)
        if k >= n - 1:
            left, right = self._lon[-1], self._lon[0] + 360.0
            return n - 1, 0, (lon_norm - left) / (right - left)
        return k, k + 1, (lon_norm - self._lon[k]) / (self._lon[k + 1] - self._lon[k])

    def query(self, lat, lon):
        """Bilinearly-interpolated distance to land (km, negative over
        land) at arbitrary (*lat*, *lon*) -- scalar or array-like,
        broadcast together. *lon* is normalized into this grid's own
        [-180, 180) domain first, so any input convention (e.g. 0-360)
        resolves correctly, including across the antimeridian. *lat*
        outside [-90, 90], or NaN input, returns NaN."""
        lat_arr = np.asarray(lat, dtype="float64")
        lon_arr = np.asarray(lon, dtype="float64")
        lon_norm = np.asarray(normalize_lon(lon_arr), dtype="float64")
        lat_b, lon_b = np.broadcast_arrays(lat_arr, lon_norm)
        scalar = lat_b.ndim == 0
        flat_lat = lat_b.reshape(-1)
        flat_lon = lon_b.reshape(-1)
        n = flat_lat.size
        if n <= _SCALAR_THRESHOLD:
            out = np.fromiter(
                (self._query_one(float(la), float(lo)) for la, lo in zip(flat_lat, flat_lon, strict=True)),
                dtype="float64",
                count=n,
            )
        else:
            out = self._query_many(flat_lat, flat_lon)
        out = out.reshape(lat_b.shape)
        return float(out) if scalar else out

    def _query_many(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        n = lat.size
        valid = ~np.isnan(lat) & ~np.isnan(lon) & (lat >= self._lat[0]) & (lat <= self._lat[-1])
        out = np.full(n, np.nan, dtype="float64")
        if not np.any(valid):
            return out
        vlat, vlon = lat[valid], lon[valid]
        i, wl = bracket_nonperiodic_many(self._lat, vlat)
        j0, j1, wo = bracket_periodic_many(self._lon, vlon)
        n_lon = self._lon.size
        if self._fits_budget(i, j0, j1, n_lon):
            result = self._interp_block(i, wl, j0, j1, wo, n_lon)
        else:
            result = self._tiled_interp(i, wl, j0, j1, wo, n_lon)
        out[valid] = result
        return out

    @staticmethod
    def _fits_budget(i_chunk, j0_chunk, j1_chunk, n_lon) -> bool:
        row0, row1 = int(i_chunk.min()), int(i_chunk.max()) + 2
        cols = np.unique(np.concatenate([j0_chunk, j1_chunk]))
        _, col_count = circular_column_range(cols, n_lon)
        return (row1 - row0) * col_count <= _MAX_WINDOW_CELLS

    def _interp_block(self, i_chunk, wl_chunk, j0_chunk, j1_chunk, wo_chunk, n_lon) -> np.ndarray:
        row0, row1 = int(i_chunk.min()), int(i_chunk.max()) + 2
        cols = np.unique(np.concatenate([j0_chunk, j1_chunk]))
        col_start, col_count = circular_column_range(cols, n_lon)
        block = self._reader.read_block(row0, row1, col_start, col_count)
        i_b = i_chunk - row0
        j0_b = (j0_chunk - col_start) % n_lon
        j1_b = (j1_chunk - col_start) % n_lon
        v00, v01 = block[i_b, j0_b], block[i_b, j1_b]
        v10, v11 = block[i_b + 1, j0_b], block[i_b + 1, j1_b]
        top = v00 * (1 - wo_chunk) + v01 * wo_chunk
        bot = v10 * (1 - wo_chunk) + v11 * wo_chunk
        return top * (1 - wl_chunk) + bot * wl_chunk

    def _tiled_interp(self, i, wl, j0, j1, wo, n_lon) -> np.ndarray:
        n = i.size
        order = np.argsort(i, kind="stable")
        i_s, wl_s, j0_s, j1_s, wo_s = i[order], wl[order], j0[order], j1[order], wo[order]
        result_sorted = np.empty(n, dtype="float64")
        self._fill_tile(i_s, wl_s, j0_s, j1_s, wo_s, 0, n, n_lon, result_sorted)
        result = np.empty(n, dtype="float64")
        result[order] = result_sorted
        return result

    def _fill_tile(self, i_s, wl_s, j0_s, j1_s, wo_s, pos, end, n_lon, out) -> None:
        if end - pos <= 1 or self._fits_budget(i_s[pos:end], j0_s[pos:end], j1_s[pos:end], n_lon):
            out[pos:end] = self._interp_block(
                i_s[pos:end],
                wl_s[pos:end],
                j0_s[pos:end],
                j1_s[pos:end],
                wo_s[pos:end],
                n_lon,
            )
            return
        mid = (pos + end) // 2
        self._fill_tile(i_s, wl_s, j0_s, j1_s, wo_s, pos, mid, n_lon, out)
        self._fill_tile(i_s, wl_s, j0_s, j1_s, wo_s, mid, end, n_lon, out)

    # -- nearest-cell land/ocean test --------------------------------------

    def is_land(self, lat, lon):
        """True where the nearest grid cell is land.

        Reads the single nearest cell's sign rather than interpolating: a
        bilinear blend across a coastline can carry a land point within
        one cell of the shore to a positive (ocean) value -- see
        :meth:`query`. Resolution-limited: a query resolves to the
        nearest cell center, so accuracy is about half a cell (roughly
        5.5 km at 0.05 degree spacing, at the equator). Out-of-range
        latitude, or NaN input, yields False."""
        values, valid, shape = self._nearest_signed_values(lat, lon)
        result = np.zeros(values.shape, dtype=bool)
        result[valid] = values[valid] < 0
        result = result.reshape(shape)
        return bool(result) if result.ndim == 0 else result

    def is_ocean(self, lat, lon):
        """True where the nearest grid cell is ocean. Not a plain negation
        of :meth:`is_land`: an out-of-range or NaN query returns False
        from both, since neither claim holds when there is no cell."""
        values, valid, shape = self._nearest_signed_values(lat, lon)
        result = np.zeros(values.shape, dtype=bool)
        result[valid] = values[valid] >= 0
        result = result.reshape(shape)
        return bool(result) if result.ndim == 0 else result

    def _nearest_signed_values(self, lat, lon) -> tuple[np.ndarray, np.ndarray, tuple]:
        lat_arr = np.asarray(lat, dtype="float64")
        lon_arr = np.asarray(lon, dtype="float64")
        lon_norm = np.asarray(normalize_lon(lon_arr), dtype="float64")
        lat_b, lon_b = np.broadcast_arrays(lat_arr, lon_norm)
        flat_lat = lat_b.reshape(-1)
        flat_lon = lon_b.reshape(-1)
        valid = ~np.isnan(flat_lat) & ~np.isnan(flat_lon) & (flat_lat >= self._lat[0]) & (flat_lat <= self._lat[-1])
        values = np.full(flat_lat.size, np.nan, dtype="float64")
        if np.any(valid):
            n_lon = self._lon.size
            i = nearest_index_many(self._lat, flat_lat[valid])
            j = nearest_index_periodic_many(self._lon, flat_lon[valid], 360.0)
            values[valid] = self._nearest_cell_values(i, j, n_lon)
        return values, valid, lat_b.shape

    def _nearest_cell_values(self, i: np.ndarray, j: np.ndarray, n_lon: int) -> np.ndarray:
        if self._fits_nearest_budget(i, j, n_lon):
            return self._read_nearest_block(i, j, n_lon)
        n = i.size
        order = np.argsort(i, kind="stable")
        i_s, j_s = i[order], j[order]
        result_sorted = np.empty(n, dtype="float64")
        self._fill_nearest_tile(i_s, j_s, 0, n, n_lon, result_sorted)
        result = np.empty(n, dtype="float64")
        result[order] = result_sorted
        return result

    def _read_nearest_block(self, i_chunk, j_chunk, n_lon) -> np.ndarray:
        row0, row1 = int(i_chunk.min()), int(i_chunk.max()) + 1
        col_start, col_count = circular_column_range(np.unique(j_chunk), n_lon)
        block = self._reader.read_block(row0, row1, col_start, col_count)
        i_b = i_chunk - row0
        j_b = (j_chunk - col_start) % n_lon
        return block[i_b, j_b]

    @staticmethod
    def _fits_nearest_budget(i_chunk, j_chunk, n_lon) -> bool:
        row0, row1 = int(i_chunk.min()), int(i_chunk.max()) + 1
        _, col_count = circular_column_range(np.unique(j_chunk), n_lon)
        return (row1 - row0) * col_count <= _MAX_WINDOW_CELLS

    def _fill_nearest_tile(self, i_s, j_s, pos, end, n_lon, out) -> None:
        if end - pos <= 1 or self._fits_nearest_budget(i_s[pos:end], j_s[pos:end], n_lon):
            out[pos:end] = self._read_nearest_block(i_s[pos:end], j_s[pos:end], n_lon)
            return
        mid = (pos + end) // 2
        self._fill_nearest_tile(i_s, j_s, pos, mid, n_lon, out)
        self._fill_nearest_tile(i_s, j_s, mid, end, n_lon, out)


class LandFraction:
    """Query wrapper around a land-fraction-within-radius grid file
    (``land_fraction`` values in ``[0, 1]``, with a ``radius`` (km)
    dimension). The trilinear-interpolation analogue of
    :class:`DistanceToLand`, extended with a third query axis. See
    ``docs/grid_format.md`` for the required file schema.

    Not thread-safe (holds an open netCDF handle) -- use one instance per
    thread, or serialize access with a lock.
    """

    def __init__(self, path: str | Path):
        self._reader = WindowReader(path, "land_fraction")
        ds = self._reader._ds
        for name in ("lat", "lon", "radius"):
            if name not in ds.variables:
                raise GridFormatError(f"{path}: missing '{name}' coordinate variable")
        self._lat = np.asarray(ds.variables["lat"][:], dtype="float64")
        self._lon = np.asarray(ds.variables["lon"][:], dtype="float64")
        self._radius = np.asarray(ds.variables["radius"][:], dtype="float64")
        _validate_lat_lon(self._lat, self._lon, str(path))
        if self._radius.ndim != 1 or self._radius.size < 1:
            raise GridFormatError(f"{path}: 'radius' must be a non-empty 1-D array")
        if self._radius.size > 1 and not np.all(np.diff(self._radius) > 0):
            raise GridFormatError(f"{path}: 'radius' must be strictly ascending")

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> LandFraction:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _bracket_lon_one(self, lon_norm: float) -> tuple[int, int, float]:
        n = self._lon.size
        k = int(np.searchsorted(self._lon, lon_norm)) - 1
        if k < 0:
            left, right = self._lon[-1] - 360.0, self._lon[0]
            return n - 1, 0, (lon_norm - left) / (right - left)
        if k >= n - 1:
            left, right = self._lon[-1], self._lon[0] + 360.0
            return n - 1, 0, (lon_norm - left) / (right - left)
        return k, k + 1, (lon_norm - self._lon[k]) / (self._lon[k + 1] - self._lon[k])

    def _query_one(self, lat: float, lon_norm: float, radius: np.ndarray) -> np.ndarray:
        if not (self._lat[0] <= lat <= self._lat[-1]):
            return np.full(radius.shape, np.nan)
        i = int(np.searchsorted(self._lat, lat)) - 1
        i = max(0, min(i, self._lat.size - 2))
        wl = (lat - self._lat[i]) / (self._lat[i + 1] - self._lat[i])
        j0, j1, wo = self._bracket_lon_one(lon_norm)
        blk = self._reader.read_2x2(i, j0, j1)  # (2, 2, n_radius)
        top = blk[0, 0, :] * (1 - wo) + blk[0, 1, :] * wo
        bot = blk[1, 0, :] * (1 - wo) + blk[1, 1, :] * wo
        at_stored_radii = top * (1 - wl) + bot * wl
        # Linear along radius, exact passthrough when radius matches a
        # stored value; NaN outside [radius.min(), radius.max()] (no
        # extrapolation). Built on bracket_nonperiodic_many rather than
        # np.interp so this (one point, many radii) path and
        # _interp_along_radius's (many points, one radius each) batched
        # path compute the identical formula in the identical order --
        # np.interp is mathematically equivalent but not bit-identical to
        # the hand-rolled weight formula the batched path needs (it has
        # no per-point vectorized equivalent), and query() must return the
        # same value regardless of which path a given call size takes.
        idx, w = bracket_nonperiodic_many(self._radius, radius)
        val = at_stored_radii[idx] * (1 - w) + at_stored_radii[idx + 1] * w
        out_of_range = (radius < self._radius[0]) | (radius > self._radius[-1])
        return np.where(out_of_range, np.nan, val)

    def query(self, lat, lon, radius):
        """Trilinearly-interpolated land fraction ``[0, 1]`` at arbitrary
        (*lat*, *lon*, *radius* km) -- scalar or array-like, broadcast
        together. *lon* is normalized into this grid's own [-180, 180)
        domain first, so any input convention resolves correctly,
        including across the antimeridian. *radius* outside the stored
        grid's own min/max returns NaN rather than extrapolating; *lat*
        outside [-90, 90], or NaN input, likewise."""
        lat_arr = np.asarray(lat, dtype="float64")
        lon_arr = np.asarray(lon, dtype="float64")
        radius_arr = np.asarray(radius, dtype="float64")
        lon_norm = np.asarray(normalize_lon(lon_arr), dtype="float64")

        if lat_arr.ndim == 0 and lon_arr.ndim == 0:
            # The common case: one (lat, lon) grid window/block read
            # serves every requested radius at once.
            result = self._query_one(float(lat_arr), float(lon_norm), np.atleast_1d(radius_arr))
            return float(result[0]) if radius_arr.ndim == 0 else result.reshape(radius_arr.shape)

        lat_b, lon_b, radius_b = np.broadcast_arrays(lat_arr, lon_norm, radius_arr)
        flat_lat, flat_lon, flat_radius = lat_b.reshape(-1), lon_b.reshape(-1), radius_b.reshape(-1)
        n = flat_lat.size
        if n <= _SCALAR_THRESHOLD:
            out = np.empty(n, dtype="float64")
            for k in range(n):
                out[k] = self._query_one(float(flat_lat[k]), float(flat_lon[k]), flat_radius[k : k + 1])[0]
        else:
            out = self._query_many(flat_lat, flat_lon, flat_radius)
        return out.reshape(lat_b.shape)

    def _query_many(self, lat: np.ndarray, lon: np.ndarray, radius: np.ndarray) -> np.ndarray:
        n = lat.size
        valid = ~np.isnan(lat) & ~np.isnan(lon) & (lat >= self._lat[0]) & (lat <= self._lat[-1])
        out = np.full(n, np.nan, dtype="float64")
        if not np.any(valid):
            return out
        vlat, vlon, vradius = lat[valid], lon[valid], radius[valid]
        i, wl = bracket_nonperiodic_many(self._lat, vlat)
        j0, j1, wo = bracket_periodic_many(self._lon, vlon)
        n_lon = self._lon.size
        n_radius = self._radius.size
        if self._fits_budget(i, j0, j1, n_lon, n_radius):
            at_stored = self._interp_block(i, wl, j0, j1, wo, n_lon)
        else:
            at_stored = self._tiled_interp(i, wl, j0, j1, wo, n_lon)
        out[valid] = self._interp_along_radius(at_stored, vradius)
        return out

    def _interp_along_radius(self, at_stored: np.ndarray, radius_query: np.ndarray) -> np.ndarray:
        idx, w = bracket_nonperiodic_many(self._radius, radius_query)
        rows = np.arange(at_stored.shape[0])
        val = at_stored[rows, idx] * (1 - w) + at_stored[rows, idx + 1] * w
        out_of_range = (radius_query < self._radius[0]) | (radius_query > self._radius[-1])
        return np.where(out_of_range, np.nan, val)

    def _fits_budget(self, i_chunk, j0_chunk, j1_chunk, n_lon, n_radius) -> bool:
        row0, row1 = int(i_chunk.min()), int(i_chunk.max()) + 2
        cols = np.unique(np.concatenate([j0_chunk, j1_chunk]))
        _, col_count = circular_column_range(cols, n_lon)
        return (row1 - row0) * col_count * n_radius <= _MAX_WINDOW_CELLS

    def _interp_block(self, i_chunk, wl_chunk, j0_chunk, j1_chunk, wo_chunk, n_lon) -> np.ndarray:
        row0, row1 = int(i_chunk.min()), int(i_chunk.max()) + 2
        cols = np.unique(np.concatenate([j0_chunk, j1_chunk]))
        col_start, col_count = circular_column_range(cols, n_lon)
        block = self._reader.read_block(row0, row1, col_start, col_count)  # (rows, cols, n_radius)
        i_b = i_chunk - row0
        j0_b = (j0_chunk - col_start) % n_lon
        j1_b = (j1_chunk - col_start) % n_lon
        v00, v01 = block[i_b, j0_b, :], block[i_b, j1_b, :]
        v10, v11 = block[i_b + 1, j0_b, :], block[i_b + 1, j1_b, :]
        top = v00 * (1 - wo_chunk)[:, None] + v01 * wo_chunk[:, None]
        bot = v10 * (1 - wo_chunk)[:, None] + v11 * wo_chunk[:, None]
        return top * (1 - wl_chunk)[:, None] + bot * wl_chunk[:, None]

    def _tiled_interp(self, i, wl, j0, j1, wo, n_lon) -> np.ndarray:
        n = i.size
        order = np.argsort(i, kind="stable")
        i_s, wl_s, j0_s, j1_s, wo_s = i[order], wl[order], j0[order], j1[order], wo[order]
        result_sorted = np.empty((n, self._radius.size), dtype="float64")
        self._fill_tile(i_s, wl_s, j0_s, j1_s, wo_s, 0, n, n_lon, result_sorted)
        result = np.empty_like(result_sorted)
        result[order] = result_sorted
        return result

    def _fill_tile(self, i_s, wl_s, j0_s, j1_s, wo_s, pos, end, n_lon, out) -> None:
        n_radius = self._radius.size
        if end - pos <= 1 or self._fits_budget(i_s[pos:end], j0_s[pos:end], j1_s[pos:end], n_lon, n_radius):
            out[pos:end] = self._interp_block(
                i_s[pos:end],
                wl_s[pos:end],
                j0_s[pos:end],
                j1_s[pos:end],
                wo_s[pos:end],
                n_lon,
            )
            return
        mid = (pos + end) // 2
        self._fill_tile(i_s, wl_s, j0_s, j1_s, wo_s, pos, mid, n_lon, out)
        self._fill_tile(i_s, wl_s, j0_s, j1_s, wo_s, mid, end, n_lon, out)


_distance_instances: dict[str, DistanceToLand] = {}
_fraction_instances: dict[str, LandFraction] = {}


def _resolve_path(kind: str, path, resolution_deg, min_island_area_km2) -> str:
    if path is not None:
        if resolution_deg is not None or min_island_area_km2 is not None:
            raise ValueError("pass either 'path' or 'resolution_deg'/'min_island_area_km2', not both")
        return str(path)
    from . import data

    res = DEFAULT_RESOLUTION_DEG if resolution_deg is None else resolution_deg
    area = DEFAULT_MIN_ISLAND_AREA_KM2 if min_island_area_km2 is None else min_island_area_km2
    return str(data.grid_path(kind, res, area))


def _get_distance_instance(path, resolution_deg, min_island_area_km2) -> DistanceToLand:
    key = _resolve_path("distance_to_land", path, resolution_deg, min_island_area_km2)
    inst = _distance_instances.get(key)
    if inst is None:
        inst = DistanceToLand(key)
        _distance_instances[key] = inst
    return inst


def _get_fraction_instance(path, resolution_deg, min_island_area_km2) -> LandFraction:
    key = _resolve_path("land_fraction", path, resolution_deg, min_island_area_km2)
    inst = _fraction_instances.get(key)
    if inst is None:
        inst = LandFraction(key)
        _fraction_instances[key] = inst
    return inst


def distance_to_land(lat, lon, path=None, *, resolution_deg=None, min_island_area_km2=None):
    """Convenience wrapper: loads (and caches, keyed by resolved path) a
    default :class:`DistanceToLand` instance the first time it's called
    for a given grid, then delegates to :meth:`DistanceToLand.query`. For
    repeated calls in a hot loop, prefer instantiating
    :class:`DistanceToLand` directly once and reusing it."""
    return _get_distance_instance(path, resolution_deg, min_island_area_km2).query(lat, lon)


def land_fraction(lat, lon, radius_km, path=None, *, resolution_deg=None, min_island_area_km2=None):
    """Convenience wrapper: loads (and caches, keyed by resolved path) a
    default :class:`LandFraction` instance the first time it's called for
    a given grid, then delegates to :meth:`LandFraction.query`. For
    repeated calls in a hot loop, prefer instantiating
    :class:`LandFraction` directly once and reusing it."""
    return _get_fraction_instance(path, resolution_deg, min_island_area_km2).query(lat, lon, radius_km)


def is_land(lat, lon, path=None, *, resolution_deg=None, min_island_area_km2=None):
    """Convenience wrapper around :meth:`DistanceToLand.is_land`, using a
    cached default instance -- see :func:`distance_to_land`."""
    return _get_distance_instance(path, resolution_deg, min_island_area_km2).is_land(lat, lon)


def is_ocean(lat, lon, path=None, *, resolution_deg=None, min_island_area_km2=None):
    """Convenience wrapper around :meth:`DistanceToLand.is_ocean`, using a
    cached default instance -- see :func:`distance_to_land`."""
    return _get_distance_instance(path, resolution_deg, min_island_area_km2).is_ocean(lat, lon)


def open_distance_to_land(*, resolution_deg=None, min_island_area_km2=None, download=True) -> DistanceToLand:
    """Open a fresh, non-cached :class:`DistanceToLand` over the grid
    matching *resolution_deg*/*min_island_area_km2* (library defaults if
    omitted), resolving/downloading it via :func:`landmetrics.data.grid_path`."""
    from . import data

    res = DEFAULT_RESOLUTION_DEG if resolution_deg is None else resolution_deg
    area = DEFAULT_MIN_ISLAND_AREA_KM2 if min_island_area_km2 is None else min_island_area_km2
    return DistanceToLand(data.grid_path("distance_to_land", res, area, download=download))


def open_land_fraction(*, resolution_deg=None, min_island_area_km2=None, download=True) -> LandFraction:
    """Open a fresh, non-cached :class:`LandFraction` over the grid
    matching *resolution_deg*/*min_island_area_km2* (library defaults if
    omitted), resolving/downloading it via :func:`landmetrics.data.grid_path`."""
    from . import data

    res = DEFAULT_RESOLUTION_DEG if resolution_deg is None else resolution_deg
    area = DEFAULT_MIN_ISLAND_AREA_KM2 if min_island_area_km2 is None else min_island_area_km2
    return LandFraction(data.grid_path("land_fraction", res, area, download=download))
