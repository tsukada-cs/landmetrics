"""Windowed netCDF reads with an LRU cache of decoded cells.

A grid file backing :class:`~landmetrics.query.DistanceToLand` or
:class:`~landmetrics.query.LandFraction` can be tens to hundreds of
megabytes on disk and tens of gigabytes once decoded to float64 -- the
0.01 degree / 6-radius ``land_fraction`` grid is 3.89 billion cells,
about 31 GB as float64. Loading it whole into memory per worker process
is the actual cause of a real multi-hundred-GB memory incident in the
pipeline this library was extracted from. Every read in this module goes
through :class:`WindowReader`, which reads only the handful of cells a
single interpolation needs, directly out of the still-closed-mostly-on-disk
netCDF variable.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np

from .exceptions import GridFormatError


class WindowReader:
    """Decoded-window reader over one netCDF variable, with a small LRU
    cache of recently-read 2x2 (lat, lon) windows.

    Not thread-safe: netCDF4/HDF5 access through a single ``Dataset`` is
    not safe to share across threads. Use one instance per thread, or
    serialize access with a lock. Safe across separate processes, each
    with its own instance -- but do not open a :class:`WindowReader`
    before forking and then use it from both parent and child.
    """

    def __init__(self, path: str | Path, varname: str, cache_size: int = 1024):
        self.path = Path(path)
        self._ds = nc.Dataset(str(self.path), "r")
        if varname not in self._ds.variables:
            self._ds.close()
            raise GridFormatError(f"{self.path}: missing '{varname}' variable")
        self._var = self._ds.variables[varname]
        # Applied by hand, never via set_auto_maskandscale(True) -- a
        # variable fetched directly off an already-open Dataset this way
        # does not auto-scale; trusting auto-scaling here has previously
        # produced values exactly 1000x too large (raw packed int16
        # returned unscaled). Always decode explicitly.
        self._var.set_auto_maskandscale(False)
        self.scale = float(getattr(self._var, "scale_factor", 1.0))
        self.offset = float(getattr(self._var, "add_offset", 0.0))
        self.n_lat, self.n_lon = self._var.shape[0], self._var.shape[1]

        self._cache: dict[tuple, np.ndarray] = {}
        self._cache_order: list[tuple] = []
        self._cache_size = cache_size

        self.reads = 0
        self.cells_read = 0

    def close(self) -> None:
        self._ds.close()

    def __enter__(self) -> WindowReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        return np.asarray(raw, dtype="float64") * self.scale + self.offset

    def read_2x2(self, i: int, j0: int, j1: int) -> np.ndarray:
        """Decoded ``(2, 2, ...)`` window at rows *i*, *i* + 1 and the two
        (possibly seam-wrapping, hence not necessarily adjacent) columns
        *j0*, *j1*. Trailing dims (e.g. ``radius``) are read in full.
        Cached by ``(i, j0, j1)``."""
        key = (i, j0, j1)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return cached

        if j1 == j0 + 1:
            raw = np.asarray(self._var[i : i + 2, j0 : j1 + 1, ...])
        else:
            left = np.asarray(self._var[i : i + 2, j0 : j0 + 1, ...])
            right = np.asarray(self._var[i : i + 2, j1 : j1 + 1, ...])
            raw = np.concatenate([left, right], axis=1)
        self.reads += 1
        self.cells_read += raw.size

        decoded = self._decode(raw)
        self._cache[key] = decoded
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_size:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        return decoded

    def read_block(self, row0: int, row1: int, col_start: int, col_count: int) -> np.ndarray:
        """Decoded ``(row1 - row0, col_count, ...)`` block, handling a
        column range that wraps past the end of the longitude axis (i.e.
        ``col_start + col_count > n_lon``) as a single logical read split
        into two netCDF slices, concatenated along the longitude axis.
        Not cached -- intended for batched multi-point reads, where the
        block itself is used once per call site rather than looked up
        repeatedly."""
        if col_start + col_count <= self.n_lon:
            raw = np.asarray(self._var[row0:row1, col_start : col_start + col_count, ...])
        else:
            first_count = self.n_lon - col_start
            second_count = col_count - first_count
            left = np.asarray(self._var[row0:row1, col_start : self.n_lon, ...])
            right = np.asarray(self._var[row0:row1, 0:second_count, ...])
            raw = np.concatenate([left, right], axis=1)
        self.reads += 1
        self.cells_read += raw.size
        return self._decode(raw)
