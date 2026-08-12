"""Longitude normalization and axis-bracketing helpers.

All longitude arithmetic in this package is confined to this module -- no
other module may normalize, wrap, or index a longitude value on its own.
This is what keeps antimeridian/dateline handling correct in exactly one
place instead of several that can silently drift apart.

Two axis kinds are handled:

* non-periodic (latitude, or a ``land_fraction`` radius axis) -- clamped
  at both ends, no wraparound.
* periodic (longitude) -- the stored axis is a half-open ``[axis[0],
  axis[0] + period)`` range with no duplicated seam column (e.g. a 0.1
  degree grid stores -180.0 .. 179.9, never a trailing 180.0), so a query
  value falling in the implicit gap between ``axis[-1]`` and ``axis[0] +
  period`` (the antimeridian, for longitude) must wrap to index 0.

Every bracketing/index function here uses ``np.searchsorted``, never an
arithmetic index computed from ``axis[1] - axis[0]``: the stored axis is
float32 on disk and is *not* perfectly uniform -- accumulated float32
rounding drifts an arithmetic index by dozens of cells by the far end of a
36,000-point longitude axis, silently reading the wrong grid cell. See
``test_axes.py`` for a regression pinning this down.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_lon",
    "bracket_nonperiodic",
    "bracket_nonperiodic_many",
    "bracket_periodic",
    "bracket_periodic_many",
    "nearest_index",
    "nearest_index_many",
    "nearest_index_periodic",
    "nearest_index_periodic_many",
    "circular_column_range",
]


def normalize_lon(lon):
    """Normalize longitude into the half-open ``[-180, 180)`` domain.

    Works for any input convention (0-360, negative-360-accumulated
    track longitudes, etc.). NaN passes through as NaN. Handles the
    float-rounding edge case where ``(lon + 180) % 360`` lands on exactly
    ``360.0`` and would otherwise hand back ``+180.0`` instead of the
    grid's own ``-180.0`` representation of that same point.
    """
    lon_arr = np.asarray(lon, dtype="float64")
    scalar = lon_arr.ndim == 0
    # +/-inf input is a caller error, not a real longitude, but must not
    # raise or crash -- it should resolve to NaN like any other
    # out-of-domain value. inf % 360 is mathematically undefined and
    # numpy warns about it; that warning is expected here and nowhere
    # else in this function, so it's suppressed only around this line.
    with np.errstate(invalid="ignore"):
        out = ((lon_arr + 180.0) % 360.0) - 180.0
    out = np.where(out >= 180.0, out - 360.0, out)
    return float(out) if scalar else out


def bracket_nonperiodic(axis: np.ndarray, value: float) -> tuple[int, float]:
    """``(idx0, weight)`` bracketing *value* within an ascending,
    non-periodic *axis*, clamped to the axis's own range (no
    extrapolation past the edges) -- ``axis[idx0]``/``axis[idx0 + 1]``
    are the two points to interpolate between, weighted
    ``(1 - weight)``/``weight`` respectively."""
    i = int(np.searchsorted(axis, value)) - 1
    i = max(0, min(i, axis.size - 2))
    weight = (value - axis[i]) / (axis[i + 1] - axis[i])
    return i, float(weight)


def bracket_nonperiodic_many(axis: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`bracket_nonperiodic` -- ``(idx0, weight)`` arrays."""
    values = np.asarray(values, dtype="float64")
    idx = np.searchsorted(axis, values) - 1
    idx = np.clip(idx, 0, axis.size - 2)
    weight = (values - axis[idx]) / (axis[idx + 1] - axis[idx])
    return idx, weight


def bracket_periodic(axis: np.ndarray, value: float, period: float = 360.0) -> tuple[int, int, float]:
    """``(idx0, idx1, weight)`` bracketing *value* within an ascending,
    periodic *axis* (longitude) -- *idx1* wraps to ``0`` when *value*
    falls in the implicit gap between ``axis[-1]`` and ``axis[0] +
    period`` (the antimeridian seam)."""
    n = axis.size
    k = int(np.searchsorted(axis, value)) - 1
    if k < 0:
        left, right = axis[-1] - period, axis[0]
        return n - 1, 0, (value - left) / (right - left)
    if k >= n - 1:
        left, right = axis[-1], axis[0] + period
        return n - 1, 0, (value - left) / (right - left)
    return k, k + 1, (value - axis[k]) / (axis[k + 1] - axis[k])


def bracket_periodic_many(
    axis: np.ndarray,
    values: np.ndarray,
    period: float = 360.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized :func:`bracket_periodic` -- ``(idx0, idx1, weight)`` arrays."""
    values = np.asarray(values, dtype="float64")
    n = axis.size
    k = np.searchsorted(axis, values) - 1

    j0 = np.empty(values.shape, dtype=np.int64)
    j1 = np.empty(values.shape, dtype=np.int64)
    w = np.empty(values.shape, dtype=np.float64)

    low = k < 0
    high = k >= n - 1
    mid = ~low & ~high

    if np.any(low):
        j0[low] = n - 1
        j1[low] = 0
        left, right = axis[-1] - period, axis[0]
        w[low] = (values[low] - left) / (right - left)

    if np.any(high):
        j0[high] = n - 1
        j1[high] = 0
        left, right = axis[-1], axis[0] + period
        w[high] = (values[high] - left) / (right - left)

    if np.any(mid):
        kk = k[mid]
        j0[mid] = kk
        j1[mid] = kk + 1
        w[mid] = (values[mid] - axis[kk]) / (axis[kk + 1] - axis[kk])

    return j0, j1, w


def nearest_index(axis: np.ndarray, value: float) -> int:
    """Index of the axis cell nearest *value*, clamped to the axis range
    (non-periodic -- latitude)."""
    n = axis.size
    i = int(np.searchsorted(axis, value))
    if i <= 0:
        return 0
    if i >= n:
        return n - 1
    if (value - axis[i - 1]) <= (axis[i] - value):
        return i - 1
    return i


def nearest_index_many(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Vectorized :func:`nearest_index`."""
    values = np.asarray(values, dtype="float64")
    n = axis.size
    i = np.searchsorted(axis, values)
    i_clamped = np.clip(i, 1, n - 1)
    left = i_clamped - 1
    right = i_clamped
    pick_left = (values - axis[left]) <= (axis[right] - values)
    result = np.where(pick_left, left, right)
    result = np.where(i <= 0, 0, result)
    result = np.where(i >= n, n - 1, result)
    return result.astype(np.int64)


def nearest_index_periodic(axis: np.ndarray, value: float, period: float = 360.0) -> int:
    """Index of the axis cell nearest *value* (periodic -- longitude),
    correctly wrapping across the seam: on a 0.1 degree grid the nearest
    cell to ``lon = 179.99`` is index 0 (``-180.0``), not the last index
    (``179.9``), because the wrapped distance (0.01 degree) is shorter
    than the in-axis distance (0.09 degree).

    Built on :func:`bracket_periodic` rather than its own index search,
    so there is exactly one implementation of the wraparound logic to
    keep correct."""
    j0, j1, w = bracket_periodic(axis, value, period)
    return j0 if w < 0.5 else j1


def nearest_index_periodic_many(axis: np.ndarray, values: np.ndarray, period: float = 360.0) -> np.ndarray:
    """Vectorized :func:`nearest_index_periodic`."""
    j0, j1, w = bracket_periodic_many(axis, values, period)
    return np.where(w < 0.5, j0, j1)


def circular_column_range(cols: np.ndarray, n_lon: int) -> tuple[int, int]:
    """``(start, length)`` of the minimal circular arc of column indices
    covering every value in *cols*, on a circle of size *n_lon*.

    Used to bound a batched read's longitude span: a set of query points
    scattered near the antimeridian must not be covered by a naive
    ``[cols.min(), cols.max()]`` range, which would span nearly the whole
    globe. Instead this finds the largest gap between consecutive
    (circularly sorted) required columns and covers everything outside
    that gap -- the shortest arc containing every point.

    *length* can exceed *n_lon - start* when the arc wraps past the end
    of the axis; the caller is responsible for splitting a wrapping read
    into two slices (see :class:`~landmetrics._reader.WindowReader`).
    Returns ``(0, n_lon)`` when *cols* has no gap large enough to make
    wrapping worthwhile (points are spread across the whole circle)."""
    cols = np.unique(np.asarray(cols, dtype=np.int64))
    if cols.size == 0:
        raise ValueError("cols must not be empty")
    if cols.size == 1:
        return int(cols[0]), 1
    if cols.size == n_lon:
        return 0, n_lon

    diffs = np.diff(cols)
    wrap_gap = (cols[0] + n_lon) - cols[-1]
    all_gaps = np.concatenate([diffs, [wrap_gap]])
    max_gap_idx = int(np.argmax(all_gaps))

    if max_gap_idx == all_gaps.size - 1:
        # The largest gap is the wraparound gap itself: the arc does not
        # need to wrap at all.
        start = int(cols[0])
        end = int(cols[-1])
        return start, end - start + 1

    start = int(cols[max_gap_idx + 1])
    end_col = int(cols[max_gap_idx])
    length = (end_col - start) % n_lon + 1
    return start, length
