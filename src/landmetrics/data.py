"""Grid file registry, path resolution, and Zenodo-backed fetch.

Resolution order for a grid file (see :func:`grid_path`):

1. An explicit path, if the caller gave one.
2. ``$LANDMETRICS_DATA_DIR`` (colon-separated, like ``$PATH``) -- the
   first directory containing a matching filename wins. This is what
   makes the library usable against a local mirror (e.g. a NAS
   ``reference/`` directory) before or instead of the Zenodo deposit.
3. The bundled copy inside the installed package, for the two grids
   small enough to ship in the wheel.
4. The local cache directory (``$LANDMETRICS_CACHE_DIR``, else a
   platform-appropriate user cache directory).
5. If *download* is true (the default), fetched from Zenodo into the
   cache directory; otherwise :class:`~landmetrics.exceptions.GridNotFoundError`.

Nothing is ever downloaded on import -- only when a query actually needs
a grid that isn't already local.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import platformdirs

from .exceptions import GridNotFoundError

logger = logging.getLogger(__name__)

__all__ = [
    "GridSpec",
    "GRIDS",
    "grid_filename",
    "grid_path",
    "available_grids",
    "cache_dir",
]

# Not yet published -- filled in once the data deposit exists (see the
# package README's release notes). Left None rather than guessed: with no
# record id, grid_path(..., download=True) raises GridNotFoundError
# instead of attempting a request to a nonexistent record.
ZENODO_RECORD_ID: str | None = None
_ZENODO_BASE_URL = "https://zenodo.org/records/{record_id}/files/{filename}?download=1"

_BUNDLED_DIR = "data"


@dataclass(frozen=True)
class GridSpec:
    kind: str  # "distance_to_land" | "land_fraction"
    resolution_deg: float
    min_island_area_km2: float
    filename: str
    size_bytes: int
    sha256: str
    bundled: bool
    radii_km: tuple[float, ...] | None = None  # land_fraction only


def grid_filename(kind: str, resolution_deg: float, min_island_area_km2: float) -> str:
    """Canonical grid filename -- ``{kind}_{res:g}deg_gt{area:g}km2.nc``,
    matching every grid file this package or its private generator has
    ever produced (e.g. ``distance_to_land_0.05deg_gt1400km2.nc``)."""
    if kind not in ("distance_to_land", "land_fraction"):
        raise ValueError(f"unknown grid kind: {kind!r}")
    return f"{kind}_{resolution_deg:g}deg_gt{min_island_area_km2:g}km2.nc"


GRIDS: tuple[GridSpec, ...] = (
    GridSpec(
        "distance_to_land",
        0.1,
        0.0,
        "distance_to_land_0.1deg_gt0km2.nc",
        5319217,
        "e5c21ba6a46f0335fed2996cdb7dc54102749eb45dac2f1efc5d7a14ea4c4fca",
        bundled=True,
    ),
    GridSpec(
        "distance_to_land",
        0.1,
        1400.0,
        "distance_to_land_0.1deg_gt1400km2.nc",
        5071983,
        "d8ae4769b432f63ad0c4fbbc797481f91a7bf7f722e7f93ac107ab929ca77230",
        bundled=False,
    ),
    GridSpec(
        "distance_to_land",
        0.1,
        4748.0,
        "distance_to_land_0.1deg_gt4748km2.nc",
        5012957,
        "0196b734d7ba9c372df157ee989fa16a7c57cbf62e7cee98b6c54ce59a09af8a",
        bundled=False,
    ),
    GridSpec(
        "distance_to_land",
        0.05,
        0.0,
        "distance_to_land_0.05deg_gt0km2.nc",
        16098635,
        "80b22f10459de534be73b2d7615c1412327fc7465ba3d275e746dee070c08b14",
        bundled=False,
    ),
    GridSpec(
        "distance_to_land",
        0.05,
        1400.0,
        "distance_to_land_0.05deg_gt1400km2.nc",
        15057967,
        "4701f662280f3f7648eaddcc4ac362d25f093ea57802c1255b4887c2b6c96817",
        bundled=False,
    ),
    GridSpec(
        "distance_to_land",
        0.05,
        4748.0,
        "distance_to_land_0.05deg_gt4748km2.nc",
        14809273,
        "7eb4cd4343c1b083555e5f86c106182e929a21b197312971651912e48cd8e9b6",
        bundled=False,
    ),
    GridSpec(
        "distance_to_land",
        0.01,
        0.0,
        "distance_to_land_0.01deg_gt0km2.nc",
        176138587,
        "e82e745d5983a1e11f70ee24768c5da37639474cbf6816292f0e49c0b3887c7b",
        bundled=False,
    ),
    GridSpec(
        "distance_to_land",
        0.01,
        1400.0,
        "distance_to_land_0.01deg_gt1400km2.nc",
        159606775,
        "710a875c3e9a7f83e5cacd6aa50a6a8e26b86ec8705fac4f97aee48c3028099a",
        bundled=False,
    ),
    GridSpec(
        "distance_to_land",
        0.01,
        4748.0,
        "distance_to_land_0.01deg_gt4748km2.nc",
        156370916,
        "9c93eacca960cf5194669fc873f9bdde1d4bd02f7099e89125239205fbda83d4",
        bundled=False,
    ),
    GridSpec(
        "land_fraction",
        0.1,
        0.0,
        "land_fraction_0.1deg_gt0km2.nc",
        11911132,
        "395a6da6ccbd61120eca788cd5ae520f09412ec3c667bde0641808b9945626f2",
        bundled=True,
        radii_km=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    ),
    GridSpec(
        "land_fraction",
        0.1,
        1400.0,
        "land_fraction_0.1deg_gt1400km2.nc",
        11669582,
        "8f75a031f69e15e754590f3994c39d7e50215c39c4c8a966e1300c11a609ad00",
        bundled=False,
        radii_km=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    ),
    GridSpec(
        "land_fraction",
        0.05,
        0.0,
        "land_fraction_0.05deg_gt0km2.nc",
        20900653,
        "ecf4a2c34bcdef5fe4d016bb359b202c6a1a0ace8c8156e14ca679762c042bb6",
        bundled=False,
        radii_km=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    ),
    GridSpec(
        "land_fraction",
        0.05,
        1400.0,
        "land_fraction_0.05deg_gt1400km2.nc",
        20350940,
        "b9423e1850295c6d8c5c748782a578b153268d5ef523ac1a2caa990f54a71d63",
        bundled=False,
        radii_km=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    ),
    GridSpec(
        "land_fraction",
        0.01,
        0.0,
        "land_fraction_0.01deg_gt0km2.nc",
        167620201,
        "ec5d24169d2d7de65b9d094cec666640975ef2f7b7a183e53ec072f67eeb0b4a",
        bundled=False,
        radii_km=(100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    ),
)

_GRIDS_BY_FILENAME = {g.filename: g for g in GRIDS}


def available_grids() -> list[GridSpec]:
    """Every grid known to this version of the package (bundled and
    Zenodo-hosted alike)."""
    return list(GRIDS)


def cache_dir() -> Path:
    """Local cache directory for downloaded grids -- ``$LANDMETRICS_CACHE_DIR``
    if set, else a platform-appropriate per-user cache directory."""
    override = os.environ.get("LANDMETRICS_CACHE_DIR")
    if override:
        path = Path(override)
    else:
        path = Path(platformdirs.user_cache_dir("landmetrics"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _data_dir_search_path() -> list[Path]:
    raw = os.environ.get("LANDMETRICS_DATA_DIR", "")
    return [Path(p) for p in raw.split(os.pathsep) if p]


def _bundled_path(filename: str) -> Path | None:
    try:
        candidate = resources.files("landmetrics") / _BUNDLED_DIR / filename
    except ModuleNotFoundError:  # pragma: no cover - defensive
        return None
    if candidate.is_file():
        return Path(str(candidate))
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(spec: GridSpec, dest: Path) -> None:
    if ZENODO_RECORD_ID is None:
        raise GridNotFoundError(
            f"{spec.filename} is not available locally and no Zenodo record is "
            f"configured for this landmetrics release yet. Set LANDMETRICS_DATA_DIR "
            f"to a directory containing this file, or pass an explicit path=.",
        )
    url = _ZENODO_BASE_URL.format(record_id=ZENODO_RECORD_ID, filename=spec.filename)
    tmp_path = dest.with_name(f"{dest.name}.part.{os.getpid()}")
    logger.info("downloading %s from Zenodo record %s ...", spec.filename, ZENODO_RECORD_ID)
    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        digest = _sha256(tmp_path)
        if digest != spec.sha256:
            raise GridNotFoundError(
                f"downloaded {spec.filename} failed checksum verification (expected {spec.sha256}, got {digest})",
            )
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def grid_path(
    kind: str,
    resolution_deg: float,
    min_island_area_km2: float,
    *,
    download: bool = True,
) -> Path:
    """Resolve a grid file's local path, searching (in order) an explicit
    ``$LANDMETRICS_DATA_DIR``, the bundled package data, and the local
    cache -- downloading from Zenodo into the cache as a last resort if
    *download* is true. Raises :class:`~landmetrics.exceptions.GridNotFoundError`
    if the grid cannot be found or (when *download* is true) fetched."""
    filename = grid_filename(kind, resolution_deg, min_island_area_km2)
    spec = _GRIDS_BY_FILENAME.get(filename)

    searched: list[str] = []
    for directory in _data_dir_search_path():
        candidate = directory / filename
        searched.append(str(candidate))
        if candidate.is_file():
            return candidate

    bundled = _bundled_path(filename)
    if bundled is not None:
        searched.append(str(bundled))
        return bundled

    cached = cache_dir() / filename
    searched.append(str(cached))
    if cached.is_file():
        return cached

    if spec is None:
        raise GridNotFoundError(
            f"{filename} is not a known landmetrics grid and was not found in any searched location: {searched}",
        )

    if not download:
        raise GridNotFoundError(
            f"{filename} not found locally (searched: {searched}) and download=False; "
            f"pass download=True, set LANDMETRICS_DATA_DIR, or run "
            f"'landmetrics fetch {kind} --resolution-deg {resolution_deg} "
            f"--min-island-area-km2 {min_island_area_km2}'",
        )

    _download(spec, cached)
    return cached
