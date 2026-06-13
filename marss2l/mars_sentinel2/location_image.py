"""Slim location/image data classes and input protocols for background selection.

These mirror marsml's ``EmissionsLocation`` and ``MarsLocationImage`` field-for-field
(same names, ``uuid.UUID`` ids, ``GeoTensor`` rasters) so that:

* the GEE-only :class:`~marss2l.mars_sentinel2.background.BackgroundImageSelector`
  can operate on them, and
* marsml's ``MarsLocationImage`` satisfies :class:`LocationImageProtocol` unmodified.

No heavy dependencies are imported at module load; GEE helpers are imported lazily
inside the constructors that need them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid importing heavy/geo deps at module load
    from georeader.geotensor import GeoTensor
    from rasterio.transform import Affine
    from shapely.geometry import Polygon


# Deterministic-id namespace so the same GEE scene always maps to the same id_loc_image.
_NAMESPACE_S2L = uuid.uuid5(uuid.NAMESPACE_URL, "marss2l/background")


@runtime_checkable
class LocationProtocol(Protocol):
    """Minimal location contract (satisfied by marsml ``EmissionsLocation``)."""

    id_location: uuid.UUID
    location_name: str
    geometry: "Polygon"
    offshore: bool


@runtime_checkable
class LocationImageProtocol(Protocol):
    """Input contract for :class:`BackgroundImageSelector`.

    Every member is an attribute marsml's ``MarsLocationImage`` already exposes, so
    that class satisfies this protocol without modification.
    """

    id_loc_image: uuid.UUID
    id_location: uuid.UUID
    location: LocationProtocol
    tile: str
    satellite: str
    tile_date: datetime
    percentage_clear: float
    observability: Optional[str]
    isplume: bool
    validated: bool
    wind_u: Optional[float]
    wind_v: Optional[float]
    image: Optional["GeoTensor"]
    cloudmask: Optional["GeoTensor"]
    band_names: Optional[list[str]]
    metadata: dict

    def haswind(self) -> bool: ...


@dataclass
class Location:
    """Slim mirror of marsml's ``EmissionsLocation`` (only fields the selector needs).

    Reusable as a lightweight monitoring-site object in notebooks/tutorials.
    """

    id_location: uuid.UUID
    location_name: str
    lat: float
    lon: float
    geometry: "Polygon"  # AOI footprint, EPSG:4326 (the ~2 km box)
    offshore: bool = False

    @classmethod
    def from_lon_lat(
        cls,
        lon: float,
        lat: float,
        margin_meters: float = 2000,
        location_name: str = "",
        offshore: bool = False,
        id_location: Optional[uuid.UUID] = None,
    ) -> "Location":
        """Build a ``Location`` with a square AOI centred at ``(lon, lat)``."""
        from marss2l.mars_sentinel2.s2lutils import center_polygon_meters

        geometry = center_polygon_meters(lon, lat, margin_meters=margin_meters)
        return cls(
            id_location=id_location or uuid.uuid4(),
            location_name=location_name,
            lat=lat,
            lon=lon,
            geometry=geometry,
            offshore=offshore,
        )


@dataclass
class S2LLocationImage:
    """Slim mirror of marsml's ``MarsLocationImage`` for background selection."""

    # --- identity (same names/types as MarsLocationImage) ---
    id_loc_image: uuid.UUID
    id_location: uuid.UUID
    location: Location
    tile: str
    satellite: str
    tile_date: datetime
    # --- selection metadata (same names as MarsLocationImage) ---
    percentage_clear: float = -1.0  # computed locally from the cloud mask; -1 = unknown
    observability: Optional[str] = None  # "clear" | "bad_retrieval" | "cloudy" | "night"
    isplume: bool = False  # GEE candidates have no plume labels
    validated: bool = False  # GEE candidates are never DB-validated
    wind_u: Optional[float] = None
    wind_v: Optional[float] = None
    # --- GEE download payload (lets download_image skip re-querying the catalog) ---
    asset_id: Optional[str] = None  # f"{collection_name}/{gee_id}"
    gee_id: Optional[str] = None
    crs: Optional[str] = None
    transform: Optional["Affine"] = None
    # --- loaded on demand, always GeoTensor ---
    image: Optional["GeoTensor"] = None
    cloudmask: Optional["GeoTensor"] = None
    band_names: Optional[list[str]] = None
    sza: Optional[float] = None
    vza: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def haswind(self) -> bool:
        return self.wind_u is not None and self.wind_v is not None

    @property
    def day(self) -> str:
        """``YYYY-MM-DD`` acquisition day (mirrors ``MarsLocationImage.day``)."""
        return self.tile_date.strftime("%Y-%m-%d")

    @staticmethod
    def _id_from_asset(asset_id: Optional[str]) -> uuid.UUID:
        return uuid.uuid5(_NAMESPACE_S2L, asset_id) if asset_id else uuid.uuid4()

    @classmethod
    def from_gee_row(cls, row: Any, location: Location) -> "S2LLocationImage":
        """Build from one row of :func:`query_images.query_gee`'s GeoDataFrame.

        ``row`` is a pandas ``Series`` whose ``name`` is the image title (the tile).
        ``percentage_clear`` is intentionally left unknown (-1); it is computed later
        from the locally downloaded cloud mask.
        """

        def _get(key: str) -> Any:
            val = row.get(key) if hasattr(row, "get") else None
            return val if (val is not None and val == val) else None  # drop NaN

        tile = str(row.name)
        asset_id = _get("asset_id")
        return cls(
            id_loc_image=cls._id_from_asset(asset_id),
            id_location=location.id_location,
            location=location,
            tile=tile,
            satellite=str(_get("satellite")),
            tile_date=_get("utcdatetime"),
            wind_u=_get("U"),
            wind_v=_get("V"),
            asset_id=asset_id,
            gee_id=_get("gee_id"),
            crs=_get("crs"),
            transform=_get("transform"),
        )

    @classmethod
    def from_tile(
        cls,
        tile: str,
        location: Location,
        tile_date: Optional[datetime] = None,
        logger: Any = None,
    ) -> "S2LLocationImage":
        """Resolve a single known tile via :func:`s2lutils.gee_info_to_download`."""
        from marss2l.mars_sentinel2.s2lutils import gee_info_to_download

        info = gee_info_to_download(
            tile, geometry=location.geometry, tile_date=tile_date, logger=logger
        )
        if info is None:
            raise ValueError(f"Could not resolve tile {tile} from GEE")
        resolved_tile = info.get("tile", tile)
        return cls(
            id_loc_image=cls._id_from_asset(info.get("asset_id")),
            id_location=location.id_location,
            location=location,
            tile=resolved_tile,
            satellite=resolved_tile.split("_")[0],
            tile_date=info.get("utcdatetime", tile_date),
            asset_id=info.get("asset_id"),
            gee_id=info.get("gee_id"),
            crs=info.get("crs"),
            transform=info.get("transform"),
        )
