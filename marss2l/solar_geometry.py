"""Solar geometry helpers for the target and the background (reference) pass.

The MBMP retrieval is a double ratio, so the solar irradiance and the geometry
factors cancel exactly and the retrieval itself never needs radiances. The
shot-noise error propagation is the first thing that does: converting the stored
ToA reflectances to radiances needs each pass's own acquisition time and solar
zenith angle,

    L = rho * E * cos(sza) / (pi * d(date) ** 2)

so the *background* pass needs its own ``sza`` and ``tile_date``, not the
target's.

Two facts about the stored angles motivate the functions below:

- **A minority of the stored angles are wrong.** About 2% of the Landsat records
  carry a solar zenith angle of ~0.05 degrees where the true value is 20-81
  degrees, and a handful exceed 90 degrees (``cos`` then goes negative and the
  radiance with it). :func:`repair_sza` substitutes a computed angle in those
  cases and reports which value was used.
- **Only Sentinel-2 tile names carry a time of day.** A Landsat identifier gives
  the acquisition *date* only, and the hour angle -- hence the solar zenith angle
  -- needs the time. :func:`datetime_from_tile` therefore returns ``None`` for
  Landsat, and the caller must fall back to the stored ``tile_date``.
"""

from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Below/above these the reflectance-to-radiance conversion is not trustworthy:
# cos(sza) is within 0.4% of 1 under 5 degrees (so a corrupt ~0 reading is
# indistinguishable from a real one by value alone), and beyond 85 degrees it
# collapses towards 0 and then changes sign.
SZA_MIN_PLAUSIBLE = 5.0
SZA_MAX_PLAUSIBLE = 85.0

# Stored and computed angles legitimately differ: the stored one is the mean over
# the granule while we compute it at the scene centre. Measured over the rows whose
# stored angle is inside the plausible range, that difference is 0.8 degrees in the
# median and never exceeds 6.3 degrees -- it grows towards the equator, where a
# given ground distance spans more of the solar arc. The corrupt records are 20
# degrees or more out, and nothing at all falls in between, so 10 degrees separates
# the two populations with room on either side.
SZA_TOLERANCE = 10.0

SZA_SOURCE_STORED = "db"
SZA_SOURCE_COMPUTED = "computed"
#: The Sun was at or below the horizon at the recorded acquisition time. Landsat
#: does acquire night scenes, and a handful of them are recorded as background
#: images. There is no reflected sunlight to work with, so no angle rescues the
#: record -- it has to be excluded from any reflectance-based statistic.
SZA_SOURCE_NIGHT = "night"

# Sentinel-2 product names carry the datacapture time as the third underscore
# separated field, e.g.
# S2B_MSIL1C_20241226T093319_N0511_R136_T33RYN_20241226T112459.
_S2_DATETIME_FORMAT = "%Y%m%dT%H%M%S"


def datetime_from_tile(tile: str) -> Optional[datetime]:
    """Acquisition datetime parsed out of a tile identifier, if it holds one.

    Sentinel-2 product names carry the sensing time to the second; Landsat
    identifiers (``LC08_L1TP_185040_20250107_20250111_02_T1``) carry the date
    only, which is not enough to place the Sun, so ``None`` is returned for them
    and the caller must use the stored ``tile_date``.

    Args:
        tile: Tile identifier as stored in the ``tile`` / ``background_image_tile``
            columns.

    Returns:
        Timezone-aware UTC datetime, or None if the identifier has no time of day.
    """
    if not isinstance(tile, str):
        return None

    parts = tile.split("_")
    if not parts[0].startswith("S2") or len(parts) < 3:
        return None

    try:
        parsed = datetime.strptime(parts[2], _S2_DATETIME_FORMAT)
    except ValueError:
        return None

    return parsed.replace(tzinfo=timezone.utc)


def satellite_from_tile(tile: str) -> Optional[str]:
    """Satellite that acquired ``tile``, from the identifier prefix.

    The background pass is frequently from a *different* satellite than the
    target (36% of the MARS-S2L pairs, both within the Landsat pair and within
    the Sentinel-2 constellation), and Landsat's SNR is about twice Sentinel-2's
    in the 1.6 and 2.3 um bands, so the noise terms of the background must be
    evaluated with its own instrument.

    Args:
        tile: Tile identifier.

    Returns:
        Satellite name (``S2A``, ``LC08``, ...), or None if unrecognised. Landsat
        products acquired by OLI alone are reported as their ``LC`` platform,
        since the platform is what sets the reference SNR.
    """
    if not isinstance(tile, str) or "_" not in tile:
        return None

    prefix = tile.split("_")[0]
    if prefix.startswith("S2") and len(prefix) == 3:
        return prefix
    if len(prefix) == 4 and prefix[:2] in {"LC", "LO", "LT", "LE"}:
        # LO/LT denote the single-instrument products of the same platform.
        return f"LC{prefix[2:]}" if prefix[:2] == "LO" else prefix
    return None


def as_utc(when) -> datetime:
    """Coerce a timestamp to a timezone-aware UTC datetime.

    ``datetime.fromisoformat`` returns a naive datetime, and ``pysolar`` silently
    treats a naive datetime as local time -- an error that shifts the hour angle
    and so the solar zenith angle without raising anything.

    Args:
        when: datetime, pandas Timestamp or ISO-8601 string.

    Returns:
        Timezone-aware UTC datetime.

    Raises:
        ValueError: if ``when`` cannot be interpreted as a timestamp.
    """
    if isinstance(when, str):
        when = pd.to_datetime(when, utc=True, format="mixed")

    if isinstance(when, pd.Timestamp):
        when = when.to_pydatetime()

    if not isinstance(when, datetime):
        raise ValueError(f"Cannot interpret {when!r} as a timestamp")

    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)

    return when.astimezone(timezone.utc)


def sza_computed(lon: float, lat: float, when) -> float:
    """Solar zenith angle in degrees at ``(lon, lat)`` and time ``when``.

    Thin wrapper over ``georeader.reflectance.compute_sza`` that enforces the UTC
    contract of :func:`as_utc`. ``pysolar`` is an optional dependency of
    ``georeader``; ``compute_sza`` raises ``ImportError`` without it.

    Args:
        lon: Longitude in degrees (EPSG:4326).
        lat: Latitude in degrees (EPSG:4326).
        when: Acquisition time, timezone-aware or assumed UTC.

    Returns:
        Solar zenith angle in degrees.
    """
    from georeader import reflectance

    return reflectance.compute_sza((lon, lat), as_utc(when))


def is_plausible_sza(sza) -> bool:
    """Whether ``sza`` is within the range where the radiance conversion holds."""
    return (
        sza is not None
        and not pd.isna(sza)
        and SZA_MIN_PLAUSIBLE <= float(sza) <= SZA_MAX_PLAUSIBLE
    )


def repair_sza(
    sza: Optional[float],
    lon: float,
    lat: float,
    when,
    tolerance: float = SZA_TOLERANCE,
) -> Tuple[Optional[float], Optional[str]]:
    """Return a trustworthy solar zenith angle, and where it came from.

    The stored angle is kept unless it is missing, outside
    ``[SZA_MIN_PLAUSIBLE, SZA_MAX_PLAUSIBLE]``, or further than ``tolerance``
    from the computed one. The computed angle agrees with the *sound* stored
    Landsat values to 0.02 degrees, so a disagreement of several degrees is the
    stored value being wrong rather than the two conventions differing.

    Idempotent: feeding back a repaired value returns it unchanged, so it is safe
    to apply defensively to a dataset that has already been through it.

    Args:
        sza: Stored solar zenith angle in degrees, or None/NaN if unknown.
        lon: Longitude in degrees (EPSG:4326).
        lat: Latitude in degrees (EPSG:4326).
        when: Acquisition time, timezone-aware or assumed UTC. If None, the
            stored value is returned as is -- there is nothing to check against.
        tolerance: Maximum tolerated disagreement in degrees.

    Returns:
        Tuple of (solar zenith angle in degrees, source) where source is ``"db"``
        if the stored value was kept, ``"computed"`` if it was substituted, and
        ``"night"`` if the Sun was below the horizon at that time -- in which case
        the returned angle exceeds 90 degrees and the record must be excluded
        rather than used. Both are None when neither value is available.
    """
    if when is None or pd.isna(when):
        return (None, None) if not is_plausible_sza(sza) else (float(sza), SZA_SOURCE_STORED)

    computed = sza_computed(lon, lat, when)

    if computed > SZA_MAX_PLAUSIBLE:
        # No stored angle can rescue a scene acquired in the dark, so this
        # dominates: report the truth and let the caller drop the record.
        return computed, SZA_SOURCE_NIGHT

    if not is_plausible_sza(sza):
        return computed, SZA_SOURCE_COMPUTED

    if abs(float(sza) - computed) > tolerance:
        return computed, SZA_SOURCE_COMPUTED

    return float(sza), SZA_SOURCE_STORED


def repair_sza_dataframe(
    dataframe: pd.DataFrame,
    *,
    sza_column: str = "sza",
    date_column: str = "tile_date",
    source_column: Optional[str] = None,
    lon_column: str = "lon",
    lat_column: str = "lat",
    tolerance: float = SZA_TOLERANCE,
) -> pd.DataFrame:
    """Apply :func:`repair_sza` to a column of a dataframe, in place.

    Args:
        dataframe: Frame carrying the angle, the acquisition date and the
            location coordinates. Modified in place and returned.
        sza_column: Column with the solar zenith angle to repair.
        date_column: Column with the acquisition date of the same pass.
        source_column: Column to write the provenance to. Defaults to
            ``f"{sza_column}_source"``.
        lon_column: Column with the longitude.
        lat_column: Column with the latitude.
        tolerance: Maximum tolerated disagreement in degrees.

    Returns:
        The same dataframe, with ``sza_column`` repaired and ``source_column``
        added.
    """
    if source_column is None:
        source_column = f"{sza_column}_source"

    dates = pd.to_datetime(dataframe[date_column], utc=True, format="mixed")

    repaired = [
        repair_sza(sza, lon, lat, when, tolerance=tolerance)
        for sza, lon, lat, when in zip(
            dataframe[sza_column],
            dataframe[lon_column],
            dataframe[lat_column],
            dates,
            strict=True,
        )
    ]

    dataframe[sza_column] = np.array([value for value, _ in repaired], dtype=float)
    dataframe[source_column] = [source for _, source in repaired]

    return dataframe
