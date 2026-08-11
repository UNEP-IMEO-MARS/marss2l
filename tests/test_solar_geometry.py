"""
Tests for marss2l.solar_geometry.

Tests cover:
- datetime_from_tile (Sentinel-2 has a time of day, Landsat does not)
- satellite_from_tile
- as_utc (the naive-datetime trap)
- sza_computed (a known geometry)
- repair_sza / repair_sza_dataframe (the corrupt stored angles)
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from marss2l import solar_geometry

S2_TILE = "S2B_MSIL1C_20241226T093319_N0511_R136_T33RYN_20241226T112459"
LANDSAT_TILE = "LC08_L1TP_185040_20250107_20250111_02_T1"

# Libya, in the middle of the MARS-S2L Landsat records.
LON, LAT = 17.368569, 28.949074


# ─────────────────────────────────────────────────────────────────────────────
# Tile identifiers
# ─────────────────────────────────────────────────────────────────────────────
def test_datetime_from_tile_sentinel2_is_exact():
    """The sensing time is in the product name, to the second."""
    assert solar_geometry.datetime_from_tile(S2_TILE) == datetime(
        2024, 12, 26, 9, 33, 19, tzinfo=timezone.utc
    )


def test_datetime_from_tile_landsat_has_no_time_of_day():
    """A Landsat identifier gives the date only, which cannot place the Sun."""
    assert solar_geometry.datetime_from_tile(LANDSAT_TILE) is None


@pytest.mark.parametrize("tile", [None, np.nan, "", "not_a_tile", "S2B_MSIL1C_notadate_N0511"])
def test_datetime_from_tile_rejects_garbage(tile):
    assert solar_geometry.datetime_from_tile(tile) is None


@pytest.mark.parametrize(
    "tile,expected",
    [
        (S2_TILE, "S2B"),
        (LANDSAT_TILE, "LC08"),
        ("LO09_L1TP_185040_20241011_20241011_02_T1", "LC09"),
        ("nonsense", None),
    ],
)
def test_satellite_from_tile(tile, expected):
    assert solar_geometry.satellite_from_tile(tile) == expected


# ─────────────────────────────────────────────────────────────────────────────
# UTC handling
# ─────────────────────────────────────────────────────────────────────────────
def test_as_utc_assumes_utc_for_naive_datetimes():
    """datetime.fromisoformat returns a naive datetime; it must not shift."""
    naive = datetime.fromisoformat("2024-12-26T09:33:19")
    assert solar_geometry.as_utc(naive) == datetime(2024, 12, 26, 9, 33, 19, tzinfo=timezone.utc)


def test_as_utc_converts_other_timezones():
    aware = pd.Timestamp("2024-12-26T11:33:19+02:00")
    assert solar_geometry.as_utc(aware) == datetime(2024, 12, 26, 9, 33, 19, tzinfo=timezone.utc)


def test_as_utc_rejects_non_timestamps():
    with pytest.raises(ValueError):
        solar_geometry.as_utc(object())


# ─────────────────────────────────────────────────────────────────────────────
# Computed angle
# ─────────────────────────────────────────────────────────────────────────────
def test_sza_computed_equatorial_equinox_noon_is_overhead():
    """Sun near the zenith at the equator, local solar noon, equinox."""
    sza = solar_geometry.sza_computed(0.0, 0.0, datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc))
    assert sza < 3.0


def test_sza_computed_is_larger_away_from_the_subsolar_point():
    when = datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc)
    assert solar_geometry.sza_computed(0.0, 60.0, when) > solar_geometry.sza_computed(
        0.0, 20.0, when
    )


# ─────────────────────────────────────────────────────────────────────────────
# Repair
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("offset", [0.0, 1.0, 6.3])
def test_repair_sza_keeps_a_sound_stored_value(offset):
    """The stored angle wins when it agrees with the computed one.

    6.3 degrees is the largest disagreement measured across the two published
    datasets among rows whose stored angle is inside the plausible range -- the
    granule-mean versus point-value difference, which must not be "repaired".
    """
    when = datetime(2024, 12, 30, 9, 20, 1, tzinfo=timezone.utc)
    computed = solar_geometry.sza_computed(LON, LAT, when)
    stored = computed + offset

    sza, source = solar_geometry.repair_sza(stored, LON, LAT, when)

    assert sza == pytest.approx(stored)
    assert source == solar_geometry.SZA_SOURCE_STORED


@pytest.mark.parametrize("stored", [0.053040, 135.54, np.nan, None])
def test_repair_sza_substitutes_implausible_values(stored):
    """The ~0.05 degree Landsat records and the >90 degree ones are replaced."""
    when = datetime(2024, 12, 30, 9, 20, 1, tzinfo=timezone.utc)

    sza, source = solar_geometry.repair_sza(stored, LON, LAT, when)

    assert source == solar_geometry.SZA_SOURCE_COMPUTED
    assert solar_geometry.SZA_MIN_PLAUSIBLE <= sza <= solar_geometry.SZA_MAX_PLAUSIBLE
    assert sza == pytest.approx(solar_geometry.sza_computed(LON, LAT, when))


def test_repair_sza_substitutes_a_plausible_but_disagreeing_value():
    """A value inside the plausible range can still be the wrong one."""
    when = datetime(2024, 12, 30, 9, 20, 1, tzinfo=timezone.utc)
    computed = solar_geometry.sza_computed(LON, LAT, when)

    sza, source = solar_geometry.repair_sza(computed + 20.0, LON, LAT, when)

    assert source == solar_geometry.SZA_SOURCE_COMPUTED
    assert sza == pytest.approx(computed)


def test_repair_sza_is_idempotent():
    """Re-running the repair on repaired data changes nothing."""
    when = datetime(2024, 12, 30, 9, 20, 1, tzinfo=timezone.utc)

    once, _ = solar_geometry.repair_sza(0.05, LON, LAT, when)
    twice, source = solar_geometry.repair_sza(once, LON, LAT, when)

    assert twice == pytest.approx(once)
    assert source == solar_geometry.SZA_SOURCE_STORED


def test_repair_sza_flags_a_night_acquisition():
    """A background image acquired after dark cannot be repaired, only labelled."""
    # 18:40 UTC at longitude 47.7 is 21:40 local: the Sun is well below the horizon.
    when = datetime(2023, 9, 15, 18, 40, 12, tzinfo=timezone.utc)

    sza, source = solar_geometry.repair_sza(135.55, 47.743480, 30.273164, when)

    assert source == solar_geometry.SZA_SOURCE_NIGHT
    assert sza > 90.0


def test_repair_sza_night_dominates_a_plausible_stored_value():
    """A sound-looking stored angle does not make a dark scene usable."""
    when = datetime(2023, 9, 15, 18, 40, 12, tzinfo=timezone.utc)

    _, source = solar_geometry.repair_sza(32.25, 47.743480, 30.273164, when)

    assert source == solar_geometry.SZA_SOURCE_NIGHT


def test_repair_sza_without_a_date_keeps_what_it_has():
    """With no acquisition time there is nothing to check the angle against."""
    assert solar_geometry.repair_sza(41.0, LON, LAT, None) == (
        41.0,
        solar_geometry.SZA_SOURCE_STORED,
    )
    assert solar_geometry.repair_sza(0.05, LON, LAT, None) == (None, None)


def test_repair_sza_dataframe_repairs_only_the_bad_rows():
    when = datetime(2024, 12, 30, 9, 20, 1, tzinfo=timezone.utc)
    computed = solar_geometry.sza_computed(LON, LAT, when)
    dataframe = pd.DataFrame(
        {
            "sza": [computed, 0.053040],
            "tile_date": [when.isoformat(), when.isoformat()],
            "lon": [LON, LON],
            "lat": [LAT, LAT],
        }
    )

    out = solar_geometry.repair_sza_dataframe(dataframe)

    assert out["sza_source"].tolist() == [
        solar_geometry.SZA_SOURCE_STORED,
        solar_geometry.SZA_SOURCE_COMPUTED,
    ]
    assert out["sza"].tolist() == pytest.approx([computed, computed])


def test_repair_sza_dataframe_names_the_source_after_the_column():
    when = datetime(2024, 12, 30, 9, 20, 1, tzinfo=timezone.utc)
    dataframe = pd.DataFrame(
        {
            "sza_bg": [0.053040],
            "tile_date_bg": [when.isoformat()],
            "lon": [LON],
            "lat": [LAT],
        }
    )

    out = solar_geometry.repair_sza_dataframe(
        dataframe, sza_column="sza_bg", date_column="tile_date_bg"
    )

    assert out["sza_bg_source"].tolist() == [solar_geometry.SZA_SOURCE_COMPUTED]
