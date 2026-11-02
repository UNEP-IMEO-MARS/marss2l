"""Tests for marss2l.mars_sentinel2.location_image (slim data classes + protocols)."""

import uuid
from datetime import datetime, timezone

import pandas as pd
import pytest
from shapely.geometry import box

from marss2l.mars_sentinel2.location_image import (
    Location,
    LocationImageProtocol,
    LocationProtocol,
    S2LLocationImage,
)


def make_location() -> Location:
    return Location(
        id_location=uuid.uuid4(),
        location_name="test_site",
        lat=32.0,
        lon=-102.0,
        geometry=box(-102.01, 31.99, -101.99, 32.01),
        offshore=False,
    )


def make_image(location: Location, **kwargs) -> S2LLocationImage:
    defaults = dict(
        id_loc_image=uuid.uuid4(),
        id_location=location.id_location,
        location=location,
        tile="S2A_MSIL1C_20250101T000000_N0511_R055_T13SGR_20250101T000000",
        satellite="S2A",
        tile_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return S2LLocationImage(**defaults)


class TestLocation:
    def test_satisfies_protocol(self):
        assert isinstance(make_location(), LocationProtocol)

    def test_fields(self):
        loc = make_location()
        assert loc.location_name == "test_site"
        assert loc.offshore is False
        assert isinstance(loc.id_location, uuid.UUID)


class TestS2LLocationImage:
    def test_satisfies_protocol(self):
        assert isinstance(make_image(make_location()), LocationImageProtocol)

    def test_haswind(self):
        loc = make_location()
        assert make_image(loc).haswind() is False
        assert make_image(loc, wind_u=1.0, wind_v=2.0).haswind() is True
        assert make_image(loc, wind_u=1.0).haswind() is False

    def test_day_property(self):
        img = make_image(make_location(), tile_date=datetime(2024, 7, 3, 12, tzinfo=timezone.utc))
        assert img.day == "2024-07-03"

    def test_defaults(self):
        img = make_image(make_location())
        assert img.percentage_clear == -1.0
        assert img.isplume is False
        assert img.validated is False
        assert img.image is None
        assert img.metadata == {}


class TestFromGeeRow:
    def _row(self, asset_id="COPERNICUS/S2_HARMONIZED/abc", title="S2B_TILE_X"):
        return pd.Series(
            {
                "satellite": "S2B",
                "utcdatetime": datetime(2025, 2, 2, tzinfo=timezone.utc),
                "cloudcoverpercentage": 12.0,
                "asset_id": asset_id,
                "gee_id": "abc",
                "crs": "EPSG:32613",
                "transform": None,
                "U": 3.0,
                "V": -4.0,
            },
            name=title,
        )

    def test_field_mapping(self):
        loc = make_location()
        img = S2LLocationImage.from_gee_row(self._row(), location=loc)
        assert img.tile == "S2B_TILE_X"
        assert img.satellite == "S2B"
        assert img.tile_date == datetime(2025, 2, 2, tzinfo=timezone.utc)
        assert img.asset_id == "COPERNICUS/S2_HARMONIZED/abc"
        assert img.wind_u == 3.0 and img.wind_v == -4.0
        assert img.id_location == loc.id_location
        # percentage_clear is NOT taken from GEE — it stays unknown until local cloud mask
        assert img.percentage_clear == -1.0

    def test_deterministic_id_from_asset(self):
        loc = make_location()
        a = S2LLocationImage.from_gee_row(self._row(asset_id="X/1"), location=loc)
        b = S2LLocationImage.from_gee_row(self._row(asset_id="X/1", title="other"), location=loc)
        c = S2LLocationImage.from_gee_row(self._row(asset_id="X/2"), location=loc)
        assert a.id_loc_image == b.id_loc_image  # same asset -> same id
        assert a.id_loc_image != c.id_loc_image

    def test_nan_wind_becomes_none(self):
        loc = make_location()
        row = self._row()
        row["U"] = float("nan")
        img = S2LLocationImage.from_gee_row(row, location=loc)
        assert img.wind_u is None

    def test_captures_s2_angles_in_fields(self):
        loc = make_location()
        row = self._row()
        row["MEAN_SOLAR_ZENITH_ANGLE"] = 30.0
        row["MEAN_INCIDENCE_ZENITH_ANGLE_B12"] = 5.0
        img = S2LLocationImage.from_gee_row(row, location=loc)
        assert img.sza == 30.0
        assert img.vza == 5.0
        assert img.metadata == {}


class TestFromTile:
    """from_tile must derive tile_date the same way query_gee does, NOT from system:time_start.

    Otherwise the target and its candidates disagree on tile_date for the same scene by several
    minutes and the same-acquisition filter keeps the target as its own background.
    """

    def _patch_info(self, monkeypatch, info):
        import marss2l.mars_sentinel2.s2lutils as s2lutils

        monkeypatch.setattr(s2lutils, "gee_info_to_download", lambda *a, **k: info)

    def test_s2_tile_date_is_datatake_stamp_not_time_start(self, monkeypatch):
        loc = make_location()
        product_id = "S2B_MSIL1C_20250529T172859_N0511_R055_T13SGR_20250529T210525"
        # system:time_start (granule sensing time) is ~16 min after the datatake stamp.
        self._patch_info(
            monkeypatch,
            {
                "tile": product_id,
                "asset_id": f"COPERNICUS/S2_HARMONIZED/{product_id}",
                "gee_id": product_id,
                "utcdatetime": datetime(2025, 5, 29, 17, 45, 4, tzinfo=timezone.utc),
                "crs": "EPSG:32613",
                "transform": None,
            },
        )
        img = S2LLocationImage.from_tile(product_id, location=loc)
        # datatake stamp 20250529T172859, NOT the 17:45:04 system:time_start
        assert img.tile_date == datetime(2025, 5, 29, 17, 28, 59, tzinfo=timezone.utc)

    def test_landsat_tile_date_uses_utcdatetime(self, monkeypatch):
        loc = make_location()
        tile = "LC08_L1TP_031037_20250529_20250529_02_T1"
        ts = datetime(2025, 5, 29, 17, 30, 0, tzinfo=timezone.utc)
        self._patch_info(
            monkeypatch,
            {
                "tile": tile,
                "asset_id": f"LANDSAT/LC08/C02/T1_L2/{tile}",
                "gee_id": tile,
                "utcdatetime": ts,
                "crs": "EPSG:32613",
                "transform": None,
            },
        )
        img = S2LLocationImage.from_tile(tile, location=loc)
        assert img.satellite == "LC08"
        assert img.tile_date == ts  # Landsat keeps system:time_start (consistent with query_gee)
