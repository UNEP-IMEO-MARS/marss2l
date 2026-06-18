"""Tests for marss2l.mars_sentinel2.background (BackgroundImageSelector + cache).

No network / GEE is used: the GEE candidate query and image download are monkeypatched
or overridden, and rasters are synthetic GeoTensors.
"""

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from georeader.geotensor import GeoTensor
from rasterio.transform import Affine
from shapely.geometry import box

from marss2l.mars_sentinel2.background import (
    BackgroundImageSelector,
    InMemorySimilarityCache,
)
from marss2l.mars_sentinel2.location_image import Location, S2LLocationImage

TRANSFORM = Affine.translation(0, 0) * Affine.scale(10, -10)
CRS = "EPSG:32630"
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
TARGET_DATE = datetime(2025, 6, 1, tzinfo=timezone.utc)
_LOADED = object()  # sentinel for "pixels present" when actual pixels are irrelevant


def gt(values: np.ndarray) -> GeoTensor:
    return GeoTensor(values, transform=TRANSFORM, crs=CRS, fill_value_default=0)


def make_location(offshore=False) -> Location:
    return Location(
        id_location=uuid.uuid4(), location_name="site", lat=32.0, lon=-102.0,
        geometry=box(-102.01, 31.99, -101.99, 32.01), offshore=offshore,
    )


def make_image(location, satellite="S2A", day_offset=10, percentage_clear=100.0,
               observability="clear", tile=None, **kwargs) -> S2LLocationImage:
    tile = tile or f"{satellite}_MSIL1C_X_N0511_R055_T13SGR_{day_offset}"
    return S2LLocationImage(
        id_loc_image=uuid.uuid4(), id_location=location.id_location, location=location,
        tile=tile, satellite=satellite,
        tile_date=TARGET_DATE + timedelta(days=day_offset),
        percentage_clear=percentage_clear, observability=observability, **kwargs,
    )


# --------------------------------------------------------------------------- helpers


class TestHelpers:
    def test_validmask_is_cloudmask_eq_zero(self):
        sel = BackgroundImageSelector()
        loc = make_location()
        img = make_image(loc)
        img.cloudmask = gt(np.array([[0, 1], [3, 0]], dtype=np.uint8))
        vm = sel.validmask(img)
        assert vm.values.tolist() == [[True, False], [False, True]]
        assert vm.fill_value_default is False

    def test_compute_percentage_clear(self):
        sel = BackgroundImageSelector()
        cm = gt(np.array([[0, 0], [0, 3]], dtype=np.uint8))  # 3/4 clear
        assert sel.compute_percentage_clear(cm) == pytest.approx(75.0)

    def test_band_index_s2(self):
        sel = BackgroundImageSelector()
        img = make_image(make_location())
        img.band_names = S2_BANDS
        assert sel.band_index(img, "B11") == 11
        assert sel.band_index(img, "B02") == 1

    def test_band_index_landsat_maps_logical_name(self):
        sel = BackgroundImageSelector()
        img = make_image(make_location(), satellite="LC08")
        img.band_names = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B09"]
        assert sel.band_index(img, "B11") == 5  # B11 -> B06
        assert sel.band_index(img, "B12") == 6  # B12 -> B07


# --------------------------------------------------------------------------- filtering


class TestFilter:
    def setup_method(self):
        self.sel = BackgroundImageSelector()
        self.loc = make_location()
        self.target = make_image(self.loc, satellite="S2A", day_offset=0)

    def test_keeps_valid_candidate(self):
        bg = make_image(self.loc, satellite="S2B", day_offset=10)
        assert self.sel.filter_background_image(self.target, bg) is False

    def test_rejects_different_constellation(self):
        bg = make_image(self.loc, satellite="LC08", day_offset=10)
        assert self.sel.filter_background_image(self.target, bg) is True

    def test_accepts_lc_lo_same_landsat8(self):
        # LC08 and LO08 are both Landsat-8 — a plain satellite[:2] check would wrongly reject.
        target = make_image(self.loc, satellite="LC08", day_offset=0)
        bg = make_image(self.loc, satellite="LO08", day_offset=10)
        assert self.sel.filter_background_image(target, bg) is False

    def test_groups_landsat8_and_9(self):
        # same_satellite_constellation=True groups Landsat-8 and Landsat-9 (as in marsml).
        t8 = make_image(self.loc, satellite="LC08", day_offset=0)
        t9 = make_image(self.loc, satellite="LC09", day_offset=0)
        bg9 = make_image(self.loc, satellite="LC09", day_offset=10)
        bg8 = make_image(self.loc, satellite="LC08", day_offset=10)
        assert self.sel.filter_background_image(t8, bg9) is False  # L8 target accepts L9 bg
        assert self.sel.filter_background_image(t9, bg8) is False  # L9 target accepts L8 bg

    def test_producttype_groups_landsat8_and_9(self):
        # The GEE query for a Landsat target fetches the whole L8/L9 family.
        t8 = make_image(self.loc, satellite="LC08")
        assert self.sel._producttype(
            t8, same_satellite=False, same_satellite_constellation=True
        ) == ("Landsat", False)

    def test_cross_constellation_allowed_when_disabled(self):
        bg = make_image(self.loc, satellite="LC08", day_offset=10)
        assert (
            self.sel.filter_background_image(
                self.target, bg, same_satellite_constellation=False
            )
            is False
        )

    def test_rejects_cloudy_when_check_cloud(self):
        bg = make_image(self.loc, satellite="S2B", day_offset=10,
                        percentage_clear=90.0, observability="clear")
        assert self.sel.filter_background_image(self.target, bg) is True  # 90 < 95

    def test_check_cloud_false_ignores_clouds(self):
        bg = make_image(self.loc, satellite="S2B", day_offset=10,
                        percentage_clear=10.0, observability="cloudy")
        assert self.sel.filter_background_image(self.target, bg, check_cloud=False) is False

    def test_rejects_cloudy_observability(self):
        bg = make_image(self.loc, satellite="S2B", day_offset=10,
                        percentage_clear=99.0, observability="cloudy")
        assert self.sel.filter_background_image(self.target, bg) is True

    def test_rejects_same_image(self):
        bg = make_image(self.loc, satellite="S2A", day_offset=0)  # same sat, same instant
        assert self.sel.filter_background_image(self.target, bg) is True

    def test_rejects_tandem_twin_same_instant(self):
        # Regression: S2A and S2C flew in tandem, so an S2C scene acquired at the same instant
        # as the S2A target must be discarded (the plume has not moved). The old check compared
        # exact satellite (S2A != S2C) and wrongly kept it.
        bg = make_image(self.loc, satellite="S2C", day_offset=0)
        assert self.sel.filter_background_image(self.target, bg) is True

    def test_rejects_tandem_twin_within_5min(self):
        # Same constellation within the ±5 min window (as in marsml) is the same acquisition.
        three_min = make_image(
            self.loc, satellite="S2C", day_offset=3 / (24 * 60), tile="S2C_tandem_3min"
        )
        assert self.sel.filter_background_image(self.target, three_min) is True

    def test_keeps_same_constellation_outside_5min(self):
        # 8 min apart is a genuinely separate pass → kept (other filters pass).
        eight_min = make_image(
            self.loc, satellite="S2C", day_offset=8 / (24 * 60), tile="S2C_8min"
        )
        assert self.sel.filter_background_image(self.target, eight_min) is False

    def test_old_landsat_missions_are_distinct_constellations(self):
        # LT04/LT05/LE07 are decades apart and not interchangeable: an L4 scene is not a
        # background for an L5 target, even with same_satellite_constellation=True.
        target = make_image(self.loc, satellite="LT05", day_offset=0)
        bg_l4 = make_image(self.loc, satellite="LT04", day_offset=10)
        bg_l5 = make_image(self.loc, satellite="LT05", day_offset=10)
        assert self.sel.filter_background_image(target, bg_l4) is True   # different constellation
        assert self.sel.filter_background_image(target, bg_l5) is False  # same mission, kept

    def test_rejects_outside_date_window(self):
        bg = make_image(self.loc, satellite="S2A", day_offset=200)
        assert self.sel.filter_background_image(self.target, bg) is True

    def test_force_same_orbit(self):
        same = make_image(self.loc, satellite="S2A", day_offset=10,
                          tile="S2A_MSIL1C_20250101T000000_N0511_R055_T13SGR_END")
        diff = make_image(self.loc, satellite="S2A", day_offset=10,
                          tile="S2A_MSIL1C_20250101T000000_N0511_R104_T13SGR_END")
        target = make_image(self.loc, satellite="S2A", day_offset=0,
                            tile="S2A_MSIL1C_20250101T000000_N0511_R055_T13SGR_TGT")
        assert self.sel.filter_background_image(target, same, force_same_orbit=True) is False
        assert self.sel.filter_background_image(target, diff, force_same_orbit=True) is True

    def test_wind_plume_exclusion(self):
        target = make_image(self.loc, satellite="S2A", day_offset=0, wind_u=1.0, wind_v=0.0)
        plume_same_wind = make_image(self.loc, satellite="S2B", day_offset=10,
                                     isplume=True, wind_u=1.0, wind_v=0.0)
        plume_cross_wind = make_image(self.loc, satellite="S2B", day_offset=10,
                                      isplume=True, wind_u=0.0, wind_v=1.0)
        assert self.sel.filter_background_image(target, plume_same_wind) is True
        assert self.sel.filter_background_image(target, plume_cross_wind) is False

    def test_wind_zero_vector_does_not_raise(self):
        # A zero wind vector has no direction; must not raise ZeroDivisionError or reject.
        target = make_image(self.loc, satellite="S2A", day_offset=0, wind_u=0.0, wind_v=0.0)
        plume = make_image(
            self.loc, satellite="S2B", day_offset=10, isplume=True, wind_u=0.0, wind_v=0.0
        )
        assert self.sel.filter_background_image(target, plume) is False


class TestFilterAndSortTwoPass:
    def test_two_pass_loosening_and_sort(self):
        sel = BackgroundImageSelector()
        loc = make_location()
        target = make_image(loc, satellite="S2A", day_offset=0)
        # all candidates are 90% clear: fail the 5% pass, pass the 35% pass
        far = make_image(loc, satellite="S2B", day_offset=40, percentage_clear=90.0)
        near = make_image(loc, satellite="S2B", day_offset=20, percentage_clear=90.0)
        out = sel._filter_and_sort_background_images(target, [far, near])
        assert out == [near, far]  # sorted by |Δdate|

    def test_returns_empty_when_all_too_cloudy(self):
        sel = BackgroundImageSelector()
        loc = make_location()
        target = make_image(loc, satellite="S2A", day_offset=0)
        bg = make_image(loc, satellite="S2B", day_offset=20,
                        percentage_clear=10.0, observability="cloudy")
        assert sel._filter_and_sort_background_images(target, [bg]) == []


# --------------------------------------------------------------------------- cache


class TestSimilarityCache:
    def test_symmetric_and_param_sensitive(self):
        cache = InMemorySimilarityCache()
        a, b = uuid.uuid4(), uuid.uuid4()
        cache.put(a, b, ("B02",), True, 0.5, {})
        assert cache.get(a, b, ("B02",), True) == 0.5
        assert cache.get(b, a, ("B02",), True) == 0.5  # symmetric
        assert cache.get(a, b, ("B02",), False) is None  # corregister differs
        assert cache.get(a, b, ("B11",), True) is None  # bands differ

    def test_miss(self):
        assert InMemorySimilarityCache().get(uuid.uuid4(), uuid.uuid4(), ("B02",), True) is None


# --------------------------------------------------------------------------- similarity ranking


class _MapSelector(BackgroundImageSelector):
    """Selector whose download_image restores pixels from an in-memory map."""

    def __init__(self, pixels, candidates=None, **kw):
        super().__init__(**kw)
        self.pixels = pixels  # tile -> (image_gt, cloudmask_gt, band_names)
        self.candidates = candidates
        self.download_count = 0

    def download_image(self, image):
        self.download_count += 1
        img, cm, names = self.pixels[image.tile]
        image.image, image.cloudmask, image.band_names = img, cm, names
        if image.percentage_clear < 0:
            image.percentage_clear = self.compute_percentage_clear(cm)
            image.observability = (
                "cloudy" if image.percentage_clear < self.threshold_max_noclear else "clear"
            )

    def query_background_images(self, image_to_process, **kw):
        return list(self.candidates)


def _bands(values):  # (13,H,W) image + all-clear cloudmask
    return gt(values.astype(np.float64)), gt(np.zeros(values.shape[1:], dtype=np.uint8)), list(S2_BANDS)


class TestMostSimilarSorted:
    def _setup(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(500, 5000, (13, 16, 16))
        loc = make_location()
        target = make_image(loc, satellite="S2A", day_offset=0)
        bg_same = make_image(loc, satellite="S2B", day_offset=10, tile="bg_same")
        bg_diff = make_image(loc, satellite="S2B", day_offset=11, tile="bg_diff")
        pixels = {
            target.tile: _bands(base),
            "bg_same": _bands(base.copy()),  # identical -> difference 0
            "bg_diff": _bands(rng.uniform(500, 5000, (13, 16, 16))),  # independent -> larger
        }
        return target, bg_same, bg_diff, pixels

    def test_ranks_most_similar_first(self):
        target, bg_same, bg_diff, pixels = self._setup()
        sel = _MapSelector(pixels)
        ranked = sel.background_images_most_similar_sorted(
            target, [bg_diff, bg_same], corregister=False
        )
        assert [bg for bg, _ in ranked] == [bg_same, bg_diff]
        assert ranked[0][1] <= ranked[1][1]
        assert ranked[0][1] == pytest.approx(0.0, abs=1e-9)

    def test_top_truncates(self):
        target, bg_same, bg_diff, pixels = self._setup()
        sel = _MapSelector(pixels)
        ranked = sel.background_images_most_similar_sorted(
            target, [bg_diff, bg_same], corregister=False, top=1
        )
        assert len(ranked) == 1 and ranked[0][0] is bg_same

    def test_cache_is_used(self):
        target, bg_same, bg_diff, pixels = self._setup()
        cache = InMemorySimilarityCache()
        sel = _MapSelector(pixels, cache=cache)
        sel.background_images_most_similar_sorted(target, [bg_same, bg_diff], corregister=False)
        # both pairs now cached
        assert cache.get(target.id_loc_image, bg_same.id_loc_image,
                         tuple(sel.bands_differences), False) is not None

    def test_compute_background_image_picks_most_similar(self):
        target, bg_same, bg_diff, pixels = self._setup()
        sel = _MapSelector(pixels, candidates=[bg_diff, bg_same])
        # candidates carry percentage_clear=100 from make_image, so the filter keeps them
        chosen = sel.compute_background_image(target, method_bg_image="most_similar")
        assert chosen is bg_same
        assert target.metadata["background_image"] is bg_same

    def test_compute_background_image_caches_choice(self):
        target, bg_same, bg_diff, pixels = self._setup()
        sel = _MapSelector(pixels, candidates=[bg_diff, bg_same])
        first = sel.compute_background_image(target)
        sel.candidates = []  # would yield nothing if recomputed
        again = sel.compute_background_image(target)
        assert again is first  # reused from metadata, not recomputed


# --------------------------------------------------------------------------- expanding download


class _ExpandSelector(BackgroundImageSelector):
    """download_image populates percentage_clear from a per-tile map and counts calls."""

    def __init__(self, clear_by_tile, **kw):
        super().__init__(**kw)
        self.clear_by_tile = clear_by_tile
        self.download_count = 0

    def download_image(self, image):
        self.download_count += 1
        if image.image is not None:
            return
        image.image = _LOADED
        image.percentage_clear = self.clear_by_tile[image.tile]
        image.observability = (
            "cloudy" if image.percentage_clear < self.threshold_max_noclear else "clear"
        )


def _patch_gee(monkeypatch, rows):
    import marss2l.mars_sentinel2.background as bg_mod
    import marss2l.mars_sentinel2.query_images as qi

    # background.py does `from ...ee import ee_initialize`, so the name is bound in
    # the background module namespace; patch it there (not on the ee module).
    monkeypatch.setattr(bg_mod, "ee_initialize", lambda *a, **k: None)
    df = pd.DataFrame(rows).set_index("title")
    monkeypatch.setattr(qi, "query_gee", lambda *a, **k: df)


def _row(title, day_offset, satellite="S2A"):
    return {
        "title": title, "satellite": satellite,
        "utcdatetime": TARGET_DATE + timedelta(days=day_offset),
        "cloudcoverpercentage": 10.0, "asset_id": f"COLL/{title}",
        "gee_id": title, "crs": CRS, "transform": None,
    }


class TestQueryBackgroundImagesExpand:
    def test_happy_path_stops_at_limit(self, monkeypatch):
        rows = [_row(f"c{i}", day_offset=i) for i in range(1, 6)]
        _patch_gee(monkeypatch, rows)
        clear = {f"c{i}": 100.0 for i in range(1, 6)}  # all clear
        sel = _ExpandSelector(clear, limit_images_most_similar=2)
        target = make_image(make_location(), satellite="S2A", day_offset=0)
        out = sel.query_background_images(target)
        assert sel.download_count == 2  # stopped once 2 clear survivors found
        assert len(out) == 2

    def test_expands_until_clear_found(self, monkeypatch):
        rows = [_row(f"c{i}", day_offset=i) for i in range(1, 6)]
        _patch_gee(monkeypatch, rows)
        clear = {"c1": 10.0, "c2": 10.0, "c3": 10.0, "c4": 10.0, "c5": 100.0}
        sel = _ExpandSelector(clear, limit_images_most_similar=2)
        target = make_image(make_location(), satellite="S2A", day_offset=0)
        out = sel.query_background_images(target)
        assert sel.download_count == 5  # expanded through all to find the clear one
        # query_background_images returns the filtered+sorted survivors: only the clear one.
        assert [c.tile for c in out] == ["c5"]
