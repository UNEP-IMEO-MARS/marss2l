"""Tests for marss2l.mars_sentinel2.query_images helpers (no network)."""

from datetime import datetime, timezone

from marss2l.mars_sentinel2.query_images import utcdatetime_from_s2_title


class TestUtcdatetimeFromS2Title:
    def test_modern_naming(self):
        # Datatake stamp at chars 11:26 of the modern product name.
        title = "S2B_MSIL1C_20250529T172859_N0511_R055_T13SGR_20250529T210525"
        assert utcdatetime_from_s2_title(title) == datetime(
            2025, 5, 29, 17, 28, 59, tzinfo=timezone.utc
        )

    def test_oper_legacy_naming(self):
        # Datatake stamp at chars 25:40 of the legacy OPER product name.
        title = "S2A_OPER_PRD_MSIL1C_PDMC_20160526T083006_R022_V20160526T083006_20160526T083006"
        assert utcdatetime_from_s2_title(title) == datetime(
            2016, 5, 26, 8, 30, 6, tzinfo=timezone.utc
        )
