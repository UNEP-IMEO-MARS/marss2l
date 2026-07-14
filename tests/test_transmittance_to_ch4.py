"""
Tests for marss2l.mars_sentinel2.transmittance_to_ch4 module.

Tests cover:
- TransmittanceCH4Interpolation (base class)
- TransmittanceCH4InterpolationFromLUT (LUT-based interpolation)
- TransmittanceCH4InterpolationFromDict (dict-based interpolation)
- export_transmittances (export function)
- Module constants (MIN_MBMP_VALUE, MAX_MBMP_VALUE)
"""

import warnings
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor
from scipy import interpolate


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_amf_arr():
    """Sample air mass factor array."""
    return np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])


@pytest.fixture
def sample_mr_ch4_arr():
    """Sample methane mixing ratio array."""
    return np.array([[1700, 1800, 1850, 1900, 1950, 2000, 2100, 2200, 2500]] * 8)  # (8, 9)


@pytest.fixture
def sample_geotensor():
    """Create a sample GeoTensor for testing."""
    H, W = 50, 30
    np.random.seed(42)
    values = np.random.rand(H, W) * 500  # deltaCH4 values in ppb
    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    return GeoTensor(values, transform=transform, crs="EPSG:32630", fill_value_default=-1)


@pytest.fixture
def sample_ratio_il():
    """Create a sample ratio_IL GeoTensor."""
    H, W = 50, 30
    np.random.seed(42)
    # Typical MBMP ratio values around 1
    values = np.random.rand(H, W) * 0.3 + 0.85  # Values between 0.85 and 1.15
    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    return GeoTensor(values, transform=transform, crs="EPSG:32630", fill_value_default=0)


def _mock_load_srfinterpfun(satellite: str):
    """Return mock SRF interpolation functions for any satellite (no file I/O)."""
    wvl = np.arange(1500, 2501, 1, dtype=float)
    # Use deterministic offsets per satellite so tests can distinguish them
    _offsets = {"S2A": 0, "S2B": 10, "LC08": 25, "LC09": 35}
    offset = _offsets.get(satellite, sum(satellite.encode()) % 50)
    b12_srf = np.exp(-((wvl - (2200 + offset)) ** 2) / 20000)
    b11_srf = np.exp(-((wvl - (1650 + offset)) ** 2) / 20000)
    b12_func = interpolate.interp1d(wvl, b12_srf, bounds_error=False, fill_value=0)
    b11_func = interpolate.interp1d(wvl, b11_srf, bounds_error=False, fill_value=0)
    return b12_func, b11_func


# ─────────────────────────────────────────────────────────────────────────────
# Tests for module constants
# ─────────────────────────────────────────────────────────────────────────────
class TestModuleConstants:
    """Tests for module-level constants."""

    def test_min_mbmp_value(self):
        """Test MIN_MBMP_VALUE constant is defined."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        assert t2ch4.MIN_MBMP_VALUE == 0.3

    def test_max_mbmp_value(self):
        """Test MAX_MBMP_VALUE constant is defined."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        assert t2ch4.MAX_MBMP_VALUE == 1.08

    def test_lut_public_file_defined(self):
        """Test LUT_PUBLIC_FILE constant is defined."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        assert t2ch4.LUT_PUBLIC_FILE is not None
        assert "integrated_transmittances.json" in t2ch4.LUT_PUBLIC_FILE

    def test_lut_public_file_is_local_path(self):
        """Test LUT_PUBLIC_FILE points to a local package file."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        assert not t2ch4.LUT_PUBLIC_FILE.startswith("az://")
        assert os.path.exists(t2ch4.LUT_PUBLIC_FILE)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for TransmittanceCH4Interpolation base class
# ─────────────────────────────────────────────────────────────────────────────
class TestTransmittanceCH4Interpolation:
    """Tests for TransmittanceCH4Interpolation base class."""

    def test_init_default_params(self, sample_amf_arr, sample_mr_ch4_arr):
        """Test initialization with default parameters."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4Interpolation(sample_amf_arr, sample_mr_ch4_arr)

        assert interp.clip_values_retrieval is True
        assert interp.allow_negative_ch4 is True
        assert len(interp.cache) == 0

    def test_init_custom_params(self, sample_amf_arr, sample_mr_ch4_arr):
        """Test initialization with custom parameters."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4Interpolation(
            sample_amf_arr,
            sample_mr_ch4_arr,
            background_concentration=1900,
            clip_values_retrieval=False,
            allow_negative_ch4=False,
        )

        assert interp.background_concentration == 1900
        assert interp.clip_values_retrieval is False
        assert interp.allow_negative_ch4 is False

    def test_air_mass_factor(self, sample_amf_arr, sample_mr_ch4_arr):
        """Test air_mass_factor method."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4Interpolation(sample_amf_arr, sample_mr_ch4_arr)

        # Test with typical sun/view angles
        amf = interp.air_mass_factor(sza=30, vza=10)

        assert isinstance(amf, (int, float, np.floating))
        assert amf > 0
        # AMF should be clipped to max value
        assert amf <= interp.amf_arr_max

    def test_air_mass_factor_extreme_angles(self, sample_amf_arr, sample_mr_ch4_arr):
        """Test air_mass_factor with extreme angles."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4Interpolation(sample_amf_arr, sample_mr_ch4_arr)

        # Very high zenith angle should be clipped
        amf = interp.air_mass_factor(sza=80, vza=60)

        assert amf <= interp.amf_arr_max

    def test_deltach4_to_ch4_numpy(self, sample_amf_arr, sample_mr_ch4_arr):
        """Test _deltach4_to_ch4 with numpy array."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4Interpolation(
            sample_amf_arr, sample_mr_ch4_arr, background_concentration=1900
        )

        deltach4 = np.array([[100, 200], [300, 400]])

        ch4 = interp._deltach4_to_ch4(deltach4)

        np.testing.assert_array_equal(ch4, deltach4 + 1900)

    def test_deltach4_to_ch4_geotensor(self, sample_amf_arr, sample_mr_ch4_arr):
        """Test _deltach4_to_ch4 with GeoTensor."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4Interpolation(
            sample_amf_arr, sample_mr_ch4_arr, background_concentration=1900
        )

        values = np.array([[100.0, 200.0], [300.0, -1.0]])  # -1 is fill value
        transform = Affine.identity()
        deltach4 = GeoTensor(values, transform=transform, crs="EPSG:4326", fill_value_default=-1)

        ch4 = interp._deltach4_to_ch4(deltach4)

        assert isinstance(ch4, GeoTensor)
        # Non-fill values should have background added
        assert ch4.values[0, 0] == 2000.0  # 100 + 1900
        assert ch4.values[0, 1] == 2100.0  # 200 + 1900
        # Fill value should be set to 0
        assert ch4.values[1, 1] == 0

    def test_load_interpolation_functions_not_implemented(self, sample_amf_arr, sample_mr_ch4_arr):
        """Test that base class raises NotImplementedError."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4Interpolation(sample_amf_arr, sample_mr_ch4_arr)

        with pytest.raises(NotImplementedError):
            interp.load_interpolation_functions_satellite("S2A")


# ─────────────────────────────────────────────────────────────────────────────
# Tests for TransmittanceCH4InterpolationFromLUT
# ─────────────────────────────────────────────────────────────────────────────

@patch(
    "marss2l.mars_sentinel2.mixing_ratio_methane.load_srfinterpfun",
    side_effect=_mock_load_srfinterpfun,
)
class TestTransmittanceCH4InterpolationFromLUT:
    """Tests for TransmittanceCH4InterpolationFromLUT class."""

    def test_init_default_params(self, mock_srf):
        """Test initialization with default parameters."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        assert interp.with_ltoa_correction is True
        assert interp.trans_tot_as_tbg is False
        assert interp.wvl_mod is not None
        assert interp.t_full_arr is not None

    def test_init_custom_params(self, mock_srf):
        """Test initialization with custom parameters."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT(
            with_ltoa_correction=False, background_concentration=1850
        )

        assert interp.with_ltoa_correction is False
        assert interp.background_concentration == 1850

    def test_init_invalid_params(self, mock_srf):
        """Test initialization with invalid parameter combination."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        with pytest.raises(ValueError, match="total atmospheric transmittance"):
            t2ch4.TransmittanceCH4InterpolationFromLUT(
                with_ltoa_correction=False, trans_tot_as_tbg=True
            )

    def test_t_background_transmittance_property(self, mock_srf):
        """Test t_background_transmittance lazy property."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        t_bg = interp.t_background_transmittance

        assert t_bg is not None
        assert t_bg.shape[0] == len(interp.amf_arr)
        assert t_bg.shape[1] == len(interp.wvl_mod)

        # Second call should return cached value
        t_bg2 = interp.t_background_transmittance
        assert t_bg is t_bg2

    def test_load_interpolation_functions_satellite_s2a(self, mock_srf):
        """Test loading interpolation functions for S2A."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        funcs = interp.load_interpolation_functions_satellite("S2A")

        assert "interpolation_funtion_amf_transmittance_b12" in funcs
        assert "interpolation_funtion_amf_transmittance_b11" in funcs
        assert "interpolation_funtion_amf_transmittance_b12_bg" in funcs
        assert "interpolation_funtion_amf_transmittance_b11_bg" in funcs
        assert callable(funcs["interpolation_funtion_amf_transmittance_b12"])

    def test_load_interpolation_functions_caching(self, mock_srf):
        """Test that interpolation functions are cached."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        funcs1 = interp.load_interpolation_functions_satellite("S2A")
        funcs2 = interp.load_interpolation_functions_satellite("S2A")

        assert funcs1 is funcs2

    def test_load_interpolation_functions_landsat(self, mock_srf):
        """Test loading interpolation functions for Landsat."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        funcs = interp.load_interpolation_functions_satellite("LC08")

        assert "interpolation_funtion_amf_transmittance_b12" in funcs

    def test_deltach4_from_ratio_transmittance_numpy(self, mock_srf):
        """Test deltach4_from_ratio_transmittance with numpy array."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        H, W = 20, 10
        np.random.seed(42)
        ratio_il = np.random.rand(H, W) * 0.2 + 0.9  # Values around 1

        result = interp.deltach4_from_ratio_transmittance(
            satellite="S2A", sza=30, vza=10, ratio_il=ratio_il
        )

        assert result.shape == (H, W)
        assert np.all(np.isfinite(result))

    def test_deltach4_from_ratio_transmittance_geotensor(self, mock_srf, sample_ratio_il):
        """Test deltach4_from_ratio_transmittance with GeoTensor."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        result = interp.deltach4_from_ratio_transmittance(
            satellite="S2A", sza=30, vza=10, ratio_il=sample_ratio_il
        )

        assert isinstance(result, GeoTensor)
        assert result.shape == sample_ratio_il.shape

    def test_deltach4_from_ratio_transmittance_no_negative(self, mock_srf):
        """Test deltach4_from_ratio_transmittance with allow_negative_ch4=False."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        H, W = 20, 10
        ratio_il = np.ones((H, W)) * 1.05  # High ratio -> negative delta

        result = interp.deltach4_from_ratio_transmittance(
            satellite="S2A",
            sza=30,
            vza=10,
            ratio_il=ratio_il,
            allow_negative_ch4=False,
        )

        # With allow_negative_ch4=False, should be clipped to >= 0
        assert np.all(result >= 0)

    def test_transmittance_B12_B11_numpy(self, mock_srf):
        """Test transmittance_B12_B11 with numpy array."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        H, W = 20, 10
        deltach4 = np.random.rand(H, W) * 500

        t_b12, t_b11 = interp.transmittance_B12_B11(
            satellite="S2A", sza=30, vza=10, deltach4=deltach4
        )

        assert t_b12.shape == (H, W)
        assert t_b11.shape == (H, W)
        # Transmittances should be positive
        assert np.all(t_b12 > 0)
        assert np.all(t_b11 > 0)

    def test_transmittance_B12_B11_geotensor(self, mock_srf, sample_geotensor):
        """Test transmittance_B12_B11 with GeoTensor."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        t_b12, t_b11 = interp.transmittance_B12_B11(
            satellite="S2A", sza=30, vza=10, deltach4=sample_geotensor
        )

        assert isinstance(t_b12, GeoTensor)
        assert isinstance(t_b11, GeoTensor)

    def test_transmittance_B12_B11_with_amf(self, mock_srf):
        """Test transmittance_B12_B11 with explicit AMF."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        H, W = 10, 10
        deltach4 = np.ones((H, W)) * 200

        t_b12, t_b11 = interp.transmittance_B12_B11(
            satellite="S2A", sza=None, vza=None, deltach4=deltach4, amf=3.0
        )

        assert t_b12.shape == (H, W)
        assert t_b11.shape == (H, W)

    def test_deltach4_from_B12_transmittance(self, mock_srf):
        """Test deltach4_from_B12_transmittance."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        H, W = 10, 10
        trans_b12 = np.ones((H, W)) * 0.95

        result = interp.deltach4_from_B12_transmittance(
            satellite="S2A", sza=30, vza=10, trans_b12=trans_b12
        )

        assert result.shape == (H, W)

    def test_deltach4_from_B11_transmittance(self, mock_srf):
        """Test deltach4_from_B11_transmittance."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        H, W = 10, 10
        trans_b11 = np.ones((H, W)) * 0.98

        result = interp.deltach4_from_B11_transmittance(
            satellite="S2A", sza=30, vza=10, trans_b11=trans_b11
        )

        assert result.shape == (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for TransmittanceCH4InterpolationFromDict
# ─────────────────────────────────────────────────────────────────────────────
class TestTransmittanceCH4InterpolationFromDict:
    """Tests for TransmittanceCH4InterpolationFromDict class."""

    def test_init_from_local_json_file(self, tmp_path):
        """Test initialization from a local JSON file path."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        data = {
            "amf_arr": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "mr_ch4_arr": [[1700, 1800, 1850, 1900, 1950, 2000, 2100, 2200, 2500]] * 8,
            "background_concentration": 1900,
            "S2A": {
                "transmittance_b12": [[0.95] * 9] * 8,
                "transmittance_b11": [[0.98] * 9] * 8,
                "transmittance_b12_bg": [0.96] * 8,
                "transmittance_b11_bg": [0.99] * 8,
            },
        }
        json_path = tmp_path / "integrated_transmittances.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        interp = t2ch4.TransmittanceCH4InterpolationFromDict(str(json_path))

        assert interp.background_concentration == 1900
        assert len(interp.amf_arr) == 8

    def test_init_from_dict(self):
        """Test initialization from dictionary."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        # Create a minimal valid dictionary
        data = {
            "amf_arr": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "mr_ch4_arr": [[1700, 1800, 1850, 1900, 1950, 2000, 2100, 2200, 2500]] * 8,
            "background_concentration": 1900,
            "S2A": {
                "transmittance_b12": [[0.95] * 9] * 8,
                "transmittance_b11": [[0.98] * 9] * 8,
                "transmittance_b12_bg": [0.96] * 8,
                "transmittance_b11_bg": [0.99] * 8,
            },
        }

        interp = t2ch4.TransmittanceCH4InterpolationFromDict(data)

        assert interp.background_concentration == 1900
        assert len(interp.amf_arr) == 8

    def test_load_interpolation_functions_from_dict(self):
        """Test loading interpolation functions from dict."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        data = {
            "amf_arr": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "mr_ch4_arr": [[1700, 1800, 1850, 1900, 1950, 2000, 2100, 2200, 2500]] * 8,
            "background_concentration": 1900,
            "S2A": {
                "transmittance_b12": [[0.95] * 9] * 8,
                "transmittance_b11": [[0.98] * 9] * 8,
                "transmittance_b12_bg": [0.96] * 8,
                "transmittance_b11_bg": [0.99] * 8,
            },
        }

        interp = t2ch4.TransmittanceCH4InterpolationFromDict(data)
        funcs = interp.load_interpolation_functions_satellite("S2A")

        assert "interpolation_funtion_amf_transmittance_b12" in funcs
        assert callable(funcs["interpolation_funtion_amf_transmittance_b12"])

    def test_load_interpolation_invalid_satellite(self):
        """Test loading interpolation for invalid satellite raises error."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        data = {
            "amf_arr": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "mr_ch4_arr": [[1700, 1800, 1850, 1900, 1950, 2000, 2100, 2200, 2500]] * 8,
            "background_concentration": 1900,
        }

        interp = t2ch4.TransmittanceCH4InterpolationFromDict(data)

        with pytest.raises(ValueError, match="not in the data"):
            interp.load_interpolation_functions_satellite("INVALID")


# ─────────────────────────────────────────────────────────────────────────────
# Tests for export_transmittances function
# ─────────────────────────────────────────────────────────────────────────────
@patch(
    "marss2l.mars_sentinel2.mixing_ratio_methane.load_srfinterpfun",
    side_effect=_mock_load_srfinterpfun,
)
class TestExportTransmittances:
    """Tests for export_transmittances function."""

    def test_export_transmittances_default(self, mock_srf):
        """Test export_transmittances with default parameters."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        result = t2ch4.export_transmittances(satellites=["S2A"])

        assert "amf_arr" in result
        assert "mr_ch4_arr" in result
        assert "background_concentration" in result
        assert "S2A" in result
        assert "transmittance_b12" in result["S2A"]

    def test_export_transmittances_multiple_satellites(self, mock_srf):
        """Test export_transmittances with multiple satellites."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        result = t2ch4.export_transmittances(satellites=["S2A", "S2B"])

        assert "S2A" in result
        assert "S2B" in result

    def test_export_transmittances_custom_params(self, mock_srf):
        """Test export_transmittances with custom parameters."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        result = t2ch4.export_transmittances(
            satellites=["S2A"],
            with_ltoa_correction=False,
            background_concentration=1850,
        )

        assert result["background_concentration"] == 1850
        assert result["with_ltoa_correction"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────────────────
@patch(
    "marss2l.mars_sentinel2.mixing_ratio_methane.load_srfinterpfun",
    side_effect=_mock_load_srfinterpfun,
)
class TestIntegration:
    """Integration tests for the module."""

    def test_round_trip_transmittance_ch4(self, mock_srf):
        """Test round-trip: deltaCH4 -> transmittance -> deltaCH4."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        # Start with known deltaCH4
        original_deltach4 = np.array([[100, 200, 300], [400, 500, 600]])

        # Get transmittances
        t_b12, t_b11 = interp.transmittance_B12_B11(
            satellite="S2A", sza=30, vza=10, deltach4=original_deltach4
        )

        # The transmittance values should be different for different deltaCH4
        assert not np.allclose(t_b12[0, 0], t_b12[1, 2])

    def test_different_satellites_different_results(self, mock_srf):
        """Test that different satellites give different results."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        H, W = 10, 10
        ratio_il = np.ones((H, W)) * 0.95

        result_s2a = interp.deltach4_from_ratio_transmittance(
            satellite="S2A", sza=30, vza=10, ratio_il=ratio_il
        )
        result_lc08 = interp.deltach4_from_ratio_transmittance(
            satellite="LC08", sza=30, vza=10, ratio_il=ratio_il
        )

        # Different satellites should give different results
        # due to different mock SRF centers per satellite
        assert not np.allclose(result_s2a, result_lc08)

    def test_clip_values_range(self, mock_srf):
        """Test that clip_values_retrieval clips input to expected range."""
        from marss2l.mars_sentinel2 import transmittance_to_ch4 as t2ch4

        interp = t2ch4.TransmittanceCH4InterpolationFromLUT()

        H, W = 10, 10
        # Value exactly at MIN_MBMP_VALUE should be the same with or without clipping
        ratio_il_at_min = np.ones((H, W)) * t2ch4.MIN_MBMP_VALUE

        result_at_min = interp.deltach4_from_ratio_transmittance(
            satellite="S2A",
            sza=30,
            vza=10,
            ratio_il=ratio_il_at_min.copy(),
            clip_values_retrieval=True,
        )

        # Value below MIN_MBMP_VALUE should be clipped to MIN_MBMP_VALUE
        ratio_il_below_min = np.ones((H, W)) * (t2ch4.MIN_MBMP_VALUE - 0.1)
        result_below_min = interp.deltach4_from_ratio_transmittance(
            satellite="S2A",
            sza=30,
            vza=10,
            ratio_il=ratio_il_below_min.copy(),
            clip_values_retrieval=True,
        )

        # After clipping, result_below_min should equal result_at_min
        np.testing.assert_allclose(result_at_min, result_below_min)
