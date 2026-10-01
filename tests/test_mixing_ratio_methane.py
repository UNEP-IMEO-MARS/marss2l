"""
Tests for marss2l.mars_sentinel2.mixing_ratio_methane module.

Tests cover:
- ratio_bands (band ratio calculation)
- mbsp_varon (single-pass multi-band)
- mbmp_varon (multi-band multi-pass)
- ratio_IL (Irakulis-Loritxate ratio)
- difference_bands (band difference calculation)
- apply_interpfun_to_image (interpolation helper)
- srf_landsat_band (SRF loading)
- load_srfinterpfun (SRF interpolation functions)
"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_geotensor():
    """Create a sample GeoTensor for testing."""
    H, W, C = 50, 30, 13
    np.random.seed(42)
    values = np.random.rand(C, H, W) * 5000 + 5000  # Typical S2 reflectance values
    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    return GeoTensor(values, transform=transform, crs="EPSG:32630", fill_value_default=0)


@pytest.fixture
def sample_image_pair():
    """Create a pair of GeoTensors for MBMP testing."""
    H, W, C = 50, 30, 13
    np.random.seed(42)
    current = np.random.rand(C, H, W) * 5000 + 5000
    background = np.random.rand(C, H, W) * 5000 + 5000
    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    return (
        GeoTensor(current, transform=transform, crs="EPSG:32630", fill_value_default=0),
        GeoTensor(background, transform=transform, crs="EPSG:32630", fill_value_default=0),
    )


@pytest.fixture
def sample_validmask():
    """Create a sample valid mask."""
    H, W = 50, 30
    mask = np.ones((H, W), dtype=bool)
    mask[0:5, 0:5] = False  # Some invalid pixels
    transform = Affine.translation(0, 0) * Affine.scale(10, -10)
    return GeoTensor(mask, transform=transform, crs="EPSG:32630", fill_value_default=False)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for mbsp_varon
# ─────────────────────────────────────────────────────────────────────────────
class TestMBSPVaron:
    """Tests for mbsp_varon function (Single-Pass Multi-Band)."""

    def test_mbsp_varon_basic(self):
        """Test basic MBSP calculation."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        H, W = 20, 10
        np.random.seed(42)
        signal = np.random.rand(H, W) * 0.5 + 0.5
        background = np.random.rand(H, W) * 0.5 + 0.5

        result = mrm.mbsp_varon(background, signal)

        assert result.shape == (H, W)
        assert np.all(np.isfinite(result))

    def test_mbsp_varon_with_validmask(self):
        """Test MBSP with valid mask."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        H, W = 20, 10
        np.random.seed(42)
        signal = np.random.rand(H, W) * 0.5 + 0.5
        background = np.random.rand(H, W) * 0.5 + 0.5
        validmask = np.ones((H, W), dtype=bool)
        validmask[0:5, :] = False

        result = mrm.mbsp_varon(background, signal, validmask=validmask)

        assert result.shape == (H, W)

    def test_mbsp_varon_zero_background(self):
        """Test MBSP handles near-zero background."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        H, W = 10, 10
        signal = np.ones((H, W)) * 0.5
        background = np.ones((H, W)) * 0.001  # Very small background

        result = mrm.mbsp_varon(background, signal)

        # Should not produce inf values due to clipping
        assert np.all(np.isfinite(result))

    def test_mbsp_varon_no_valid_values(self):
        """Test MBSP when no valid values exist."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        H, W = 10, 10
        signal = np.ones((H, W)) * 0.5
        background = np.ones((H, W)) * 0.5
        validmask = np.zeros((H, W), dtype=bool)  # All invalid

        result = mrm.mbsp_varon(background, signal, validmask=validmask)

        # Should use c=1 fallback
        assert result.shape == (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for ratio_bands
# ─────────────────────────────────────────────────────────────────────────────
class TestRatioBands:
    """Tests for ratio_bands function."""

    def test_ratio_bands_basic(self, sample_geotensor):
        """Test basic ratio calculation."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        result = mrm.ratio_bands(sample_geotensor, numerator_index=12, denominator_index=11)

        # Should return GeoTensor with shape (H, W)
        assert result.shape == sample_geotensor.shape[1:]

    def test_ratio_bands_with_validmask(self, sample_geotensor, sample_validmask):
        """Test ratio with valid mask."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        result = mrm.ratio_bands(
            sample_geotensor,
            numerator_index=12,
            denominator_index=11,
            validmask=sample_validmask,
        )

        assert result.shape == sample_geotensor.shape[1:]

    def test_ratio_bands_no_normalize(self, sample_geotensor):
        """Test ratio without normalization."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        result = mrm.ratio_bands(
            sample_geotensor, numerator_index=12, denominator_index=11, normalize=False
        )

        assert result.shape == sample_geotensor.shape[1:]

    def test_ratio_bands_numpy_array(self):
        """Test ratio with numpy array input."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        C, H, W = 13, 20, 10
        np.random.seed(42)
        image = np.random.rand(C, H, W) * 5000 + 5000

        result = mrm.ratio_bands(image, numerator_index=12, denominator_index=11)

        assert result.shape == (H, W)

    def test_ratio_bands_clips_values(self, sample_geotensor):
        """Test that ratio values are clipped to [0, 10]."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        result = mrm.ratio_bands(sample_geotensor, numerator_index=12, denominator_index=11)

        result_values = result.values if hasattr(result, "values") else result
        # After normalization and clipping, values should be reasonable
        assert np.all(result_values >= 0)
        assert np.all(result_values <= 10)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for apply_interpfun_to_image
# ─────────────────────────────────────────────────────────────────────────────
class TestApplyInterpfunToImage:
    """Tests for apply_interpfun_to_image function."""

    def test_apply_interpfun_numpy_array(self):
        """Test interpolation with numpy array."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        def linear_interp(x):
            return x * 2

        image = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = mrm.apply_interpfun_to_image(linear_interp, image)

        np.testing.assert_array_equal(result, image * 2)

    def test_apply_interpfun_geotensor(self):
        """Test interpolation with GeoTensor."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        def linear_interp(x):
            return x * 2

        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        transform = Affine.identity()
        geotensor = GeoTensor(values, transform=transform, crs="EPSG:4326", fill_value_default=-999)

        result = mrm.apply_interpfun_to_image(linear_interp, geotensor)

        assert isinstance(result, GeoTensor)
        np.testing.assert_array_almost_equal(result.values, values * 2)

    def test_apply_interpfun_handles_fill_value(self):
        """Test interpolation handles fill values correctly."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        def linear_interp(x):
            return x * 2

        values = np.array([[1.0, -1.0], [3.0, 4.0]])
        transform = Affine.identity()
        geotensor = GeoTensor(values, transform=transform, crs="EPSG:4326", fill_value_default=-1.0)

        result = mrm.apply_interpfun_to_image(linear_interp, geotensor)

        # Fill value pixels should be preserved
        assert result.values[0, 1] == -1.0

    def test_apply_interpfun_custom_fill_value(self):
        """Test interpolation with custom fill value."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        def linear_interp(x):
            return x * 2

        values = np.array([[1.0, np.nan], [3.0, 4.0]])
        transform = Affine.identity()
        geotensor = GeoTensor(values, transform=transform, crs="EPSG:4326", fill_value_default=0)

        result = mrm.apply_interpfun_to_image(linear_interp, geotensor, fill_value_default=-999)

        assert result.fill_value_default == -999


# ─────────────────────────────────────────────────────────────────────────────
# Tests for module constants
# ─────────────────────────────────────────────────────────────────────────────
class TestModuleConstants:
    """Tests for module-level constants and imports."""

    def test_file_lut_gas_defined(self):
        """Test FILE_LUT_GAS constant is defined."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        assert mrm.FILE_LUT_GAS is not None

    def test_fill_value_ratio_il_defined(self):
        """Test FILL_VALUE_RATIO_IL constant is defined."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        assert mrm.FILL_VALUE_RATIO_IL == 0

    def test_link_rsr_landsat_defined(self):
        """Test LINK_RSR_LANDSAT dictionary is defined."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        assert "LC08" in mrm.LINK_RSR_LANDSAT
        assert "LC09" in mrm.LINK_RSR_LANDSAT
        assert "LT05" in mrm.LINK_RSR_LANDSAT

    def test_band_to_sheet_name_defined(self):
        """Test band to sheet name mappings are defined."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        assert "B06" in mrm.BAND_TO_SHEET_NAME_L89
        assert "B07" in mrm.BAND_TO_SHEET_NAME_L89

    def test_load_all_lut_callable(self):
        """Test load_all_lut is callable."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        assert callable(mrm.load_all_lut)

    def test_air_mass_factor_callable(self):
        """Test air_mass_factor is callable."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        assert callable(mrm.air_mass_factor)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for srf_landsat_band
# ─────────────────────────────────────────────────────────────────────────────
class TestSRFLandsatBand:
    """Tests for srf_landsat_band function."""

    def test_bundled_srf_files_exist(self):
        """Every satellite in LINK_RSR_LANDSAT has its workbook shipped."""
        import os

        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        assert set(mrm.SRF_FILE_DEFAULT) == set(mrm.LINK_RSR_LANDSAT)
        for satellite, path in mrm.SRF_FILE_DEFAULT.items():
            assert os.path.exists(path), f"missing bundled SRF for {satellite}"

    def test_srf_landsat_reads_bundled_file(self):
        """The default path needs no network: it reads the bundled workbook."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        srf = mrm.srf_landsat_band("LC08", "B06", cache=False)
        assert list(srf.columns) == ["B06"]
        assert srf.index.name == "wavelength"
        assert len(srf) > 0

    def test_srf_landsat_invalid_satellite(self):
        """Test srf_landsat_band raises for invalid satellite."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        with pytest.raises(AssertionError):
            mrm.srf_landsat_band("INVALID", "B06")

    def test_srf_landsat_invalid_band(self):
        """Test srf_landsat_band raises for invalid band."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        with pytest.raises(AssertionError):
            mrm.srf_landsat_band("LC08", "B99")


# ─────────────────────────────────────────────────────────────────────────────
# Tests for load_srfinterpfun
# ─────────────────────────────────────────────────────────────────────────────
class TestLoadSRFInterpfun:
    """Tests for load_srfinterpfun function."""

    def test_load_srfinterpfun_invalid_satellite(self):
        """Test load_srfinterpfun raises for invalid satellite."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        with pytest.raises(ValueError, match="not recognized"):
            mrm.load_srfinterpfun("INVALID")


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests with mocked dependencies
# ─────────────────────────────────────────────────────────────────────────────
class TestIntegration:
    """Integration tests with mocked external dependencies."""

    def test_ratio_bands_integration(self):
        """Test ratio_bands with realistic data."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        # Create realistic S2-like data
        C, H, W = 13, 100, 100
        np.random.seed(42)

        # B11 and B12 with some correlation
        b11 = np.random.rand(H, W) * 3000 + 2000
        b12 = b11 * 0.9 + np.random.rand(H, W) * 500  # Slightly lower

        image = np.zeros((C, H, W))
        image[11] = b11
        image[12] = b12

        result = mrm.ratio_bands(image, numerator_index=12, denominator_index=11)

        assert result.shape == (H, W)
        # Ratio should be around 0.9-1.1 after normalization
        assert np.nanmean(result) < 2.0

    def test_mbsp_integration(self):
        """Test MBSP with realistic data."""
        from marss2l.mars_sentinel2 import mixing_ratio_methane as mrm

        H, W = 100, 100
        np.random.seed(42)

        # Background and signal with small difference
        background = np.random.rand(H, W) * 0.3 + 0.2
        signal = background * 0.95 + np.random.rand(H, W) * 0.02

        result = mrm.mbsp_varon(background, signal)

        assert result.shape == (H, W)
        # Result should show small enhancement
        assert np.abs(np.mean(result)) < 1.0
