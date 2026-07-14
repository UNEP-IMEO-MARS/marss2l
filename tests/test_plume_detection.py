"""Tests for marss2l.mars_sentinel2.plume_detection module.

This module tests plume detection algorithms including:
- binary_connected_prediction: Threshold and filter connected components
- count_connected_pixels: Count pixels in valid connected components
- threshold_cutoff_connected_components: Find optimal threshold via binary search
"""

import numpy as np
import pytest
from georeader.geotensor import GeoTensor
from marss2l.mars_sentinel2.plume_detection import (
    MINIMUM_NUMBER_PIXELS_PLUME,
    binary_connected_prediction,
    count_connected_pixels,
    threshold_cutoff_connected_components,
)
from rasterio.crs import CRS
from rasterio.transform import Affine


def create_test_geotensor(values: np.ndarray) -> GeoTensor:
    """Create a GeoTensor from numpy array for testing."""
    transform = Affine.translation(0, 0) * Affine.scale(1, -1)
    crs = CRS.from_epsg(4326)
    return GeoTensor(values, transform=transform, crs=crs)


class TestBinaryConnectedPrediction:
    """Tests for binary_connected_prediction function."""

    def test_returns_ndarray_for_ndarray_input(self):
        pred = np.random.rand(100, 100)
        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_returns_geotensor_for_geotensor_input(self):
        values = np.random.rand(100, 100)
        pred = create_test_geotensor(values)
        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)
        assert isinstance(result, GeoTensor)
        assert result.values.dtype == np.uint8

    def test_binary_output_values(self):
        pred = np.random.rand(100, 100)
        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)
        unique_values = np.unique(result)
        assert all(v in [0, 1] for v in unique_values)

    def test_removes_small_clusters(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[10:20, 10:30] = 0.9  # 200 pixels
        pred[50:53, 50:53] = 0.9  # 9 pixels

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=50)

        assert np.sum(result[10:20, 10:30]) > 0
        assert np.sum(result[50:53, 50:53]) == 0

    def test_keeps_large_clusters(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[20:40, 20:40] = 0.9

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=100)

        assert np.sum(result[20:40, 20:40]) == 400

    def test_binary_connected_prediction_many_small_clusters(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[5:7, 5:7] = 0.9
        pred[5:7, 30:32] = 0.9
        pred[5:7, 60:62] = 0.9
        pred[50:52, 5:7] = 0.9
        pred[50:52, 30:32] = 0.9
        pred[50:52, 60:62] = 0.9

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)

        assert np.sum(result) == 0

    def test_threshold_prediction_applied(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[20:40, 20:40] = 0.6
        pred[50:70, 50:70] = 0.4

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)

        assert np.sum(result[20:40, 20:40]) > 0
        assert np.sum(result[50:70, 50:70]) == 0

    def test_empty_prediction_returns_zeros(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)
        assert np.sum(result) == 0

    def test_all_ones_prediction(self):
        pred = np.ones((100, 100), dtype=np.float32)
        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)
        assert np.sum(result) == 10000

    def test_geotensor_preserves_metadata(self):
        values = np.random.rand(100, 100)
        transform = Affine.translation(10, 20) * Affine.scale(0.1, -0.1)
        crs = CRS.from_epsg(32632)
        pred = GeoTensor(values, transform=transform, crs=crs)

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)

        assert result.transform == transform
        assert result.crs == crs


class TestCountConnectedPixels:
    """Tests for count_connected_pixels function."""

    def test_returns_integer(self):
        pred = np.random.rand(100, 100)
        result = count_connected_pixels(pred, threshold_prediction=0.5, threshold_pixels=10)
        assert isinstance(result, int)

    def test_empty_prediction_returns_zero(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        result = count_connected_pixels(pred, threshold_prediction=0.5, threshold_pixels=10)
        assert result == 0

    def test_counts_pixels_in_large_cluster(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[20:40, 20:40] = 0.9

        result = count_connected_pixels(pred, threshold_prediction=0.5, threshold_pixels=100)

        assert result == 400

    def test_excludes_small_clusters(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[20:25, 20:25] = 0.9

        result = count_connected_pixels(pred, threshold_prediction=0.5, threshold_pixels=100)

        assert result == 0

    def test_multiple_clusters(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[10:20, 10:20] = 0.9
        pred[50:60, 50:60] = 0.9

        result = count_connected_pixels(pred, threshold_prediction=0.5, threshold_pixels=50)

        assert result == 200

    def test_works_with_geotensor(self):
        values = np.zeros((100, 100), dtype=np.float32)
        values[20:40, 20:40] = 0.9
        pred = create_test_geotensor(values)

        result = count_connected_pixels(pred, threshold_prediction=0.5, threshold_pixels=100)

        assert result == 400


class TestThresholdCutoffConnectedComponents:
    """Tests for threshold_cutoff_connected_components function."""

    def test_returns_float(self):
        pred = np.random.rand(100, 100)
        result = threshold_cutoff_connected_components(pred, threshold_pixels=100)
        assert isinstance(result, float)

    def test_threshold_within_value_range(self):
        pred = np.random.rand(100, 100) * 0.5 + 0.25
        result = threshold_cutoff_connected_components(pred, threshold_pixels=100)
        assert 0.25 <= result <= 0.75

    def test_higher_threshold_pixels_lower_threshold(self):
        pred = np.random.rand(100, 100)

        result_low = threshold_cutoff_connected_components(pred, threshold_pixels=100)
        result_high = threshold_cutoff_connected_components(pred, threshold_pixels=1000)

        assert result_high <= result_low

    def test_uniform_values_returns_that_value(self):
        pred = np.full((100, 100), 0.5, dtype=np.float32)
        result = threshold_cutoff_connected_components(pred, threshold_pixels=100, tol=1e-3)
        assert abs(result - 0.5) < 0.01

    def test_works_with_geotensor(self):
        values = np.random.rand(100, 100)
        pred = create_test_geotensor(values)

        result = threshold_cutoff_connected_components(pred, threshold_pixels=100)

        assert isinstance(result, float)

    def test_custom_tolerance(self):
        pred = np.random.rand(100, 100)

        result_tight = threshold_cutoff_connected_components(pred, threshold_pixels=100, tol=1e-6)
        result_loose = threshold_cutoff_connected_components(pred, threshold_pixels=100, tol=1e-2)

        assert isinstance(result_tight, float)
        assert isinstance(result_loose, float)

    def test_threshold_cutoff_tolerance_sensitivity(self):
        np.random.seed(42)
        pred = np.zeros((100, 100), dtype=np.float32)
        for i in range(100):
            pred[i, :] = i / 100.0

        threshold_pixels = 500

        result_tight = threshold_cutoff_connected_components(
            pred, threshold_pixels=threshold_pixels, tol=0.001
        )
        result_loose = threshold_cutoff_connected_components(
            pred, threshold_pixels=threshold_pixels, tol=0.1
        )

        assert isinstance(result_tight, (float, np.floating))
        assert isinstance(result_loose, (float, np.floating))
        assert 0.0 <= result_tight <= 1.0
        assert 0.0 <= result_loose <= 1.0
        assert result_tight != result_loose

    def test_finds_threshold_for_exact_pixel_count(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        for i in range(100):
            pred[i, :] = i / 100.0

        threshold_pixels = 500
        result = threshold_cutoff_connected_components(
            pred, threshold_pixels=threshold_pixels, tol=0.01
        )

        actual_pixels = count_connected_pixels(pred, result, threshold_pixels=threshold_pixels)

        assert actual_pixels >= threshold_pixels or actual_pixels == 0


class TestMinimumNumberPixelsConstant:
    """Tests for MINIMUM_NUMBER_PIXELS_PLUME constant."""

    def test_constant_value(self):
        assert MINIMUM_NUMBER_PIXELS_PLUME == 150

    def test_used_as_default(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[20:35, 20:35] = 0.9  # 225 pixels - above default threshold

        result = binary_connected_prediction(pred, threshold_prediction=0.5)

        assert np.sum(result) == 225


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_pixel_above_threshold(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[50, 50] = 0.9

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=2)

        assert np.sum(result) == 0

    def test_diagonal_connectivity(self):
        pred = np.zeros((10, 10), dtype=np.float32)
        for i in range(5):
            pred[i, i] = 0.9

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=3)

        assert np.sum(result) == 5

    def test_narrow_connection(self):
        pred = np.zeros((20, 20), dtype=np.float32)
        pred[2:5, 2:5] = 0.9
        pred[5, 5] = 0.9
        pred[6:9, 6:9] = 0.9

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=10)

        assert np.sum(result) == 19

    def test_very_large_array(self):
        pred = np.random.rand(1000, 1000).astype(np.float32)

        result = binary_connected_prediction(pred, threshold_prediction=0.5, threshold_pixels=1000)

        assert result.shape == (1000, 1000)

    def test_negative_values(self):
        pred = np.random.rand(100, 100) * 2 - 1

        result = binary_connected_prediction(pred, threshold_prediction=-0.5, threshold_pixels=10)

        assert result.dtype == np.uint8

    def test_nan_values(self):
        pred = np.random.rand(100, 100).astype(np.float32)
        pred[50, 50] = np.nan

        result = binary_connected_prediction(
            pred, threshold_prediction=0.5, threshold_pixels=10
        )

        assert result.shape == (100, 100)
