"""
Tests for marss2l georeader utility functions.

Tests cover:
- marss2l.mars_sentinel2.utils_torch (find_padding, padded_predict)
- marss2l.mars_sentinel2.utils (values, class_counts, get_channels_to_pred, align_images, corregister_images)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from georeader.geotensor import GeoTensor
from marss2l.mars_sentinel2.utils import (
    align_images,
    class_counts,
    corregister_images,
    get_channels_to_pred,
    values,
)
from marss2l.mars_sentinel2.utils_torch import (
    find_padding,
    padded_predict,
)
from numpy.typing import NDArray
from rasterio.transform import Affine

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_geotensor():
    """Create a sample GeoTensor for testing."""
    data = np.random.rand(4, 64, 64).astype(np.float32)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    crs = "EPSG:4326"
    return GeoTensor(data, transform=transform, crs=crs)


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    return torch.nn.Conv2d(4, 1, kernel_size=3, padding=1)


@pytest.fixture
def identity_model():
    """Create a model that returns input (identity-like for testing shapes)."""
    model = torch.nn.Conv2d(4, 4, kernel_size=1)
    with torch.no_grad():
        model.weight.fill_(0)
        for i in range(4):
            model.weight[i, i, 0, 0] = 1.0
        model.bias.fill_(0)
    return model


# =============================================================================
# TestFindPadding
# =============================================================================


class TestFindPadding:
    """Tests for find_padding function"""

    def test_find_padding_no_padding_needed(self):
        pad1, pad2 = find_padding(64, divisor=8)
        assert pad1 == 0
        assert pad2 == 0

    def test_find_padding_needs_padding(self):
        pad1, pad2 = find_padding(65, divisor=8)
        total_pad = 72 - 65
        assert pad1 + pad2 == 7
        assert pad1 == 3
        assert pad2 == 4

    def test_find_padding_small_value(self):
        pad1, pad2 = find_padding(5, divisor=8)
        assert pad1 + pad2 == 3
        assert pad1 == 1
        assert pad2 == 2

    def test_find_padding_default_divisor(self):
        pad1, pad2 = find_padding(60)
        assert pad1 + pad2 == 4

    def test_find_padding_divisor_32(self):
        pad1, pad2 = find_padding(50, divisor=32)
        assert pad1 + pad2 == 14

    def test_find_padding_exact_divisor(self):
        pad1, pad2 = find_padding(32, divisor=32)
        assert pad1 == 0
        assert pad2 == 0

    def test_find_padding_value_zero(self):
        pad1, pad2 = find_padding(0, divisor=8)
        assert pad1 + pad2 == 8

    def test_find_padding_large_value(self):
        pad1, pad2 = find_padding(1000, divisor=32)
        assert pad1 + pad2 == 24

    def test_find_padding_symmetric_when_even(self):
        pad1, pad2 = find_padding(62, divisor=8)
        assert pad1 == 1
        assert pad2 == 1

    def test_find_padding_various_divisors(self):
        for divisor in [4, 8, 16, 32, 64]:
            for v in [10, 33, 100, 255]:
                pad1, pad2 = find_padding(v, divisor)
                result = v + pad1 + pad2
                assert result % divisor == 0


# =============================================================================
# TestPaddedPredict
# =============================================================================


class TestPaddedPredict:
    """Tests for padded_predict function"""

    def test_padded_predict_basic(self, simple_model):
        tensor = np.random.rand(4, 64, 64).astype(np.float32)
        output = padded_predict(tensor, simple_model, divisor=32)
        assert output is not None
        assert isinstance(output, np.ndarray)

    def test_padded_predict_preserves_spatial_dimensions(self, simple_model):
        tensor = np.random.rand(4, 65, 70).astype(np.float32)
        output = padded_predict(tensor, simple_model, divisor=32)
        assert output.shape[-2] == 65
        assert output.shape[-1] == 70

    def test_padded_predict_no_padding_needed(self, simple_model):
        tensor = np.random.rand(4, 64, 64).astype(np.float32)
        output = padded_predict(tensor, simple_model, divisor=32)
        assert output.shape[-2] == 64
        assert output.shape[-1] == 64

    def test_padded_predict_divisor_8(self, simple_model):
        tensor = np.random.rand(4, 60, 60).astype(np.float32)
        output = padded_predict(tensor, simple_model, divisor=8)
        assert output.shape[-2] == 60
        assert output.shape[-1] == 60

    def test_padded_predict_non_square(self, simple_model):
        tensor = np.random.rand(4, 50, 100).astype(np.float32)
        output = padded_predict(tensor, simple_model, divisor=32)
        assert output.shape[-2] == 50
        assert output.shape[-1] == 100

    def test_padded_predict_cpu_device(self, simple_model):
        tensor = np.random.rand(4, 64, 64).astype(np.float32)
        output = padded_predict(tensor, simple_model, divisor=32, device=torch.device("cpu"))
        assert output is not None

    def test_padded_predict_2d_output(self):
        class Model2D(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=1)

        model = Model2D()
        tensor = np.random.rand(4, 64, 64).astype(np.float32)
        output = padded_predict(tensor, model, divisor=32)
        assert len(output.shape) == 2

    def test_padded_predict_3d_output(self, simple_model):
        tensor = np.random.rand(4, 64, 64).astype(np.float32)
        output = padded_predict(tensor, simple_model, divisor=32)
        assert len(output.shape) == 3

    def test_padded_predict_invalid_2d_input(self, simple_model):
        tensor = np.random.rand(64, 64).astype(np.float32)
        with pytest.raises(AssertionError) as exc_info:
            padded_predict(tensor, simple_model, divisor=32)
        assert "Expected 3D tensor" in str(exc_info.value)

    def test_padded_predict_invalid_4d_input(self, simple_model):
        tensor = np.random.rand(1, 4, 64, 64).astype(np.float32)
        with pytest.raises(AssertionError) as exc_info:
            padded_predict(tensor, simple_model, divisor=32)
        assert "Expected 3D tensor" in str(exc_info.value)

    def test_padded_predict_small_tensor(self, simple_model):
        tensor = np.random.rand(4, 10, 10).astype(np.float32)
        output = padded_predict(tensor, simple_model, divisor=32)
        assert output.shape[-2] == 10
        assert output.shape[-1] == 10


# =============================================================================
# TestValues
# =============================================================================


class TestValues:
    """Tests for values function"""

    def test_values_from_ndarray(self):
        arr = np.array([[1, 2], [3, 4]])
        result = values(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_values_from_geotensor(self, sample_geotensor):
        result = values(sample_geotensor)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, sample_geotensor.values)

    def test_values_preserves_shape(self, sample_geotensor):
        result = values(sample_geotensor)
        assert result.shape == sample_geotensor.values.shape

    def test_values_preserves_dtype(self):
        arr = np.array([1.5, 2.5], dtype=np.float64)
        result = values(arr)
        assert result.dtype == np.float64


# =============================================================================
# TestClassCounts
# =============================================================================


class TestClassCounts:
    """Tests for class_counts function"""

    def test_class_counts_basic(self):
        data = np.array([0, 0, 1, 1, 1, 2])
        class_names = ["class_0", "class_1", "class_2"]
        result = class_counts(data, class_names)
        assert result["class_0"] == 2
        assert result["class_1"] == 3
        assert result["class_2"] == 1

    def test_class_counts_percentage(self):
        data = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        class_names = ["class_0", "class_1", "class_2"]
        result = class_counts(data, class_names, percentage=True)
        assert result["class_0"] == 20.0
        assert result["class_1"] == 40.0
        assert result["class_2"] == 40.0

    def test_class_counts_2d_array(self):
        data = np.array([[0, 0], [1, 1], [2, 2]])
        class_names = ["class_0", "class_1", "class_2"]
        result = class_counts(data, class_names)
        assert result["class_0"] == 2
        assert result["class_1"] == 2
        assert result["class_2"] == 2

    def test_class_counts_missing_class(self):
        data = np.array([0, 0, 2, 2])
        class_names = ["class_0", "class_1", "class_2"]
        result = class_counts(data, class_names)
        assert result["class_0"] == 2
        assert result["class_1"] == 0
        assert result["class_2"] == 2

    def test_class_counts_all_same_class(self):
        data = np.zeros((10,), dtype=int)
        class_names = ["class_0", "class_1"]
        result = class_counts(data, class_names)
        assert result["class_0"] == 10
        assert result["class_1"] == 0

    def test_class_counts_returns_dict(self):
        data = np.array([0, 1])
        class_names = ["a", "b"]
        result = class_counts(data, class_names)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"a", "b"}

    def test_class_counts_percentage_sums_to_100(self):
        data = np.random.randint(0, 3, size=(100,))
        class_names = ["class_0", "class_1", "class_2"]
        result = class_counts(data, class_names, percentage=True)
        total = sum(result.values())
        assert abs(total - 100.0) < 0.01

    def test_class_counts_3d_array(self):
        data = np.zeros((10, 10, 10), dtype=int)
        data[0:5, :, :] = 1
        data[5:10, :, :] = 2
        class_names = ["class_0", "class_1", "class_2"]
        result = class_counts(data, class_names)
        assert result["class_0"] == 0
        assert result["class_1"] == 500
        assert result["class_2"] == 500


# =============================================================================
# TestGetChannelsToPred
# =============================================================================


class TestGetChannelsToPred:
    """Tests for get_channels_to_pred function"""

    def test_get_channels_same_channels(self, sample_geotensor):
        channels = ["B1", "B2", "B3", "B4"]
        channels_model = ["B1", "B2", "B3", "B4"]
        result = get_channels_to_pred(sample_geotensor, channels, channels_model)
        assert result.shape == sample_geotensor.shape

    def test_get_channels_subset(self):
        data = np.random.rand(5, 64, 64).astype(np.float32)
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:4326")
        channels = ["B1", "B2", "B3", "B4", "B5"]
        channels_model = ["B2", "B4"]
        result = get_channels_to_pred(geotensor, channels, channels_model)
        assert result.shape[0] == 2
        np.testing.assert_array_equal(result.values[0], data[1])
        np.testing.assert_array_equal(result.values[1], data[3])

    def test_get_channels_reorder(self):
        data = np.arange(3 * 4 * 4).reshape(3, 4, 4).astype(np.float32)
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:4326")
        channels = ["B1", "B2", "B3"]
        channels_model = ["B3", "B1"]
        result = get_channels_to_pred(geotensor, channels, channels_model)
        assert result.shape[0] == 2
        np.testing.assert_array_equal(result.values[0], data[2])
        np.testing.assert_array_equal(result.values[1], data[0])

    def test_get_channels_missing_channel_raises(self, sample_geotensor):
        channels = ["B1", "B2", "B3", "B4"]
        channels_model = ["B1", "B5"]
        with pytest.raises(ValueError) as exc_info:
            get_channels_to_pred(sample_geotensor, channels, channels_model)
        assert "doesn't have bands compatible" in str(exc_info.value)

    def test_get_channels_single_channel(self):
        data = np.random.rand(4, 64, 64).astype(np.float32)
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:4326")
        channels = ["B1", "B2", "B3", "B4"]
        channels_model = ["B3"]
        result = get_channels_to_pred(geotensor, channels, channels_model)
        assert result.shape[0] == 1

    def test_get_channels_preserves_spatial_dims(self, sample_geotensor):
        channels = ["B1", "B2", "B3", "B4"]
        channels_model = ["B1", "B2"]
        result = get_channels_to_pred(sample_geotensor, channels, channels_model)
        assert result.shape[1] == sample_geotensor.shape[1]
        assert result.shape[2] == sample_geotensor.shape[2]


# =============================================================================
# TestCorregisterImages
# =============================================================================


class TestCorregisterImages:
    """Tests for corregister_images function (requires satalign)"""

    @pytest.mark.skip(reason="satalign requires structured images, not random data")
    def test_corregister_basic_3d(self):
        image = np.random.rand(4, 64, 64).astype(np.float32)
        reference = np.random.rand(4, 64, 64).astype(np.float32)
        correg, warps, syncmodel = corregister_images(
            image, reference, rgb_bands=[0, 1, 2], max_translations=5
        )
        assert correg is not None
        assert correg.shape == image.shape

    def test_corregister_dimension_mismatch_raises(self):
        image = np.random.rand(4, 64, 64).astype(np.float32)
        reference = np.random.rand(64, 64).astype(np.float32)
        with pytest.raises(AssertionError):
            corregister_images(image, reference)

    def test_corregister_invalid_rgb_bands_raises(self):
        image = np.random.rand(3, 64, 64).astype(np.float32)
        reference = np.random.rand(3, 64, 64).astype(np.float32)
        with pytest.raises(ValueError) as exc_info:
            corregister_images(image, reference, rgb_bands=[0, 1, 5])
        assert "RGB bands must be less than" in str(exc_info.value)


# =============================================================================
# TestAlignImages
# =============================================================================


class TestAlignImages:
    """Tests for align_images function"""

    @pytest.fixture
    def image_geotensor(self):
        data = np.random.rand(4, 64, 64).astype(np.float32)
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        return GeoTensor(data, transform=transform, crs="EPSG:4326")

    @pytest.fixture
    def bg_geotensor(self):
        data = np.random.rand(4, 64, 64).astype(np.float32)
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
        return GeoTensor(data, transform=transform, crs="EPSG:4326")

    def test_align_images_no_corregister(self, image_geotensor, bg_geotensor):
        result = align_images(image_geotensor, bg_geotensor, corregister=False)
        assert isinstance(result, GeoTensor)
        assert result.shape == bg_geotensor.shape

    @pytest.mark.skip(reason="satalign requires structured images, not random data")
    def test_align_images_with_corregister(self, image_geotensor, bg_geotensor):
        result = align_images(
            image_geotensor, bg_geotensor, corregister=True, rgb_bands=[0, 1, 2], max_translations=5
        )
        assert isinstance(result, GeoTensor)

    def test_align_images_with_validmask(self, image_geotensor, bg_geotensor):
        validmask_data = np.ones((64, 64), dtype=bool)
        validmask = GeoTensor(
            validmask_data, transform=bg_geotensor.transform, crs=bg_geotensor.crs
        )
        result = align_images(
            image_geotensor, bg_geotensor, validmask_bg=validmask, corregister=False
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_align_images_validmask_wrong_extent_raises(self, image_geotensor, bg_geotensor):
        validmask_data = np.ones((32, 32), dtype=bool)
        validmask = GeoTensor(
            validmask_data,
            transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0),
            crs="EPSG:4326",
        )
        with pytest.raises(AssertionError):
            align_images(image_geotensor, bg_geotensor, validmask_bg=validmask, corregister=False)

    def test_align_images_validmask_3d_raises(self, image_geotensor, bg_geotensor):
        validmask_data = np.ones((4, 64, 64), dtype=bool)
        validmask = GeoTensor(
            validmask_data, transform=bg_geotensor.transform, crs=bg_geotensor.crs
        )
        with pytest.raises(AssertionError) as exc_info:
            align_images(image_geotensor, bg_geotensor, validmask_bg=validmask, corregister=False)
        assert "validmask must be 2D" in str(exc_info.value)
