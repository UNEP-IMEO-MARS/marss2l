import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from georeader.readers.ee_image import interpolate_20mbands_s2ee

from marss2l import resampling

SIGMA = 100.0


def _pipeline(native: np.ndarray, n: int, offset: int) -> np.ndarray:
    """What MARS stores: Earth Engine's nearest fill on the 10 m grid, then the bilinear step."""
    filled = np.repeat(np.repeat(native, 2, 0), 2, 1)[offset : offset + n, offset : offset + n]
    stack = GeoTensor(
        np.stack([filled, filled]).round().astype(np.uint16),
        transform=rasterio.Affine(10, 0, 500000, 0, -10, 4000000),
        crs="EPSG:32633",
        fill_value_default=0,
    )
    return interpolate_20mbands_s2ee(stack, ["B11", "B12"], inplace=False).values[0]


def _lag1(a: np.ndarray) -> float:
    a = a - a.mean()
    return float((a[:, 1:] * a[:, :-1]).mean() / a.var())


def test_axis_operator_is_the_bilinear_step():
    operator = resampling.axis_operator(202)
    assert operator.shape == (202, 101)
    np.testing.assert_allclose(operator[1, :2], [0.75, 0.25], atol=1e-4)
    np.testing.assert_allclose(operator[2, :2], [0.25, 0.75], atol=1e-4)
    # the first row straddles the zero padding: not an interior row
    assert 0 not in resampling.interior_rows(operator)
    assert 1 in resampling.interior_rows(operator)


@pytest.mark.parametrize("n", [200, 201, 202, 203])
@pytest.mark.parametrize("offset", [0, 1])
def test_round_trip_recovers_native_values(n, offset):
    rng = np.random.default_rng(n + offset)
    native = 3000 + SIGMA * rng.standard_normal((n // 2 + 2, n // 2 + 2))
    stored = _pipeline(native, n, offset).astype(float)

    recovered = resampling.invert_bilinear_2x(stored)

    # A misaligned 10 m grid picks the neighbouring native pixel: compare over a one-pixel shift.
    errors = []
    for dr in (0, 1):
        for dc in (0, 1):
            truth = native[dr : dr + recovered.shape[0], dc : dc + recovered.shape[1]]
            if truth.shape == recovered.shape:
                errors.append(np.sqrt(np.mean((recovered - truth)[1:-1, 1:-1] ** 2)))
    assert min(errors) < 1.0  # uint16 rounding, not the 60 DN the step removes

    inner = recovered[1:-1, 1:-1]
    assert inner.std() / SIGMA == pytest.approx(1.0, abs=0.04)
    assert abs(_lag1(inner)) < 0.04
    assert _lag1(stored[4:-4, 4:-4]) > 0.7  # the stored grid shares its noise


def test_forward_model_reproduces_the_pipeline():
    rng = np.random.default_rng(0)
    stored = _pipeline(3000 + SIGMA * rng.standard_normal((103, 103)), 202, 0).astype(float)
    recovered = resampling.invert_bilinear_2x(stored)
    operator = resampling.axis_operator(202)
    rows = resampling.interior_rows(operator)
    refit = np.round(operator @ recovered @ operator.T)
    assert np.abs(refit - stored)[np.ix_(rows, rows)].max() <= 1


def test_to_20m_masks_invalid_and_georeferences():
    rng = np.random.default_rng(1)
    names = ["B02", "B11", "B12"]
    b10 = 1000 + rng.integers(0, 50, (202, 202))
    b20 = _pipeline(3000 + SIGMA * rng.standard_normal((103, 103)), 202, 0)
    values = np.stack([b10, b20, b20]).astype(np.uint16)
    values[:, 100, 100] = 0
    image = GeoTensor(
        values,
        transform=rasterio.Affine(10, 0, 500000, 0, -10, 4000000),
        crs="EPSG:32633",
        fill_value_default=0,
    )

    out = resampling.to_20m(image, names)

    assert out.shape == (3, 101, 101)
    assert out.transform.a == 20 and out.transform.c == 500000
    assert (out.values[:, 50, 50] == 0).all()  # the invalid pixel, and its margin
    assert (out.values[:, 10, 10] > 0).all()
    np.testing.assert_allclose(out.values[0, 10, 10], b10[20:22, 20:22].mean(), atol=0.5)
