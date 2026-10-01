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


def _stored_chip(seed: int, start_r: int, start_c: int, n: int = 200):
    """A chip cropped from the interior of a stored image, starting at either parity."""
    rng = np.random.default_rng(seed)
    native = 3000 + SIGMA * rng.standard_normal((n // 2 + 6, n // 2 + 6))
    stored = _pipeline(native, n + 8, 0)
    r, c = 2 + start_r, 2 + start_c  # clear of the image's own clipped border
    return native, stored[r : r + n, c : c + n], (r, c)


def _truth(native, chip, first):
    """The native pixels the chip's cells should hold: cell (0, 0) starts on stored pixel
    ``first + offset``, which must open a pair."""
    (r, c), (h, w) = (first[0] + chip.row_off, first[1] + chip.col_off), chip.shape
    assert r % 2 == 0 and c % 2 == 0
    return native[r // 2 : r // 2 + h, c // 2 : c // 2 + w]


@pytest.mark.parametrize("start_r,start_c", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_chip_to_20m_finds_the_pairing_and_recovers_both_passes(start_r, start_c):
    native_t, target, first = _stored_chip(0, start_r, start_c)
    # The reference pass need not start on the same parity as the target.
    _, background, _ = _stored_chip(1, 1 - start_r, start_c)
    names = ["B11", "B12"]

    chip = resampling.chip_to_20m(np.stack([target, target, background, background]), names)

    assert chip.values.shape == (4, 100, 100)
    assert max(chip.refit_rms) < 0.5  # uint16 rounding
    truth = _truth(native_t, chip, first)
    recovered = chip.values[1, : chip.shape[0], : chip.shape[1]].astype(float)
    # A cell the reference pass does not cover is dropped from both passes.
    kept = recovered > 0
    assert kept.sum() >= 99 * 99
    assert np.sqrt(np.mean((recovered - truth)[kept] ** 2)) < 1.0
    # Native noise back at its own level and independent between pixels, where the stored
    # grid shares it between neighbours.
    assert recovered[kept].std() / SIGMA == pytest.approx(1.0, abs=0.05)
    assert abs(_lag1(recovered[2:-2, 2:-2])) < 0.04
    assert _lag1(target[4:-4, 4:-4].astype(float)) > 0.7


def test_chip_to_20m_discards_cells_near_invalid_input():
    native, target, first = _stored_chip(2, 1, 0)
    half = np.stack([target, target]).astype(np.uint16)
    half[:, :, :40] = 0  # a swath edge
    cloudmask = np.zeros((200, 200), dtype=np.uint8)
    cloudmask[100:104, 100:104] = 1

    chip = resampling.chip_to_20m(np.concatenate([half, half]), ["B11", "B12"], cloudmask=cloudmask)

    assert chip.refit_rms[0] < 0.5  # the pairing is judged away from the edge
    valid = chip.values[1] > 0
    assert not valid[:, :20].any()
    assert valid[: chip.shape[0], 23 : chip.shape[1]].all()
    # The fill that keeps zeros out of the least squares leaves the kept cells exact.
    truth = _truth(native, chip, first)
    recovered = chip.values[1, : chip.shape[0], : chip.shape[1]].astype(float)
    keep = valid[: chip.shape[0], : chip.shape[1]]
    assert np.abs(recovered - truth)[keep].max() < 3.0
    assert chip.cloudmask.shape == (100, 100) and 2 <= chip.cloudmask.sum() <= 9


def test_native_chip_mask_and_transform():
    _, target, _ = _stored_chip(3, 1, 1)
    chip = resampling.chip_to_20m(np.stack([target, target]), ["B11", "B12"])
    mask = np.zeros((200, 200), dtype=bool)
    mask[50, 60] = True

    native_mask = chip.mask_to_20m(mask)

    assert native_mask.shape == (100, 100) and native_mask.sum() == 1
    assert native_mask[(50 - chip.row_off) // 2, (60 - chip.col_off) // 2]
    transform = chip.transform(rasterio.Affine(10, 0, 500000, 0, -10, 4000000))
    assert transform.a == 20 and transform.e == -20
    assert transform.c == 500000 + 10 * chip.col_off
    assert transform.f == 4000000 - 10 * chip.row_off
