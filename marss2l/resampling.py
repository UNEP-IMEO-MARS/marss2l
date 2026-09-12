"""Recover native 20 m Sentinel-2 bands from images interpolated to 10 m.

MARS stores Sentinel-2 on a 10 m grid. Earth Engine fills the 20 m bands by nearest
neighbour on that grid, and ``georeader.readers.ee_image.interpolate_20mbands_s2ee`` then
re-interpolates them bilinearly: nearest back to 20 m, bilinear up to 10 m. Interpolation
averages neighbouring native pixels, so it lowers the photon noise of a pixel by a fixed
factor (0.62 for this step) and makes neighbours share their noise. Any noise measured on the
10 m grid is therefore not comparable with a floor computed per native pixel.

The step is linear and separable, so it can be undone. Along each axis the stored image is
``y = A x`` with ``A`` the (n_out x n_20m) matrix of the step; in two dimensions
``Y = A_r X A_c^T``. ``A`` is measured here by pushing unit impulses through
``interpolate_20mbands_s2ee`` itself, so padding and border handling are exactly the
pipeline's. Every output pixel draws on at most two native ones with weights 3/4 and 1/4, so
``A`` is well conditioned and ``X`` follows by least squares, up to the uint16 rounding of the
stored values.

One non-linearity has to be avoided: skimage's ``resize`` clips its output to the input
range, and the outermost row and column on each side are pulled towards the zero padding and
then clipped. Those rows, the ones whose weights do not sum to one, are left out of the least
squares. Every native pixel is still observed through its inner neighbours.

The recovered grid is half a native pixel (10 m) away from the true 20 m lattice when the
10 m image does not start on it; the values are the native ones either way.
"""

from functools import lru_cache
from typing import Sequence

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from georeader.readers.ee_image import _find_padding, interpolate_20mbands_s2ee
from scipy import ndimage

#: Impulse height for measuring the operator; the weights come back to about 1e-5.
_AMPLITUDE = 60_000

#: Sentinel-2 L1C bands at 20 m, which the pipeline interpolates to 10 m.
BANDS_20M = ("B05", "B06", "B07", "B8A", "B11", "B12")


@lru_cache(maxsize=None)
def axis_operator(n_out: int) -> np.ndarray:
    """The 20 m -> 10 m step along one axis of ``n_out`` output pixels, as a matrix.

    Measured through ``interpolate_20mbands_s2ee``: each column is the response to one 20 m
    row, replicated over its pair of 10 m rows the way Earth Engine's nearest-neighbour fill
    does, including the edge padding the function applies to odd sizes.

    Returns:
        ``(n_out, m)`` matrix, ``m = ceil(n_out / 2)``.
    """
    pad_before, pad_after = _find_padding(n_out, divisor=2)
    m = (n_out + pad_before + pad_after) // 2
    width = 16
    operator = np.zeros((n_out, m))
    for j in range(m):
        impulse = np.zeros((n_out, width), dtype=np.uint16)
        for padded_row in (2 * j, 2 * j + 1):
            impulse[min(max(padded_row - pad_before, 0), n_out - 1)] = _AMPLITUDE
        stack = GeoTensor(
            np.stack([impulse, impulse]),
            transform=rasterio.Affine(10, 0, 0, 0, -10, 0),
            crs="EPSG:32633",
            fill_value_default=0,
        )
        out = interpolate_20mbands_s2ee(stack, ["B11", "B12"], inplace=False).values[0]
        operator[:, j] = out[:, width // 2].astype(float) / _AMPLITUDE
    return operator


def interior_rows(operator: np.ndarray) -> np.ndarray:
    """Output rows whose weights sum to one: the ones not touched by padding or clipping."""
    return np.flatnonzero(np.abs(operator.sum(axis=1) - 1) < 1e-9)


@lru_cache(maxsize=None)
def _left_inverse(n_out: int) -> tuple:
    operator = axis_operator(n_out)
    rows = interior_rows(operator)
    return rows, np.linalg.pinv(operator[rows])


def invert_bilinear_2x(values: np.ndarray) -> np.ndarray:
    """Native 20 m values from 10 m values produced by ``interpolate_20mbands_s2ee``.

    Args:
        values: ``(H, W)`` or ``(bands, H, W)`` array on the 10 m grid.

    Returns:
        Float array ``(..., ceil(H/2), ceil(W/2))`` on the 20 m grid.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 3:
        return np.stack([invert_bilinear_2x(v) for v in values])
    rows, left_r = _left_inverse(values.shape[0])
    cols, left_c = _left_inverse(values.shape[1])
    return left_r @ values[np.ix_(rows, cols)] @ left_c.T


def _pad_pairs(values: np.ndarray) -> np.ndarray:
    """Edge-pad the last two axes to even sizes, as the pipeline does before its 2x2 step."""
    pad_r = _find_padding(values.shape[-2], divisor=2)
    pad_c = _find_padding(values.shape[-1], divisor=2)
    widths = [(0, 0)] * (values.ndim - 2) + [pad_r, pad_c]
    return np.pad(values, widths, mode="edge")


def _blocks(values: np.ndarray) -> np.ndarray:
    padded = _pad_pairs(values)
    h, w = padded.shape[-2] // 2, padded.shape[-1] // 2
    return padded.reshape(padded.shape[:-2] + (h, 2, w, 2))


def block_mean_2x(values: np.ndarray) -> np.ndarray:
    """A native 10 m band on the 20 m grid, by 2x2 mean over the pipeline's pairs."""
    return _blocks(np.asarray(values, dtype=np.float64)).mean(axis=(-3, -1))


def block_max_2x(values: np.ndarray) -> np.ndarray:
    """A mask on the 20 m grid: a 20 m pixel is flagged if any of its 10 m pixels is."""
    return _blocks(np.asarray(values)).max(axis=(-3, -1))


def transform_20m(transform_10m: rasterio.Affine, shape_10m: Sequence[int]) -> rasterio.Affine:
    """Georeferencing of the recovered 20 m grid, accounting for odd-size padding."""
    pad_r = _find_padding(shape_10m[-2], divisor=2)[0]
    pad_c = _find_padding(shape_10m[-1], divisor=2)[0]
    return transform_10m * rasterio.Affine.translation(-pad_c, -pad_r) * rasterio.Affine.scale(2)


def to_20m(image: GeoTensor, band_names: Sequence[str], invalid_margin: int = 2) -> GeoTensor:
    """A stored 10 m Sentinel-2 image on the native 20 m grid.

    20 m bands are recovered by :func:`invert_bilinear_2x`; native 10 m bands are averaged
    over 2x2 blocks. A 20 m pixel within ``invalid_margin`` pixels of any zero-valued input
    pixel -- how the products encode invalid data -- is set to zero in every band, since the
    inversion couples neighbours and a zero is not a measurement.

    Args:
        image: ``(bands, H, W)`` GeoTensor at 10 m, reflectance x 10,000.
        band_names: Sentinel-2 L1C names of the bands, in order.
        invalid_margin: Dilation of the invalid mask, in 20 m pixels.

    Returns:
        uint16 GeoTensor ``(bands, ceil(H/2), ceil(W/2))`` at 20 m.
    """
    values = np.asarray(image.values)
    out = np.stack(
        [
            invert_bilinear_2x(values[i]) if name in BANDS_20M else block_mean_2x(values[i])
            for i, name in enumerate(band_names)
        ]
    )
    invalid = block_max_2x((values == 0).any(axis=0))
    if invalid.any() and invalid_margin > 0:
        invalid = ndimage.binary_dilation(invalid, iterations=invalid_margin)
    out[:, invalid] = 0
    return GeoTensor(
        np.clip(np.round(out), 0, 65_535).astype(np.uint16),
        transform=transform_20m(image.transform, values.shape),
        crs=image.crs,
        fill_value_default=0,
    )
