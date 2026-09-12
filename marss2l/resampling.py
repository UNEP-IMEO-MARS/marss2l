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

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from georeader.readers.ee_image import _find_padding, interpolate_20mbands_s2ee
from scipy import ndimage, sparse
from scipy.sparse import linalg as sparse_linalg

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


# ---------------------------------------------------------------------------------------
# Chips: crops of a stored image, target and reference pass stacked.
# ---------------------------------------------------------------------------------------
#
# A chip starts wherever the crop fell, so which pairs of 10 m rows (and columns) make up
# a 20 m pixel is not known in advance: either the first row of the chip opens a pair or it
# closes one. Both pairings are fitted and the one that reproduces the chip is kept -- the
# right one reproduces it to the uint16 rounding, the wrong one cannot. Only the 20 m pixels
# whose two own rows lie inside the chip are kept; the half-seen ones at the edges are not.
#
# The reference pass was aligned to the target after interpolation, by a sub-pixel warp, so
# its half is not an exact bilinear upsample and its recovery is approximate. Each half is
# fitted on its own, and both are returned on the target's cells.


@dataclass(frozen=True)
class NativeChip:
    """A chip on its native 20 m grid.

    Attributes:
        values: ``(bands, size, size)`` uint16, the halves stacked as in the input. Cells
            outside the recovered area, and cells near invalid input, are zero.
        cloudmask: ``(size, size)``, a cell flagged if any of its 10 m pixels is.
        row_off, col_off: The 10 m chip pixel where 20 m cell ``(0, 0)`` starts.
        shape: Recovered cells, rows by columns, before padding to ``size``.
        refit_rms: Per half, the RMS in DN by which the fitted 20 m values fail to
            reproduce the chip's B12: about 0.3 for an exact recovery.
    """

    values: np.ndarray
    cloudmask: Optional[np.ndarray]
    row_off: int
    col_off: int
    shape: tuple
    refit_rms: tuple

    def transform(self, transform_10m: rasterio.Affine) -> rasterio.Affine:
        """Georeferencing of the 20 m cells, from the chip's own."""
        return (
            transform_10m
            * rasterio.Affine.translation(self.col_off, self.row_off)
            * rasterio.Affine.scale(2)
        )

    def mask_to_20m(self, mask_10m: np.ndarray) -> np.ndarray:
        """A 10 m mask on the 20 m cells: a cell is set if any of its 10 m pixels is."""
        return _pad(_cells(mask_10m, self.row_off, self.col_off, self.shape, "max"),
                    self.values.shape[-1]).astype(mask_10m.dtype)


def _crop_operator(n: int, start: int) -> tuple:
    """The step for a crop of ``n`` 10 m pixels whose first one is pixel ``start`` of a
    stored image whose pairs begin at 0; ``start`` 1 or 2 keeps the crop clear of the
    clipped border rows.

    Returns:
        ``(operator, cells, full)``: the operator on the 20 m cells the crop touches, those
        cells' indices, and the indices of the cells both of whose own rows are in the crop.
    """
    operator = axis_operator(n + 4 + n % 2)[start : start + n]
    cells = np.flatnonzero(operator.any(axis=0))
    own_rows = np.isclose(operator, 0.75).sum(axis=0)
    return operator[:, cells], cells, np.flatnonzero(own_rows == 2)


def _cells(values: np.ndarray, row_off: int, col_off: int, shape: tuple, how: str):
    """Aggregate 10 m pixels over the 2x2 blocks of the cells starting at the offsets."""
    n_r, n_c = shape
    block = values[..., row_off : row_off + 2 * n_r, col_off : col_off + 2 * n_c]
    block = block.reshape(values.shape[:-2] + (n_r, 2, n_c, 2))
    return block.mean(axis=(-3, -1)) if how == "mean" else block.max(axis=(-3, -1))


def _pad(values: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(values.shape[:-2] + (size, size), dtype=values.dtype)
    h, w = min(size, values.shape[-2]), min(size, values.shape[-1])
    out[..., :h, :w] = values[..., :h, :w]
    return out


#: Stored pixels this close to a zero are left out of the fit as well: the step, and the
#: reference pass's alignment, mixed the zero into them.
_CONTAMINATED = 2


def _usable(invalid: np.ndarray) -> np.ndarray:
    """Pixels the least squares may use: away from zeros, which are not measurements."""
    if not invalid.any():
        return np.ones_like(invalid)
    return ~ndimage.binary_dilation(invalid, iterations=_CONTAMINATED)


def _solve(op_r: np.ndarray, op_c: np.ndarray, band: np.ndarray, usable: np.ndarray):
    """Least-squares 20 m values of one band from its usable 10 m pixels.

    Separable when every pixel is usable. Otherwise the two-dimensional problem restricted
    to the usable pixels, by sparse least squares; a cell no usable pixel sees comes back
    as zero, and the caller discards it.
    """
    if usable.all():
        return np.linalg.pinv(op_r) @ band @ np.linalg.pinv(op_c).T
    op_r, op_c = (np.where(np.abs(op) < 1e-3, 0.0, op) for op in (op_r, op_c))
    operator = sparse.kron(sparse.csr_matrix(op_r), sparse.csr_matrix(op_c), format="csr")
    solution = sparse_linalg.lsqr(
        operator[usable.ravel()], band[usable], atol=1e-10, btol=1e-10, iter_lim=1000
    )[0]
    return solution.reshape(op_r.shape[1], op_c.shape[1])


def _fit_half(b12: np.ndarray, usable: np.ndarray) -> tuple:
    """Choose the pairing of one half from its B12 band.

    Returns:
        ``((start_r, start_c), rms)``: the pairing that refits the band best, and that
        refit's RMS in DN over the pixels used.
    """
    best = None
    for start_r in (1, 2):
        op_r = _crop_operator(b12.shape[0], start_r)[0]
        for start_c in (1, 2):
            op_c = _crop_operator(b12.shape[1], start_c)[0]
            residual = (op_r @ _solve(op_r, op_c, b12, usable) @ op_c.T - b12)[usable]
            rms = float(np.sqrt(np.mean(residual**2))) if residual.size else float("inf")
            if best is None or rms < best[1]:
                best = ((start_r, start_c), rms)
    return best


def _invert_half(values: np.ndarray, band_names: Sequence[str], starts: tuple,
                 usable: np.ndarray) -> tuple:
    """One half on its fully observed 20 m cells, and where those cells start on the chip."""
    (op_r, cells_r, full_r), (op_c, cells_c, full_c) = (
        _crop_operator(n, start) for n, start in zip(values.shape[-2:], starts, strict=True)
    )
    keep = np.ix_(np.searchsorted(cells_r, full_r), np.searchsorted(cells_c, full_c))
    # Chip pixel of the first own row of the first fully observed cell.
    row_off, col_off = int(2 * full_r[0] - starts[0]), int(2 * full_c[0] - starts[1])
    shape = (len(full_r), len(full_c))
    out = []
    for band, name in zip(values.astype(np.float64), band_names, strict=True):
        if name in BANDS_20M:
            out.append(_solve(op_r, op_c, band, usable)[keep])
        else:
            out.append(_cells(band, row_off, col_off, shape, "mean"))
    return np.stack(out), (row_off, col_off), shape


def chip_to_20m(
    values: np.ndarray,
    band_names: Sequence[str],
    cloudmask: Optional[np.ndarray] = None,
    size: Optional[int] = None,
    invalid_margin: int = 1,
) -> NativeChip:
    """A 10 m Sentinel-2 chip -- one or more passes stacked -- on its native 20 m grid.

    Args:
        values: ``(passes * len(band_names), H, W)``, reflectance x 10,000; zero where
            invalid.
        band_names: Sentinel-2 L1C names of one pass's bands, in order. Must include B12,
            from which the pairing is read.
        cloudmask: ``(H, W)``, optional, aggregated by maximum onto the target's cells.
        size: Side of the output, zero-padded. Defaults to ``ceil(max(H, W) / 2)``.
        invalid_margin: Zeros, and the pixels within two of them, are left out of the fit;
            a cell that has one of them among its own pixels, in any pass, is set to zero in
            every band, and so is every cell within this many cells of it.

    Returns:
        The chip on the target pass's 20 m cells; see :class:`NativeChip`.
    """
    values = np.asarray(values)
    nbands = len(band_names)
    if values.shape[0] % nbands:
        raise ValueError(f"{values.shape[0]} bands is not a whole number of {nbands}-band passes")
    size = size or -(-max(values.shape[-2:]) // 2)
    b12 = list(band_names).index("B12")

    halves, invalid_cells, refits, grid = [], [], [], None
    for first in range(0, values.shape[0], nbands):
        half = values[first : first + nbands]
        usable = _usable((half == 0).any(axis=0))
        starts, rms = _fit_half(half[b12].astype(np.float64), usable)
        cells, offsets, shape = _invert_half(half, band_names, starts, usable)
        grid = grid or (offsets, shape)
        halves.append(_pad(cells, size))
        invalid_cells.append(_pad(_cells(~usable, *offsets, shape, "max"), size))
        filled = np.zeros((size, size), dtype=bool)
        filled[: shape[0], : shape[1]] = True
        invalid_cells.append(~filled)
        refits.append(rms)

    invalid = np.logical_or.reduce(invalid_cells)
    if invalid.any() and invalid_margin > 0:
        # Only input invalidity spreads: the padding beyond the recovered cells is not a
        # neighbour of anything the inversion used.
        spread = ndimage.binary_dilation(np.logical_or.reduce(invalid_cells[::2]),
                                         iterations=invalid_margin)
        invalid |= spread
    out = np.concatenate(halves)
    out[:, invalid] = 0
    (row_off, col_off), shape = grid

    native_cloudmask = None
    if cloudmask is not None:
        native_cloudmask = _pad(_cells(np.asarray(cloudmask), row_off, col_off, shape, "max"),
                                size).astype(np.asarray(cloudmask).dtype)
    return NativeChip(
        values=np.clip(np.round(out), 0, 65_535).astype(np.uint16),
        cloudmask=native_cloudmask,
        row_off=row_off,
        col_off=col_off,
        shape=shape,
        refit_rms=tuple(refits),
    )
