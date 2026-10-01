from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from georeader import read
from georeader.geotensor import GeoTensor
from numpy.typing import NDArray

GeoTensorOrNDArray = Union[GeoTensor, NDArray]


def values(x: GeoTensorOrNDArray) -> NDArray:
    if isinstance(x, GeoTensor):
        return x.values
    return x


def class_counts(
    data: NDArray, class_names: List[str], percentage: bool = False
) -> Dict[str, float]:
    values, counts = np.unique(data, return_counts=True)
    meta_iter = {k: 0 for k in class_names}
    if percentage:
        normalization_factor = np.prod(data.shape) / 100
    else:
        normalization_factor = 1
    for v, c in zip(values, counts, strict=False):
        meta_iter[class_names[v]] = c / normalization_factor

    return meta_iter


def get_channels_to_pred(
    img: GeoTensor, channels: List[str], channels_model: List[str]
) -> GeoTensor:
    if channels != channels_model:
        try:
            indexes = [channels.index(band) for band in channels_model]
            image_input = img.isel({"band": indexes})
        except ValueError:
            raise ValueError(
                "Image doesn't have bands compatible with the model: "
                f"\n channels image: {channels} \n channels model: {channels_model}"
            )
    else:
        image_input = img
    return image_input


def align_images(
    image: GeoTensor,
    bg: GeoTensor,
    validmask_bg: Optional[GeoTensor] = None,
    corregister: bool = True,
    rgb_bands: Optional[List[int]] = None,
    max_translations: float = 5,
) -> Union[GeoTensor, Tuple[GeoTensor, GeoTensor]]:
    """
    Aligns the background image to the input image.

    Args:
        image_s2 (GeoTensor): S2 image
        background_s2 (GeoTensor): S2 image to align
        validmask_bg (Optional[GeoTensor], optional): mask with valid values. Defaults to None.
        corregister (bool, optional): If True, it uses satalign to do corregister the images. Defaults to True.
        max_translations (float, optional): max translation allowed in pixels when corregistering. Defaults to 5.

    Returns:
        GeoTensor: aligned background image or a tuple with the aligned background image and the validmask_bg
    """
    if validmask_bg is not None:
        assert validmask_bg.same_extent(bg), "background_s2 and validmask must have the same shape"
        assert (
            len(validmask_bg.shape) == 2
        ), f"validmask must be 2D but has shape {validmask_bg.shape}"

    if not bg.same_extent(image):
        bg = read.read_reproject_like(bg, image)
        if validmask_bg is not None:
            validmask_bg = read.read_reproject_like(validmask_bg, image).squeeze()

    if validmask_bg is not None:
        assert (
            len(validmask_bg.shape) == 2
        ), f"validmask must be 2D but has shape {validmask_bg.shape}"

    if corregister:
        bg_corr, warps, syncmodel = corregister_images(
            bg.values.astype(np.float32),
            image.values.astype(np.float32),
            rgb_bands=rgb_bands,
            max_translations=max_translations,
        )

        if validmask_bg is not None:
            validmask_bg_warped = (
                syncmodel.warp_feature(
                    img=validmask_bg.values[np.newaxis].astype(np.float32),
                    warp_matrix=warps,
                )[0]
                > 0.5
            )

            assert (
                len(validmask_bg_warped.shape) == 2
            ), f"validmask must be 2D but has shape {validmask_bg_warped.shape}"

            validmask_bg = GeoTensor(
                validmask_bg_warped,
                validmask_bg.transform,
                validmask_bg.crs,
                fill_value_default=False,
            )
        bg = GeoTensor(bg_corr, bg.transform, bg.crs, fill_value_default=bg.fill_value_default)

    if validmask_bg is not None:
        bg.values[..., ~validmask_bg.values] = bg.fill_value_default
        return bg, validmask_bg

    return bg


def corregister_images(
    image: GeoTensorOrNDArray,
    reference: GeoTensorOrNDArray,
    rgb_bands: Optional[List[int]] = None,
    max_translations: float = 5,
) -> Tuple[GeoTensorOrNDArray, Any, Any]:
    """
    Corregister the image to the reference

    Args:
        image (NDArray): image to corregister
        reference (NDArray): reference image

    Returns:
        Tuple[NDArray, Any, Any]: corregistered image, warps and syncmodel for warping.
    """
    image_values = values(image)
    reference = values(reference)
    import satalign

    assert (
        image.ndim == reference.ndim
    ), "image and reference must have the same number of dimensions"
    assert image.ndim in [2, 3], "image and reference must have 2 or 3 dimensions"

    flat = False
    if image_values.ndim == 2:
        image_values = image_values[np.newaxis]
        reference = reference[np.newaxis]
        rgb_bands = [0]
        flat = True

    if rgb_bands is None:
        if image_values.shape[0] < 3:
            rgb_bands = [0]
        if image_values.shape[0] == 3:
            rgb_bands = [2, 1, 0]
        else:
            rgb_bands = [3, 2, 1]

    if any([b >= image_values.shape[0] for b in rgb_bands]):
        raise ValueError("RGB bands must be less than the number of bands in the image")

    syncmodel = satalign.PCC(
        datacube=image_values[np.newaxis],
        reference=reference,
        channel="mean",  # mean of RGB
        rgb_bands=rgb_bands,
        crop_center=round(min(image_values.shape[1], image_values.shape[2]) * 0.8),
        max_translations=max_translations,
        # upsample_factor=10,
        num_threads=2,
    )
    news2cube, warps = syncmodel.run_multicore()

    correg = news2cube[0]
    if flat:
        correg = correg[0]

    if isinstance(image, GeoTensor):
        correg = GeoTensor(
            correg,
            image.transform,
            image.crs,
            fill_value_default=image.fill_value_default,
        )
    return correg, warps[0], syncmodel
