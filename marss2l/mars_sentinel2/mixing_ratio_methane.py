""" "
This code is a modified version of a code provided by Javier Gorroño (UPV),
The original code is commented at the bottom of this file.

This code produces methane enhancement images (in ppb) from Sentinel-2 and Landsat images.

The descrition of the method is provided in the following paper:
Understanding the potential of Sentinel-2 for monitoring methane point emissions, Gorroño et al. (2023)
https://amt.copernicus.org/articles/16/89/2023/

The LUT `output_Tch4_LUT_AMF_VZA_0_v2.nc` was also provided by Javier Gorroño (UPV).

"""

import os
import warnings
from functools import cache
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from georeader.geotensor import GeoTensor
from georeader.readers import S2_SAFE_reader
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate
from marshsi.lut import FILE_LUT_GAS

from .utils import align_images, corregister_images, values

# filter UserWarning: Unknown extension is not supported and will be removed
warnings.filterwarnings(
    "ignore", "Unknown extension is not supported and will be removed", UserWarning
)



def ratio_bands(
    image_s2: Union[GeoTensor, ArrayLike],
    numerator_index=12,
    denominator_index=11,
    validmask: Optional[Union[GeoTensor, ArrayLike]] = None,
    plumemaskbool: Optional[Union[GeoTensor, ArrayLike]] = None,
    normalize: bool = True,
    corregister: bool = False,
    fill_value_default: float = 0,
) -> Union[GeoTensor, ArrayLike]:
    """
    Method to calculate the ratio between bands of Sentinel-2 or Landsat (using B12 and B11 by default which are the bands
    used to calculate methane enhancement).

    Returns the ratio normalized:
     (b12 / b11) / np.mean(b12 / b11)

    This function takes into account the validmask and clip the values of the ratio to reasonable values (0, 10).

    Args:
        image_s2 (GeoTensor): S2 image
        numerator_index (int, optional): . Defaults to 12.
        denominator_index (int, optional): . Defaults to 11.
        validmask (Optional[GeoTensor], optional):  mask with valid values. Defaults to None.
        plumemaskbool (Optional[GeoTensor], optional): mask with potential plumes. It will mask out these values from the normalization factor. Defaults to None.
        normalize (bool, optional): If True, the ratio is normalized by the mean of the ratio. Defaults to True.
        corregister (bool, optional): If True, it uses satalign to do corregister the images. Defaults to False.
        fill_value_default (float, optional): value to use for invalid pixels. Defaults to 0.

    Returns:
        GeoTensor: ratio image (H, W) (b12 / b11) / np.mean(b12 / b11) values clipped to [0, 10]
    """
    if validmask is None:
        validmaskarray = np.ones(image_s2.shape[1:], dtype=bool)
    elif isinstance(validmask, GeoTensor):
        validmaskarray = validmask.values
    else:
        validmaskarray = validmask

    assert (
        validmaskarray.shape == image_s2.shape[1:]
    ), "validmask and image_s2 must have the same shape"

    image_s2_values = values(image_s2)

    numerator = image_s2_values[numerator_index]
    denominator = image_s2_values[denominator_index]
    if corregister:
        denominator, _, _ = corregister_images(reference=numerator, image=denominator)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_b12b11 = numerator / denominator

    finite_mask = np.isfinite(ratio_b12b11)

    ratio_b12b11 = ratio_b12b11.clip(0, 10)

    # ratio_b12b11 = image_s2.isel({"band": numerator_index}) / image_s2.isel({"band": denominator_index})
    # ratio_b12b11 = ratio_b12b11.clip(0, 10)

    # extend invalid mask with nans
    validmask = validmaskarray & finite_mask
    invalidmask = ~validmask

    if normalize:
        if plumemaskbool is not None:
            plumemaskbool = values(plumemaskbool)
            assert (
                plumemaskbool.shape == ratio_b12b11.shape
            ), "plumemask and ratio_b12b11 must have the same shape"
            normalization_factor = np.mean(ratio_b12b11[validmask & ~plumemaskbool])
        else:
            normalization_factor = np.mean(ratio_b12b11[validmask])

        ratio_b12b11[validmask] /= normalization_factor

    ratio_b12b11[invalidmask] = fill_value_default

    if isinstance(image_s2, GeoTensor):
        ratio_b12b11 = GeoTensor(
            ratio_b12b11,
            image_s2.transform,
            image_s2.crs,
            fill_value_default=fill_value_default,
        )

    return ratio_b12b11


def mbmp_varon(
    image_s2: GeoTensor,
    background_s2: GeoTensor,
    b12_index: int = 12,
    b11_index: int = 11,
    validmask: Optional[GeoTensor] = None,
    corregister: bool = False,
) -> GeoTensor:
    """
    Multi-band multi-pass Method of Varon et al 2021.

    Args:
        image_s2 (GeoTensor): S2 image
        background_s2 (GeoTensor): S2 image to use as background (i.e. without observed methane)
        b12_index (int, optional): Channel signal. Defaults to 12.
        b11_index (int, optional): Channel background. Defaults to 11.
        validmask (Optional[GeoTensor], optional): mask with valid values. Defaults to None.
        corregister (bool, optional): If True, it uses satalign to do corregister the images. Defaults to False.

    Returns:
        GeoTensor: methane enhancement image
    """
    assert image_s2.shape[-2:] == validmask.shape, "image and validmask must have the same shape"
    assert (
        image_s2.shape[0] == background_s2.shape[0]
    ), "image and background must have the same number of bands"

    if validmask is None:
        validmaskarray = None
    else:
        validmaskarray = validmask.values.copy()

    b11 = image_s2.isel({"band": b11_index}) / 10_000.0
    b12 = image_s2.isel({"band": b12_index}) / 10_000.0
    ratio_varon_MBSP = mbsp_varon(
        background_channel=b11.values, signal=b12.values, validmask=validmaskarray
    )

    background_s2 = align_images(image_s2, background_s2, corregister=corregister)

    b11_background = background_s2.isel({"band": b11_index}) / 10_000.0
    b12_background = background_s2.isel({"band": b12_index}) / 10_000.0

    ratio_varon_MBSP_background = mbsp_varon(
        background_channel=b11_background.values,
        signal=b12_background.values,
        validmask=validmaskarray,
    )

    if validmaskarray is None:
        validmaskarray = (b11.values > 1e-4) & (b11_background.values > 1e-4)
    else:
        validmaskarray &= (b11.values > 1e-4) & (b11_background.values > 1e-4)

    ratio_varon_MBMP = ratio_varon_MBSP - ratio_varon_MBSP_background

    ratio_varon_MBMP[~validmaskarray] = 0

    return GeoTensor(ratio_varon_MBMP, image_s2.transform, image_s2.crs, fill_value_default=-1)


def mbsp_varon(
    background_channel: NDArray,
    signal: NDArray,
    validmask: Optional[NDArray] = None,
    diff_mask_value=np.inf,
) -> NDArray:
    """
    Single-Pass Multi-Band method proposed in Varon et al 2021

    returns (c * background - signal) / background

    where c is matched by least squares

    Args:
        background_channel: background image
        signal: signal image
        cloudmask: mask with valid values. Defaults to None.
        diff_mask_value: pixels with difference between background and signal greater
          than this value are masked out.

    Returns:
        ratio image [(c * background - signal) / background]

    """
    signal_flat = signal.flatten()
    background_flat = background_channel.flatten()

    differences_flat = background_flat - signal_flat
    valid_values = differences_flat < diff_mask_value

    if validmask is not None:
        valid_values &= validmask.flatten()

    if np.any(valid_values):
        numerator = np.sum(signal_flat[valid_values] * background_flat[valid_values])
        denominator = np.sum(background_flat[valid_values] ** 2)

        c = numerator / denominator
    else:
        c = 1

    R = (c * background_channel - signal) / np.clip(background_channel, 1e-3, None)

    return R


FILL_VALUE_RATIO_IL = 0


def ratio_IL(
    image_s2: Union[GeoTensor, NDArray],
    background_s2: Union[GeoTensor, NDArray],
    b12_index: int = 12,
    b11_index: int = 11,
    b12_index_bg: Optional[int] = None,
    b11_index_bg: Optional[int] = None,
    validmask: Optional[Union[GeoTensor, NDArray]] = None,
    validmask_bg: Optional[Union[GeoTensor, NDArray]] = None,
    plumemaskbool: Optional[Union[GeoTensor, NDArray]] = None,
    fill_value_ratio_il: float = FILL_VALUE_RATIO_IL,
    normalize: bool = True,
    corregister: bool = True,
    corregister_bands: bool = False,
) -> Union[GeoTensor, NDArray]:
    """
    Ratio to enhance methane of Irakullis-Loritxate et al 2022:
    (b12ch4 / b11ch4) / (b12noch4 / b11noch4) * np.mean(b12noch4 / b11noch4) / np.mean(b12ch4 / b11ch4)

    This function takes into account the validmask and clip the values of the ratio to reasonable values.

    Args:
        image_s2 (GeoTensor): S2 image
        background_s2 (GeoTensor): S2 image to use as background
        b12_index (int, optional): index of band 12. Defaults to 12.
        b11_index (int, optional): index of band 11. Defaults to 11.
        validmask (Optional[GeoTensor], optional): mask with valid values. Defaults to None.
        validmask_bg (Optional[GeoTensor], optional): mask with valid values for the background. Defaults to None.
        fill_value_ratio_il (float, optional): value to use for invalid pixels. Defaults to 0.
        normalize (bool, optional): If True, each ratio is normalized by the mean of the ratio. Defaults to True.
        corregister (bool, optional): If True, it uses satalign to do corregister the images. Defaults to True.
        corregister_bands (bool, optional): If True, it uses satalign to do corregister the B11 and B12 bands for each image. Defaults to False.

    Returns:
        GeoTensor: ratio image (H, W) values clipped to [0, 10]
    """

    assert (validmask is None) or (
        image_s2.shape[-2:] == validmask.shape
    ), "image and validmask must have the same shape"
    assert (validmask_bg is None) or (
        background_s2.shape[-2:] == validmask_bg.shape
    ), "background and validmask_bg must have the same shape"

    # This is not needed as band indexes are specified for each image.
    # assert image_s2.shape[0] == background_s2.shape[0], "image and background must have the same number of bands"

    fill_value_ratio = 0
    ratio_b12b11 = ratio_bands(
        image_s2,
        numerator_index=b12_index,
        denominator_index=b11_index,
        validmask=validmask,
        normalize=normalize,
        plumemaskbool=plumemaskbool,
        corregister=corregister_bands,
        fill_value_default=fill_value_ratio,
    )

    if isinstance(image_s2, GeoTensor) and isinstance(background_s2, GeoTensor):
        if validmask_bg is None:
            background_s2 = align_images(
                image_s2, background_s2, validmask_bg, corregister=corregister
            )
        else:
            background_s2, validmask_bg = align_images(
                image_s2, background_s2, validmask_bg, corregister=corregister
            )

    if b12_index_bg is None:
        b12_index_bg = b12_index
    if b11_index_bg is None:
        b11_index_bg = b11_index

    ratio_b12b11_bg = ratio_bands(
        background_s2,
        numerator_index=b12_index_bg,
        denominator_index=b11_index_bg,
        validmask=validmask_bg,
        normalize=normalize,
        corregister=corregister_bands,
        fill_value_default=fill_value_ratio,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        mbmp_ratio_raster = ratio_b12b11 / ratio_b12b11_bg
        mbmp_ratio_raster = mbmp_ratio_raster.clip(0, 10)

    if isinstance(mbmp_ratio_raster, GeoTensor):
        mbmp_ratio_raster_values = mbmp_ratio_raster.values
        ratio_b12b11_values = ratio_b12b11.values
        ratio_b12b11_bg_values = ratio_b12b11_bg.values
    else:
        mbmp_ratio_raster_values = mbmp_ratio_raster
        ratio_b12b11_values = ratio_b12b11
        ratio_b12b11_bg_values = ratio_b12b11_bg

    mbmp_ratio_raster_values[
        (ratio_b12b11_values == fill_value_ratio) | (ratio_b12b11_bg_values == fill_value_ratio)
    ] = fill_value_ratio_il

    if isinstance(mbmp_ratio_raster, GeoTensor):
        mbmp_ratio_raster.fill_value_default = fill_value_ratio_il

    return mbmp_ratio_raster


def difference_bands(
    curr_img: GeoTensor,
    bg_img: GeoTensor,
    bands_indexes: List[int],
    valid_mask_curr: Optional[GeoTensor] = None,
    valid_mask_bg: Optional[GeoTensor] = None,
    corregister: bool = False,
) -> GeoTensor:
    """
    Computes a measure of difference between chrr_img and bf_img in the bands specified by bands_indexes.
    This measure is the absoute mean difference normalized by the mean brightness of each band. Efectively
    it is the mean absolute difference correcting by the brightness of the images. The mean is computed across the
    channel dimension.

    Args:
        curr_img (GeoTensor): Current image
        bg_img (GeoTensor): Background image
        bands_indexes (List[int]): indexes of the bands to use to compute the difference
        valid_mask_curr (Optional[GeoTensor], optional): mask with valid values for the current image. Defaults to None.
        valid_mask_bg (Optional[GeoTensor], optional): mask with valid values for the background image. Defaults to None.
        corregister (bool, optional): If True, it uses satalign to do corregister the images. Defaults to False.

    Returns:
        float: difference measure. Absoute mean difference normalized by the mean brightness of each band.
         (np.mean(np.abs(curr_img/np.mean(curr_img,axis=0) - bg_img/np.mean(curr_img,axis=0))))
    """
    # assert both images have the same number of bands

    if valid_mask_curr is not None:
        assert valid_mask_curr.same_extent(
            curr_img
        ), "valid_mask_curr must have the same extent as curr_img_bands"

        assert (
            len(valid_mask_curr.shape) == 2
        ), f"valid_mask_curr must be a 2D mask {valid_mask_curr.shape}"

    if valid_mask_bg is not None:
        valid_mask_bg.same_extent(bg_img), "valid_mask_bg must have the same extent as bg_img_bands"

        assert (
            len(valid_mask_bg.shape) == 2
        ), f"valid_mask_bg must be a 2D mask {valid_mask_bg.shape}"

    if valid_mask_bg is None:
        bg_img = align_images(curr_img, bg_img, None, corregister=corregister)
    else:
        bg_img, valid_mask_bg = align_images(
            curr_img, bg_img, valid_mask_bg, corregister=corregister
        )

    curr_img_bands = curr_img.isel({"band": bands_indexes}) / 10_000.0
    bg_img_bands = bg_img.isel({"band": bands_indexes}) / 10_000.0

    assert (
        curr_img_bands.shape[0] == bg_img_bands.shape[0]
    ), "curr_img and bg_img must have the same number of bands"

    if (valid_mask_curr is not None) and valid_mask_curr.values.any():
        curr_img_bands.values[:, ~valid_mask_curr.values] = 0
        numerator = np.nansum(
            curr_img_bands.values / np.sum(valid_mask_curr.values),
            axis=(1, 2),
            keepdims=True,
        )
    else:
        valid_mask_curr = None
        numerator = np.nanmean(curr_img_bands.values, axis=(1, 2), keepdims=True)

    if (valid_mask_bg is not None) and valid_mask_bg.values.any():
        bg_img_bands.values[:, ~valid_mask_bg.values] = 0
        denominator = np.nansum(
            bg_img_bands.values / np.sum(valid_mask_bg.values),
            axis=(1, 2),
            keepdims=True,
        )
    else:
        valid_mask_bg = None
        denominator = np.nanmean(bg_img_bands.values, axis=(1, 2), keepdims=True)

    # print(numerator, denominator)

    ratio_normalize_bg = numerator / denominator
    if np.isnan(ratio_normalize_bg).any():
        ratio_normalize_bg[np.isnan(ratio_normalize_bg)] = 1

    curr_img_bands_norm = curr_img_bands

    # Normalize the background by the brightness of the current image
    bg_img_bands_norm = bg_img_bands * ratio_normalize_bg

    if valid_mask_curr is not None and valid_mask_bg is not None:
        valid_mask = valid_mask_curr.values & valid_mask_bg.values
    elif valid_mask_curr is not None:
        valid_mask = valid_mask_curr.values
    elif valid_mask_bg is not None:
        valid_mask = valid_mask_bg.values
    else:
        valid_mask = None

    diffimage = GeoTensor(
        np.abs(curr_img_bands_norm.values - bg_img_bands_norm.values),
        transform=curr_img_bands_norm.transform,
        crs=curr_img_bands_norm.crs,
        fill_value_default=-1,
    )

    # Average difference across the bands
    diffimage.values = np.mean(diffimage.values, axis=0)

    if valid_mask is not None:
        diffimage.values[~valid_mask] = diffimage.fill_value_default

    return diffimage


def apply_interpfun_to_image(
    interpfun: Callable[[NDArray], NDArray],
    image: Union[NDArray, GeoTensor],
    fill_value_default: Optional[float] = None,
) -> Union[NDArray, GeoTensor]:
    """
    Applies the interpolation function interpfun to the image.

    Args:
        interpfun (Callable[[np.ndarray], np.ndarray]): interpolation function
        image (Union[np.array, GeoTensor]): image to apply the interpolation function. if GeoTensor, it will be returned as GeoTensor
        fill_value_default (Optional[float], optional): fill value to use for masking invalid pixels. Defaults to None.

    Returns:
        Union[np.array, GeoTensor]: image with the interpolation function applied.
            Ff image is GeoTensor, it will be returned as GeoTensor
    """
    if isinstance(image, GeoTensor):
        values = image.values
        invalids = (values == image.fill_value_default) | np.isnan(values)
    else:
        values = image

    values = interpfun(values.flatten()).reshape(values.shape)

    if isinstance(image, GeoTensor):
        if fill_value_default is None:
            fill_value_default = image.fill_value_default
        values[invalids] = fill_value_default
        return GeoTensor(
            values,
            transform=image.transform,
            crs=image.crs,
            fill_value_default=fill_value_default,
        )

    return values


LINK_RSR_LANDSAT = {
    "LC09": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L9_OLI2_RSR.xlsx",
    "LC08": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L8_OLI_RSR.xlsx",
    "LT05": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L5_TM_RSR.xlsx",
    "LT04": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L4_TM_RSR.xlsx",
    "LE07": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L7_ETM_RSR.xlsx",
}

BAND_TO_SHEET_NAME_L89 = {
    "B01": "CoastalAerosol",
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B05": "NIR",
    "B06": "SWIR1",
    "B07": "SWIR2",
    "B08": "Pan",
    "B09": "Cirrus",
}

BANDS_TO_SHEET_NAME_L7 = {
    "B01": "Blue-L7",
    "B02": "Green-L7",
    "B03": "Red-L7",
    "B04": "NIR-L7",
    "B05": "SWIR(5)-L7",
    "B07": "SWIR(7)-L7",
    "B08": "Pan-L7",
}

BANDS_TO_SHEET_NAME_L5 = {
    "B01": "Blue-L5 TM",
    "B02": "Green-L5 TM",
    "B03": "Red-L5 TM",
    "B04": "NIR-L5 TM",
    "B05": "SWIR(5)-L5 TM",
    "B07": "SWIR(7)-L5 TM",
}

BANDS_TO_SHEET_NAME_L4 = {
    "B01": "Blue-L4 TM",
    "B02": "Green-L4 TM",
    "B03": "Red-L4 TM",
    "B04": "NIR-L4 TM",
    "B05": "SWIR(5)-L4 TM",
    "B07": "SWIR(7)-L4 TM",
}

SRF_LANDSAT = {"LC09": {}, "LC08": {}, "LT05": {}, "LT04": {}, "LE07": {}}


def srf_landsat_band(satellite: str, band: str, cache: bool = True) -> pd.DataFrame:
    """
    Loads the Spectral Response Function (SRF) for a given Landsat band.

    It uses the SRF provided by USGS:
     - L8: https://landsat.usgs.gov/landsat/spectral_viewer/bands/L8_OLI_RSR.xlsx
     - L9: https://landsat.usgs.gov/landsat/spectral_viewer/bands/L9_OLI2_RSR.xlsx
     - L7: https://landsat.usgs.gov/landsat/spectral_viewer/bands/L7_ETM_RSR.xlsx
     - L5: https://landsat.usgs.gov/landsat/spectral_viewer/bands/L5_TM_RSR.xlsx
     - L4: https://landsat.usgs.gov/landsat/spectral_viewer/bands/L4_TM_RSR.xlsx

    Args:
        satellite (str): satellite acronym (LC08, LC09, LT05, LT04, LE07)
        band (str): band name (B01, B02, ..., B09)
        cache (bool, optional): If True, it caches the SRF in memory. Defaults to True.

    Returns:
        pd.DataFrame: SRF for the given band
    """
    assert (
        satellite in LINK_RSR_LANDSAT.keys()
    ), f"satellite must be one of {LINK_RSR_LANDSAT.keys()}"
    assert (
        band in BAND_TO_SHEET_NAME_L89.keys()
    ), f"band must be one of {BAND_TO_SHEET_NAME_L89.keys()}"

    if cache:
        global SRF_LANDSAT
        if band in SRF_LANDSAT[satellite]:
            return SRF_LANDSAT[satellite][band]

    link = LINK_RSR_LANDSAT[satellite]
    home_dir = os.path.join(os.path.expanduser("~"), ".georeader")
    srf_file_local = os.path.join(home_dir, os.path.basename(link))

    if not os.path.exists(srf_file_local):
        os.makedirs(home_dir, exist_ok=True)
        import fsspec

        with fsspec.open(link, "rb") as f:
            with open(srf_file_local, "wb") as f2:
                f2.write(f.read())

    if satellite == "LE07":
        sheet_name = BANDS_TO_SHEET_NAME_L7[band]
    elif satellite == "LT05":
        sheet_name = BANDS_TO_SHEET_NAME_L5[band]
    elif satellite == "LT04":
        sheet_name = BANDS_TO_SHEET_NAME_L4[band]
    elif satellite in ["LC08", "LC09"]:
        sheet_name = BAND_TO_SHEET_NAME_L89[band]
    else:
        raise ValueError(f"Satellite {satellite} not supported")

    dat = pd.read_excel(srf_file_local, engine="openpyxl", sheet_name=sheet_name)
    dat = dat.iloc[:, [0, 1]]
    dat.columns = ["wavelength", band]
    dat = dat.set_index("wavelength")
    if cache:
        SRF_LANDSAT[satellite][band] = dat

    return dat


@cache
def load_all_lut(
    lut_file: str = FILE_LUT_GAS,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Loads all the variables from a LUT file.

    Args:
        lut_file (str): LUT file path

    Returns:
        Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
            wvl_mod, t_ch4_arr, mr_ch4_arr, amf_arr, eg_arr, trans_tot_arr
    """
    from netCDF4 import Dataset

    nc_lut = Dataset(lut_file, "r", format="NETCDF4")
    wvl_mod = np.array(nc_lut.variables["wvl_mod"])
    t_ch4_arr = np.array(nc_lut.variables["t_ch4_arr"]).T  # (8, 9, len(wvl_mod))
    mr_ch4_arr = np.array(nc_lut.variables["mr_ch4_arr"]).T  # (8, 9)
    amf_arr = np.array(nc_lut.variables["amf_arr"])  # (8,)

    wavelength = np.array(nc_lut.variables["wavelength"])
    ediff = np.array(nc_lut.variables["ediff"])
    edir = np.array(nc_lut.variables["edir"])
    trans_tot = np.array(nc_lut.variables["trans_tot"])

    eg = ediff + edir

    eg_interp = interpolate.interp1d(wavelength, eg, kind="cubic", bounds_error=False, fill_value=0)
    trans_tot_interp = interpolate.interp1d(
        wavelength, trans_tot, kind="cubic", bounds_error=False, fill_value=0
    )

    eg_arr = eg_interp(wvl_mod)
    trans_tot_arr = trans_tot_interp(wvl_mod)

    return wvl_mod, t_ch4_arr, mr_ch4_arr, amf_arr, eg_arr, trans_tot_arr


def air_mass_factor(sza: float, vza: float) -> float:
    """
    Air Mass Factor (AMF) for a given Solar Zenith Angle (SZA) and Viewing Zenith Angle (VZA).
    The AMF is given by the formula:
    AMF = 1 / cos(VZA) + 1 / cos(SZA)

    Args:
        sza (float): Solar Zenith Angle in degrees
        vza (float): Viewing Zenith Angle in degrees

    Returns:
        float: Air Mass Factor
    """
    return 1.0 / np.cos(np.radians(vza)) + 1.0 / np.cos(np.radians(sza))


@cache
def load_srfinterpfun(
    satellite: str,
) -> Tuple[Callable[[NDArray], NDArray], Callable[[NDArray], NDArray]]:
    """
    Loads the SRF interpolation functions for bands 12 and 11 for a given satellite.

    Args:
        satellite (str): satellite acronym (S2A, S2B, LC08, LC09, LT04, LT05, LE07)

    Returns:
        b12srf_interpfunc : interpolation function for the SRF of band 12. Its input is the wavelength and its output is the SRF value.
        b11srf_interpfunc : interpolation function for the SRF of band 11. Its input is the wavelength and its output is the SRF value.
    """

    # STEP 4: Convolve results to S2 B12 SRF
    if satellite.startswith("S2"):  # select S2 SRF
        srf = S2_SAFE_reader.read_srf(satellite)
        b12srf_interpfunc = interpolate.interp1d(
            srf.index.values,
            srf["B12"].values,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        b11srf_interpfunc = interpolate.interp1d(
            srf.index.values,
            srf["B11"].values,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
    elif satellite in ["LC08", "LC09"]:  # select Landsat SRF
        srf_b11 = srf_landsat_band(satellite, "B06")
        srf_b12 = srf_landsat_band(satellite, "B07")
        b12srf_interpfunc = interpolate.interp1d(
            srf_b12.index.values,
            srf_b12["B07"].values,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        b11srf_interpfunc = interpolate.interp1d(
            srf_b11.index.values,
            srf_b11["B06"].values,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
    elif satellite in ["LT04", "LT05", "LE07"]:  # select Landsat SRF
        srf_b11 = srf_landsat_band(satellite, "B05")
        srf_b12 = srf_landsat_band(satellite, "B07")
        b12srf_interpfunc = interpolate.interp1d(
            srf_b12.index.values,
            srf_b12["B07"].values,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        b11srf_interpfunc = interpolate.interp1d(
            srf_b11.index.values,
            srf_b11["B05"].values,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
    else:
        raise ValueError(f"Satellite {satellite} not recognized")

    return b12srf_interpfunc, b11srf_interpfunc


# Original code from Gorroño et al (2023)
# # import xlrd

# FILE_LUT_GAS = './t_mr_ch4/output_Tch4_LUT_AMF_VZA_0_v2.nc'
# S2SRF = '/home/gorrono/Desktop/UPV/L2ARUT/data/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0.xlsx'
# L8OLISRF = '/home/gorrono/Desktop/UPV/Methane/L8_OLI_RSR.xlsx'
# L9OLISRF = '/home/gorrono/Desktop/UPV/Methane/L9_OLI2_RSR.xlsx'
# PROD_CH4 = 'S2B_MSIL1C_20220930T100729_N0400_R022_T33UWA_20220930T121519.zip'

# def get_mixingratio(vza,sza):
#     '''

#     Here below a snippet on how you can use the output of this function to generate an enhancement map.

#     # you will use this option if the area is sparse enough and does not affect the mean value.
#     delta_txall = (b12ch4 / b11ch4) / (b12noch4 / b11noch4) * np.nanmean(b12noch4 / b11noch4) / np.nanmean(
#         b12ch4 / b11ch4)
#     tx_all = mr_trat(1800) * delta_txall.flatten()
#     mrch4 = []
#     for tx in tx_all:
#         mrch4.append(trat_mr(tx))  # this would account for B12/B11

#     delta_mrch4 = np.array(mrch4).reshape(b12ch4.shape[0], b12ch4.shape[1]) - 1800

#     :param vza: numpy array with viewing zenith angle values per pixel in the selected region of interest
#     :param sza: numpy array with sun zenith angle values per pixel in the selected region of interest
#     :return: trat_mr,mr_trat
#     '''

#     # STEP 1: obtain the AMF for S2 B12. TODO - you could refine to specific AMF per pixel
#     amf = 1. / np.cos(np.radians(vza)) + 1. / np.cos(np.radians(sza))

#     # STEP 2: Extract wavelength (wvl_mod), Xch4 (sc_arr), mixing ratio (mr_arr) and transmission (t_arr)
#     nc_lut = Dataset(FILE_LUT_GAS, 'r', format='NETCDF4')
#     wvl_mod = np.array(nc_lut.variables['wvl_mod'])

#     tmp = nc_lut.variables['t_ch4_arr']
#     t_arr = np.copy(tmp).T

#     sc_arr = np.array(nc_lut.variables['ch4_sc_arr'])
#     tmp = np.array(nc_lut.variables['mr_ch4_arr'])
#     mr_arr = np.copy(tmp).T

#     # STEP 3: Interpolate results to specific AMF
#     amf_arr = np.array(nc_lut.variables['amf_arr'])
#     if amf > amf_arr.max():
#         amf = amf_arr.max()

#     f_t = interpolate.interp1d(amf_arr, t_arr, axis=0, kind='cubic')
#     f_mr = interpolate.interp1d(amf_arr, mr_arr, axis=0, kind='cubic')
#     t_arr = f_t(amf)
#     mr_arr = f_mr(amf)

#     # STEP 4: Convolve results to S2 B12 SRF
#     if PROD_CH4[0:7] == 'LC08_L1' or PROD_CH4[0:7] == 'LO09_L1' or PROD_CH4[0:7] == 'LC09_L1':  # select L8/L9 SRF
#         if PROD_CH4[0:7] == 'LC08_L1':
#             l8srfbook = xlrd.open_workbook(L8OLISRF)
#         elif PROD_CH4[0:7] == 'LO09_L1' or PROD_CH4[0:7] == 'LC09_L1':
#             l8srfbook = xlrd.open_workbook(L9OLISRF)
#         l8srfsheet = l8srfbook.sheet_by_name('SWIR2')
#         b12srf = l8srfsheet.col_values(1, 1)
#         b12srfcw = l8srfsheet.col_values(0, 1)
#         b12srf_interpfunc = interpolate.interp1d(b12srfcw, b12srf, kind='cubic', bounds_error=False, fill_value=0)
#     else:
#         s2srfbook = xlrd.open_workbook(S2SRF)
#         s2srfsheet = s2srfbook.sheet_by_name('Spectral Responses (' + PROD_CH4[0:3] + ')')
#         b12srf = s2srfsheet.col_values(13, 1)  # it contains the SRF of S2 13 bands in order
#         b12srfcw = np.array(s2srfsheet.col_values(0, 1))  # first column are the wavelengths
#         b12srf_interpfunc = interpolate.interp1d(b12srfcw, b12srf, kind='cubic', bounds_error=False, fill_value=0)

#     b12srf_interp = b12srf_interpfunc(wvl_mod)

#     t_s2b12 = []
#     for i in range(len(mr_arr)):
#         t_s2b12.append(np.sum(t_arr[i] * b12srf_interp) / np.sum(b12srf_interp))


#     if PROD_CH4[0:7] == 'LC08_L1' or PROD_CH4[0:7] == 'LO09_L1' or PROD_CH4[0:7] == 'LC09_L1':  # select L8/L9 SRF
#         l8srfsheet = l8srfbook.sheet_by_name('SWIR1')
#         b11srf = l8srfsheet.col_values(1, 1)
#         b11srfcw = l8srfsheet.col_values(0, 1)
#         b11srf_interpfunc = interpolate.interp1d(b11srfcw, b11srf, kind='cubic', bounds_error=False, fill_value=0)
#     else:
#         b11srf = s2srfsheet.col_values(12, 1)  # it contains the SRF of S2 13 bands in order
#         b11srfcw = np.array(s2srfsheet.col_values(0, 1))  # first column are the wavelengths
#         b11srf_interpfunc = interpolate.interp1d(b11srfcw, b11srf, kind='cubic', bounds_error=False, fill_value=0)

#     b11srf_interp = b11srf_interpfunc(wvl_mod)
#     t_s2b11 = []
#     for i in range(len(mr_arr)):
#         t_s2b11.append(np.sum(t_arr[i] * b11srf_interp) / np.sum(b11srf_interp))


#     trat_mr = interpolate.interp1d(np.array(t_s2b12) / np.array(t_s2b11), mr_arr, axis=0,
#                                         fill_value="extrapolate", kind='cubic')  # transmittance to mixing ratio
#     mr_trat = interpolate.interp1d(mr_arr, np.array(t_s2b12) / np.array(t_s2b11), axis=0,
#                                    fill_value="extrapolate", kind='cubic')  # mixing ratio to transmittance

#     return(trat_mr,mr_trat)
