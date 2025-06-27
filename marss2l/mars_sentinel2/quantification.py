from georeader.geotensor import GeoTensor
from typing import Optional, Tuple, Union, Dict
import numpy as np

from numpy.typing import NDArray

"""

References:
 * [Roger 2024] Roger, J., Irakulis-Loitxate, I., Valverde, A., Gorroño, J., Chabrillat, S., Brell, M., 
 & Guanter, L. (2024). 
 High-Resolution Methane Mapping With the EnMAP Satellite Imaging Spectroscopy Mission. 
 IEEE Transactions on Geoscience and Remote Sensing, 62, 1–12. https://doi.org/10.1109/TGRS.2024.3352403
 * [Thorpe 2023] Thorpe, A. K., Green, R. O., Thompson, D. R., Brodrick, P. G., Chapman, J. W., 
 Elder, C. D., Irakulis-Loitxate, I., Cusworth, D. H., Ayasse, A. K., Duren, R. M., 
 Frankenberg, C., Guanter, L., Worden, J. R., Dennison, P. E., Roberts, D. A., 
 Chadwick, K. D., Eastwood, M. L., Fahlen, J. E., & Miller, C. E. (2023). 
 Attribution of individual methane and carbon dioxide emission sources using EMIT observations from space. 
 Science Advances, 9(46), eadh2391. https://doi.org/10.1126/sciadv.adh2391 
* [Gorroño 2023] Gorroño, J., Varon, D. J., Irakulis-Loitxate, I., & Guanter, L. (2023). 
  Understanding the potential of Sentinel-2 for monitoring methane point emissions. 
  Atmospheric Measurement Techniques, 16(1), 89–107. https://doi.org/10.5194/amt-16-89-2023
  * [Guanter 2021] Guanter, L., Irakulis-Loitxate, I., Gorroño, J., Sánchez-García, E., Cusworth, D. H., 
 Varon, D. J., Cogliati, S., & Colombo, R. (2021). 
 Mapping methane point emissions with the PRISMA spaceborne imaging spectrometer. 
 Remote Sensing of Environment, 265, 112671. https://doi.org/10.1016/j.rse.2021.112671
* [Varon 2018] Varon, D. J., Jacob, D. J., McKeever, J., Jervis, D., Durak, B. O. A., Xia, Y., 
  & Huang, Y. (2018). 
  Quantifying methane point sources from fine-scale satellite observations of atmospheric methane plumes. 
  Atmospheric Measurement Techniques, 11(10), 5673–5686. https://doi.org/10.5194/amt-11-5673-2018
 

"""

ATMOSPHERE_HEIGHT_METHANE = 8_000 # m
SIGMA_CH4_S2_PPB = 205.64270005117675

A_UEFF_S2 = 0.33 # [Varon 2018]
B_UEFF_S2 = 0.45 # [Varon 2018]
MAX_CH4_CONCENTRATION_PPB = 100_000 # ppb


def convert_units(data: Union[NDArray, GeoTensor, float, int], units_src:str, units_dst:str) -> Union[NDArray, GeoTensor, float, int]:
    """
    Converts the units of the data from `units_src` to `units_dst`.

    Args:
        data (Union[NDArray, GeoTensor, float, int]): data to convert
        units_src (str): one of "ppm", "ppb", or "ppm x m"
        units_dst (str): one of "ppm", "ppb", or "ppm x m"

    Raises:
        ValueError: if `units_src` or `units_dst` are not one of "ppm", "ppb", or "ppm x m"

    Returns:
        Union[NDArray, GeoTensor, float]: converted data
    """
    if units_src == units_dst:
        return data
    factor_to_ppmxm = 1
    if units_src != "ppm x m":
        if units_src == "ppm":
            factor_to_ppmxm = ATMOSPHERE_HEIGHT_METHANE
        elif units_src == "ppb":
            factor_to_ppmxm =  ATMOSPHERE_HEIGHT_METHANE  / 1000
        else:
            raise ValueError(f"units_src must be one of 'ppm', 'ppb', or 'ppm x m' found {units_src}")
    
    if units_dst == "ppm x m":
        return data * factor_to_ppmxm
    if units_dst == "ppm":
        return data * (factor_to_ppmxm / ATMOSPHERE_HEIGHT_METHANE)
    elif units_dst == "ppb":
        return data * (factor_to_ppmxm * 1000 / ATMOSPHERE_HEIGHT_METHANE)
   
    raise ValueError(f"units_dst must be one of 'ppm', 'ppb', or 'ppm x m'found {units_dst}")


def obtain_flux_rate(methane_enhancement_image:  Union[NDArray, GeoTensor], plume_mask_binary: Union[NDArray, GeoTensor], 
                     wind_speed:float,
                     resolution:Optional[Tuple[float,float]]=None,
                     a_u_eff:float=A_UEFF_S2, b_u_eff:float=B_UEFF_S2,
                     a_std:float=0.01, b_std:float=0.01,
                     wind_speed_unc:float=0.5,
                     n_samp:int=100_000,
                     sig_xch4:Optional[float]=SIGMA_CH4_S2_PPB,
                     return_std:bool=False,
                     seed:Optional[int]=None) -> Dict[str, Union[float,int]]:
    """
    Calculates the flux rate of a methane plume given the methane enhancement image, the binary plume mask, and the wind speed.

    This function assumes that methane_enhancement_image is is UTM coordinates (i.e. in meters). 

    Effective wind speed for Sentinel-2:
        0.33 * wind_speed + 0.45 Based on 3.5m/s (Cusworth 2019) used for simulations and ueff model in Varon 2020.

    Adapted from uncertainty propagation code of Javier Gorrono 30th June 2022 (adapted from original Luis Guanter)

    Args:
        methane_enhancement_image ( Union[NDArray, GeoTensor]): in ppb
        plume_mask_binary ( Union[NDArray, GeoTensor]): binary mask of the plume
        wind_speed (float): wind speed in m/s. 
        resolution (Optional[Tuple[float,float]], optional): resolution of the methane_enhancement_image in meters if it is a np.ndarray. Defaults to None.
        a_u_eff (float, optional): Slope of the linear relation between wind speed and effective wind speed. Defaults to 1.
        b_u_eff (float, optional): Intercept of the linear relation between wind speed and effective wind speed. Defaults to 0.
        a_std (float, optional): Standard deviation of the slope of the linear relation between wind speed and effective wind speed. Defaults to 0.01.
        b_std (float, optional): Standard deviation of the intercept of the linear relation between wind speed and effective wind speed. Defaults to 0.01.
        wind_speed_unc (float, optional): Relative error of the wind speed in m/s. 
            Defaults to 0.5.
        n_samp (int, optional): Number of samples for MonteCarlo propagation. Defaults to 100000.
        sig_xch4 (float, optional): Standard deviation of the methane enhancement retrieval. In ppb.
            This value is satellite specific.
        return_std (bool, optional): If True, returns the standard deviation of the flux rate. Defaults to False.
        seed (Optional[int], optional): Seed for the random number generator. Defaults to None.

    Returns:
        Dict[str, Union[float,int]]: dictionary with the following fields:
            - Q: flux rate in kg/h
            - L: sqrt(npix_plume * np.prod(plume_mask_binary.res)) in m
            - npix_plume: number of pixels in the plume
            - IME: ime in kg
            - u_eff: effective wind speed in m/s
            - pixel_size: pixel size in m²
            - sigma_q: standard deviation of the flux rate in kg/h if `return_std` is True
            - sig_xch4: standard deviation of the methane enhancement retrieval in ppm x m if `return_std` is True
    """

    assert methane_enhancement_image.shape == plume_mask_binary.shape, f"methane_enhancement_image and plume_mask_binary must have the same shape {methane_enhancement_image.shape} != {plume_mask_binary.shape}"
    assert len(methane_enhancement_image.shape) == 2, f"methane_enhancement_image and plume_mask_binary must be 2D images {methane_enhancement_image.shape}"

    if isinstance(methane_enhancement_image, GeoTensor):
        assert isinstance(plume_mask_binary, GeoTensor), "methane_enhancement_image and plume_mask_binary must be both GeoTensor or both np.ndarray"
        assert methane_enhancement_image.same_extent(plume_mask_binary), f"methane_enhancement_image and plume_mask_binary must have the same extent {methane_enhancement_image.transform} != {plume_mask_binary.transform}"
        resolution = methane_enhancement_image.res
    else:
        assert resolution is not None, "res must be provided if methane_enhancement_image is a np.ndarray"
        assert len(resolution) == 2, "res must be a tuple with two elements"

    
    max_ch4_concentration = MAX_CH4_CONCENTRATION_PPB
    
    if isinstance(methane_enhancement_image, GeoTensor):
        methane_enhancement_image = methane_enhancement_image.copy()
        methane_enhancement_image_values = methane_enhancement_image.values
        methane_enhancement_image.values[methane_enhancement_image.values == methane_enhancement_image.fill_value_default] = 0
        plume_mask_binary_values = plume_mask_binary.values
    else:
        methane_enhancement_image_values = methane_enhancement_image
        plume_mask_binary_values = plume_mask_binary

    methane_enhancement_image_values = np.clip(methane_enhancement_image_values, 0, max_ch4_concentration)

    methane_enhancement_image_values = convert_units(methane_enhancement_image_values, "ppb", "ppm x m")
    sig_xch4 = convert_units(sig_xch4, "ppb", "ppm x m")

    if not plume_mask_binary_values.dtype == bool:
        binary_mask = plume_mask_binary_values != 0
    else:
        binary_mask = plume_mask_binary_values

    ime = (np.sum(methane_enhancement_image_values[binary_mask]) * np.prod(resolution) * 1_000 * 0.01604) / (1e6 * 22.4)
    
    npix_plume = np.sum(binary_mask)

    L = np.sqrt(npix_plume * np.prod(resolution))

    effective_wind_speed = a_u_eff * wind_speed + b_u_eff

    Q = 3600 * effective_wind_speed * ime / L

    out_dict = {"Q": Q, 
                "L": L,
                "npix_plume": npix_plume,
                "IME": ime,
                "pixel_size": np.prod(resolution),
                "u_eff": effective_wind_speed}
    if not return_std:
        return out_dict

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # https://stats.stackexchange.com/questions/164505/standard-error-for-sum
    sig_IME = (sig_xch4 * np.sqrt(npix_plume) * np.prod(resolution) * 1_000 * 0.01604) / (1e6 * 22.4)

    # wind_speed_n = np.random.normal(1, wind_speed_unc, n_samp)

    noise_arr = rng.normal(0, 1, n_samp)

    # MonteCarlo propagation of the U10 to Ueff
    wind_speed_n = wind_speed * \
        (1 + noise_arr * wind_speed_unc) # uncertainty proportional to the wind speed because it is relative error
    # wind_speed_n = np.random.normal(wind_speed, wind_speed_unc*wind_speed, n_samp) # equivalent to the previous line

    ueff_n = wind_speed_n * rng.normal(a_u_eff, a_std, n_samp) + rng.normal(b_u_eff, b_std, n_samp)

    # MonteCarlo propagation of the Ueff and IME to Q
    sigma_q = np.std(3600. * ueff_n * rng.normal(ime, sig_IME, n_samp) / L)
    out_dict["sigma_Q"] = sigma_q
    out_dict["sig_xch4"] = sig_xch4

    return out_dict

