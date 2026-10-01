"""
Re-exports from marshsi.quantification for backward compatibility.

The quantification logic now lives in marshsi.quantification. ``obtain_flux_rate_s2l89`` wraps
``obtain_flux_rate`` with the units and coefficients of MARS-S2L.
"""
from marshsi.quantification import (  # noqa: F401
    A_UEFF_S2,
    ATMOSPHERE_HEIGHT_METHANE,
    B_UEFF_S2,
    BACKGROUND_CONCENTRATION,
    MAX_CH4_CONCENTRATION_LUT,
    MAX_CH4_CONCENTRATION_PPB,
    MIN_CH4_CONCENTRATION_PPB,
    SIGMA_CH4_S2_PPB,
    convert_units,
    obtain_flux_rate,
)


def obtain_flux_rate_s2l89(methane_enhancement_image, plume_mask_binary, wind_speed, **kwargs):
    """Flux rate of a MARS-S2L (Sentinel-2, Landsat 8/9) plume from a ΔXCH₄ image in ppb.

    Use this instead of ``obtain_flux_rate`` for MARS-S2L enhancements. ``obtain_flux_rate``
    defaults to ppm units, ``a_u_eff=1``, ``b_u_eff=0`` and no retrieval noise. Passing a ppb image
    with those defaults inflates the flux rate by a factor of 1000 and moves the -600 ppb
    clip to -0.6 ppb. This wrapper sets the defaults that fit MARS-S2L:

    - ``units_methane_enhancement="ppb"``
    - ``a_u_eff=A_UEFF_S2`` and ``b_u_eff=B_UEFF_S2`` (Sentinel-2 effective wind speed)
    - ``sig_xch4=SIGMA_CH4_S2_PPB``

    Each one can be overridden through ``kwargs``. ``resolution`` has no default here: a
    GeoTensor input keeps its own pixel size, and array inputs must pass ``resolution``.

    Args:
        methane_enhancement_image: ΔXCH₄ image in ppb (array or GeoTensor).
        plume_mask_binary: binary mask of the plume.
        wind_speed: wind speed in m/s.
        **kwargs: other arguments of ``obtain_flux_rate``.

    Returns:
        Dict[str, Union[float, int]]: the output of ``obtain_flux_rate``.
    """
    kwargs.setdefault("units_methane_enhancement", "ppb")
    kwargs.setdefault("a_u_eff", A_UEFF_S2)
    kwargs.setdefault("b_u_eff", B_UEFF_S2)
    kwargs.setdefault("sig_xch4", SIGMA_CH4_S2_PPB)
    return obtain_flux_rate(methane_enhancement_image, plume_mask_binary, wind_speed, **kwargs)
