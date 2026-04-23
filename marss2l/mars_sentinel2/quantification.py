"""
Re-exports from marshsi.quantification for backward compatibility.

The quantification logic now lives in marshsi.quantification.
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
