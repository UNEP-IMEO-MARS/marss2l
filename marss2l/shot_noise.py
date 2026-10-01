r"""Photon-noise floors for the MBMP methane retrieval.

This module implements the error budget of *On the limits of methane detection for
Sentinel-2 and Landsat* (Mateo-García et al.). Equation labels below refer to that paper.

**Signal-to-noise at any radiance** (``eq:snrscale``). For a shot-noise-limited detector,
the SNR at radiance :math:`L` follows from one published reference point:

.. math:: \mathrm{SNR}_L = \mathrm{SNR}_\mathrm{ref}\sqrt{L / L_\mathrm{ref}}

**Propagation through the double ratio** (``eq:eta``). MBMP is a ratio of four radiances,
so to first order its relative noise is

.. math:: \eta = \sqrt{s_{23} + s_{16} + s'_{23} + s'_{16}}, \qquad s = \mathrm{SNR}^{-2}

Which terms enter :math:`\eta` defines the three floors (:func:`eta_ladder`).

**Conversion to ppb.** :func:`sigma_delta_xch4` gives the standard deviation of the
retrieved enhancement (``eq:sigmaxch``) and :func:`epsilon` the minimum significant
enhancement at a confidence level (``eq:eps``). Both are first-order and are checked
against Monte Carlo by :func:`monte_carlo_mbmp` and :func:`monte_carlo_delta_xch4`.

**Resolution.** The floors describe one *native* pixel. :data:`NOISE_FACTOR_10M` gives how
much interpolating to the published 10 m grid lowers per-pixel noise, which is what is
needed to compare a floor with noise measured on the 10 m chips.

**Detectable source rate** (``eq:q50``). :func:`q50_from_noise` turns a noise level into the
source rate detected half the time.

These are *floors*: what photon statistics alone permit, given a background estimate that
contributes nothing but its own photon noise.
"""

from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Instruments
# ─────────────────────────────────────────────────────────────────────────────
#: The two SWIR bands, in the Sentinel-2 naming used across ``marss2l``: ``B11`` is the
#: 1.6 um band (no methane absorption) and ``B12`` the 2.3 um band (absorbing).
BAND_16 = "B11"
BAND_23 = "B12"

#: Landsat's names for the same two bands, needed to look up its spectral response.
_LANDSAT_BAND = {BAND_16: "B06", BAND_23: "B07"}

#: Reference radiance (W m-2 sr-1 um-1) and measured SNR per band (Table 1 of the paper).
#: Sentinel-2: ESA *MSI S2 Annual Performance Report, Year 2025*, Tables 20 and 21.
#: Landsat: USGS ECCOE quarterly calibration report Q1 2025, Table 3 (reference radiance)
#: and Figures 1 and 40 (median SNR at that radiance, March 2025).
SNR_REFERENCE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "S2A": {BAND_16: (4.0, 157.0), BAND_23: (1.7, 165.0)},
    "S2B": {BAND_16: (4.0, 164.0), BAND_23: (1.7, 169.0)},
    "S2C": {BAND_16: (4.0, 182.0), BAND_23: (1.7, 183.0)},
    "LC08": {BAND_16: (4.0, 267.0), BAND_23: (1.7, 327.0)},
    "LC09": {BAND_16: (4.0, 286.0), BAND_23: (1.7, 339.0)},
}


def snr_at_radiance(radiance: ArrayLike, satellite: str, band: str) -> NDArray:
    r"""SNR at a given radiance by shot-noise scaling of the reference point (``eq:snrscale``).

    Assumes a shot-noise-limited, linear detector. At low radiance read-out and dark noise
    matter and the true SNR is lower than this, so floors over dark ground are optimistic.

    Args:
        radiance: Radiance in W m-2 sr-1 um-1.
        satellite: One of :data:`SNR_REFERENCE`.
        band: :data:`BAND_16` or :data:`BAND_23`.
    """
    radiance_ref, snr_ref = SNR_REFERENCE[satellite][band]
    return snr_ref * np.sqrt(np.asarray(radiance, dtype=np.float64) / radiance_ref)


def eta_ladder(
    radiance_23: ArrayLike,
    radiance_16: ArrayLike,
    radiance_23_bg: Optional[ArrayLike] = None,
    radiance_16_bg: Optional[ArrayLike] = None,
    *,
    satellite: str,
    satellite_bg: Optional[str] = None,
) -> Dict[str, NDArray]:
    r"""The three floors as relative noise levels :math:`\eta` (``eq:eta``, Table ``tab:ladder``).

    ==== ===================================================== ==================================
    rung the background estimate may replace                     terms in :math:`\eta^2`
    ==== ===================================================== ==================================
    L1   everything but the signal band: bounds *any* retrieval  :math:`s_{23}`
    L2   the reference pass                                      :math:`s_{23} + s_{16}`
    L3   nothing: the floor of MBMP as operated                  :math:`s_{23} + s_{16} + s'_{23} + s'_{16}`
    ==== ===================================================== ==================================

    Each rung adds a non-negative term, so ``L1 <= L2 <= L3`` pixel by pixel.

    Args:
        radiance_23: 2.3 um radiance of the target pass, W m-2 sr-1 um-1.
        radiance_16: 1.6 um radiance of the target pass.
        radiance_23_bg: 2.3 um radiance of the reference pass.
        radiance_16_bg: 1.6 um radiance of the reference pass.
        satellite: Instrument of the target pass.
        satellite_bg: Instrument of the reference pass, which may differ from the target's
            (a third of MARS-S2L pairs cross platforms). Defaults to ``satellite``.

    Returns:
        ``{"L1": ..., "L2": ..., "L3": ...}``. ``L3`` is absent without a reference pass:
        offshore scenes use a single-pass retrieval, for which it is undefined.
    """
    s_23 = snr_at_radiance(radiance_23, satellite, BAND_23) ** -2.0
    s_16 = snr_at_radiance(radiance_16, satellite, BAND_16) ** -2.0
    ladder = {"L1": np.sqrt(s_23), "L2": np.sqrt(s_23 + s_16)}

    if radiance_23_bg is not None and radiance_16_bg is not None:
        satellite_bg = satellite_bg or satellite
        s_23_bg = snr_at_radiance(radiance_23_bg, satellite_bg, BAND_23) ** -2.0
        s_16_bg = snr_at_radiance(radiance_16_bg, satellite_bg, BAND_16) ** -2.0
        ladder["L3"] = np.sqrt(s_23 + s_16 + s_23_bg + s_16_bg)
    return ladder


# ─────────────────────────────────────────────────────────────────────────────
# Reflectance -> radiance
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def band_irradiance(satellite: str, band: str) -> float:
    """Band-integrated solar irradiance, W m-2 um-1: the band's spectral response over the
    Thuillier spectrum.

    ``georeader``'s ``integrated_irradiance`` returns mW m-2 nm-1 and its
    ``reflectance_to_radiance`` treats that as W m-2 nm-1. The two factors of 1000 cancel,
    and ``1 mW m-2 nm-1 == 1 W m-2 um-1``, so the value can be used as is in the units of
    :data:`SNR_REFERENCE`. Converting only one of them would put every radiance off by 1000.
    """
    from georeader import reflectance

    if satellite.startswith("S2"):
        from georeader.readers import S2_SAFE_reader

        srf = S2_SAFE_reader.read_srf(satellite)
        irradiance = np.atleast_1d(reflectance.integrated_irradiance(srf))
        return float(irradiance[list(srf.columns).index(band)])

    from marss2l.mars_sentinel2.mixing_ratio_methane import srf_landsat_band

    srf = srf_landsat_band(satellite, _LANDSAT_BAND[band])
    return float(np.atleast_1d(reflectance.integrated_irradiance(srf))[0])


def radiance_from_reflectance(
    reflectance_values: ArrayLike, satellite: str, band: str, sza: float, date_of_acquisition
) -> NDArray:
    r"""Top-of-atmosphere reflectance to radiance, :math:`L = \rho E \cos\theta_s / (\pi d^2)`.

    Each pass needs its *own* solar zenith angle and date: the reference pass is weeks or
    months from the target, so its illumination and Earth-Sun distance differ.

    Args:
        reflectance_values: ToA reflectance, dimensionless.
        satellite: Instrument of this pass.
        band: :data:`BAND_16` or :data:`BAND_23`.
        sza: Solar zenith angle of this pass, degrees.
        date_of_acquisition: Acquisition time of this pass.

    Returns:
        Radiance in W m-2 sr-1 um-1.
    """
    from georeader import reflectance as georeader_reflectance

    from marss2l.solar_geometry import as_utc

    earth_sun = georeader_reflectance.earth_sun_distance_correction_factor(
        as_utc(date_of_acquisition)
    )
    scale = band_irradiance(satellite, band) * np.cos(np.radians(sza)) / (np.pi * earth_sun**2)
    return np.asarray(reflectance_values, dtype=np.float64) * scale


# ─────────────────────────────────────────────────────────────────────────────
# From relative noise to ppb
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def default_lut():
    """The transmittance look-up table bundled with ``marss2l``, loaded once."""
    from marss2l.mars_sentinel2.transmittance_to_ch4 import TransmittanceCH4InterpolationFromDict

    return TransmittanceCH4InterpolationFromDict()


def significance_ratio(eta: ArrayLike, p: float = 0.95) -> NDArray:
    r"""The transmittance ratio a pixel must fall below to be significant at level ``p``,
    :math:`1 / (\Phi^{-1}(p)\,\eta + 1)`. Below 1, and tending to 1 as the noise vanishes."""
    return 1.0 / (stats.norm.ppf(p) * np.asarray(eta, dtype=np.float64) + 1.0)


def epsilon(eta: ArrayLike, satellite: str, sza: float, vza: float, *, p: float = 0.95) -> NDArray:
    r"""Minimum significant enhancement in ppb (``eq:eps``, Appendix A).

    .. math:: \epsilon = \Delta\tau_{23/16}^{-1}\left(\frac{1}{\Phi^{-1}(p)\,\eta + 1}\right)

    The enhancement a pixel with an ideal reference needs for the retrieval to read above
    zero with probability ``p``. Same shape as ``eta``.
    """
    ratio = significance_ratio(eta, p=p)
    ppb = default_lut().deltach4_from_ratio_transmittance(
        satellite, sza=sza, vza=vza, ratio_il=ratio
    )
    return np.asarray(ppb).reshape(np.shape(ratio))


#: Range of MBMP over which the look-up-table inverse is fitted by a quadratic. Local on
#: purpose: the inverse is strongly convex, and a fit over the retrieval's whole clip range
#: [0.3, 1.08] would even get the sign of the slope at 1 wrong.
_FIT_RANGE = (0.90, 1.05)


@lru_cache(maxsize=4096)
def _lut_slope_at_one(satellite: str, sza: float, vza: float) -> float:
    """Slope of the look-up-table inverse at MBMP = 1, in ppb per unit ratio, from a quadratic
    :math:`a m^2 + b m + c` fitted over :data:`_FIT_RANGE`: :math:`2a + b`. Negative, since a
    smaller ratio means more methane. Within 0.2 % of a numerical derivative of the table."""
    mbmp = np.linspace(*_FIT_RANGE, 101)
    ppb = default_lut().deltach4_from_ratio_transmittance(
        satellite, sza=sza, vza=vza, ratio_il=mbmp
    )
    a, b, _ = np.polyfit(mbmp, np.asarray(ppb), 2)
    return 2.0 * a + b


def sigma_delta_xch4(eta: ArrayLike, satellite: str, sza: float, vza: float) -> NDArray:
    r"""Standard deviation of the retrieved enhancement, in ppb (``eq:sigmaxch``).

    .. math:: \sigma(\Delta\mathrm{XCH_4}) \approx \mathrm{MBMP}\,\eta\,|2a\,\mathrm{MBMP} + b|

    evaluated at MBMP = 1, which is where the retrieval reads on plume-free ground once each
    pass is normalised by its own mean band ratio (the factor :math:`\kappa` of ``eq:mbmp``).
    The absolute value is needed because the inverse is decreasing.

    Args:
        eta: Relative noise level, any rung of :func:`eta_ladder`.
        satellite: Instrument of the target pass (the look-up table is per instrument).
        sza: Solar zenith angle of the target pass, degrees.
        vza: View zenith angle, degrees.
    """
    return np.asarray(eta, dtype=np.float64) * abs(
        _lut_slope_at_one(satellite, float(sza), float(vza))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation against Monte Carlo (Appendix B)
# ─────────────────────────────────────────────────────────────────────────────
def monte_carlo_mbmp(
    radiance_23: float,
    radiance_16: float,
    radiance_23_bg: float,
    radiance_16_bg: float,
    *,
    satellite: str,
    satellite_bg: Optional[str] = None,
    n_samples: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Draw MBMP samples with Gaussian shot noise on all four radiances, with no expansion.

    Identical target and reference radiances give a true MBMP of exactly 1.
    """
    rng = rng or np.random.default_rng()
    satellite_bg = satellite_bg or satellite

    def draw(radiance: float, sat: str, band: str) -> NDArray:
        return rng.normal(radiance, radiance / snr_at_radiance(radiance, sat, band), size=n_samples)

    return (
        draw(radiance_23, satellite, BAND_23)
        / draw(radiance_16, satellite, BAND_16)
        * draw(radiance_16_bg, satellite_bg, BAND_16)
        / draw(radiance_23_bg, satellite_bg, BAND_23)
    )


def monte_carlo_delta_xch4(
    radiance_23: float,
    radiance_16: float,
    radiance_23_bg: float,
    radiance_16_bg: float,
    *,
    satellite: str,
    sza: float,
    vza: float,
    satellite_bg: Optional[str] = None,
    n_samples: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """:func:`monte_carlo_mbmp` pushed through the full look-up-table inversion, in ppb.

    The retrieval's clip of the ratio to [0.3, 1.08] is switched off, since it would truncate
    the tail whose width is being measured.
    """
    samples = monte_carlo_mbmp(
        radiance_23,
        radiance_16,
        radiance_23_bg,
        radiance_16_bg,
        satellite=satellite,
        satellite_bg=satellite_bg,
        n_samples=n_samples,
        rng=rng,
    )
    return np.asarray(
        default_lut().deltach4_from_ratio_transmittance(
            satellite, sza=sza, vza=vza, ratio_il=samples, clip_values_retrieval=False
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────
#: Side of a native pixel of the SWIR bands, in metres, keyed by platform family.
PIXEL_SIZE = {"S2": 20.0, "LC": 30.0}

#: Per-pixel photon noise on the published 10 m grid relative to a native pixel.
#: Interpolation makes each 10 m pixel a weighted mean of native ones, which lowers its
#: independent noise by a fixed factor:
#:
#: * ``S2``, 20 -> 10 m bilinear: weights 3/4 and 1/4 per axis, so the variance shrinks by
#:   :math:`(9/16 + 1/16)^2` and the standard deviation by 0.625.
#: * ``LC``, 30 -> 10 m cubic spline (``georeader.read.resize``): 0.48, measured by passing
#:   white noise through the resize, since the spline has no short closed form.
#:
#: A floor is per native pixel; multiply it by this factor before comparing it with noise
#: measured on the 10 m chips (``stats_dataset.csv``). Noise measured on the native grid
#: (``stats_dataset_native.csv``) needs no factor.
NOISE_FACTOR_10M = {"S2": 0.625, "LC": 0.48}


# ─────────────────────────────────────────────────────────────────────────────
# From noise to a detectable source rate (eq:q50)
# ─────────────────────────────────────────────────────────────────────────────
#: Background methane mixing ratio the noise is expressed relative to, ppb.
BACKGROUND_XCH4_PPB = 1800.0

#: Observability at 50 % probability of detection, per instrument. Fitted against noise
#: measured on the published products (Sentinel-2 at 10 m, Landsat at 30 m), so the noise
#: passed to :func:`q50_from_noise` must be on those grids. **Provisional**: to be replaced
#: by a fit in the observability of Bruno et al. (2024) on the native grids.
OBSERVABILITY_50: Dict[str, float] = {
    "S2A": 0.059,
    "S2B": 0.0638,
    "S2C": 0.059,
    "LC08": 0.059,
    "LC09": 0.059,
}


def q50_from_noise(
    noise_ppb: ArrayLike, satellite: ArrayLike, wind_speed: ArrayLike = 1.0
) -> NDArray:
    r"""Source rate detected with 50 % probability, in kg h-1 (``eq:q50``).

    .. math:: Q_{50} = 3600\, O_{50}\, U\, W\, \sigma / 1800\,\mathrm{ppb}

    Args:
        noise_ppb: Standard deviation of the retrieval, ppb, on the grid of
            :data:`OBSERVABILITY_50`.
        satellite: Instrument, one per noise value.
        wind_speed: 10 m wind speed, m s-1.
    """
    satellites = np.atleast_1d(satellite)
    o50 = np.array([OBSERVABILITY_50[s] for s in satellites])
    pixel = np.array([PIXEL_SIZE[s[:2]] for s in satellites])
    relative_noise = np.asarray(noise_ppb, dtype=np.float64) / BACKGROUND_XCH4_PPB
    q50 = 3600.0 * o50 * np.asarray(wind_speed) * pixel * relative_noise
    return q50.reshape(
        np.broadcast_shapes(np.shape(satellite), np.shape(noise_ppb), np.shape(wind_speed))
    )
