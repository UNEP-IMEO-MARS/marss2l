r"""Shot-noise propagation for the MBMP methane retrieval.

Photon arrival is a counting process, so a shot-noise-limited image has
:math:`\mathrm{SNR} = \sqrt{N}` and the SNR at an arbitrary radiance follows from
a single published reference point :math:`(\mathrm{SNR}_\mathrm{ref},
L_\mathrm{ref})`:

.. math::
    \mathrm{SNR}_L = \mathrm{SNR}_\mathrm{ref}\sqrt{L / L_\mathrm{ref}}

The MBMP retrieval is a double ratio of four radiances,

.. math::
    \mathrm{MBMP} = \frac{L_{23}}{L_{16}}\cdot\frac{L'_{16}}{L'_{23}}

so first-order propagation, treating the four as independent, gives
:math:`\sigma(\mathrm{MBMP}) \approx \mathrm{MBMP}\,\eta` with

.. math::
    \eta = \sqrt{s_{23} + s_{16} + s'_{23} + s'_{16}},
    \qquad s = 1/\mathrm{SNR}^2

:math:`\eta` carries every noise assumption in the model, and which terms enter it
is the whole of the L1/L2/L3 ladder (:func:`eta_ladder`). From it come the two
quantities this module exists to compute: the **minimum significant enhancement**
:math:`\epsilon` at confidence :math:`p` (:func:`epsilon`) and the **standard
deviation of the retrieved enhancement** (:func:`sigma_delta_xch4`), both in ppb.

Both are first-order approximations, so both are checked against Monte Carlo:
:func:`monte_carlo_mbmp` and :func:`monte_carlo_delta_xch4` draw shot noise on the
four radiances and push the samples through the full retrieval. See
``scripts/figure_monte_carlo.py``.

**These are floors, not predictions of what the retrieval achieves.** They say what
photon statistics alone permit, assuming a background estimate that contributes
nothing but its own shot noise.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

#: The two SWIR bands, in the Sentinel-2 naming used across ``marss2l``: ``B11``
#: is the 1.6 um band and ``B12`` the 2.3 um methane-absorbing band.
BAND_16 = "B11"
BAND_23 = "B12"

#: Landsat carries the same two bands under its own names. ``srf_landsat_band`` is
#: keyed by these, so the translation has to happen before any SRF lookup.
LANDSAT_BAND_NAMES = {BAND_16: "B06", BAND_23: "B07"}

#: Reference radiance (W m-2 sr-1 um-1) and measured SNR per band, from the ESA and
#: NASA periodic quality assessments -- Table 1 of the shot-noise draft. Landsat's
#: SNR is roughly twice Sentinel-2's in these bands, which is why its floors come
#: out lower despite the coarser pixel.
SNR_REFERENCE: Dict[str, Dict[str, Tuple[float, float]]] = {
    "S2A": {BAND_16: (4.0, 156.0), BAND_23: (1.7, 165.0)},
    "S2B": {BAND_16: (4.0, 164.0), BAND_23: (1.7, 169.0)},
    "LC08": {BAND_16: (4.0, 267.0), BAND_23: (1.7, 327.0)},
    "LC09": {BAND_16: (4.0, 286.0), BAND_23: (1.7, 339.0)},
}

#: Satellites with no published reference point of their own, mapped to the closest
#: platform. **Provisional**: Sentinel-2C postdates the quality assessment the table
#: comes from. It is 0.2% of the MARS-S2L target images, but it also appears as the
#: *reference* pass for others, so refusing it outright would drop scenes for a
#: reason unrelated to their noise. Revisit when ESA publishes an S2C figure.
SATELLITE_ALIASES = {"S2C": "S2B"}

#: The retrieval's own clip range for the observed ratio (see
#: ``transmittance_to_ch4.MIN_MBMP_VALUE`` / ``MAX_MBMP_VALUE``). The quadratic fit
#: of :func:`lut_inverse_quadratic` covers the plume-free neighbourhood of 1 rather
#: than this whole span -- see that function.
DEFAULT_MBMP_FIT_RANGE = (0.90, 1.05)


def band_name(satellite: str, band: str) -> str:
    """The instrument's own name for one of the two SWIR bands.

    The exported rasters are labelled per instrument -- a Sentinel-2 scene carries
    ``B11`` / ``B12`` and a Landsat scene ``B06`` / ``B07`` -- so anything selecting
    a band by name has to translate first. Reading ``B12`` out of a Landsat stack
    silently picks the wrong band or fails, depending on how it is indexed.

    Args:
        satellite: Instrument name.
        band: :data:`BAND_16` or :data:`BAND_23` (Sentinel-2 naming).

    Returns:
        The band name as it appears in that instrument's imagery and SRF tables.
    """
    return band if satellite.startswith("S2") else LANDSAT_BAND_NAMES[band]


def _resolve_satellite(satellite: str) -> str:
    """Map a satellite to the one whose reference point is used for it."""
    resolved = SATELLITE_ALIASES.get(satellite, satellite)
    if resolved not in SNR_REFERENCE:
        raise KeyError(
            f"No SNR reference point for satellite {satellite!r}. "
            f"Known: {sorted(SNR_REFERENCE)} (aliased: {sorted(SATELLITE_ALIASES)})"
        )
    return resolved


def snr_reference(satellite: str, band: str) -> Tuple[float, float]:
    """Reference radiance and SNR for one band of one instrument.

    Args:
        satellite: ``S2A``, ``S2B``, ``S2C``, ``LC08`` or ``LC09``.
        band: :data:`BAND_16` or :data:`BAND_23` (Sentinel-2 naming).

    Returns:
        ``(L_ref, SNR_ref)`` -- radiance in W m-2 sr-1 um-1 and the SNR measured there.
    """
    return SNR_REFERENCE[_resolve_satellite(satellite)][band]


def snr_at_radiance(radiance: ArrayLike, satellite: str, band: str) -> NDArray:
    r"""SNR at an arbitrary radiance, by shot-noise rescaling of the reference point.

    .. math:: \mathrm{SNR}_L = \mathrm{SNR}_\mathrm{ref}\sqrt{L/L_\mathrm{ref}}

    The rescaling assumes a shot-noise-limited, linear regime. It breaks at low
    radiance, where read-out noise dominates, and at high radiance, where the
    detector is non-linear; over water and deep shadow the true noise is therefore
    **higher** than this returns, which keeps the derived floors floors.

    Args:
        radiance: Radiance in W m-2 sr-1 um-1. Non-positive values give SNR 0.
        satellite: Instrument name.
        band: :data:`BAND_16` or :data:`BAND_23`.

    Returns:
        SNR, same shape as ``radiance``.
    """
    radiance_ref, snr_ref = snr_reference(satellite, band)
    radiance = np.asarray(radiance, dtype=np.float64)
    return snr_ref * np.sqrt(np.clip(radiance, 0.0, None) / radiance_ref)


def _noise_term(radiance: ArrayLike, satellite: str, band: str) -> NDArray:
    r"""One :math:`s = 1/\mathrm{SNR}^2` term, infinite where the SNR is zero."""
    snr = snr_at_radiance(radiance, satellite, band)
    with np.errstate(divide="ignore"):
        return np.where(snr > 0, 1.0 / np.square(snr), np.inf)


def eta_from_terms(*terms: ArrayLike) -> NDArray:
    r"""Combine :math:`s` terms into :math:`\eta = \sqrt{\sum s}`.

    Args:
        *terms: One :math:`1/\mathrm{SNR}^2` term per contributing radiance.

    Returns:
        The relative noise level, dimensionless.
    """
    return np.sqrt(np.sum([np.asarray(t, dtype=np.float64) for t in terms], axis=0))


def eta_ladder(
    radiance_23: ArrayLike,
    radiance_16: ArrayLike,
    radiance_23_bg: Optional[ArrayLike] = None,
    radiance_16_bg: Optional[ArrayLike] = None,
    *,
    satellite: str,
    satellite_bg: Optional[str] = None,
) -> Dict[str, NDArray]:
    r"""The three theoretical floors, as relative noise levels.

    Each rung assumes its background estimate is free of everything except the
    photon noise it must itself carry:

    ==== ============================================= =====================
    rung assumption                                    terms
    ==== ============================================= =====================
    L1   background known exactly, at no cost          :math:`s_{23}`
    L2   background of the band ratio known exactly    :math:`s_{23}+s_{16}`
    L3   background taken from a reference image       all four
    ==== ============================================= =====================

    **L1 bounds any retrieval, not just MBMP**: for any :math:`r` that is a function
    of the observed radiances, :math:`\mathrm{Var}(r) \geq (\partial r / \partial
    L_{23})^2\sigma^2_{L_{23}}` -- however good the background estimate, the signal
    band's own photon noise passes through it. **L3 is where the baseline MBMP
    sits**: it is that method's floor, not its performance.

    ``L1 <= L2 <= L3`` holds pixel by pixel by construction, since each rung adds a
    non-negative term.

    Args:
        radiance_23: 2.3 um radiance of the target pass, W m-2 sr-1 um-1.
        radiance_16: 1.6 um radiance of the target pass.
        radiance_23_bg: 2.3 um radiance of the reference pass. Needed for L3.
        radiance_16_bg: 1.6 um radiance of the reference pass. Needed for L3.
        satellite: Instrument of the target pass.
        satellite_bg: Instrument of the reference pass. Defaults to ``satellite``,
            but 36% of the MARS-S2L pairs are cross-satellite and Landsat's SNR is
            about twice Sentinel-2's here, so passing it matters.

    Returns:
        ``{"L1": ..., "L2": ..., "L3": ...}``. ``L3`` is absent when either
        reference radiance is None -- offshore scenes use the single-pass SBMP
        retrieval and have no reference image, so L3 is undefined for them rather
        than merely unknown.
    """
    term_23 = _noise_term(radiance_23, satellite, BAND_23)
    term_16 = _noise_term(radiance_16, satellite, BAND_16)

    ladder = {"L1": eta_from_terms(term_23), "L2": eta_from_terms(term_23, term_16)}

    if radiance_23_bg is not None and radiance_16_bg is not None:
        satellite_bg = satellite if satellite_bg is None else satellite_bg
        ladder["L3"] = eta_from_terms(
            term_23,
            term_16,
            _noise_term(radiance_23_bg, satellite_bg, BAND_23),
            _noise_term(radiance_16_bg, satellite_bg, BAND_16),
        )

    return ladder


# ─────────────────────────────────────────────────────────────────────────────
# Reflectance -> radiance
# ─────────────────────────────────────────────────────────────────────────────
def band_irradiance(satellite: str, band: str) -> float:
    """Solar irradiance in one band, from its SRF over the Thuillier spectrum.

    Sentinel-2 SRFs come from ``georeader``; Landsat's from ``marss2l``'s own
    ``srf_landsat_band``, which pulls the USGS response files (and caches them under
    ``~/.georeader``) -- no second SRF source.

    **The units cancel, and that is the trap.** ``integrated_irradiance`` returns
    mW m-2 nm-1 while ``reflectance_to_radiance`` documents its argument as
    W m-2 nm-1 and returns W m-2 sr-1 nm-1, whereas :data:`SNR_REFERENCE` is in
    W m-2 sr-1 um-1. Both conversions are 1e3 in opposite directions and
    ``1 mW m-2 nm-1 == 1 W m-2 um-1`` exactly, so passing the irradiance through
    unconverted lands directly in the reference table's units. Apply both or
    neither: applying one gives radiances off by 1e3, hence a factor ~32 in every
    noise number, which looks plausible rather than crashing.

    Args:
        satellite: Instrument name.
        band: :data:`BAND_16` or :data:`BAND_23` (Sentinel-2 naming; translated for
            Landsat).

    Returns:
        Band-integrated solar irradiance in W m-2 um-1.
    """
    from georeader import reflectance

    if satellite.startswith("S2"):
        from georeader.readers import S2_SAFE_reader

        srf = S2_SAFE_reader.read_srf(satellite)
        irradiance = reflectance.integrated_irradiance(srf)
        return float(np.atleast_1d(irradiance)[list(srf.columns).index(band)])

    from marss2l.mars_sentinel2.mixing_ratio_methane import srf_landsat_band

    srf = srf_landsat_band(satellite, band_name(satellite, band))
    return float(np.atleast_1d(reflectance.integrated_irradiance(srf))[0])


def radiance_from_reflectance(
    reflectance_values: ArrayLike,
    satellite: str,
    band: str,
    sza: float,
    date_of_acquisition,
) -> NDArray:
    r"""Convert ToA reflectance to radiance for one band and one pass.

    .. math:: L = \rho E \cos(\mathrm{SZA}) / (\pi d^2)

    Each pass needs its own SZA and date: for the reference pass those are
    ``sza_bg`` / ``tile_date_bg``, not the target's. Mixing a stored angle for one
    pass with a computed one for the other puts a spurious asymmetry straight into
    :math:`\eta`, where the two enter as separate terms.

    Args:
        reflectance_values: ToA reflectance, dimensionless.
        satellite: Instrument name.
        band: :data:`BAND_16` or :data:`BAND_23`.
        sza: Solar zenith angle in degrees, for **this** pass.
        date_of_acquisition: Acquisition time of this pass; anything
            ``solar_geometry.as_utc`` accepts.

    Returns:
        Radiance in W m-2 sr-1 um-1, the units of :data:`SNR_REFERENCE`.
    """
    from georeader import reflectance as georeader_reflectance

    from marss2l.solar_geometry import as_utc

    distance_factor = georeader_reflectance.earth_sun_distance_correction_factor(
        as_utc(date_of_acquisition)
    )
    correction = np.pi * distance_factor**2 / np.cos(np.radians(sza))

    return np.asarray(reflectance_values, dtype=np.float64) * (
        band_irradiance(satellite, band) / correction
    )


# ─────────────────────────────────────────────────────────────────────────────
# The two derived quantities
# ─────────────────────────────────────────────────────────────────────────────
def _default_lut():
    """The public transmittance look-up table bundled with ``marss2l``."""
    from marss2l.mars_sentinel2.transmittance_to_ch4 import (
        TransmittanceCH4InterpolationFromDict,
    )

    return TransmittanceCH4InterpolationFromDict()


def significance_ratio(eta: ArrayLike, p: float = 0.95) -> NDArray:
    r"""The transmittance ratio a signal must reach to be significant at level ``p``.

    .. math:: \frac{1}{\Phi^{-1}(p)\,\eta + 1}

    Args:
        eta: Relative noise level.
        p: Confidence level.

    Returns:
        Threshold ratio, below 1 and approaching it as ``eta`` goes to 0.
    """
    return 1.0 / (stats.norm.ppf(p) * np.asarray(eta, dtype=np.float64) + 1.0)


def epsilon(
    eta: ArrayLike,
    satellite: str,
    sza: float,
    vza: float,
    *,
    p: float = 0.95,
    lut=None,
) -> NDArray:
    r"""Minimum statistically significant enhancement, in ppb.

    .. math::
        \epsilon = \Delta\tau_{23/16}^{-1}\!\left(\frac{1}{\Phi^{-1}(p)\eta+1}\right)

    Treating the transmittance ratio as normal about the observed MBMP and
    inverting through the monotone look-up table. Monotone increasing in ``eta``,
    and 0 as ``eta`` goes to 0.

    Args:
        eta: Relative noise level -- any rung of :func:`eta_ladder`.
        satellite: Instrument name (the LUT covers S2A/S2B/S2C/LC08/LC09).
        sza: Solar zenith angle in degrees, for the air-mass factor.
        vza: View zenith angle in degrees.
        p: Confidence level. 0.95 is what the figures report.
        lut: Transmittance interpolation object. Defaults to the bundled public LUT.

    Returns:
        Enhancement in ppb, same shape as ``eta`` (a scalar for a scalar).
    """
    lut = _default_lut() if lut is None else lut
    ratio = significance_ratio(eta, p=p)
    result = np.asarray(
        lut.deltach4_from_ratio_transmittance(satellite, sza=sza, vza=vza, ratio_il=ratio)
    )
    # The LUT always returns at least 1-d; give a scalar back for a scalar input.
    return result.reshape(np.shape(ratio))


def lut_inverse_quadratic(
    satellite: str,
    sza: float,
    vza: float,
    *,
    lut=None,
    mbmp_range: Tuple[float, float] = DEFAULT_MBMP_FIT_RANGE,
    n_points: int = 101,
) -> NDArray:
    r"""Quadratic fit of the LUT inverse, :math:`\Delta \text{XCH}_4(\mathrm{MBMP})`.

    Returns coefficients :math:`(a, b, c)` of :math:`a\,m^2 + b\,m + c`, whose
    derivative :math:`2am + b` is what :func:`sigma_delta_xch4` propagates through.

    **The fit is local, over the plume-free neighbourhood of 1, not over the
    retrieval's whole clip range.** The LUT inverse is strongly convex -- an MBMP of
    0.8 is already ~20,000 ppb -- so a quadratic over [0.3, 1.08] would be a poor
    fit exactly where the noise lives. What the propagation needs is the local
    slope near the observed MBMP, and near 1 a quadratic captures it well.

    Args:
        satellite: Instrument name.
        sza: Solar zenith angle in degrees.
        vza: View zenith angle in degrees.
        lut: Transmittance interpolation object. Defaults to the bundled public LUT.
        mbmp_range: Range to fit over.
        n_points: Samples of the LUT inside that range.

    Returns:
        Array ``[a, b, c]``, highest power first, as ``numpy.polyfit`` returns.
    """
    lut = _default_lut() if lut is None else lut
    mbmp = np.linspace(mbmp_range[0], mbmp_range[1], n_points)
    delta_xch4 = np.asarray(
        lut.deltach4_from_ratio_transmittance(satellite, sza=sza, vza=vza, ratio_il=mbmp)
    )
    return np.polyfit(mbmp, delta_xch4, 2)


def sigma_delta_xch4(
    mbmp: ArrayLike,
    eta: ArrayLike,
    satellite: str,
    sza: float,
    vza: float,
    *,
    lut=None,
    mbmp_range: Tuple[float, float] = DEFAULT_MBMP_FIT_RANGE,
) -> NDArray:
    r"""Standard deviation of the retrieved enhancement, in ppb.

    Propagating :math:`\sigma(\mathrm{MBMP}) = \mathrm{MBMP}\,\eta` once more
    through the LUT inverse fitted as a quadratic:

    .. math::
        \sigma(\Delta \text{XCH}_4) \approx
        \mathrm{MBMP}\,\eta\,\lvert 2a\,\mathrm{MBMP} + b\rvert

    The absolute value is not in the draft's equation but is required: the LUT
    inverse is monotone **decreasing** (a smaller ratio means more methane), so
    :math:`2am+b` is negative and the expression as written returns a negative
    standard deviation.

    Args:
        mbmp: Observed transmittance ratio. 1 on plume-free ground, since
            ``ratio_IL(normalize=True)`` divides by the scene mean.
        eta: Relative noise level -- any rung of :func:`eta_ladder`.
        satellite: Instrument name.
        sza: Solar zenith angle in degrees.
        vza: View zenith angle in degrees.
        lut: Transmittance interpolation object. Defaults to the bundled public LUT.
        mbmp_range: Range the quadratic is fitted over.

    Returns:
        Standard deviation in ppb, broadcast over ``mbmp`` and ``eta``.
    """
    coefficients = lut_inverse_quadratic(
        satellite, sza=sza, vza=vza, lut=lut, mbmp_range=mbmp_range
    )
    mbmp = np.asarray(mbmp, dtype=np.float64)
    slope = 2.0 * coefficients[0] * mbmp + coefficients[1]
    return mbmp * np.asarray(eta, dtype=np.float64) * np.abs(slope)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo -- the check the whole epic rests on
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
    r"""Draw MBMP samples with shot noise on all four radiances.

    Each radiance is perturbed by Gaussian noise of relative width
    :math:`1/\mathrm{SNR}_L` -- the shot-noise limit at that radiance -- and the
    samples are pushed through the full double ratio, with no expansion anywhere.
    The true MBMP is :math:`(L_{23}/L_{16})(L'_{16}/L'_{23})`, so passing identical
    target and reference radiances gives a plume-free truth of exactly 1.

    Args:
        radiance_23: 2.3 um radiance of the target pass, W m-2 sr-1 um-1.
        radiance_16: 1.6 um radiance of the target pass.
        radiance_23_bg: 2.3 um radiance of the reference pass.
        radiance_16_bg: 1.6 um radiance of the reference pass.
        satellite: Instrument of the target pass.
        satellite_bg: Instrument of the reference pass. Defaults to ``satellite``.
        n_samples: Number of draws.
        rng: Generator, for reproducibility.

    Returns:
        Array of ``n_samples`` MBMP values.
    """
    rng = np.random.default_rng() if rng is None else rng
    satellite_bg = satellite if satellite_bg is None else satellite_bg

    def draw(radiance: float, sat: str, band: str) -> NDArray:
        snr = snr_at_radiance(radiance, sat, band)
        return rng.normal(radiance, radiance / snr, size=n_samples)

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
    lut=None,
) -> NDArray:
    """Draw retrieved-enhancement samples, through the full LUT inversion.

    :func:`monte_carlo_mbmp` followed by the operational inversion, so the result
    carries both the double-ratio non-linearity and the curvature of the LUT --
    the two approximations :func:`sigma_delta_xch4` makes.

    Args:
        radiance_23: 2.3 um radiance of the target pass, W m-2 sr-1 um-1.
        radiance_16: 1.6 um radiance of the target pass.
        radiance_23_bg: 2.3 um radiance of the reference pass.
        radiance_16_bg: 1.6 um radiance of the reference pass.
        satellite: Instrument of the target pass.
        sza: Solar zenith angle in degrees.
        vza: View zenith angle in degrees.
        satellite_bg: Instrument of the reference pass. Defaults to ``satellite``.
        n_samples: Number of draws.
        rng: Generator, for reproducibility.
        lut: Transmittance interpolation object. Defaults to the bundled public LUT.

    Returns:
        Array of ``n_samples`` enhancements in ppb.
    """
    lut = _default_lut() if lut is None else lut
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
    # clip_values_retrieval=False: the operational clip to [0.3, 1.08] would
    # truncate the very tail whose width is being measured.
    return np.asarray(
        lut.deltach4_from_ratio_transmittance(
            satellite,
            sza=sza,
            vza=vza,
            ratio_il=samples,
            clip_values_retrieval=False,
        )
    )
