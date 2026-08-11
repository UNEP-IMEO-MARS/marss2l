"""
Tests for marss2l.shot_noise.

The Monte-Carlo checks at the end validate the *model* -- that the first-order
expansion is adequate. They do not catch a wrong constant or a flipped ratio,
because a plausible wrong number still produces a plausible figure. The cheap
by-hand tests above them are what catch those.
"""

import numpy as np
import pytest

from marss2l import shot_noise as sn

SZA, VZA = 38.5, 6.1  # the medians of the MARS-S2L target images
LUT = sn._default_lut()


# ─────────────────────────────────────────────────────────────────────────────
# SNR rescaling
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("satellite", ["S2A", "S2B", "LC08", "LC09"])
@pytest.mark.parametrize("band", [sn.BAND_16, sn.BAND_23])
def test_snr_at_the_reference_radiance_is_the_reference_snr(satellite, band):
    radiance_ref, snr_ref = sn.snr_reference(satellite, band)
    assert sn.snr_at_radiance(radiance_ref, satellite, band) == pytest.approx(snr_ref)


@pytest.mark.parametrize("satellite", ["S2A", "LC09"])
def test_snr_doubles_at_four_times_the_reference_radiance(satellite):
    """SNR goes as sqrt(L), so 4x the radiance is exactly 2x the SNR."""
    radiance_ref, snr_ref = sn.snr_reference(satellite, sn.BAND_23)
    assert sn.snr_at_radiance(4 * radiance_ref, satellite, sn.BAND_23) == pytest.approx(2 * snr_ref)


def test_snr_is_zero_at_zero_radiance():
    """Invalid pixels are stored as zero reflectance; they must not divide by zero."""
    assert sn.snr_at_radiance(0.0, "S2A", sn.BAND_23) == 0.0
    assert np.isinf(sn._noise_term(0.0, "S2A", sn.BAND_23))


def test_landsat_snr_is_about_twice_sentinel2():
    """The reason Landsat's floors come out lower despite the coarser pixel."""
    for band in [sn.BAND_16, sn.BAND_23]:
        s2 = sn.snr_at_radiance(5.0, "S2A", band)
        landsat = sn.snr_at_radiance(5.0, "LC09", band)
        assert 1.7 < landsat / s2 < 2.2


def test_band_names_are_translated_for_landsat():
    """A Landsat raster is labelled B06/B07, a Sentinel-2 one B11/B12."""
    assert sn.band_name("S2A", sn.BAND_23) == "B12"
    assert sn.band_name("S2C", sn.BAND_16) == "B11"
    assert sn.band_name("LC08", sn.BAND_23) == "B07"
    assert sn.band_name("LC09", sn.BAND_16) == "B06"


def test_s2c_falls_back_to_s2b_and_unknown_satellites_raise():
    assert sn.snr_reference("S2C", sn.BAND_23) == sn.snr_reference("S2B", sn.BAND_23)
    with pytest.raises(KeyError, match="No SNR reference point"):
        sn.snr_reference("EMIT", sn.BAND_23)


# ─────────────────────────────────────────────────────────────────────────────
# eta and the ladder
# ─────────────────────────────────────────────────────────────────────────────
def test_eta_with_four_equal_snrs_is_two_over_snr():
    """Four equal terms: eta = sqrt(4/SNR^2) = 2/SNR."""
    snr = 200.0
    term = 1.0 / snr**2
    assert sn.eta_from_terms(term, term, term, term) == pytest.approx(2.0 / snr)


def test_dropping_the_primed_pair_is_exactly_sqrt2_smaller():
    """Two of four equal terms removed: sqrt(2) less noise, by construction."""
    term = 1.0 / 200.0**2
    four = sn.eta_from_terms(term, term, term, term)
    two = sn.eta_from_terms(term, term)
    assert four / two == pytest.approx(np.sqrt(2.0))


def test_the_ladder_is_ordered_on_random_inputs():
    """L1 <= L2 <= L3 pixel by pixel, by construction.

    The cheapest guard against a term landing in the wrong rung, and the one
    worth writing first: it holds for any radiances at all.
    """
    rng = np.random.default_rng(0)
    radiances = rng.uniform(0.2, 40.0, size=(4, 500))
    ladder = sn.eta_ladder(*radiances, satellite="S2A", satellite_bg="LC08")

    assert np.all(ladder["L1"] <= ladder["L2"])
    assert np.all(ladder["L2"] <= ladder["L3"])


def test_the_ladder_omits_l3_without_a_reference_pass():
    """Offshore scenes use single-pass SBMP: L3 is undefined, not merely unknown."""
    ladder = sn.eta_ladder(5.0, 15.0, satellite="S2A")
    assert set(ladder) == {"L1", "L2"}


def test_the_reference_pass_uses_its_own_instrument():
    """36% of pairs are cross-satellite, and Landsat is ~2x quieter here."""
    quiet_reference = sn.eta_ladder(5.0, 15.0, 5.0, 15.0, satellite="S2A", satellite_bg="LC09")
    noisy_reference = sn.eta_ladder(5.0, 15.0, 5.0, 15.0, satellite="S2A", satellite_bg="S2A")
    assert quiet_reference["L3"] < noisy_reference["L3"]


# ─────────────────────────────────────────────────────────────────────────────
# Reflectance -> radiance
# ─────────────────────────────────────────────────────────────────────────────
def test_radiance_from_reflectance_lands_in_the_expected_range():
    """The 10^3 unit trap: applying one of the two conversions and not the other.

    Bound the ratio to the band's own reference radiance rather than an absolute
    window -- a bright desert legitimately reaches ~8x L_ref at 1.6 um. A unit
    error is a factor of 1000, so this catches it with room to spare.
    """
    # An Algerian desert scene from MARS-S2L: bright ground, mid-morning Sun.
    for band, reflectance in [(sn.BAND_16, 0.477), (sn.BAND_23, 0.389)]:
        radiance = sn.radiance_from_reflectance(
            reflectance, "S2A", band, sza=23.15, date_of_acquisition="2024-08-22T10:00:21+00:00"
        )
        radiance_ref, _ = sn.snr_reference("S2A", band)
        assert 0.1 <= radiance / radiance_ref <= 15.0


def test_radiance_scales_with_reflectance_and_the_cosine():
    """Linear in reflectance, and lower when the Sun is lower."""
    kwargs = dict(satellite="S2A", band=sn.BAND_23, date_of_acquisition="2024-08-22T10:00:21+00:00")
    low_sun = sn.radiance_from_reflectance(0.3, sza=70.0, **kwargs)
    high_sun = sn.radiance_from_reflectance(0.3, sza=20.0, **kwargs)
    doubled = sn.radiance_from_reflectance(0.6, sza=20.0, **kwargs)

    assert low_sun < high_sun
    assert doubled == pytest.approx(2 * high_sun)


# ─────────────────────────────────────────────────────────────────────────────
# epsilon
# ─────────────────────────────────────────────────────────────────────────────
def test_epsilon_is_monotone_increasing_in_eta():
    eta = np.array([0.001, 0.002, 0.005, 0.01, 0.02])
    values = sn.epsilon(eta, "S2A", SZA, VZA, lut=LUT)
    assert np.all(np.diff(values) > 0)


def test_epsilon_goes_to_zero_with_the_noise():
    assert sn.epsilon(1e-9, "S2A", SZA, VZA, lut=LUT) == pytest.approx(0.0, abs=1.0)


def test_epsilon_grows_with_the_confidence_level():
    eta = 0.005
    assert sn.epsilon(eta, "S2A", SZA, VZA, p=0.99, lut=LUT) > sn.epsilon(
        eta, "S2A", SZA, VZA, p=0.95, lut=LUT
    )


def test_epsilon_keeps_the_shape_of_its_input():
    assert np.shape(sn.epsilon(0.005, "S2A", SZA, VZA, lut=LUT)) == ()
    assert np.shape(sn.epsilon(np.zeros(3) + 0.005, "S2A", SZA, VZA, lut=LUT)) == (3,)


def test_landsat_floor_is_lower_than_sentinel2_at_the_same_radiance():
    """The headline instrument comparison, in ppb rather than in eta."""
    radiances = (9.5, 33.0, 9.5, 33.0)
    eta_s2 = sn.eta_ladder(*radiances, satellite="S2A")["L3"]
    eta_landsat = sn.eta_ladder(*radiances, satellite="LC09")["L3"]

    assert sn.epsilon(eta_landsat, "LC09", SZA, VZA, lut=LUT) < sn.epsilon(
        eta_s2, "S2A", SZA, VZA, lut=LUT
    )


# ─────────────────────────────────────────────────────────────────────────────
# sigma(delta XCH4)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_lut_inverse_is_decreasing_so_the_fitted_slope_is_negative():
    """Less transmittance means more methane; this is why sigma needs the abs."""
    a, b, _ = sn.lut_inverse_quadratic("S2A", SZA, VZA, lut=LUT)
    assert 2 * a * 1.0 + b < 0


def test_sigma_delta_xch4_is_positive_and_grows_with_eta():
    values = sn.sigma_delta_xch4(1.0, np.array([0.002, 0.005, 0.01]), "S2A", SZA, VZA, lut=LUT)
    assert np.all(values > 0)
    assert np.all(np.diff(values) > 0)


def test_the_quadratic_matches_the_lut_slope_near_one():
    """The fit is local for a reason: check it against a numerical derivative."""
    a, b, _ = sn.lut_inverse_quadratic("S2A", SZA, VZA, lut=LUT)
    fitted = 2 * a * 1.0 + b

    step = 1e-3
    numerical = np.diff(
        LUT.deltach4_from_ratio_transmittance(
            "S2A", sza=SZA, vza=VZA, ratio_il=np.array([1 - step, 1 + step])
        )
    ) / (2 * step)

    assert fitted == pytest.approx(float(numerical[0]), rel=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo -- validating the model rather than the arithmetic
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("satellite", ["S2A", "LC09"])
@pytest.mark.parametrize("radiance_23", [1.0, 9.5])
def test_monte_carlo_agrees_with_sigma_mbmp(satellite, radiance_23):
    """sigma(MBMP) = MBMP * eta, against the full double ratio with no expansion."""
    radiance_16 = 3.5 * radiance_23
    radiances = (radiance_23, radiance_16, radiance_23, radiance_16)
    eta = float(sn.eta_ladder(*radiances, satellite=satellite)["L3"])

    samples = sn.monte_carlo_mbmp(
        *radiances, satellite=satellite, n_samples=100_000, rng=np.random.default_rng(0)
    )

    assert samples.mean() == pytest.approx(1.0, abs=5e-4)
    assert samples.std() == pytest.approx(eta, rel=0.02)


@pytest.mark.parametrize("satellite", ["S2A", "LC09"])
@pytest.mark.parametrize("radiance_23", [1.0, 9.5])
def test_monte_carlo_agrees_with_sigma_delta_xch4(satellite, radiance_23):
    """The curvature-sensitive one: propagation through the LUT inverse."""
    radiance_16 = 3.5 * radiance_23
    radiances = (radiance_23, radiance_16, radiance_23, radiance_16)
    eta = float(sn.eta_ladder(*radiances, satellite=satellite)["L3"])

    closed_form = float(sn.sigma_delta_xch4(1.0, eta, satellite, SZA, VZA, lut=LUT))
    samples = sn.monte_carlo_delta_xch4(
        *radiances,
        satellite=satellite,
        sza=SZA,
        vza=VZA,
        n_samples=100_000,
        rng=np.random.default_rng(0),
        lut=LUT,
    )

    assert samples.std() == pytest.approx(closed_form, rel=0.03)


@pytest.mark.parametrize("p", [0.90, 0.95, 0.99])
def test_epsilon_has_the_false_alarm_rate_it_claims(p):
    """The meaning of epsilon: on plume-free ground it is exceeded 1-p of the time."""
    radiances = (9.5, 33.25, 9.5, 33.25)
    eta = float(sn.eta_ladder(*radiances, satellite="S2A")["L3"])
    threshold = float(sn.epsilon(eta, "S2A", SZA, VZA, p=p, lut=LUT))

    samples = sn.monte_carlo_delta_xch4(
        *radiances,
        satellite="S2A",
        sza=SZA,
        vza=VZA,
        n_samples=200_000,
        rng=np.random.default_rng(0),
        lut=LUT,
    )

    assert (samples > threshold).mean() == pytest.approx(1 - p, rel=0.15)
