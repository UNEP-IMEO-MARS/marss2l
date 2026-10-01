"""
Tests for marss2l.shot_noise.

The Monte-Carlo checks validate the *model* -- that the first-order expansion is adequate.
They do not catch a wrong constant or a flipped ratio, because a plausible wrong number
still produces a plausible figure. The cheap by-hand tests above them are what catch those.
"""

import numpy as np
import pytest

from marss2l import shot_noise as sn

SZA, VZA = 38.5, 6.1  # the medians of the MARS-S2L target images


# ─────────────────────────────────────────────────────────────────────────────
# SNR rescaling
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("satellite", list(sn.SNR_REFERENCE))
@pytest.mark.parametrize("band", [sn.BAND_16, sn.BAND_23])
def test_snr_at_the_reference_radiance_is_the_reference_snr(satellite, band):
    radiance_ref, snr_ref = sn.SNR_REFERENCE[satellite][band]
    assert sn.snr_at_radiance(radiance_ref, satellite, band) == pytest.approx(snr_ref)


@pytest.mark.parametrize("satellite", ["S2A", "LC09"])
def test_snr_doubles_at_four_times_the_reference_radiance(satellite):
    """SNR goes as sqrt(L), so 4x the radiance is exactly 2x the SNR."""
    radiance_ref, snr_ref = sn.SNR_REFERENCE[satellite][sn.BAND_23]
    assert sn.snr_at_radiance(4 * radiance_ref, satellite, sn.BAND_23) == pytest.approx(2 * snr_ref)


def test_landsat_snr_is_about_twice_sentinel2():
    """The reason Landsat's floors come out lower despite the coarser pixel."""
    for band in [sn.BAND_16, sn.BAND_23]:
        s2 = sn.snr_at_radiance(5.0, "S2A", band)
        landsat = sn.snr_at_radiance(5.0, "LC09", band)
        assert 1.7 < landsat / s2 < 2.2


def test_s2c_is_quieter_than_s2b_at_the_same_reference_radiance():
    """The 2025 ESA report (Table 21) gives S2C its own figures, above S2B's in both bands."""
    for band in (sn.BAND_16, sn.BAND_23):
        radiance_s2c, snr_s2c = sn.SNR_REFERENCE["S2C"][band]
        radiance_s2b, snr_s2b = sn.SNR_REFERENCE["S2B"][band]
        assert radiance_s2c == radiance_s2b
        assert snr_s2c > snr_s2b


# ─────────────────────────────────────────────────────────────────────────────
# eta and the ladder
# ─────────────────────────────────────────────────────────────────────────────
def test_l1_at_the_reference_radiance_is_one_over_the_reference_snr():
    radiance_ref, snr_ref = sn.SNR_REFERENCE["S2A"][sn.BAND_23]
    ladder = sn.eta_ladder(radiance_ref, 4.0, satellite="S2A")
    assert ladder["L1"] == pytest.approx(1.0 / snr_ref)


def test_an_identical_reference_pass_costs_exactly_sqrt2():
    """Two equal pairs of terms: the multi-pass construction doubles the variance."""
    ladder = sn.eta_ladder(5.0, 16.0, 5.0, 16.0, satellite="S2A")
    assert ladder["L3"] / ladder["L2"] == pytest.approx(np.sqrt(2.0))


def test_the_ladder_is_ordered_on_random_inputs():
    """L1 <= L2 <= L3 pixel by pixel, by construction: it holds for any radiances at all."""
    rng = np.random.default_rng(0)
    radiances = rng.uniform(0.2, 40.0, size=(4, 500))
    ladder = sn.eta_ladder(*radiances, satellite="S2A", satellite_bg="LC08")

    assert np.all(ladder["L1"] <= ladder["L2"])
    assert np.all(ladder["L2"] <= ladder["L3"])


def test_the_ladder_omits_l3_without_a_reference_pass():
    """Offshore scenes use a single-pass retrieval: L3 is undefined, not merely unknown."""
    ladder = sn.eta_ladder(5.0, 15.0, satellite="S2A")
    assert set(ladder) == {"L1", "L2"}


def test_the_reference_pass_uses_its_own_instrument():
    """A third of the pairs are cross-platform, and Landsat is ~2x quieter here."""
    quiet_reference = sn.eta_ladder(5.0, 15.0, 5.0, 15.0, satellite="S2A", satellite_bg="LC09")
    noisy_reference = sn.eta_ladder(5.0, 15.0, 5.0, 15.0, satellite="S2A", satellite_bg="S2A")
    assert quiet_reference["L3"] < noisy_reference["L3"]


# ─────────────────────────────────────────────────────────────────────────────
# Reflectance -> radiance
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("band", [sn.BAND_16, sn.BAND_23])
def test_landsat_and_sentinel2_irradiances_agree(band):
    """Landsat is looked up by its own band names (B06/B07) under the Sentinel-2 ones."""
    s2 = sn.band_irradiance("S2A", band)
    landsat = sn.band_irradiance("LC08", band)
    assert landsat == pytest.approx(s2, rel=0.1)
    assert sn.band_irradiance("S2A", sn.BAND_16) > sn.band_irradiance("S2A", sn.BAND_23)


def test_radiance_from_reflectance_lands_in_the_expected_range():
    """The 10^3 unit trap: applying one of the two conversions and not the other.

    Bound the ratio to the band's own reference radiance rather than an absolute window --
    a bright desert legitimately reaches ~8x L_ref at 1.6 um. A unit error is a factor of
    1000, so this catches it with room to spare.
    """
    # An Algerian desert scene from MARS-S2L: bright ground, mid-morning Sun.
    for band, reflectance in [(sn.BAND_16, 0.477), (sn.BAND_23, 0.389)]:
        radiance = sn.radiance_from_reflectance(
            reflectance, "S2A", band, sza=23.15, date_of_acquisition="2024-08-22T10:00:21+00:00"
        )
        radiance_ref, _ = sn.SNR_REFERENCE["S2A"][band]
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
    values = sn.epsilon(eta, "S2A", SZA, VZA)
    assert np.all(np.diff(values) > 0)


def test_epsilon_goes_to_zero_with_the_noise():
    assert sn.epsilon(1e-9, "S2A", SZA, VZA) == pytest.approx(0.0, abs=1.0)


def test_epsilon_grows_with_the_confidence_level():
    eta = 0.005
    assert sn.epsilon(eta, "S2A", SZA, VZA, p=0.99) > sn.epsilon(eta, "S2A", SZA, VZA, p=0.95)


def test_epsilon_keeps_the_shape_of_its_input():
    assert np.shape(sn.epsilon(0.005, "S2A", SZA, VZA)) == ()
    assert np.shape(sn.epsilon(np.zeros(3) + 0.005, "S2A", SZA, VZA)) == (3,)


def test_landsat_floor_is_lower_than_sentinel2_at_the_same_radiance():
    """The headline instrument comparison, in ppb rather than in eta."""
    radiances = (9.5, 33.0, 9.5, 33.0)
    eta_s2 = sn.eta_ladder(*radiances, satellite="S2A")["L3"]
    eta_landsat = sn.eta_ladder(*radiances, satellite="LC09")["L3"]

    assert sn.epsilon(eta_landsat, "LC09", SZA, VZA) < sn.epsilon(eta_s2, "S2A", SZA, VZA)


# ─────────────────────────────────────────────────────────────────────────────
# sigma(delta XCH4)
# ─────────────────────────────────────────────────────────────────────────────
def test_the_lut_inverse_is_decreasing_so_the_fitted_slope_is_negative():
    """Less transmittance means more methane; this is why sigma needs the abs."""
    assert sn._lut_slope_at_one("S2A", SZA, VZA) < 0


def test_sigma_delta_xch4_is_positive_and_grows_with_eta():
    values = sn.sigma_delta_xch4(np.array([0.002, 0.005, 0.01]), "S2A", SZA, VZA)
    assert np.all(values > 0)
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("satellite", ["S2A", "LC08"])
def test_the_quadratic_matches_the_lut_slope_at_one(satellite):
    """The fit is local for a reason: check it against a numerical derivative."""
    step = 1e-3
    numerical = np.diff(
        sn.default_lut().deltach4_from_ratio_transmittance(
            satellite, sza=SZA, vza=VZA, ratio_il=np.array([1 - step, 1 + step])
        )
    ) / (2 * step)

    assert sn._lut_slope_at_one(satellite, SZA, VZA) == pytest.approx(float(numerical[0]), rel=0.01)


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
    """The curvature-sensitive one: propagation through the look-up-table inverse."""
    radiance_16 = 3.5 * radiance_23
    radiances = (radiance_23, radiance_16, radiance_23, radiance_16)
    eta = float(sn.eta_ladder(*radiances, satellite=satellite)["L3"])

    closed_form = float(sn.sigma_delta_xch4(eta, satellite, SZA, VZA))
    samples = sn.monte_carlo_delta_xch4(
        *radiances,
        satellite=satellite,
        sza=SZA,
        vza=VZA,
        n_samples=100_000,
        rng=np.random.default_rng(0),
    )

    assert samples.std() == pytest.approx(closed_form, rel=0.03)


@pytest.mark.parametrize("p", [0.90, 0.95, 0.99])
def test_epsilon_has_the_false_alarm_rate_it_claims(p):
    """The meaning of epsilon: on plume-free ground it is exceeded 1-p of the time."""
    radiances = (9.5, 33.25, 9.5, 33.25)
    eta = float(sn.eta_ladder(*radiances, satellite="S2A")["L3"])
    threshold = float(sn.epsilon(eta, "S2A", SZA, VZA, p=p))

    samples = sn.monte_carlo_delta_xch4(
        *radiances,
        satellite="S2A",
        sza=SZA,
        vza=VZA,
        n_samples=200_000,
        rng=np.random.default_rng(0),
    )

    assert (samples > threshold).mean() == pytest.approx(1 - p, rel=0.15)


# ─────────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────────
def test_the_sentinel2_noise_factor_is_the_bilinear_step():
    """Photon noise on the 10 m grid over the native 20 m one, from the step itself."""
    from marss2l.resampling import axis_operator

    operator = axis_operator(204)
    interior = np.abs(operator.sum(axis=1) - 1) < 1e-9  # rows clear of the edge padding
    per_axis = (operator[interior] ** 2).sum(axis=1).mean()
    # Separable: variance shrinks by per_axis**2, so the standard deviation by per_axis.
    assert per_axis == pytest.approx(sn.NOISE_FACTOR_10M["S2"], abs=1e-3)


def test_the_landsat_noise_factor_is_the_cubic_spline_resize():
    """White noise on a 30 m grid through the resize the Landsat chips went through."""
    import rasterio
    from georeader import read
    from georeader.geotensor import GeoTensor
    from rasterio.enums import Resampling

    noise = np.random.default_rng(0).standard_normal((1, 200, 200))
    native = GeoTensor(
        noise, transform=rasterio.Affine(30, 0, 500000, 0, -30, 4000000), crs="EPSG:32633"
    )
    resized = read.resize(native, resolution_dst=(10, 10), resampling=Resampling.cubic_spline)
    interior = resized.values[0, 30:-30, 30:-30]
    assert interior.std() / noise.std() == pytest.approx(sn.NOISE_FACTOR_10M["LC"], abs=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# From noise to a detectable flux
# ─────────────────────────────────────────────────────────────────────────────
def test_q50_uses_each_platforms_observability_and_pixel():
    q = sn.q50_from_noise([300, 300, 300], ["S2A", "S2B", "LC09"])
    assert q[1] / q[0] == pytest.approx(0.0638 / 0.059)
    assert q[2] / q[0] == pytest.approx(30 / 20)


def test_q50_is_linear_in_wind_and_noise():
    base = sn.q50_from_noise(250, "LC08")
    assert sn.q50_from_noise(250, "LC08", wind_speed=3.5) == pytest.approx(3.5 * base)
    assert sn.q50_from_noise(500, "LC08") == pytest.approx(2 * base)
    assert np.shape(base) == ()
