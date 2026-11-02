"""
Tests for marss2l.mars_sentinel2.quantification.obtain_flux_rate_s2l89.
"""

import numpy as np
import pytest

from marss2l.mars_sentinel2 import quantification

S2L89_DEFAULTS = {
    "units_methane_enhancement": "ppb",
    "a_u_eff": quantification.A_UEFF_S2,
    "b_u_eff": quantification.B_UEFF_S2,
    "sig_xch4": quantification.SIGMA_CH4_S2_PPB,
}


@pytest.fixture
def recorded_call(monkeypatch):
    """Replace ``obtain_flux_rate`` with a stub that records its arguments."""
    calls = []

    def _fake(methane_enhancement_image, plume_mask_binary, wind_speed, **kwargs):
        calls.append(kwargs)
        return {"Q": 0.0}

    monkeypatch.setattr(quantification, "obtain_flux_rate", _fake)
    return calls


def test_wrapper_passes_s2l89_defaults(recorded_call):
    quantification.obtain_flux_rate_s2l89(
        np.zeros((3, 3)), np.ones((3, 3)), wind_speed=3.0, resolution=(10, 10)
    )

    assert recorded_call == [{**S2L89_DEFAULTS, "resolution": (10, 10)}]


@pytest.mark.parametrize(
    "name, value",
    [("units_methane_enhancement", "ppm"), ("a_u_eff", 1.0), ("b_u_eff", 0.0), ("sig_xch4", 50.0)],
)
def test_explicit_argument_overrides_default(recorded_call, name, value):
    quantification.obtain_flux_rate_s2l89(
        np.zeros((3, 3)), np.ones((3, 3)), wind_speed=3.0, **{name: value}
    )

    assert recorded_call[0][name] == value
    assert {k: v for k, v in recorded_call[0].items() if k != name} == {
        k: v for k, v in S2L89_DEFAULTS.items() if k != name
    }


def test_wrapper_matches_explicit_ppb_call():
    ch4 = np.zeros((20, 20))
    ch4[5:15, 5:15] = 1_000.0  # ppb
    mask = (ch4 > 0).astype(np.uint8)

    wrapped = quantification.obtain_flux_rate_s2l89(
        ch4, mask, wind_speed=3.0, resolution=(10, 10), return_std=True, seed=0
    )
    explicit = quantification.obtain_flux_rate(
        ch4,
        mask,
        wind_speed=3.0,
        resolution=(10, 10),
        return_std=True,
        seed=0,
        units_methane_enhancement="ppb",
        a_u_eff=quantification.A_UEFF_S2,
        b_u_eff=quantification.B_UEFF_S2,
        sig_xch4=quantification.SIGMA_CH4_S2_PPB,
    )

    assert wrapped["Q"] > 0
    assert wrapped["Q"] == pytest.approx(explicit["Q"])
    assert wrapped["sigma_Q"] == pytest.approx(explicit["sigma_Q"])
