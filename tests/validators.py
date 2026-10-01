"""
Shared assertion helpers for marss2l tests.
"""

import numpy as np


def assert_valid_flux_result(result, *, expect_std=False):
    """Assert that a flux rate result dict has all required keys with valid values."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    for key in ("Q", "L", "IME", "u_eff", "npix_plume", "pixel_size"):
        assert key in result, f"Missing key '{key}' in flux result"

    assert result["Q"] > 0, f"Q should be > 0, got {result['Q']}"
    assert np.isfinite(result["Q"]), f"Q should be finite, got {result['Q']}"
    assert result["L"] > 0, f"L should be > 0, got {result['L']}"
    assert result["IME"] > 0, f"IME should be > 0, got {result['IME']}"
    assert result["u_eff"] > 0, f"u_eff should be > 0, got {result['u_eff']}"
    assert result["npix_plume"] > 0, f"npix_plume should be > 0, got {result['npix_plume']}"
    assert result["pixel_size"] > 0, f"pixel_size should be > 0, got {result['pixel_size']}"

    if expect_std:
        assert "sigma_Q" in result, "Missing key 'sigma_Q' in flux result"
        assert "sig_xch4" in result, "Missing key 'sig_xch4' in flux result"
        assert result["sigma_Q"] > 0, f"sigma_Q should be > 0, got {result['sigma_Q']}"
