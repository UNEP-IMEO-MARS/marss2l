"""
Tests for the shot-noise additions to marss2l.stats_dataset.

compute_stats and its helpers are pure functions of tensors -- no IO, no model --
which is what makes them cheap to pin down here rather than in an integration run.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from marss2l import stats_dataset

BANDS = ["MBMP", "B11", "B12", "B11_bg", "B12_bg", "cloudmask"]
SZA, VZA = 38.5, 6.1
DATE = "2024-08-22T10:00:21+00:00"


def make_stack(reflectance: float = 0.3, size: int = 8) -> torch.Tensor:
    """A uniform, cloud-free, all-valid image stack in the loader's units."""
    x = torch.full((len(BANDS), size, size), reflectance * 2.0)
    x[BANDS.index("MBMP")] = 1.0
    x[BANDS.index("cloudmask")] = 0.0
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Masking
# ─────────────────────────────────────────────────────────────────────────────
def test_valid_mask_keeps_a_clean_scene():
    assert stats_dataset.valid_mask(BANDS, make_stack()).all()


def test_valid_mask_drops_zero_pixels_in_any_band():
    """Zero is how the loader encodes invalid data, and it gives an infinite eta."""
    x = make_stack()
    x[BANDS.index("B12_bg"), 0, 0] = 0.0

    mask = stats_dataset.valid_mask(BANDS, x)

    assert not mask[0, 0]
    assert mask.sum() == mask.numel() - 1


def test_valid_mask_drops_cloudy_pixels():
    x = make_stack()
    x[BANDS.index("cloudmask"), :, 0] = 1.0

    assert stats_dataset.valid_mask(BANDS, x)[:, 0].sum() == 0


def test_valid_mask_keeps_dark_ground():
    """Dark surfaces stay in: the floor is optimistic there, which keeps it a floor."""
    assert stats_dataset.valid_mask(BANDS, make_stack(reflectance=0.002)).all()


# ─────────────────────────────────────────────────────────────────────────────
# meanabs
# ─────────────────────────────────────────────────────────────────────────────
def test_meanabs_survives_cancellation_where_the_mean_does_not():
    """The reason meanabs exists: on ground centred at zero the mean says nothing."""
    values = torch.tensor([-100.0, 100.0, -50.0, 50.0])

    summary = stats_dataset._summary(values, "q")

    assert summary["q_mean"] == pytest.approx(0.0)
    assert summary["q_meanabs"] == pytest.approx(75.0)


def test_summary_of_nothing_is_all_nan():
    summary = stats_dataset._summary(torch.tensor([]), "q")
    assert set(summary) == {f"q_{s}" for s in ("mean", "meanabs", "std", "min", "max")}
    assert all(np.isnan(v) for v in summary.values())


# ─────────────────────────────────────────────────────────────────────────────
# The floors, per scene
# ─────────────────────────────────────────────────────────────────────────────
def _shot_noise_stats(satellite="S2A", **kwargs):
    x = make_stack()
    return stats_dataset.compute_shot_noise_stats(
        BANDS,
        x=x,
        mask=stats_dataset.valid_mask(BANDS, x),
        satellite=satellite,
        sza=SZA,
        vza=VZA,
        tile_date=DATE,
        **kwargs,
    )


def test_scene_floors_are_ordered_and_in_a_plausible_range():
    stats = _shot_noise_stats(satellite_bg="S2A", sza_bg=25.0, tile_date_bg=DATE)

    assert stats["eta_L1_mean"] <= stats["eta_L2_mean"] <= stats["eta_L3_mean"]
    assert stats["epsilon_L1_mean"] <= stats["epsilon_L2_mean"] <= stats["epsilon_L3_mean"]
    # A few hundred ppb on ordinary ground -- the range the draft quotes.
    assert 10 < stats["epsilon_L3_mean"] < 5_000
    assert stats["sigma_ch4_L3_mean"] > 0


def test_l3_is_absent_without_a_reference_pass():
    """Offshore: single-pass SBMP, so the primed terms do not exist."""
    stats = _shot_noise_stats()

    assert "eta_L2_mean" in stats
    assert not any(key.startswith(("eta_L3", "epsilon_L3", "sigma_ch4_L3")) for key in stats)


def test_radiance_is_reported_for_the_current_image_only():
    stats = _shot_noise_stats(satellite_bg="S2A", sza_bg=25.0, tile_date_bg=DATE)

    assert "radiance_B12_mean" in stats and "radiance_B11_mean" in stats
    assert not any("_bg" in key for key in stats)


def test_a_quieter_reference_instrument_lowers_l3():
    """satellite_bg has to drive the primed terms; Landsat is ~2x quieter."""
    with_landsat = _shot_noise_stats(satellite_bg="LC09", sza_bg=SZA, tile_date_bg=DATE)
    with_s2 = _shot_noise_stats(satellite_bg="S2A", sza_bg=SZA, tile_date_bg=DATE)

    assert with_landsat["eta_L3_mean"] < with_s2["eta_L3_mean"]


def test_brighter_ground_gives_a_lower_floor():
    x_bright, x_dark = make_stack(0.5), make_stack(0.05)
    common = dict(satellite="S2A", sza=SZA, vza=VZA, tile_date=DATE)

    bright = stats_dataset.compute_shot_noise_stats(
        BANDS, x=x_bright, mask=stats_dataset.valid_mask(BANDS, x_bright), **common
    )
    dark = stats_dataset.compute_shot_noise_stats(
        BANDS, x=x_dark, mask=stats_dataset.valid_mask(BANDS, x_dark), **common
    )

    assert bright["epsilon_L1_mean"] < dark["epsilon_L1_mean"]


# ─────────────────────────────────────────────────────────────────────────────
# Measured noise, and image selection
# ─────────────────────────────────────────────────────────────────────────────
def test_measured_noise_is_reported_over_valid_pixels_only():
    x = make_stack()
    x[BANDS.index("B12"), 0, 0] = 0.0  # invalid pixel
    ch4 = torch.zeros(8, 8)
    ch4[0, 0] = 10_000.0  # a wild value that must not reach the statistics

    stats = stats_dataset.measured_noise_stats(
        BANDS, x=x, ch4=ch4, target=torch.zeros(8, 8), isplume=0
    )

    assert stats["npixelsvalid"] == 63
    assert stats["ch4_valid_meanabs"] == pytest.approx(0.0)


def test_log_mbmp_is_not_halved():
    """The loader's reflectance x 2 does not apply to a dimensionless ratio."""
    stats = stats_dataset.measured_noise_stats(
        BANDS, x=make_stack(), ch4=torch.zeros(8, 8), target=torch.zeros(8, 8), isplume=0
    )
    # MBMP == 1 everywhere, so log MBMP == 0. Halving it would give log(0.5).
    assert stats["log_mbmp_valid_mean"] == pytest.approx(0.0)


@pytest.fixture
def stub_csv(monkeypatch):
    """Stand in for the CSV read, so select_images can be exercised without one."""
    dataframe = pd.DataFrame(
        {
            "isplume": [True] * 50 + [False] * 50,
            "s2path": [f"img_{i}.tif" for i in range(100)],
        }
    )
    monkeypatch.setattr(
        stats_dataset.loaders, "read_csv_images", lambda *args, **kwargs: dataframe.copy()
    )
    return dataframe


def test_smoke_test_sample_is_balanced(stub_csv):
    """20 images, 10 with plumes and 10 without -- not head(), and not 100 + 100."""
    sample = stats_dataset.select_images("unused.csv", fs=None, smoke_test=True)

    assert len(sample) == 20
    assert int(sample.isplume.sum()) == 10


def test_smoke_test_sample_is_reproducible(stub_csv):
    first = stats_dataset.select_images("unused.csv", fs=None, smoke_test=True)
    second = stats_dataset.select_images("unused.csv", fs=None, smoke_test=True)

    assert first.s2path.tolist() == second.s2path.tolist()


def test_without_smoke_test_every_image_is_swept(stub_csv):
    assert len(stats_dataset.select_images("unused.csv", fs=None)) == 100
