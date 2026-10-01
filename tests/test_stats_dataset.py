"""
Tests for the shot-noise statistics of marss2l.stats_dataset.

compute_stats and its helpers are pure functions of tensors -- no IO, no model -- which is
what makes them cheap to pin down here rather than in an integration run.
"""

import fsspec
import fsspec.asyn
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
    """Zero is how the loader encodes invalid data."""
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
    """Dark surfaces stay in: a brightness cut would make the floors depend on its threshold."""
    assert stats_dataset.valid_mask(BANDS, make_stack(reflectance=0.002)).all()


# ─────────────────────────────────────────────────────────────────────────────
# The floors, per scene
# ─────────────────────────────────────────────────────────────────────────────
NO_REFERENCE = dict(satellite_bg="", sza_bg=float("nan"), tile_date_bg="")


def _shot_noise_stats(x=None, satellite="S2A", **reference):
    x = make_stack() if x is None else x
    return stats_dataset.compute_shot_noise_stats(
        BANDS,
        x=x,
        mask=stats_dataset.valid_mask(BANDS, x),
        satellite=satellite,
        sza=SZA,
        vza=VZA,
        tile_date=DATE,
        **(reference or NO_REFERENCE),
    )


def test_scene_floors_are_ordered_and_in_a_plausible_range():
    stats = _shot_noise_stats(satellite_bg="S2A", sza_bg=25.0, tile_date_bg=DATE)

    assert stats["sigma_ch4_L1_mean"] <= stats["sigma_ch4_L2_mean"] <= stats["sigma_ch4_L3_mean"]
    assert stats["epsilon_L1_mean"] <= stats["epsilon_L2_mean"] <= stats["epsilon_L3_mean"]
    # A few hundred ppb on ordinary ground.
    assert 10 < stats["epsilon_L3_mean"] < 5_000


def test_l3_is_absent_without_a_reference_pass():
    """Offshore: a single-pass retrieval, so the reference-pass terms do not exist."""
    stats = _shot_noise_stats()

    assert "sigma_ch4_L2_mean" in stats
    assert not any("L3" in key for key in stats)


def test_radiance_is_reported_for_the_2300nm_band_of_the_target_only():
    stats = _shot_noise_stats(satellite_bg="S2A", sza_bg=25.0, tile_date_bg=DATE)

    assert {k for k in stats if k.startswith("radiance")} == {
        "radiance_B12_mean",
        "radiance_B12_std",
    }
    assert stats["radiance_B12_std"] == pytest.approx(0.0, abs=1e-9)  # uniform scene


def test_a_scene_without_valid_pixels_reports_nothing():
    x = make_stack()
    x[BANDS.index("cloudmask")] = 1.0
    assert _shot_noise_stats(x) == {}


def test_a_quieter_reference_instrument_lowers_l3():
    """satellite_bg has to drive the reference-pass terms; Landsat is ~2x quieter."""
    with_landsat = _shot_noise_stats(satellite_bg="LC09", sza_bg=SZA, tile_date_bg=DATE)
    with_s2 = _shot_noise_stats(satellite_bg="S2A", sza_bg=SZA, tile_date_bg=DATE)

    assert with_landsat["sigma_ch4_L3_mean"] < with_s2["sigma_ch4_L3_mean"]


def test_brighter_ground_gives_a_lower_floor():
    bright = _shot_noise_stats(make_stack(0.5))
    dark = _shot_noise_stats(make_stack(0.05))

    assert bright["epsilon_L1_mean"] < dark["epsilon_L1_mean"]


# ─────────────────────────────────────────────────────────────────────────────
# Measured noise
# ─────────────────────────────────────────────────────────────────────────────
def test_measured_noise_excludes_invalid_and_plume_pixels():
    x = make_stack()
    x[BANDS.index("B12"), 0, 0] = 0.0  # invalid pixel
    ch4 = torch.zeros(8, 8)
    ch4[0, 0] = 10_000.0  # a wild value on the invalid pixel
    ch4[4, :] = 500.0  # a plume
    target = torch.zeros(8, 8)
    target[4, :] = 1.0
    ch4[1, :4], ch4[1, 4:] = -1.0, 1.0  # the noise the background reads

    stats = stats_dataset.measured_noise_stats(BANDS, x=x, ch4=ch4, target=target)

    assert stats["npixelsvalid"] == 63
    background = ch4[(target == 0) & stats_dataset.valid_mask(BANDS, x)]
    assert stats["ch4_valid_noplume_std"] == pytest.approx(background.std().item())
    assert stats["ch4_valid_noplume_std"] < 1.0


def test_mbmp_is_reported_unhalved_but_reflectance_is_not():
    """The /2 undoes the loader's reflectance x 2, which a ratio never carried."""
    stats = stats_dataset.compute_stats(
        BANDS,
        isplume=0,
        ch4=torch.zeros(8, 8),
        target=torch.zeros(8, 8),
        x=make_stack(reflectance=0.3),
        wind_vector=np.zeros(2),
    )

    assert stats["MBMP_mean"] == pytest.approx(1.0)
    assert stats["B12_mean"] == pytest.approx(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Image selection and output
# ─────────────────────────────────────────────────────────────────────────────
def test_without_a_split_every_image_is_swept(monkeypatch):
    dataframe = pd.DataFrame({"s2path": [f"img_{i}.tif" for i in range(100)]})
    monkeypatch.setattr(
        stats_dataset.loaders, "read_csv_images", lambda *args, **kwargs: dataframe.copy()
    )
    assert len(stats_dataset.select_images("unused.csv", fs=None)) == 100


def test_write_replaces_the_file_whole(tmp_path):
    fs = fsspec.filesystem("file")
    output = str(tmp_path / "stats.csv")

    stats_dataset._write(pd.DataFrame({"a": [1, 2]}), fs, output)
    stats_dataset._write(pd.DataFrame({"a": [1, 2, 3]}), fs, output)

    assert len(pd.read_csv(output)) == 3
    assert not fs.exists(output + ".part")


# ─────────────────────────────────────────────────────────────────────────────
# Forked workers over an object store
# ─────────────────────────────────────────────────────────────────────────────
class _StubDataset:
    def __init__(self, fs):
        self.fs = fs


def _stub_worker(monkeypatch, dataset):
    monkeypatch.setattr(
        stats_dataset.torch.utils.data,
        "get_worker_info",
        lambda: type("Info", (), {"dataset": dataset})(),
    )


def test_worker_init_is_a_noop_in_the_main_process(monkeypatch):
    monkeypatch.setattr(stats_dataset.torch.utils.data, "get_worker_info", lambda: None)
    stats_dataset.reopen_filesystem_in_worker(0)  # must not raise


def test_worker_init_leaves_a_local_filesystem_alone(monkeypatch):
    dataset = _StubDataset(fsspec.filesystem("file"))
    _stub_worker(monkeypatch, dataset)

    stats_dataset.reopen_filesystem_in_worker(0)

    assert dataset.fs is not None


def test_worker_init_drops_an_async_filesystem(monkeypatch):
    """An async filesystem does not survive fork, so the worker must rebuild it.

    Dropping the handle is what makes ``load_image_method`` fall back to
    ``fs_from_path``, which builds one inside the worker.
    """
    dataset = _StubDataset(fsspec.asyn.AsyncFileSystem())
    _stub_worker(monkeypatch, dataset)

    stats_dataset.reopen_filesystem_in_worker(0)

    assert dataset.fs is None
