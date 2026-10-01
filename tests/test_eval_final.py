"""Integration test for the eval_final evaluation routine.

Imports ``marss2l.eval_final`` and calls its ``run_eval`` entry point directly with a simple
Linear model on the balanced CSV fixture (test split, on CPU). Model weight loading is mocked to
a no-op, so the test does not need a real checkpoint: it only verifies that ``run_eval`` reads the
config + (dummy) weights, runs inference, and writes the expected predictions CSV.

It runs with ``smoke_test=False`` (required for the preds CSV to be written) on the small 30/30
test split of the fixture, so it stays light. Per-image rasters are read from the public MARS-S2L
Hugging Face dataset; only the heavy images CSV is replaced by the fixture.

The legacy ``film`` evaluation branch is intentionally not exercised.
"""

import json

import loguru
import pandas as pd
import pytest

from marss2l import eval_final, train_final

from tests.conftest import HF_PATH_PREPEND


@pytest.mark.integration
def test_eval_final_linear_smoke(tmp_path, monkeypatch, images_fixture_path, hf_raster_fs):
    """run_eval with the Linear model on the fixture test split, with weights loading mocked.

    The local fixture CSV is read locally (via the ``hf_raster_fs`` monkeypatches) while the rasters
    are loaded from Hugging Face through the returned ``HfFileSystem``.
    """
    # Mock weight loading: eval_final imports load_weights into its own namespace.
    monkeypatch.setattr(eval_final, "load_weights", lambda *args, **kwargs: None)

    output_dir = tmp_path / "model"
    output_dir.mkdir()

    # A weights file only needs to exist (run_eval checks fsout.exists); load_weights is mocked.
    (output_dir / "best_epoch").write_bytes(b"dummy")

    # Minimal config matching what train_final would write for a Linear model. The keys read by
    # run_eval are sourced from train_final's DEFAULT_* constants to stay in sync.
    config = {
        "model": "Linear",
        "multipass": train_final.DEFAULT_MULTIPASS,
        "do_simulation": train_final.DEFAULT_DO_SIMULATION,
        "wind": train_final.DEFAULT_WIND,
        "cloud_mask": train_final.DEFAULT_CLOUD_MASK,
        "classification_head": train_final.DEFAULT_CLASSIFICATION_HEAD,
        "norm_wind": train_final.DEFAULT_NORM_WIND,
        "cat_mbmp": train_final.DEFAULT_CAT_MBMP,
        "bands_l8": train_final.DEFAULT_BANDS_L8,
        "batch_norm": train_final.DEFAULT_BATCH_NORM,
    }
    with open(output_dir / "config_experiment.json", "w") as f:
        json.dump(config, f)

    eval_final.run_eval(
        output_dir=str(output_dir),
        split="test_2023",
        csv_path=images_fixture_path,
        device_name="cpu",
        num_workers=2,
        batch_size=4,
        smoke_test=False,
        fs=hf_raster_fs,  # rasters from HF; CSV read locally via the monkeypatched reader
        path_prepend_data=HF_PATH_PREPEND,
        logger=loguru.logger,  # avoid creating a ./log directory via the file logger
    )

    # The predictions CSV must have been written for the test split.
    preds_file = output_dir / "preds_test_2023.csv"
    assert preds_file.exists(), f"preds_test_2023.csv not found in {output_dir}"

    preds = pd.read_csv(preds_file)
    assert len(preds) > 0, "predictions CSV is empty"
    for col in ("id_loc_image", "scene_pred"):
        assert col in preds.columns, f"expected column {col!r} missing from predictions CSV"
