"""Integration test for the train_final training routine.

Imports ``marss2l.train_final`` and calls its ``run`` entry point directly
(instead of spawning ``python -m``) with a simple Linear model and the realistic
default ``do_simulation=True`` (reduced batch size, on CPU). It runs in
``smoke_test`` mode: a subset of the data, two short epochs plus evaluation,
and then exits.

Image and plume metadata are read from the committed balanced CSV fixtures
(``tests/fixtures/``, see ``scripts/create_fixture.py``) instead of the ~100 MB
HuggingFace CSV; per-image rasters are still fetched from HuggingFace.

The test asserts the run actually trains and persists a model: the weights
checkpoints (``best_epoch`` / ``last_epoch``) and the ``config_experiment.json``
must be written to the output directory. ``smoke_test`` runs wandb in its
disabled (no-op) mode, so the test never contacts wandb; ``WANDB_MODE`` is also
forced to ``disabled`` as a safeguard.

The MARS-S2L Hugging Face dataset is public, so no token is required; this test
runs whenever the ``integration`` marker is selected (i.e. in the label-gated
integration CI job).
"""

import json

import pytest
import torch

from marss2l import train_final

from tests.conftest import HF_PATH_PREPEND


@pytest.mark.integration
def test_train_final_linear_smoke(
    tmp_path, monkeypatch, images_fixture_path, plumes_fixture_path, hf_raster_fs
):
    """A couple of steps of train_final with the Linear model on the CSV fixtures.

    Uses the realistic default ``do_simulation=True`` (which reads the plumes CSV) with the
    lightweight Linear model. Image/plume metadata are read locally from the committed fixtures
    (via the ``hf_raster_fs`` monkeypatches); per-image rasters are read from Hugging Face.
    """
    # Keep torch.compile a no-op so the test stays fast and portable, and make
    # sure wandb never tries to reach the server.
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    output_dir = tmp_path / "train_output"

    result = train_final.run(
        model_name="Linear",
        smoke_test=True,
        do_simulation=True,
        device_name="cpu",
        data_parallel=False,
        cache_all=True,
        batch_size=4,
        batch_size_val=4,
        num_workers=2,
        num_workers_val=2,
        n_samples_per_epoch_train=8,
        csv_path=images_fixture_path,
        csv_plume_path=plumes_fixture_path,
        fsread=hf_raster_fs,  # rasters from HF; CSVs read locally via the monkeypatched readers
        path_prepend_data=HF_PATH_PREPEND,
        output_dir=str(output_dir),
    )

    assert result is True

    # The training config must have been written and be valid JSON.
    config_file = output_dir / "config_experiment.json"
    assert config_file.exists(), f"config_experiment.json not found in {output_dir}"
    with open(config_file) as f:
        config = json.load(f)
    assert config["model"] == "Linear"

    # The model weights must have been saved and be loadable checkpoints.
    best_epoch = output_dir / "best_epoch"
    last_epoch = output_dir / "last_epoch"
    assert best_epoch.exists(), f"best_epoch weights not found in {output_dir}"
    assert last_epoch.exists(), f"last_epoch weights not found in {output_dir}"

    checkpoint = torch.load(last_epoch, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert len(checkpoint["model_state_dict"]) > 0
