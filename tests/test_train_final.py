"""Integration test for the train_final training script.

Runs a couple of training steps of ``marss2l.train_final`` with a simple
linear model on the real MARS-S2L Hugging Face dataset (reduced batch size,
on CPU). It calls the ``train_final`` script directly via ``python -m`` so the
whole CLI entry point is exercised end-to-end.

Gated on the Hugging Face credentials (``HF_TOKEN``) being set — mirroring the
notebook tests — so it only runs where the dataset is reachable (e.g. CI with
secrets configured).
"""

import os
import subprocess
import sys

import pytest

from marss2l.config import HFConfig


@pytest.mark.integration
@pytest.mark.skipif(
    not HFConfig.is_available(),
    reason=(
        "missing required environment variables: "
        f"{', '.join(HFConfig.required_env_vars)}"
    ),
)
def test_train_final_linear_smoke(tmp_path):
    """A couple of steps of train_final with the Linear model on the HF dataset."""
    env = dict(os.environ)
    # Keep torch.compile a no-op so the test stays fast and portable.
    env["TORCHDYNAMO_DISABLE"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "marss2l.train_final",
        "--model-name",
        "Linear",
        "--smoke-test",
        "--no-do-simulation",
        "--device-name",
        "cpu",
        "--no-data-parallel",
        "--cache-all",
        "--batch-size",
        "4",
        "--batch-size-val",
        "4",
        "--num-workers",
        "2",
        "--num-workers-val",
        "2",
        "--n-samples-per-epoch-train",
        "8",
        "--output-dir",
        str(tmp_path / "train_output"),
    ]

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert result.returncode == 0, (
        "train_final exited with a non-zero status\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
