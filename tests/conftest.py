"""Pytest configuration for marss2l.

Notebook (and other credential-dependent) tests are gated on the
environment, mirroring the georeader pattern: the credential dataclasses in
:mod:`marss2l.config` *define the dependencies* a test needs, and a test is
skipped unless every config it depends on is configured via environment
variables.

In CI the credentials are provided through GitHub Actions secrets. Locally,
copy ``.env.sample`` to ``.env``, fill it in, and load it before running the
tests. Tests whose dependencies are unmet are skipped (not failed) so the
core unit-test suite always runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type

import pytest

from marss2l.config import AzureConfig, GEEConfig, HFConfig


@dataclass(frozen=True)
class NotebookDependency:
    """Declares the credential configs a notebook needs in order to run.

    Parameters
    ----------
    path:
        Repo-relative path (or path suffix) of the notebook.
    requires:
        Tuple of credential config classes from :mod:`marss2l.config`. The
        notebook is skipped unless every one of them ``is_available()``.
    """

    path: str
    requires: Tuple[Type, ...] = field(default_factory=tuple)

    def missing_env_vars(self) -> List[str]:
        """Return the env vars that are required but not set."""
        missing: List[str] = []
        for cfg in self.requires:
            if not cfg.is_available():
                missing.extend(cfg.required_env_vars)
        return missing


# Maps each notebook exercised by the test-suite to the credentials it needs.
# HF notebooks read the (gated) MARS-S2L dataset; the figures that pull rasters
# from Azure also need the Azure SAS token; download_and_inference needs GEE.
NOTEBOOK_DEPENDENCIES: Tuple[NotebookDependency, ...] = (
    NotebookDependency("notebooks/examples/plot_images_dataset_train.ipynb", (HFConfig,)),
    NotebookDependency("notebooks/examples/plot_plumes_dataset_test.ipynb", (HFConfig,)),
    NotebookDependency("notebooks/examples/run_inference.ipynb", (HFConfig,)),
    NotebookDependency("notebooks/examples/download_and_inference.ipynb", (GEEConfig,)),
    NotebookDependency(
        "notebooks/figures/dataset_stats_by_split_and_geopackage_locations.ipynb",
        (HFConfig, AzureConfig),
    ),
    NotebookDependency("notebooks/figures/figure_number_of_images_per_country.ipynb", (HFConfig,)),
    NotebookDependency("notebooks/figures/mdl_exploration_by_case_study.ipynb", (HFConfig,)),
    NotebookDependency("notebooks/figures/mdl_exploration_adapted.ipynb", (HFConfig,)),
    NotebookDependency("notebooks/figures/figure_wind_speed.ipynb", (HFConfig,)),
    NotebookDependency("notebooks/figures/stats_dataset_toareflectances.ipynb", (HFConfig,)),
    NotebookDependency(
        "notebooks/figures/eval_model_and_figure_prob_vs_emission_rate.ipynb",
        (HFConfig, AzureConfig),
    ),
    NotebookDependency("notebooks/figures/figure_controlled_releases.ipynb", (HFConfig,)),
    NotebookDependency("notebooks/figures/cloudsen12_experiment.ipynb", (HFConfig,)),
    NotebookDependency(
        "notebooks/figures/ablation_threshold_pixels.ipynb",
        (HFConfig, AzureConfig),
    ),
)


def _dependency_for(item_path: Path) -> NotebookDependency | None:
    """Match a collected item path to its declared dependency, if any."""
    posix = item_path.as_posix()
    for dep in NOTEBOOK_DEPENDENCIES:
        if posix.endswith(dep.path):
            return dep
    return None


def pytest_collection_modifyitems(config, items):
    """Skip notebooks whose credential dependencies are not configured."""
    for item in items:
        item_path = Path(str(getattr(item, "fspath", item.nodeid)))
        if item_path.suffix != ".ipynb":
            continue
        dep = _dependency_for(item_path)
        if dep is None:
            continue
        missing = dep.missing_env_vars()
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"missing required environment variables: {', '.join(missing)}"
                    )
                )
            )
