"""Pytest configuration for the notebook integration tests under ``notebooks/``.

The notebooks double as integration tests: ``make test-notebooks`` (and the CI
``integration`` workflow) execute them with ``pytest --nbmake``. Some of them
need a cloud credential that is not always available, so a notebook is *skipped
automatically* unless everything it needs is configured.

This file lives **next to the notebooks** on purpose. pytest only loads a
``conftest.py`` for the directory subtree it sits in, so the skip hook has to be
under ``notebooks/`` to be applied when collecting ``notebooks/**/*.ipynb`` —
the same pattern georeader and marshsi use (their notebook-gating conftest lives
in ``docs/`` alongside their notebooks). The credential dataclasses in
:mod:`marss2l.config` *define the dependencies* a notebook needs, and a notebook
is skipped unless every config it depends on is configured via environment
variables.

In CI the credentials are provided through GitHub Actions secrets. Locally,
copy ``.env.sample`` to ``.env``, fill it in, and load it before running the
tests. Notebooks whose dependencies are unmet are skipped (not failed) so the
suite stays green on a machine with no credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Type

import pytest

from marss2l.config import AzureConfig, GEEConfig


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
        """Return the required env vars that are actually unset (or empty)."""
        missing: List[str] = []
        for cfg in self.requires:
            for var in cfg.required_env_vars:
                if not os.environ.get(var):
                    missing.append(var)
        return missing


# Maps each notebook to the credentials it needs. The MARS-S2L Hugging Face
# dataset is public, so HF-only notebooks need no gating and are omitted here
# (an unlisted notebook always runs). download_and_inference needs GEE; the
# figures that pull rasters from Azure need the Azure SAS token.
NOTEBOOK_DEPENDENCIES: Tuple[NotebookDependency, ...] = (
    NotebookDependency("notebooks/examples/download_and_inference.ipynb", (GEEConfig,)),
    NotebookDependency(
        "notebooks/figures/dataset_stats_by_split_and_geopackage_locations.ipynb",
        (AzureConfig,),
    ),
    NotebookDependency(
        "notebooks/figures/eval_model_and_figure_prob_vs_emission_rate.ipynb",
        (AzureConfig,),
    ),
    NotebookDependency(
        "notebooks/figures/ablation_threshold_pixels.ipynb",
        (AzureConfig,),
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
