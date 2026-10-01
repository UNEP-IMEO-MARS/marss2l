"""Shared pytest fixtures for the marss2l test suite."""

from pathlib import Path

import pytest

# Hugging Face dataset repo whose rasters the fixtures point at.
HF_REPO_ID = "UNEP-IMEO/MARS-S2L"
# Prepended to the fixtures' relative raster paths so HfFileSystem can resolve them
# (e.g. "data/..." -> "datasets/UNEP-IMEO/MARS-S2L/data/...").
HF_PATH_PREPEND = f"datasets/{HF_REPO_ID}"

# Balanced CSV fixtures used by the train_final / eval_final integration tests.
# Generate/refresh them with: python -m scripts.create_fixture
FIXTURES_DIR = Path(__file__).parent / "fixtures"
IMAGES_FIXTURE = FIXTURES_DIR / "images_fixture.csv"
PLUMES_FIXTURE = FIXTURES_DIR / "plumes_fixture.csv"


@pytest.fixture
def images_fixture_path() -> str:
    """Return the path to the committed balanced images CSV fixture.

    Skips the test if the fixture has not been generated yet.
    """
    if not IMAGES_FIXTURE.exists():
        pytest.skip(
            f"Images fixture not found at {IMAGES_FIXTURE}. "
            "Generate it with `python -m scripts.create_fixture`."
        )
    return str(IMAGES_FIXTURE)


@pytest.fixture
def plumes_fixture_path() -> str:
    """Return the path to the committed balanced plumes CSV fixture.

    Used by train_final (do_simulation=True). Skips if not generated yet.
    """
    if not PLUMES_FIXTURE.exists():
        pytest.skip(
            f"Plumes fixture not found at {PLUMES_FIXTURE}. "
            "Generate it with `python -m scripts.create_fixture`."
        )
    return str(PLUMES_FIXTURE)


@pytest.fixture
def hf_raster_fs(monkeypatch):
    """Filesystem for reading rasters from Hugging Face while CSVs are read locally.

    The committed fixtures are local CSVs whose raster paths are *relative* (``data/...``). The
    rasters themselves live on Hugging Face. ``train_final``/``eval_final`` use a single filesystem
    for both the CSV and the rasters, so we:

    - return an ``HfFileSystem`` for the tests to pass as that filesystem (it loads the rasters, with
      ``path_prepend_data=HF_PATH_PREPEND`` turning ``data/...`` into ``datasets/<repo>/data/...``);
    - monkeypatch ``read_csv_images``/``read_csv_plumes`` (where they are bound in the entry-point
      modules) so the *local* fixture CSV is read with a local filesystem, while everything else
      (derived columns, ``path_prepend_data``, plume merge) still runs in the real functions.
    """
    from huggingface_hub import HfFileSystem

    import marss2l.dataframe_image_plumes as dip
    from marss2l import eval_final, train_final
    from marss2l.utils import fs_from_path

    real_read_csv_images = dip.read_csv_images
    real_read_csv_plumes = dip.read_csv_plumes

    def _read_csv_images_local(csv_path, fs=None, **kwargs):
        # Ignore the (HF) fs for the CSV read; keep path_prepend_data and other kwargs.
        return real_read_csv_images(csv_path, fs=fs_from_path(csv_path), **kwargs)

    def _read_csv_plumes_local(csv_path, fs=None, **kwargs):
        return real_read_csv_plumes(csv_path, fs=fs_from_path(csv_path), **kwargs)

    monkeypatch.setattr(eval_final, "read_csv_images", _read_csv_images_local)
    monkeypatch.setattr(train_final, "read_csv_images", _read_csv_images_local)
    monkeypatch.setattr(train_final, "read_csv_plumes", _read_csv_plumes_local)

    return HfFileSystem()
