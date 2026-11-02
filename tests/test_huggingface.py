"""
Tests for the Hugging Face hub-cache filesystems in marss2l.huggingface.

``hf_hub_download`` is monkeypatched and the Hub API made unreachable, so no test touches the
network.
"""

import pytest
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import HfFileSystem, constants

from marss2l import huggingface
from marss2l.huggingface import REPO_ID, HfCachedFileSystem, HfCachedHTTPFileSystem


@pytest.fixture
def fake_download(monkeypatch, tmp_path):
    """Replace ``hf_hub_download`` with a stub that records its arguments and returns a local file."""
    calls = []

    def _fake(**kwargs):
        calls.append(kwargs)
        local_path = tmp_path / "hub" / kwargs["filename"]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"from-hub")
        return str(local_path)

    monkeypatch.setattr(huggingface, "hf_hub_download", _fake)
    return calls


@pytest.fixture
def offline_hub(monkeypatch):
    """Make the Hub API unreachable: table reads must not depend on ``repo_info`` validation."""

    def _unreachable(self, *args, **kwargs):
        raise ConnectionError("Hub API unreachable")

    monkeypatch.setattr(HfFileSystem, "_repo_and_revision_exist", _unreachable)


@pytest.fixture
def parent_open(monkeypatch):
    """Record calls that reach ``HfFileSystem._open`` / ``HTTPFileSystem._open``."""
    calls = []

    def _recorder(self, path, mode="rb", **kwargs):
        calls.append((path, mode))
        return "parent-file"

    monkeypatch.setattr(HfFileSystem, "_open", _recorder)
    monkeypatch.setattr(HTTPFileSystem, "_open", _recorder)
    return calls


class TestHfCachedFileSystem:
    @pytest.mark.parametrize(
        "path, revision, filename",
        [
            (f"datasets/{REPO_ID}/validated_images_all.csv", "main", "validated_images_all.csv"),
            (f"datasets/{REPO_ID}/data/stats.parquet", "main", "data/stats.parquet"),
            (f"datasets/{REPO_ID}@refs/pr/1/train.csv", "refs/pr/1", "train.csv"),
            (f"hf://datasets/{REPO_ID}@v1.0/data/test.csv", "v1.0", "data/test.csv"),
        ],
    )
    def test_table_read_uses_hub_cache(self, fake_download, offline_hub, path, revision, filename):
        fs = HfCachedFileSystem(skip_instance_cache=True)

        with fs.open(path, "rb") as f:
            assert f.read() == b"from-hub"

        assert fake_download == [
            dict(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                token=None,
                endpoint=constants.ENDPOINT,
            )
        ]

    def test_token_and_endpoint_reach_hf_hub_download(self, fake_download, offline_hub):
        fs = HfCachedFileSystem(
            token="hf_test", endpoint="https://hub.example.org", skip_instance_cache=True
        )

        with fs.open(f"datasets/{REPO_ID}/train.csv", "rb"):
            pass

        assert fake_download[0]["token"] == "hf_test"
        assert fake_download[0]["endpoint"] == "https://hub.example.org"

    def test_other_files_and_writes_go_to_hffilesystem(self, fake_download, parent_open):
        fs = HfCachedFileSystem(skip_instance_cache=True)
        npy_path = f"datasets/{REPO_ID}/data/train/0/chip.npy"
        csv_path = f"datasets/{REPO_ID}/validated_images_all.csv"

        assert fs._open(npy_path, "rb") == "parent-file"
        assert fs._open(csv_path, "wb") == "parent-file"

        assert parent_open == [(npy_path, "rb"), (csv_path, "wb")]
        assert fake_download == []

    def test_local_dir_serves_present_file(self, fake_download, offline_hub, tmp_path):
        local_dir = tmp_path / "release"
        (local_dir / "data").mkdir(parents=True)
        (local_dir / "data" / "stats.csv").write_bytes(b"from-local-dir")
        fs = HfCachedFileSystem(local_dir=str(local_dir), skip_instance_cache=True)

        with fs.open(f"datasets/{REPO_ID}/data/stats.csv", "rb") as f:
            assert f.read() == b"from-local-dir"
        assert fake_download == []

    def test_local_dir_falls_through_for_absent_file(self, fake_download, offline_hub, tmp_path):
        fs = HfCachedFileSystem(local_dir=str(tmp_path / "release"), skip_instance_cache=True)

        with fs.open(f"datasets/{REPO_ID}/train.csv", "rb") as f:
            assert f.read() == b"from-hub"
        assert [c["filename"] for c in fake_download] == ["train.csv"]


class TestHfCachedHTTPFileSystem:
    @pytest.mark.parametrize(
        "url, revision, filename",
        [
            (huggingface.CSV_PATH_DEFAULT_HF, "main", "validated_images_all.csv"),
            (huggingface.PARQUET_PLUME_PATH_DEFAULT_HF, "main", "validated_images_plumes.parquet"),
            (
                f"https://huggingface.co/datasets/{REPO_ID}/resolve/refs%2Fpr%2F1/data/train.csv",
                "refs/pr/1",
                "data/train.csv",
            ),
        ],
    )
    def test_hf_url_uses_hub_cache(self, fake_download, url, revision, filename):
        fs = HfCachedHTTPFileSystem(skip_instance_cache=True)

        with fs.open(url, "rb") as f:
            assert f.read() == b"from-hub"

        assert fake_download == [
            dict(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                token=None,
                endpoint=None,
            )
        ]

    def test_other_urls_go_to_httpfilesystem(self, fake_download, parent_open):
        fs = HfCachedHTTPFileSystem(skip_instance_cache=True)
        url = "https://example.org/data/table.csv"

        assert fs._open(url, "rb") == "parent-file"
        assert parent_open == [(url, "rb")]
        assert fake_download == []

    def test_fs_from_path_returns_cached_http_filesystem(self):
        from marss2l.utils import fs_from_path

        assert isinstance(fs_from_path(huggingface.CSV_PATH_DEFAULT_HF), HfCachedHTTPFileSystem)
