import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("download_datasets", Path(__file__).parents[1] / "scripts/download_datasets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response(io.BytesIO):
    def __init__(self, content, length=None):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content) if length is None else length)}


def test_download_checksum_and_idempotency(tmp_path):
    content = b"a,b\n1,2\n"
    artifact = dict(url="https://example.invalid/data.csv", expected_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    path = tmp_path / "data.csv"
    assert module.fetch(artifact, path, 100, lambda *a, **kw: Response(content)) == "downloaded"
    assert module.fetch(artifact, path, 100, lambda *a, **kw: pytest.fail("Should not fetch again")) == "verified_existing"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.fetch(artifact, path, 100)


@pytest.mark.parametrize("content,name,match", [
    (b"<!doctype html><html>login", "data.csv", "HTML"),
    (b"version https://git-lfs.github.com/spec/v1", "data.csv", "LFS pointer"),
    (b"abc", "data.zip", "ZIP magic"),
    (b"abc", "data.csv.gz", "GZIP magic"),
])
def test_bad_payload_removed(tmp_path, content, name, match):
    path = tmp_path / name
    with pytest.raises(ValueError, match=match):
        module.fetch(dict(url="https://example.invalid"), path, 100, lambda *a, **kw: Response(content))
    assert not path.exists()
    assert not path.with_name(name+".part").exists()


def test_size_limits_and_digest_failures(tmp_path):
    with pytest.raises(ValueError, match="byte limit"):
        module.fetch(dict(url="https://example.invalid", expected_bytes=1000), tmp_path/"a", 10)
    with pytest.raises(ValueError, match="sha256"):
        module.fetch(dict(url="https://example.invalid", sha256="wrong"), tmp_path/"b", 100, lambda *a, **kw: Response(b"data"))
    with pytest.raises(ValueError, match="Truncated"):
        module.fetch(dict(url="https://example.invalid"), tmp_path/"c", 100, lambda *a, **kw: Response(b"data", length=8))
    assert not list(tmp_path.iterdir())


def test_path_traversal_and_concurrent_download_lock(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        module.destination(tmp_path, "dataset", "../../escape")
    with module.directory_lock(tmp_path):
        with pytest.raises(FileExistsError):
            with module.directory_lock(tmp_path):
                pytest.fail("Concurrent writer entered")
    assert not (tmp_path / ".download.lock").exists()
