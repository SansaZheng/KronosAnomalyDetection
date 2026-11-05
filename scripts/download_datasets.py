#!/usr/bin/env python3
"""Download reviewed catalog artifacts without extracting archives or executing data."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def destination(root, dataset_id, relative):
    base = (Path(root) / dataset_id).resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Artifact path escapes dataset directory")
    return target


def fetch(artifact, path, limit, opener=urllib.request.urlopen):
    """Atomic bounded download; skip verified copies and preserve existing files."""
    if path.exists():
        expected = artifact.get("sha256")
        if expected and digest(path) == expected:
            return "verified_existing"
        raise ValueError(f"Existing file is unverified or changed: {path}")
    if artifact.get("expected_bytes", 0) > limit:
        raise ValueError("Source exceeds per-file byte limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise ValueError(f"Partial file already exists: {temporary}")
    try:
        request = urllib.request.Request(artifact["url"], headers={"User-Agent": "KronosAnomalyDetection-dataset-downloader/1.0"})
        with opener(request, timeout=60) as response, temporary.open("xb") as out:
            length = response.headers.get("Content-Length")
            if length and int(length) > limit:
                raise ValueError("Response exceeds per-file byte limit")
            size = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                if size == 0:
                    prefix = block.lstrip()[:200].lower()
                    if prefix.startswith((b"<!doctype html", b"<html")):
                        raise ValueError("HTML/login page returned instead of data")
                    if block.startswith(b"version https://git-lfs.github.com/spec/"):
                        raise ValueError("Git LFS pointer returned instead of data")
                    if path.suffix == ".zip" and not block.startswith(b"PK"):
                        raise ValueError("ZIP magic mismatch")
                    if path.suffix == ".gz" and not block.startswith(b"\x1f\x8b"):
                        raise ValueError("GZIP magic mismatch")
                size += len(block)
                if size > limit:
                    raise ValueError("Stream exceeds per-file byte limit")
                out.write(block)
        if not size:
            raise ValueError("Empty download")
        if length and size != int(length):
            raise ValueError("Truncated response")
        if artifact.get("expected_bytes") is not None and size != artifact["expected_bytes"]:
            raise ValueError("Source size differs from reviewed metadata")
        for algorithm in ("sha256", "md5"):
            if artifact.get(algorithm) and digest(temporary, algorithm) != artifact[algorithm]:
                raise ValueError(f"{algorithm} mismatch")
        temporary.rename(path)
        return "downloaded"
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def directory_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".download.lock"
    with lock.open("x") as handle:
        handle.write("A download process is writing this directory.\n")
    try:
        yield
    finally:
        lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, default=ROOT / "data/catalog.json")
    p.add_argument("--output", type=Path, default=ROOT / "data/raw")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--list", action="store_true")
    p.add_argument("--include-large", action="store_true", help="Include reviewed large sources; use a separate /data or /date output and explicit byte limits")
    p.add_argument("--max-file-mib", type=int, default=95)
    p.add_argument("--max-total-mib", type=int, default=256)
    a = p.parse_args()
    catalog = json.loads(a.catalog.read_text())
    entries = catalog["datasets"]
    if a.ids:
        unknown = set(a.ids) - {e["id"] for e in entries}
        if unknown:
            p.error(f"Unknown dataset IDs: {sorted(unknown)}")
        entries = [e for e in entries if e["id"] in a.ids]
    if a.list:
        for e in entries:
            print(f"{e['id']:24} {e['access']:22} {len(e['artifacts'])} artifacts")
        return
    if a.max_file_mib < 1 or a.max_total_mib < 1:
        p.error("Download limits must be positive")
    with directory_lock(a.output):
        a.output.mkdir(parents=True, exist_ok=True)
        receipt_path = a.output / "download_manifest.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"schema_version": 1, "files": []}
        old = {f["path"]: f for f in receipt["files"]}
        budget = a.max_total_mib * 1024**2
        used = 0
        failures = []
        for e in entries:
            if e["access"] not in ("direct", "direct_partial") and not (a.include_large and e["access"] == "large_download"):
                print(f"SKIP {e['id']}: {e['access']} — see catalog and data/README.md")
                continue
            for artifact in e["artifacts"] + (e.get("large_artifacts", []) if a.include_large else []):
                path = destination(a.output, e["id"], artifact["path"])
                relative = str(path.relative_to(a.output.resolve()))
                spec = dict(artifact)
                if relative in old:
                    if old[relative]["source_url"] != artifact["url"]:
                        raise ValueError(f"Existing receipt points to a different source: {relative}")
                    spec["sha256"] = old[relative]["sha256"]
                try:
                    status = fetch(spec, path, min(a.max_file_mib * 1024**2, budget-used))
                    if status == "downloaded":
                        used += path.stat().st_size
                    old[relative] = dict(dataset=e["id"], path=relative, source_url=artifact["url"],
                                         role=artifact["role"], bytes=path.stat().st_size,
                                         sha256=digest(path), license=e["license"],
                                         retrieved_at=old.get(relative, {}).get("retrieved_at", datetime.now(timezone.utc).isoformat()),
                                         source_revision=e.get("source_revision"))
                    receipt["files"] = list(old.values())
                    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
                    print(status.upper(), relative, flush=True)
                except Exception as exc:
                    failures.append(dict(dataset=e["id"], path=artifact["path"], error=str(exc)))
                    print("FAILED", e["id"], artifact["path"], str(exc), file=sys.stderr, flush=True)
        report = dict(checked_at=datetime.now(timezone.utc).isoformat(), downloaded_bytes_this_run=used, failures=failures)
        (a.output / "last_download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
