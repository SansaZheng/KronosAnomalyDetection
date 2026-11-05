#!/usr/bin/env python3
"""Offline integrity/schema audit. Never unpickle remote data or extract archives."""
import argparse
from collections import Counter
import csv
import gzip
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from download_datasets import digest

ROOT = Path(__file__).resolve().parents[1]


def csv_summary(handle, label_candidates=()):
    reader = csv.DictReader(handle)
    names = reader.fieldnames
    if not names:
        raise ValueError("Missing CSV header")
    counts = {label: Counter() for label in label_candidates if label in names}
    n = 0
    for row in reader:
        if None in row:
            raise ValueError("CSV row has more fields than header")
        n += 1
        for label in counts:
            counts[label][row[label]] += 1
    return dict(rows=n, columns=names, labels={k: dict(v) for k, v in counts.items()})


def audit(root):
    root = Path(root)
    receipt = json.loads((root / "download_manifest.json").read_text())
    results = []
    for record in receipt["files"]:
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Receipt mismatch: {record['path']}")
        detail = dict(dataset=record["dataset"], path=record["path"], bytes=record["bytes"], sha256_verified=True)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
                if bad:
                    raise ValueError(f"ZIP CRC failed: {bad}")
                detail["members"] = [dict(path=i.filename, bytes=i.file_size) for i in z.infolist() if not i.is_dir()]
                if record["dataset"] in ("ulb_creditcard", "banksim", "s_ffsd"):
                    detail["tables"] = {}
                    for member in z.namelist():
                        if member.endswith(".csv"):
                            with z.open(member) as f:
                                detail["tables"][member] = csv_summary(io.TextIOWrapper(f, encoding="utf-8-sig"), ("Class", "fraud", "Labels"))
        elif path.name.endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8-sig") as f:
                detail["table"] = csv_summary(f, ("label", "labels", "is_pump", "pump", "class"))
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as f:
                detail["table"] = csv_summary(f)
        elif path.suffix == ".npy":
            # Inspect headers only for object arrays, so no pickle is executed.
            with path.open("rb") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            detail["array"] = dict(shape=list(shape), dtype=str(dtype), object_array=bool(dtype.hasobject))
            if not dtype.hasobject:
                array = np.load(path, allow_pickle=False)
                detail["array"]["finite"] = bool(np.isfinite(array).all())
                if "label" in path.name.lower():
                    values, counts = np.unique(array, return_counts=True)
                    detail["array"]["values"] = dict(zip(map(str, values), map(int, counts)))
        results.append(detail)
    return dict(files=len(results), total_bytes=sum(r["bytes"] for r in results),
                dataset_ids=sorted({r["dataset"] for r in results}), artifacts=results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "data/raw")
    p.add_argument("--report", type=Path, default=ROOT / "data/integrity_report.json")
    a = p.parse_args()
    report = audit(a.root)
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"Verified {report['files']} files, {report['total_bytes']} bytes, {len(report['dataset_ids'])} source packages")


if __name__ == "__main__":
    main()
