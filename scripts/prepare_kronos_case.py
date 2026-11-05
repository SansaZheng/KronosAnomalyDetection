#!/usr/bin/env python3
"""Reproduce the exact local market/model inputs pinned by the experiment manifest."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'results/2026-11-06/input_manifest.json').read_text())
    for record in manifest['files']:
        p=ROOT/record['path']
        if p.exists():
            if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError(f'Existing input has changed: {p}')
            print('VERIFIED',record['path'],flush=True)
            continue
        p.parent.mkdir(parents=True,exist_ok=True)
        request=urllib.request.Request(record['url'],headers={'User-Agent':'KronosAnomalyDetection-reproduction'})
        raw=urllib.request.urlopen(request,timeout=60).read()
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:
            raise ValueError(f'Source version differs from published experiment: {record["url"]}')
        p.write_bytes(raw)
        print('DOWNLOADED',record['path'],flush=True)


if __name__=='__main__':main()
