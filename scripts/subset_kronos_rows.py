#!/usr/bin/env python3
"""Size study: cut an existing Kronos inference cache down to the row subset the larger models are run on.

`run_kronos_rows.py --rows-from <cache> --caps a,b,c` runs another model on a subset of a cache's
rows (`caps` rows per set plus every labelled anomaly).  This script slices the source cache itself
to the same subset, so the source model (Kronos-mini) is compared on identical rows at no compute.

    python scripts/subset_kronos_rows.py results/2026-11-10/kronos_rows/banksim results/2026-11-14/mini/kronos_rows/banksim --caps 4000,4000,8000
"""
import argparse
import json
from pathlib import Path

import numpy as np

from kronos_states import subset_positions

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('source')
p.add_argument('out')
p.add_argument('--caps', default='4000,4000,8000')
p.add_argument('--study-caps', default='6000,20000')
a = p.parse_args()
src, out = Path(a.source), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((src / 'meta.json').read_text())
meta['subset_of'] = str(src)
plan = list(zip(('ref', 'cal', 'test'), map(int, a.caps.split(',')))) + list(zip(('study_cal', 'study_test'), map(int, a.study_caps.split(','))))
for i, (name, cap) in enumerate(plan):
    if not (src / f'{name}.npz').exists():
        continue
    seed = 100 + (i if i < 3 else i - 3)
    with np.load(src / f'{name}.npz') as z:
        n = len(z['targets'])
        keep = subset_positions(z['labels'], cap, seed)
        np.savez_compressed(out / f'{name}.npz', **{k: (z[k][keep] if z[k].shape[:1] == (n,) else z[k]) for k in z.files})
        positives = int((z['labels'][keep] == 1).sum())
    meta.setdefault('sets', {}).setdefault(name, {}).update(rows=int(len(keep)), positives=positives)
    print(f'{name}: {n} -> {len(keep)} rows ({positives} anomalies)')
(out / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
