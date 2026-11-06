#!/usr/bin/env python3
"""Post-hoc analyses, run after the frozen test evaluation and reported as such.

1. Online threshold step size. The frozen gamma=0.005 was taken from the adaptive
   conformal literature, where alpha is 0.1. With alpha=0.01 one alarm lowers the
   level by 0.00495, and recovery takes ~100 quiet rows, so the detector goes blind
   after any burst. Smaller steps are compared here; the frozen value stays in the
   main tables.
2. Alarm audit on bars with no trades.
"""
import json
import warnings

import numpy as np
import pandas as pd

from fusion_data import NATIVE, PROXY_NAMES, RESULTS, native_bundle, proxy_bundle
from kronos_ad.fusion import track_threshold
from run_fusion import ALPHA, OUT, build, evaluate, subset

GAMMAS = (0.005, 0.001, 0.0005, 0.0001)


def main():
    warnings.filterwarnings('ignore')
    rows = []
    bundles = [native_bundle()] + [proxy_bundle(n) for n in PROXY_NAMES]
    for b in bundles:
        test = b['test']
        model, forecasts = build(b)
        score = model.score(b['x'][test], b['windows'][test], subset(forecasts, test))
        fixed = score > model.threshold_
        y = b['y'][test]
        rows.append(dict(dataset=b['name'], gamma='fixed', **evaluate(y, score, fixed),
                         blind_rows=0, pump_alarm=bool(fixed[np.flatnonzero(y == 1)[0]]) if b['name'] == NATIVE else None))
        for gamma in GAMMAS:
            flags, thresholds = track_threshold(score, model.calibration_scores_, ALPHA, gamma, 512)
            rows.append(dict(dataset=b['name'], gamma=gamma, **evaluate(y, score, flags),
                             blind_rows=int(np.isinf(thresholds).sum()),
                             pump_alarm=bool(flags[np.flatnonzero(y == 1)[0]]) if b['name'] == NATIVE else None))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'posthoc_online_gamma.csv', index=False)
    native = pd.read_csv(OUT / 'native_rows.csv.gz')
    quiet = native.volume == 0
    audit = dict(zero_volume_bars=int(quiet.sum()), test_bars=len(native),
                 original_alarms_on_zero_volume=int((native.original_alarm & quiet).sum()),
                 original_price_only_alarms_on_zero_volume=int((native.original_price_alarm & ~native.original_flow_alarm & quiet).sum()),
                 fusion_alarms_on_zero_volume=int((native.fusion_alarm & quiet).sum()),
                 fusion_alarms=int(native.fusion_alarm.sum()), original_alarms=int(native.original_alarm.sum()))
    (OUT / 'posthoc_zero_volume_audit.json').write_text(json.dumps(audit, indent=1) + '\n')
    print(table.pivot(index='dataset', columns='gamma', values='alarm_rate').round(4))
    print(table.pivot(index='dataset', columns='gamma', values='blind_rows'))
    print(table[table.dataset == NATIVE][['gamma', 'alarm_rate', 'pump_alarm']])
    print(audit)


if __name__ == '__main__':
    main()
