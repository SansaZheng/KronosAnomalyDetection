#!/usr/bin/env python3
"""List the Binance pump-and-dump events and the public kline archives they need.

The Sapienza list has 1,110 announced pumps; 520 target Binance, whose historical
5-minute bars are public (data.binance.vision, monthly ZIPs with CHECKSUM files).
Each event needs the event month and the month before it, so the detector has
reference and calibration history that ends at 00:00 UTC on the pump day.

Outputs (no network access needed):
  data/binance_pump/events.csv  one row per event with its archive names
  data/binance_pump/urls.txt    unique archive and checksum URLs, one per line

Download on a machine with access, then verify:
  wget -q -P data/binance_pump/raw -i data/binance_pump/urls.txt
  python scripts/binance_pump_manifest.py --verify
Archives for delisted pairs may be missing (HTTP 404); the verify step lists them.
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/binance_pump'
BASE = 'https://data.binance.vision/data/spot/monthly/klines'


def archives(symbol, date):
    pair = f'{symbol}BTC'
    months = [date.to_period('M'), (date - pd.offsets.MonthBegin(1)).to_period('M')]
    return [f'{pair}-5m-{m}.zip' for m in sorted(set(months))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(ROOT / 'data/raw/sapienza_pump_dump/pump_telegram.csv')
    events = events[events.exchange.str.lower() == 'binance'].copy()
    events['date'] = pd.to_datetime(events.date)
    events['announced_utc'] = pd.to_datetime(events.date.dt.strftime('%Y-%m-%d') + ' ' + events.hour)
    events['archives'] = [';'.join(archives(s, d)) for s, d in zip(events.symbol, events.date)]
    events = events.sort_values('announced_utc').reset_index(drop=True)
    events.to_csv(OUT / 'events.csv', index=False)
    names = sorted({name for row in events.archives for name in row.split(';')})
    urls = []
    for name in names:
        pair = name.split('-5m-')[0]
        urls += [f'{BASE}/{pair}/5m/{name}', f'{BASE}/{pair}/5m/{name}.CHECKSUM']
    (OUT / 'urls.txt').write_text('\n'.join(urls) + '\n')
    print(len(events), 'Binance events,', events.symbol.nunique(), 'symbols,', len(names), 'monthly archives')
    if a.verify:
        raw = OUT / 'raw'
        present, bad, missing = 0, [], []
        for name in names:
            path = raw / name
            if not path.exists():
                missing.append(name)
                continue
            checksum = path.with_name(name + '.CHECKSUM')
            if not checksum.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum.read_text().split()[0]:
                bad.append(name)
            else:
                present += 1
        print('verified', present, 'archives; checksum failures', len(bad), '; missing', len(missing))
        (OUT / 'verify.txt').write_text('\n'.join(['BAD ' + n for n in bad] + ['MISSING ' + n for n in missing]) + '\n')


if __name__ == '__main__':
    main()
