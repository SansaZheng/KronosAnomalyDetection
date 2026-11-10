"""Fast paired bootstrap of AP and recall@FPR.

Each score is sorted once; a resample is a vector of multiplicities, and both metrics become
cumulative sums in the pre-sorted order.  The numbers equal sklearn's
``average_precision_score`` (ties grouped at one threshold) and ``run_fusion.recall_at_fpr``
(``positive > np.quantile(negative, 1 - fpr)`` with linear interpolation) on the resampled rows.
"""
import numpy as np


class ScoreOrder:
    """Pre-sorted view of one score vector over rows with binary labels."""

    def __init__(self, y, s):
        y, s = np.asarray(y).astype(int), np.asarray(s, dtype=np.float64)
        self.y = y
        self.n = len(y)
        self.desc = np.argsort(-s, kind='stable')
        s_desc = s[self.desc]
        self.ends = np.r_[np.flatnonzero(np.diff(s_desc) != 0), self.n - 1]   # last index of every tie group
        self.y_desc = y[self.desc]
        self.neg_rows = np.flatnonzero(y == 0)
        self.pos_rows = np.flatnonzero(y == 1)
        self.neg_asc = self.neg_rows[np.argsort(s[self.neg_rows], kind='stable')]
        self.neg_values = s[self.neg_asc]
        self.pos_asc = self.pos_rows[np.argsort(s[self.pos_rows], kind='stable')]
        self.pos_values = s[self.pos_asc]

    def average_precision(self, w):
        wd = w[self.desc]
        tps = np.cumsum(wd * self.y_desc)[self.ends]
        total = np.cumsum(wd)[self.ends]
        P = tps[-1]
        if P <= 0:
            return np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            precision = np.where(total > 0, tps / total, 0.0)
        recall = tps / P
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def recall_at_fpr(self, w, fpr=0.01):
        wn = w[self.neg_asc]
        cum = np.cumsum(wn)
        N = cum[-1] if len(cum) else 0
        wp = w[self.pos_asc]
        P = wp.sum()
        if N <= 0 or P <= 0:
            return np.nan
        h = (N - 1) * (1 - fpr)
        lo = int(np.floor(h))
        frac = h - lo
        x_lo = self.neg_values[np.searchsorted(cum, lo, side='right')]
        x_hi = self.neg_values[np.searchsorted(cum, min(lo + 1, N - 1), side='right')]
        threshold = x_lo + frac * (x_hi - x_lo)
        start = np.searchsorted(self.pos_values, threshold, side='right')
        return float(wp[start:].sum() / P)


def paired_bootstrap(y, scores, index_iter, fpr=0.01):
    """(AP, recall@fpr) per resample for every score, as {name: array (resamples, 2)}.

    ``index_iter`` yields row-index arrays (with repeats) over the rows of ``y``; a resample
    without both classes is skipped for every score alike, so the arrays stay paired.
    """
    y = np.asarray(y).astype(int)
    orders = {k: ScoreOrder(y, v) for k, v in scores.items()}
    out = {k: [] for k in scores}
    for idx in index_iter:
        w = np.bincount(idx, minlength=len(y)).astype(np.float64)
        pos = (w * (y == 1)).sum()
        if not 0 < pos < w.sum():
            continue
        for k, o in orders.items():
            out[k].append((o.average_precision(w), o.recall_at_fpr(w, fpr)))
    return {k: np.array(v) for k, v in out.items()}
