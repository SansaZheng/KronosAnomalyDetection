#!/usr/bin/env python3
"""Replay real Kronos forecasts with robust joint calibration; no new inference."""
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kronos_ad.joint import JointResidualCalibrator
from kronos_ad.proxy import select_targets
from run_kronos_proxy import load_job
from run_benchmarks import encode, metrics

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results/2026-11-06/kronos_proxy'
OUT = ROOT / 'results/2026-11-06/kronos_tuning'
DEFAULT = dict(floor=.1, aggregate='top3', bias=True, risk=.01)


def evaluate(labels, score, flags):
    result = metrics(labels, score, flags)
    if not result['positives']:
        result.update(recall=None, f1=None)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, searches, comparisons, manifest = [], [], [], []
    for old in json.loads((OLD / 'all_metrics.json').read_text()):
        name = old['dataset']
        bundle, group_type = load_job(name)
        frame, labels, parts, ids, _ = bundle
        groups = np.array([str(v).split(':', 1)[0] for v in ids]) if group_type else np.repeat('stream', len(frame))
        targets, histories, _ = select_targets(parts, groups, 32, 512, 1024)
        values, features = encode(frame, parts)
        with np.load(OLD / (name + '__predictions.npz'), allow_pickle=False) as cached:
            np.testing.assert_array_equal(targets, cached['targets'])
            predictions = cached['predictions'].copy()
        residual = values[targets] - predictions
        std = values[histories].std(axis=1)
        details = json.loads((OLD / (name + '__details.json')).read_text())
        reference_scale = np.asarray(details['reference_scale'])
        cal = np.flatnonzero(parts[targets] == 'calibration')
        test = np.flatnonzero(parts[targets] == 'test')
        # Preserve whole event/case groups when splitting the calibration period.
        cal_groups = groups[targets[cal]]
        unique = list(dict.fromkeys(cal_groups))
        if len(unique) > 1:
            boundary = targets[cal[len(cal) // 2]]
            fit_groups = [g for g in unique if targets[cal[cal_groups == g]].max() < boundary]
            val_groups = [g for g in unique if targets[cal[cal_groups == g]].min() >= boundary]
            fit = cal[np.isin(cal_groups, fit_groups)]
            validation = cal[np.isin(cal_groups, val_groups)]
        else:
            fit, validation = np.array_split(cal, 2)
        if min(len(fit), len(validation)) < 100:
            raise ValueError('Insufficient separated calibration partitions: ' + name)
        np.savez_compressed(OUT / (name + '__score_inputs.npz'), targets=targets,
                            residual=residual, local_std=std, reference_scale=reference_scale,
                            fit_indices=fit, validation_indices=validation, test_indices=test)
        # Both methods fit thresholds using only the earlier calibration partition.
        # Validation labels are used solely by the explicitly non-unsupervised arm.
        known_val = labels[targets[validation]]
        eligible = (known_val == 1).sum() >= 3 and (known_val == 0).sum() >= 20
        chosen = DEFAULT.copy()
        best = None
        for floor, aggregate, bias, risk in itertools.product([.01, .1, .5], ['max', 'top3', 'mean'], [False, True], [.005, .01, .02, .05, .1, .2]):
            config = dict(floor=floor, aggregate=aggregate, bias=bias, risk=risk)
            if np.ceil((len(fit) + 1) * (1 - risk)) > len(fit):
                searches.append(dict(dataset=name, **config, status='insufficient_calibration_for_risk'))
                continue
            model = JointResidualCalibrator(**config).fit(residual[fit], std[fit], reference_scale)
            score = model.score(residual[validation], std[validation])
            result = evaluate(known_val, score, score > model.threshold_)
            searches.append(dict(dataset=name, **config, status='evaluated', **result))
            if eligible:
                # Predeclared selection: validation F1, AP, fewer alarms, then
                # smaller floor/risk. No test metric participates in selection.
                key = (result['f1'], result['average_precision'], -result['alarm_rate'], -floor, -risk)
                if best is None or key > best:
                    best, chosen = key, config
        old_rows = pd.read_csv(OLD / (name + '__detections.csv.gz'))
        old_test = old_rows[old_rows.split == 'test']
        baseline = evaluate(labels[targets[test]], old_test.max_score_ratio.to_numpy(), old_test.anomaly.to_numpy())
        for method, config in [('joint_unsupervised', DEFAULT), ('validation_selected', chosen)]:
            model = JointResidualCalibrator(**config).fit(residual[fit], std[fit], reference_scale)
            # The model/threshold is frozen before touching any test score or label.
            score = model.score(residual, std)
            flags = score > model.threshold_
            result = evaluate(labels[targets[test]], score[test], flags[test])
            val_result = evaluate(known_val, score[validation], flags[validation])
            state = model.state()
            summary = dict(dataset=name, method=method, config=config, fitted_state=state,
                           fit_rows=len(fit), validation_rows=len(validation), test_rows=len(test),
                           validation_positives=int((known_val == 1).sum()), validation_selection_eligible=bool(eligible),
                           uses_validation_labels=bool(method == 'validation_selected' and eligible),
                           validation_metrics=val_result, **result)
            summaries.append(summary)
            split = np.repeat('purged_calibration', len(targets)).astype('U20')
            split[test] = 'test'
            split[fit] = 'calibration_fit'; split[validation] = 'validation'
            pd.DataFrame(dict(source_row=targets, row_id=ids[targets], group=groups[targets], split=split,
                              label=labels[targets], score=score, threshold=model.threshold_, anomaly=flags)).to_csv(
                OUT / f'{name}__{method}__detections.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
            comparisons.append(dict(dataset=name, method=method, config=json.dumps(config, sort_keys=True),
                                    validation_selection_eligible=bool(eligible),
                                    **{'before_' + k: baseline[k] for k in ['precision', 'recall', 'f1', 'average_precision', 'alarm_rate', 'tp', 'fp', 'fn', 'tn']},
                                    **{'after_' + k: result[k] for k in ['precision', 'recall', 'f1', 'average_precision', 'alarm_rate', 'tp', 'fp', 'fn', 'tn']}))
        for suffix in ['__predictions.npz', '__cache_parameters.json', '__details.json']:
            source = OLD / (name + suffix)
            manifest.append(dict(path=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        print('Finished', name, 'fit/validation/test', len(fit), len(validation), len(test), 'validation positives', int((known_val == 1).sum()), flush=True)
    (OUT / 'all_metrics.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
    (OUT / 'input_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    pd.DataFrame(searches).to_csv(OUT / 'validation_search.csv', index=False)
    table = pd.DataFrame(comparisons)
    table.to_csv(OUT / 'comparison.csv', index=False)
    plot(table)
    report(table, summaries)


def plot(table):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained')
    names = table.dataset.drop_duplicates().tolist()
    x = np.arange(len(names))
    before = table.drop_duplicates('dataset').set_index('dataset').reindex(names)
    for ax, metric in zip(axes, ['f1', 'alarm_rate']):
        ax.bar(x - .25, before['before_' + metric], width=.25, label='Original per-feature POT')
        for offset, method, label in [(0, 'joint_unsupervised', 'Joint unsupervised'), (.25, 'validation_selected', 'Validation selected (uses labels if eligible)')]:
            subset = table[table.method == method].set_index('dataset').reindex(names)
            ax.bar(x + offset, subset['after_' + metric], width=.25, label=label)
        ax.set_xticks(x, names, rotation=60, ha='right')
        ax.set_ylabel('F1 (NA when no positives)' if metric == 'f1' else 'Fraction of test positions alerted')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Kronos numeric proxy: retrospective scoring comparison on identical test positions')
    fig.savefig(OUT / 'before_after.png', dpi=180)
    fig.savefig(OUT / 'before_after.pdf')
    plt.close(fig)


def report(table, summaries):
    lines = ['# Kronos 参数与方法调整：回顾性实验', '',
             '本次复用已保存的真实 Kronos-mini 单步预测，修改误差评分与阈值，不冒充重新训练或新增模型推理。原生 OHLCVA 实验保持不变；以下全部是已授权的逐特征数值代理实验。', '',
             '## 改动与选参协议', '',
             '- 残差尺度下限：max(历史窗口标准差, floor × 参考期特征标准差)，防止小分母放大。',
             '- 可选校准残差中位数偏差修正，并用拟合分区的残差中位数平衡特征尺度。',
             '- 将逐维 POT 后取或，改成 max、top3 均值或全维均值的联合分数，再对一个联合分数设置经验次序统计阈值；判定使用严格大于，处理并列分数。',
             '- 原校准位置按时间/文件顺序中点分成前后两部分；事件/案例分组保持完整，跨越中点的分组整体剔除。拟合区最晚位置必须早于验证区最早位置；前半拟合评分统计和阈值，后半作验证。没有重新拟合全部校准集。',
             '- 完全无监督方案固定 floor=0.1、top3、偏差修正、risk=0.01，不使用任何标签。它是本次新增方法；该默认配置不是通过本次测试成绩选择的。',
             '- 验证集选参方案名义搜索 3 floors × 3 聚合 × 2 偏差设置 × 6 risks = 108 配置；校准样本不足以支持对应 risk 的配置明确跳过。仅当验证区有至少3个已知异常和20个已知正常时，按验证 F1、AP、较低报警率等预定顺序选择；否则回退固定无监督配置。此方案使用标签选参，不能称为完全无监督。',
             '- 阈值为排序后第 ceil((n+1)(1-risk)) 项，risk 是名义报警风险；金融序列有相关性，且拟合统计与阈值共享拟合分区，不宣称具有分布无关或严格误报保证。',
             '- 所有配置选择仅访问拟合/验证分区。测试位置与旧实验完全相同，并已在之前实验中查看，因此这里只能做回顾性比较；真正独立验证仍需要新时间段/数据集。',
             '- 512个校准与1024个测试位置为原实验上限；SD 小案例扣除预热后更少，未运行全量数据。窗口32、采样1、T=1、top_p=0.9、冻结模型及输入保持不变。unknown=2 只排除评估、不伪装正常；无正例时 F1/Recall/AP 为 NA。', '',
             '## 相同测试位置的结果', '',
             '| Dataset | 方法 | 原 F1 | 新 F1 | 原报警率 | 新报警率 | 验证标签选参 |',
             '|---|---|---:|---:|---:|---:|---|']
    def fmt(v): return 'NA' if pd.isna(v) else f'{v:.4f}'
    by_key = {(r['dataset'], r['method']): r for r in summaries}
    for _, r in table.iterrows():
        used = by_key[r.dataset, r.method]['uses_validation_labels']
        lines.append(f'| {r.dataset} | {r.method} | {fmt(r.before_f1)} | {fmt(r.after_f1)} | {r.before_alarm_rate:.2%} | {r.after_alarm_rate:.2%} | {"是" if used else "否"} |')
    lines += ['', '![比较图](before_after.png)', '',
              '改进不保证每个数据集都成立；完整 Precision/Recall/AP、混淆矩阵与配置见 comparison.csv、all_metrics.json 和 validation_search.csv。无监督方案在 Sapienza-25s、SD-prices 的 F1 提升，但 ULB、BankSim、S-FFSD、SD-crypto、SD-synthetic 的召回/F1 下降，不建议全局替换原方案。验证选参方案在 SD-stock、SD-prices 提升 F1，但伴随较高报警率，且 SD-synthetic 仍退步。以上结论应与正例数量一起理解，Sapienza-25s 测试仅2个正例。不同采样版本不视为独立语料；固定阈值可能无法适应分布切换。代理方法存在领域偏移，不能替代原生行情有效性验证。', '',
              '复现前安装 `pip install -e ".[test,experiments,kronos]"`，保留仓库内原始数据与已发布的 Kronos-Proxy 预测缓存。运行 `PYTHONPATH=.:scripts python scripts/tune_kronos_proxy.py`；验证：`PYTHONPATH=.:scripts python scripts/verify_kronos_tuning.py`。评分重放不需要重新下载模型权重。输入预测哈希见 input_manifest.json；完整结果不覆盖原始实验。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
