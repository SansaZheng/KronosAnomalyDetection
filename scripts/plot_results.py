#!/usr/bin/env python3
"""Publish tables and scientific figures from measured detector outputs only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2026-11-06'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','axes.facecolor':'white','savefig.facecolor':'#f8fafc'})
COLORS={'IsolationForest':'#2563eb','RobustMAD':'#e46a37'}


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def baselines():
    table=pd.read_csv(OUT/'baselines/summary.csv')
    display=table[['dataset','model','evaluated','positives','precision','recall','f1','average_precision']].copy()
    display.columns=['Dataset','Model','N evaluated','N positive','Precision','Recall','F1','AP']
    for c in ['Precision','Recall','F1','AP']:display[c]=display[c].map(lambda v:f'{v:.3f}')
    fig,ax=plt.subplots(figsize=(14,10));ax.axis('off')
    artist=ax.table(cellText=display.values,colLabels=display.columns,cellLoc='center',loc='center',colWidths=[.16,.15,.12,.11,.1,.1,.1,.1])
    artist.auto_set_font_size(False);artist.set_fontsize(10);artist.scale(1,1.8)
    for (row,col),cell in artist.get_celld().items():
        cell.set_edgecolor('#e2e8f0')
        if row==0:cell.set_facecolor('#172554');cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#eff6ff' if row%4 in [1,2] else 'white')
    ax.set_title('Downloaded-data unsupervised baselines\nFrozen POT thresholds; test labels used only for evaluation',fontsize=16,weight='bold',pad=15)
    fig.text(.5,.055,'AP = average precision (not trapezoidal PR-AUC). Variants share sources and are not independent datasets.\nSD arrays: file-order case split, not verified chronological validation. Unknown labels excluded.',ha='center',fontsize=10,color='#475569')
    save(fig,'baseline_results_table')
    names=table.dataset.drop_duplicates().tolist()
    fig,axes=plt.subplots(1,2,figsize=(13,7.5),sharey=True)
    y=np.arange(len(names))
    for ax,metric,title in zip(axes,['average_precision','f1'],['Average precision','F1 at frozen POT threshold']):
        for model,offset in [('IsolationForest',-.18),('RobustMAD',.18)]:
            part=table[table.model==model].set_index('dataset').loc[names]
            ax.barh(y+offset,part[metric],height=.32,color=COLORS[model],label=model)
        if metric=='average_precision':
            base=table.drop_duplicates('dataset').set_index('dataset').loc[names]
            ax.scatter(base.prevalence,y,c='#111827',marker='|',s=150,label='Test prevalence (chance AP)',zorder=5)
        ax.set_xlim(0,1);ax.set_title(title,weight='bold');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].legend(loc='lower right',fontsize=9);axes[1].legend(loc='lower right',fontsize=9)
    fig.suptitle('Baseline performance on held-out data',fontsize=17,weight='bold');fig.tight_layout()
    save(fig,'baseline_comparison')
    chosen=['ulb_creditcard','banksim','s_ffsd','sapienza_25s','sd_crypto','sd_stock']
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,name in zip(axes.flat,chosen):
        for model in COLORS:
            rows=pd.read_csv(OUT/f'baselines/{name}__{model}__detections.csv.gz')
            rows=rows[rows.evaluation_known]
            precision,recall,_=precision_recall_curve(rows.label,rows.score)
            idx=np.unique(np.r_[np.arange(0,len(recall),max(1,len(recall)//1500)),len(recall)-1])
            ap=table.loc[(table.dataset==name)&(table.model==model),'average_precision'].iloc[0]
            ax.step(recall[idx],precision[idx],where='post',color=COLORS[model],label=f'{model} AP={ap:.3f}')
        ax.axhline(rows.label.mean(),ls=':',color='#64748b',label='Prevalence')
        ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Recall',ylabel='Precision',title=name);ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Precision–recall curves: evaluation labels only',fontsize=17,weight='bold');fig.tight_layout()
    save(fig,'baseline_precision_recall')
    # Markdown table has the same source numbers as the CSV and table image.
    header='| '+' | '.join(display.columns)+' |\n| '+' | '.join(['---']*len(display.columns))+' |\n'
    body='\n'.join('| '+' | '.join(map(str,row))+' |' for row in display.values)
    (OUT/'BASELINE_TABLE.md').write_text('# 无监督基线检测结果\n\n'+header+body+'\n')


def kronos():
    path=OUT/'kronos/detections.csv.gz'
    if not path.exists():return
    rows=pd.read_csv(path,parse_dates=['timestamps']);s=json.loads((OUT/'kronos/summary.json').read_text())
    features=s['features'];test=rows[rows.split=='test'];alerts=test[test.anomaly]
    fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True,gridspec_kw={'height_ratios':[2,1,1.4]})
    axes[0].plot(rows.timestamps,rows.close,color='#334155',lw=.9,label='Observed close')
    axes[0].plot(rows.timestamps,rows.prediction_close,color='#0d9488',lw=.7,alpha=.7,label='One-step Kronos forecast')
    axes[0].scatter(alerts.timestamps,alerts.close,color='#dc2626',s=16,label='Test alarm (any feature)',zorder=4)
    weak=test[test.weak_label==1]
    for t in weak.timestamps:
        for ax in axes:ax.axvline(t,color='#a855f7',ls='--',lw=1.2)
    axes[0].set_ylabel('Price (BTC / BRD)');axes[0].legend(loc='upper left',ncol=3,fontsize=9)
    axes[1].bar(rows.timestamps,rows.volume,width=4/(24*60),color='#94a3b8');axes[1].set_ylabel('Volume (BRD)')
    axes[2].plot(rows.timestamps,rows.max_score_ratio.clip(lower=1e-6),color='#2563eb',lw=.8)
    axes[2].axhline(1,color='#dc2626',ls='--',label='Alarm threshold = 1')
    axes[2].set_yscale('log');axes[2].set_ylabel('max(score / threshold)');axes[2].legend(loc='upper left')
    split=pd.Timestamp(s['test_start'])
    for ax in axes:
        ax.axvspan(rows.timestamps.min(),split,color='#dbeafe',alpha=.35)
        ax.axvline(split,color='#64748b',ls=':');ax.grid(alpha=.15)
    axes[2].xaxis.set_major_formatter(DateFormatter('%m-%d\n%H:%M UTC'))
    axes[0].set_title(f"Actual Kronos-mini forecasts — BRD/BTC, 5-minute bars\nCalibration: Dec 22–27 | Test: Dec 28–31 | {s['test_alarms']} / {s['test_rows']} test bars flagged",fontsize=15,weight='bold')
    fig.text(.5,.01,'Blue background: calibration. Purple dashed line: published weak pump-chunk annotation.\nRetrospective single-case study; unannotated market bars are not confirmed negatives.',ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=[0,.045,1,1]);save(fig,'kronos_timeseries')
    ratio=np.array([test['score_'+f]/test['threshold_'+f] for f in features])
    fig,ax=plt.subplots(figsize=(14,4))
    im=ax.imshow(np.clip(ratio,1e-3,1e3),aspect='auto',interpolation='nearest',norm=LogNorm(vmin=.01,vmax=100),cmap='RdYlBu_r')
    ax.set_yticks(range(6),features);ticks=np.linspace(0,len(test)-1,6,dtype=int)
    ax.set_xticks(ticks,[test.timestamps.iloc[i].strftime('%m-%d %H:%M') for i in ticks]);ax.set_xlabel('Test bar time (UTC)')
    ax.set_title('Kronos anomaly dimension localization: score / per-feature POT threshold\nValues > 1 trigger that dimension; this is not causal attribution',weight='bold')
    fig.colorbar(im,ax=ax,label='Score / threshold (log color scale)');fig.tight_layout();save(fig,'kronos_feature_heatmap')
    summary=pd.read_csv(OUT/'kronos/feature_summary.csv')
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(summary.feature,summary.test_alarms,color=['#2563eb']*4+['#0d9488','#d97706']);ax.set(ylabel='Number of test bars flagged',title='Kronos per-feature test alarms (overlap allowed)');ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'kronos_feature_alarms')


if __name__=='__main__':
    baselines();kronos();print('Saved tables, PNG and PDF figures to',OUT)
