#!/usr/bin/env python3
"""Figures for explicit Kronos-Proxy experiments, separate from native OHLCVA."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2026-11-06/kronos_proxy'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f8fafc','savefig.facecolor':'#f8fafc'})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    results=json.loads((OUT/'all_metrics.json').read_text())
    table=pd.DataFrame([dict(Dataset=r['dataset'],Dimensions=f"{r['active_dimensions']}/{r['dimensions']}",Calibration=r['calibration_scored'],Test=r['test_rows'],Known=r['evaluated'],Positive=r['positives'],Precision=r['precision'],Recall=r['recall'],F1=r['f1'],AP=r['average_precision'],AlarmRate=r['alarm_rate']) for r in results])
    table.loc[table.Positive==0,['Recall','F1','AP']]=np.nan
    table.to_csv(OUT/'results_table.csv',index=False)
    display=table.copy()
    for c in ['Precision','Recall','F1','AP']:display[c]=display[c].map(lambda x:'—' if pd.isna(x) else f'{x:.3f}')
    display.AlarmRate=display.AlarmRate.map(lambda x:f'{100*x:.2f}%')
    fig,ax=plt.subplots(figsize=(15,6.2));ax.axis('off')
    tab=ax.table(cellText=display.values,colLabels=display.columns,cellLoc='center',loc='center',colWidths=[.15,.09,.09,.075,.075,.075,.08,.08,.08,.08,.09])
    tab.auto_set_font_size(False);tab.set_fontsize(9.5);tab.scale(1,2.)
    for (i,j),cell in tab.get_celld().items():
        cell.set_edgecolor('#dbe2ef');cell.set_facecolor('#172554' if i==0 else '#eff6ff' if i%2 else 'white')
        if i==0:cell.set_text_props(color='white',weight='bold')
    ax.set_title('Kronos-mini-Proxy: actual frozen-weight inference on NON-OHLCVA data\nPer-feature numeric proxies | Deterministic label-free subsampling | Not native financial K-lines',fontsize=14,weight='bold',pad=20)
    fig.text(.5,.01,'Dimensions = calibrated / inferred features. Known = test points with labels 0 or 1.\nRecall/F1/AP are shown as N/A for subsets with no positives. Each feature has its own POT threshold.',ha='center',fontsize=9,color='#475569')
    save(fig,'kronos_proxy_table')
    fig,axes=plt.subplots(1,2,figsize=(13,6.5),sharey=True);y=np.arange(len(table))
    for ax,col,title in zip(axes,['AP','F1'],['Average precision on known selected test points','F1 at frozen per-feature POT thresholds']):
        val=table[col].to_numpy(dtype=float);ax.barh(y,np.nan_to_num(val),color='#0d9488')
        for i,v in enumerate(val):ax.text(min((v if np.isfinite(v) else 0)+.015,.95),i,'N/A' if not np.isfinite(v) else f'{v:.3f}',va='center',fontsize=9)
        ax.set_xlim(0,1.05);ax.set_title(title,fontsize=11,weight='bold');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yticks(y,table.Dataset);axes[0].invert_yaxis();fig.suptitle('Kronos-Proxy pilot results — not full-dataset native-Kronos benchmarks',weight='bold');fig.tight_layout();save(fig,'kronos_proxy_metrics')
    fig,axes=plt.subplots(5,2,figsize=(14,14))
    for ax,r in zip(axes.flat,results):
        d=pd.read_csv(OUT/(r['dataset']+'__detections.csv.gz'));d=d[d.split=='test'].reset_index(drop=True)
        ax.plot(np.maximum(d.max_score_ratio,1e-6),lw=.65,color='#2563eb')
        ax.axhline(1,color='#dc2626',ls='--',lw=.8)
        positive=np.flatnonzero(d.label==1)
        ax.scatter(positive,np.maximum(d.max_score_ratio.iloc[positive],1e-6),color='#a855f7',s=12,label='Known positive',zorder=4)
        ax.set_yscale('log');ax.set_title(f"{r['dataset']}: {len(d)} selected test points",fontsize=10)
        ax.set_xlabel('Selected test-point index');ax.set_ylabel('max(score / threshold)');ax.grid(alpha=.15)
    fig.suptitle('Proxy anomaly scores on selected test points; red line = alert threshold\nPurple markers are evaluation labels, never model inputs',weight='bold');fig.tight_layout(rect=[0,0,1,.965]);save(fig,'kronos_proxy_timelines')
    r=next(x for x in results if x['dataset']=='ulb_creditcard');d=pd.read_csv(OUT/'ulb_creditcard__detections.csv.gz');d=d[d.split=='test']
    active=[f for f in r['features'] if d['threshold_'+f].notna().all()]
    ratio=np.array([d['score_'+f]/d['threshold_'+f] for f in active])
    fig,ax=plt.subplots(figsize=(14,8));im=ax.imshow(np.clip(ratio,.001,1000),aspect='auto',interpolation='nearest',norm=LogNorm(vmin=.01,vmax=100),cmap='RdYlBu_r')
    ax.set_yticks(np.arange(len(active)),active,fontsize=9);ax.set_xlabel('Selected ULB test-point index');ax.set_title('ULB feature localization using Kronos numeric proxies\nThese are PCA/amount features, not market OHLCVA dimensions',weight='bold');fig.colorbar(im,ax=ax,label='score / threshold');fig.tight_layout();save(fig,'kronos_proxy_ulb_dimensions')
    header='| '+' | '.join(display.columns)+' |\n| '+' | '.join(['---']*len(display.columns))+' |\n'
    body='\n'.join('| '+' | '.join(map(str,row))+' |' for row in display.values)
    (OUT/'RESULTS_TABLE.md').write_text('# Kronos-Proxy 抽样适配检测结果\n\n'+header+body+'\n')
    print('Saved proxy tables and 4 PNG/PDF figure pairs.')


if __name__=='__main__':main()
