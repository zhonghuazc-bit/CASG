#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Figure 2: 超参数调优三联图（renorm16 口径）
(a) 损失函数 L(α)（pdsigma01 相对偏差）
(b) 与人工评分的一致性: Pearson r、QWK10(10 bins)、MAE(右轴) vs α
移除原 QWK5(5 bins) 面板 —— 该指标论文未汇报（中英文、投稿、LaTeX 均一致）。
数据源: data/results/casg_experiment/renormalized/renormalized_compare.json
         -> variants.base_rn.grid_report (8 个网格点, renorm16 口径)
         -> variants.base_rn.opt.pdsigma01 (连续最优 α*=0.8246)
基线: llm_v2_baseline_metrics.json -> llm_v2_fewshot_baseline (FEW-LLM, 与论文表 3 注一致)
输出: paper/figures/alpha_tuning_pdsigma01.png (300dpi)
"""

import json
from pathlib import Path

import matplotlib

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fig_font import apply_figure_font

apply_figure_font()

ROOT = Path(r'C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育')
DATA = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'renormalized' / 'renormalized_compare.json'
BASE = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'llm_v2_baseline_metrics.json'
OUT = ROOT / 'paper' / 'figures' / 'alpha_tuning_pdsigma01.png'

data = json.load(open(DATA, encoding='utf-8'))
brn = data['variants']['base_rn']
gr = brn['grid_report']
opt = brn['opt']['pdsigma01']
base = json.load(open(BASE, encoding='utf-8'))['llm_v2_fewshot_baseline']

# 网格点按 α 升序
alphas = sorted(gr.keys(), key=float)
alpha = [float(a) for a in alphas]
loss = [gr[a]['L_pdsigma01'] for a in alphas]
r = [gr[a]['pearson_r'] for a in alphas]
q10 = [gr[a]['qwk_10'] for a in alphas]
mae = [gr[a]['mae'] for a in alphas]

a_opt = opt['alpha_star']          # 0.8246
loss_min = opt['L_pdsigma01']      # 0.4375

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
fig.suptitle('Hyperparameter Tuning: Loss and Agreement Metrics vs. Power Exponent $\\alpha$',
             fontsize=13, fontweight='bold')

# (a) Loss
ax = axes[0]
ax.plot(alpha, loss, '-o', color='#185FA5', lw=1.6, ms=4, label=r'$\mathcal{L}(\alpha)$')
ax.axvline(a_opt, color='#993C1D', ls='--', lw=1.2)
ax.annotate(f'$\\alpha^*$ = {a_opt:.3f}\nmin $\\mathcal{{L}}$ = {loss_min:.4f}',
            xy=(a_opt, loss_min), xytext=(1.25, 0.62),
            arrowprops=dict(arrowstyle='->', color='#993C1D', lw=1.0),
            fontsize=10, color='#993C1D')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('Loss')
ax.set_title('(a) Loss function (relative deviation)', fontsize=11)
ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
ax.set_ylim(0.3, 1.35)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# (b) Pearson r
ax = axes[1]
ax.plot(alpha, r, '-o', color='#0F6E56', lw=1.6, ms=4, label='CASG $r$')
ax.axhline(base['pearson_r'], color='#888780', ls=':', lw=1.2)
ax.text(0.12, base['pearson_r'] + 0.006, f"CASG $r$ (baseline {base['pearson_r']:.3f})",
        fontsize=9, color='#444441')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('Pearson $r$')
ax.set_title('(b) Pearson correlation with human scores', fontsize=11)
ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
ax.set_ylim(0.55, 0.70)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)

# (c) QWK10 + MAE (双轴)
ax = axes[2]
ax2 = ax.twinx()
l1 = ax.plot(alpha, q10, '-o', color='#BA7517', lw=1.6, ms=4, label='QWK (10 bins)')
l2 = ax2.plot(alpha, mae, '-s', color='#7F77DD', lw=1.6, ms=4, label='MAE (right)')
ax.axhline(base['qwk_10'], color='#888780', ls=':', lw=1.2)
ax.text(0.12, base['qwk_10'] + 0.015, f"QWK10 baseline {base['qwk_10']:.3f}",
        fontsize=9, color='#444441')
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('QWK (10 bins)')
ax2.set_ylabel('MAE')
ax.set_title('(c) QWK10 and MAE vs human scores', fontsize=11)
ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
ax.grid(alpha=0.3)
lines = l1 + l2
ax.legend(lines, [l.get_label() for l in lines], loc='center left', fontsize=9, framealpha=0.9)

fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print('saved:', OUT)