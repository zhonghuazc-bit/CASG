#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Figure 3: 权重优先可达性分析（renorm16 口径）
(a) P/D/σ 随 α 的变化(整卷扫描): 显示 P 可精确调节(穿越 0.20), D/σ 单调增但达不到目标
(b) 权重优先实验(equal/pdsigma01/prio_P/prio_D/prio_sigma)的 P/D/σ 相对偏差 bar
数据源: data/results/casg_experiment/renormalized/renormalized_compare.json
         -> variants.base_rn.grid_report (扫描, 8 网格点)
         -> variants.base_rn.opt (五变体)
输出: paper/figures/weight_priority_analysis.png (300dpi)
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
import numpy as np

ROOT = Path(r'C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育')
DATA = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'renormalized' / 'renormalized_compare.json'
OUT = ROOT / 'paper' / 'figures' / 'weight_priority_analysis.png'

T_P, T_D, T_S = 0.20, 0.30, 10.0

data = json.load(open(DATA, encoding='utf-8'))
brn = data['variants']['base_rn']
gr = brn['grid_report']
opt = brn['opt']

# 扫描: 8 网格点 (renorm16)
alphas = sorted(gr.keys(), key=float)
alphas_f = [float(a) for a in alphas]
P_sweep = [gr[a]['P'] for a in alphas]
D_sweep = [gr[a]['D'] for a in alphas]
S_sweep = [gr[a]['sigma'] for a in alphas]

# 五变体
cfg_order = ['equal', 'pdsigma01', 'prio_P', 'prio_D', 'prio_sigma']
labels = {'equal': 'equal', 'pdsigma01': 'pdsigma01', 'prio_P': 'P-priority',
          'prio_D': 'D-priority', 'prio_sigma': r'$\sigma$-priority'}
P_cfg = [opt[c]['P'] for c in cfg_order]
D_cfg = [opt[c]['D'] for c in cfg_order]
S_cfg = [opt[c]['sigma'] for c in cfg_order]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
fig.suptitle('Weight-Priority Analysis: Reachability of Institutional Targets', fontsize=13, fontweight='bold')

# ---- (a) P/D/sigma vs alpha ----
ax = axes[0]
ax2 = ax.twinx()
l1 = ax.plot(alphas_f, S_sweep, '-o', color='#7F77DD', lw=1.6, ms=4, label=r'$\sigma(\alpha)$ (left)')
l2 = ax2.plot(alphas_f, P_sweep, '-s', color='#185FA5', lw=1.6, ms=4, label=r'$P(\alpha)$ (right)')
l3 = ax2.plot(alphas_f, D_sweep, '-^', color='#0F6E56', lw=1.6, ms=4, label=r'$D(\alpha)$ (right)')
ax.axhline(T_S, color='#888780', ls=':', lw=1.3)
ax.text(0.13, T_S + 0.12, r'$\sigma_{target}$=10', fontsize=9, color='#444441')
ax2.axhline(T_P, color='#185FA5', ls=':', lw=1.1, alpha=0.7)
ax2.axhline(T_D, color='#0F6E56', ls=':', lw=1.1, alpha=0.7)
ax2.text(0.13, T_P + 0.012, r'$P_{target}$=0.20', fontsize=9, color='#0C447C')
ax2.text(0.13, T_D + 0.012, r'$D_{target}$=0.30', fontsize=9, color='#085041')
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\sigma$')
ax2.set_ylabel('P / D')
ax.set_title(r'(a) $P$, $D$ and $\sigma$ vs. $\alpha$ (whole-exam sweep)', fontsize=11)
ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
ax.set_ylim(6.2, 10.6)
ax2.set_ylim(0.05, 0.42)
ax.grid(alpha=0.3)
lines = l1 + l2 + l3
ax.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=9, framealpha=0.9)
ax.annotate(r'$P$ reaches 0.20 at $\alpha\approx0.825$', xy=(0.83, 0.198), xytext=(1.05, 0.34),
            arrowprops=dict(arrowstyle='->', color='#185FA5', lw=1.0), fontsize=9, color='#0C447C')
ax.annotate(r'$D$ and $\sigma$ rise monotonically but stay below targets',
            xy=(1.95, 8.90), xytext=(0.5, 9.7),
            arrowprops=dict(arrowstyle='->', color='#534AB7', lw=1.0), fontsize=9, color='#3C3489')

# ---- (b) relative deviations by priority config ----
ax = axes[1]
cfgs = cfg_order
xs = np.arange(len(cfgs))
w = 0.26
P_dev = np.abs(np.array(P_cfg) - T_P) / T_P
D_dev = np.abs(np.array(D_cfg) - T_D) / T_D
S_dev = np.abs(np.array(S_cfg) - T_S) / T_S
b1 = ax.bar(xs - w, P_dev, w, color='#85B7EB', label='P rel. dev.')
b2 = ax.bar(xs, D_dev, w, color='#97C459', label='D rel. dev.')
b3 = ax.bar(xs + w, S_dev, w, color='#B5B4FF', label=r'$\sigma$ rel. dev.')
for bars, vals in [(b1, P_dev), (b2, D_dev), (b3, S_dev)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.2f}', ha='center', fontsize=8)
ax.axhline(0, color='#888780', lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([labels[c] for c in cfgs], fontsize=9)
ax.set_ylabel('Relative deviation from target')
ax.set_title('(b) Weight-priority loss variants (target weight ×10)', fontsize=11)
ax.set_ylim(0, 0.85)
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.annotate('only P is reachable', xy=(2, 0.0), xytext=(1.3, 0.72),
            arrowprops=dict(arrowstyle='->', color='#993C1D', lw=1.0), fontsize=9, color='#993C1D')

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print('saved:', OUT)