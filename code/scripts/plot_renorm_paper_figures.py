#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按重归一化数据重绘论文图 2 与图 3（覆盖 paper/figures/ 下的同名文件）。
数据源: data/results/casg_experiment/renormalized/renormalized_compare.json
        （variant base_rn：16 题权重重归一化后的 B2 主实验细网格与各损失变体最优）
基线参照: data/results/casg_experiment/llm_v2_baseline_metrics.json（FEW-LLM 不变）
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
import numpy as np

ROOT = REPO_ROOT
RES = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'renormalized' / 'renormalized_compare.json'
BASE = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'llm_v2_baseline_metrics.json'

T_P, T_D, T_S = 0.20, 0.30, 10.0
A_OPT = 0.8246
L_OPT = 0.4375

d = json.load(open(RES, encoding='utf-8'))['variants']['base_rn']
fine = {float(a): m for a, m in d['fine'].items()}
alphas = np.array(sorted(fine.keys()))
P = np.array([fine[a]['P'] for a in alphas])
D = np.array([fine[a]['D'] for a in alphas])
sig = np.array([fine[a]['sigma'] for a in alphas])
r = np.array([fine[a]['pearson_r'] for a in alphas])
q10 = np.array([fine[a]['qwk_10'] for a in alphas])
q5 = np.array([fine[a]['qwk_5'] for a in alphas])
L = np.array([fine[a]['L_pdsigma01'] for a in alphas])
opt = d['opt']
base = json.load(open(BASE, encoding='utf-8'))['llm_v2_fewshot_baseline']


def fig_alpha_tuning():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    fig.suptitle('Hyperparameter Tuning: Loss and Agreement Metrics vs. Power Exponent $\\alpha$',
                 fontsize=13, fontweight='bold')
    ax = axes[0][0]
    ax.plot(alphas, L, '-o', color='#185FA5', lw=1.6, ms=4, label=r'$\mathcal{L}(\alpha)$')
    ax.axvline(A_OPT, color='#993C1D', ls='--', lw=1.2)
    ax.annotate(f'$\\alpha^*$ = {A_OPT:.2f}\nmin $\\mathcal{{L}}$ = {L_OPT:.4f}',
                xy=(A_OPT, L_OPT), xytext=(1.25, 0.72),
                arrowprops=dict(arrowstyle='->', color='#993C1D', lw=1.0),
                fontsize=10, color='#993C1D')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('Loss')
    ax.set_title('(a) Loss function (relative deviation)', fontsize=11)
    ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0][1]
    ax.plot(alphas, r, '-o', color='#0F6E56', lw=1.6, ms=4, label='CASG $r$')
    ax.axhline(base['pearson_r'], color='#888780', ls=':', lw=1.2)
    ax.text(0.12, base['pearson_r'] + 0.006, f"FEW-LLM baseline = {base['pearson_r']:.3f}",
            fontsize=9, color='#444441')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('Pearson $r$')
    ax.set_title('(b) Pearson correlation with human scores', fontsize=11)
    ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0]); ax.set_ylim(0.55, 0.70)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1][0]
    ax.plot(alphas, q10, '-o', color='#BA7517', lw=1.6, ms=4, label='CASG QWK(10 bins)')
    ax.axhline(base['qwk_10'], color='#888780', ls=':', lw=1.2)
    ax.text(0.12, base['qwk_10'] + 0.012, f"FEW-LLM baseline = {base['qwk_10']:.3f}",
            fontsize=9, color='#444441')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('QWK (10 bins)')
    ax.set_title('(c) Quadratic weighted kappa, 10 bins', fontsize=11)
    ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0]); ax.set_ylim(0.0, 0.75)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1][1]
    ax.plot(alphas, q5, '-o', color='#7F77DD', lw=1.6, ms=4, label='CASG QWK(5 bins)')
    ax.axhline(base['qwk_5'], color='#888780', ls=':', lw=1.2)
    ax.text(0.12, base['qwk_5'] + 0.012, f"FEW-LLM baseline = {base['qwk_5']:.3f}",
            fontsize=9, color='#444441')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('QWK (5 bins)')
    ax.set_title('(d) Quadratic weighted kappa, 5 bins', fontsize=11)
    ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / 'paper' / 'figures' / 'alpha_tuning_pdsigma01.png'
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print('saved:', out)


def fig_weight_priority():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.suptitle('Weight-Priority Analysis: Reachability of Institutional Targets', fontsize=13, fontweight='bold')
    ax = axes[0]
    ax2 = ax.twinx()
    l1 = ax.plot(alphas, sig, '-o', color='#7F77DD', lw=1.6, ms=3.5, label=r'$\sigma(\alpha)$ (left)')
    l2 = ax2.plot(alphas, P, '-s', color='#185FA5', lw=1.6, ms=3.5, label=r'$P(\alpha)$ (right)')
    l3 = ax2.plot(alphas, D, '-^', color='#0F6E56', lw=1.6, ms=3.5, label=r'$D(\alpha)$ (right)')
    ax.axhline(T_S, color='#888780', ls=':', lw=1.3)
    ax.text(0.13, T_S + 0.12, r'$\sigma_{target}$=10', fontsize=9, color='#444441')
    ax2.axhline(T_P, color='#185FA5', ls=':', lw=1.1, alpha=0.7)
    ax2.axhline(T_D, color='#0F6E56', ls=':', lw=1.1, alpha=0.7)
    ax2.text(0.13, T_P + 0.012, r'$P_{target}$=0.20', fontsize=9, color='#0C447C')
    ax2.text(0.13, T_D + 0.012, r'$D_{target}$=0.30', fontsize=9, color='#085041')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$\sigma$'); ax2.set_ylabel('P / D')
    ax.set_title('(a) P, D and σ vs. α (whole-exam sweep)', fontsize=11)
    ax.set_xticks([0.2, 0.5, 0.8, 1.0, 1.5, 2.0])
    ax.set_ylim(6.2, 10.6); ax2.set_ylim(0.05, 0.42)
    ax.grid(alpha=0.3)
    lines = l1 + l2 + l3
    ax.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=9, framealpha=0.9)
    ax.annotate(r'$P$ reaches 0.20 at $\alpha\approx0.82$', xy=(0.82, 0.2005), xytext=(1.0, 0.34),
                arrowprops=dict(arrowstyle='->', color='#185FA5', lw=1.0), fontsize=9, color='#0C447C')
    ax.annotate(r'$D$ and $\sigma$ rise monotonically but stay below targets',
                xy=(2.0, 8.90), xytext=(0.5, 9.7),
                arrowprops=dict(arrowstyle='->', color='#534AB7', lw=1.0), fontsize=9, color='#3C3489')

    ax = axes[1]
    order = ['equal', 'pdsigma01', 'prio_P', 'prio_D', 'prio_sigma']
    labels = {'equal': 'equal', 'pdsigma01': 'pdsigma01', 'prio_P': 'P-priority',
              'prio_D': 'D-priority', 'prio_sigma': r'$\sigma$-priority'}
    rows = [(k, opt[k]) for k in order]
    xs = np.arange(len(rows)); w = 0.26
    P_dev = np.array([abs(o['P'] - T_P) / T_P for _, o in rows])
    D_dev = np.array([abs(o['D'] - T_D) / T_D for _, o in rows])
    S_dev = np.array([abs(o['sigma'] - T_S) / T_S for _, o in rows])
    b1 = ax.bar(xs - w, P_dev, w, color='#85B7EB', label='P rel. dev.')
    b2 = ax.bar(xs, D_dev, w, color='#97C459', label='D rel. dev.')
    b3 = ax.bar(xs + w, S_dev, w, color='#B5B4FF', label=r'$\sigma$ rel. dev.')
    for bars, vals in [(b1, P_dev), (b2, D_dev), (b3, S_dev)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015, f'{v:.2f}', ha='center', fontsize=8)
    ax.axhline(0, color='#888780', lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([labels[k] for k, _ in rows], fontsize=9)
    ax.set_ylabel('Relative deviation from target')
    ax.set_title('(b) Weight-priority loss variants (target weight ×10)', fontsize=11)
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.annotate('only P is reachable', xy=(2, 0.0), xytext=(1.3, 0.72),
                arrowprops=dict(arrowstyle='->', color='#993C1D', lw=1.0), fontsize=9, color='#993C1D')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / 'paper' / 'figures' / 'weight_priority_analysis.png'
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print('saved:', out)


if __name__ == '__main__':
    fig_alpha_tuning()
    fig_weight_priority()
