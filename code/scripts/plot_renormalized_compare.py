#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重归一化对比图：读 renormalized_compare.json 的细网格画两组图。

图 1 renormalized_alpha_metrics.png   2x2: L(α) / P(α) / D(α) / σ(α)
图 2 renormalized_consistency.png     2x2: Pearson r / QWK10 / QWK5 / MAE vs α

四条曲线：基线（蓝）与方案A（橙）；虚线为原权重，实线为重归一化。
L 面板标注两种重归一化变体的最优 α*。
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
RES = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'renormalized'

plt.rcParams.update({
    'font.family': 'Microsoft YaHei',
    'axes.titlesize': 10.5,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 200,
})

VARIANTS = [
    ('base_orig', 'Baseline, orig. w', 'tab:blue', '--', 0.9),
    ('base_rn', 'Baseline, renorm. w', 'tab:blue', '-', 1.8),
    ('llm_orig', 'Scheme A, orig. w', 'tab:orange', '--', 0.9),
    ('llm_rn', 'Scheme A, renorm. w', 'tab:orange', '-', 1.8),
]

# 同一 α 横轴（取基线细网格的 α 列表）
ALPHAS = 'alphas'


def load():
    d = json.load(open(RES / 'renormalized_compare.json', encoding='utf-8'))
    fine = {}
    alphas = None
    for key, *_ in VARIANTS:
        f = d['variants'][key]['fine']
        xs = np.array([float(a) for a in sorted(f.keys())])
        ys = {m: np.array([f[str(a)][m] for a in sorted(f.keys())])
              for m in ['P', 'D', 'sigma', 'pearson_r', 'qwk_10', 'qwk_5',
                        'mae', 'L_pdsigma01']}
        fine[key] = (xs, ys)
        if alphas is None:
            alphas = xs
    opt = {key: d['variants'][key]['opt']['pdsigma01']['alpha_star']
           for key, *_ in VARIANTS}
    return d, fine, opt


def draw_curve(ax, xs, ys, key, color, ls, lw, label, ms='.', every=None):
    ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, label=label)
    if every:
        ax.plot(xs[::every], ys[::every], color=color, marker=ms,
                linestyle='None', markersize=2.2)


def make_fig1(d, fine, opt):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    ax = axes[0][0]
    for key, lab, c, ls, lw in VARIANTS:
        xs, ys = fine[key]
        ax.plot(xs, ys['L_pdsigma01'], color=c, linestyle=ls, linewidth=lw, label=lab)
    for key, c in (('base_rn', 'tab:blue'), ('llm_rn', 'tab:orange')):
        xs, ys = fine[key]
        i = int(np.argmin(ys['L_pdsigma01']))
        ax.plot(xs[i], ys['L_pdsigma01'][i], marker='*', color=c, markersize=9,
                zorder=5)
        ax.axvline(xs[i], color=c, linestyle=':', linewidth=1.0, alpha=0.8)
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('Loss $L(\\alpha)$')
    ax.set_title('(a) Loss function (pdsigma01)')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0.05, 2.05)
    ax.grid(alpha=0.25, linewidth=0.4)

    # P / D / σ
    panels = [('P', 0.20, '(b) Difficulty P', 'P'),
              ('D', 0.30, '(c) Discrimination D', 'D'),
              ('sigma', 10.0, '(d) SD σ', 'σ')]
    for (mkey, tgt, title, ylab), a in zip(panels, [axes[0][1], axes[1][0], axes[1][1]]):
        for key, lab, c, ls, lw in VARIANTS:
            xs, ys = fine[key]
            draw_curve(a, xs, ys[mkey], key, c, ls, lw, lab, every=6)
        a.axhline(tgt, color='k', linestyle='--', linewidth=0.8, alpha=0.65)
        a.text(2.03, tgt, f'  target {tgt}', fontsize=7.5, va='center',
               ha='right', color='0.25')
        a.set_xlabel(r'$\alpha$'); a.set_ylabel(ylab)
        a.set_title(title)
        a.set_xlim(0.05, 2.05)
        a.grid(alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    out = RES / 'fig1_renormalized_alpha_metrics.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def make_fig2(d, fine, opt):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    panels = [('pearson_r', 'Pearson r', '(a) Pearson r'),
              ('qwk_10', 'QWK10', '(b) QWK10'),
              ('qwk_5', 'QWK5', '(c) QWK5'),
              ('mae', 'MAE', '(d) MAE')]
    for (mkey, ylab, title), a in zip(panels, axes.ravel()):
        for key, lab, c, ls, lw in VARIANTS:
            xs, ys = fine[key]
            draw_curve(a, xs, ys[mkey], key, c, ls, lw, lab, every=6)
        a.set_xlabel(r'$\alpha$'); a.set_ylabel(ylab)
        a.set_title(title)
        a.set_xlim(0.05, 2.05)
        a.grid(alpha=0.25, linewidth=0.4)
        if mkey == 'mae':
            a.legend(loc='upper right', framealpha=0.95)
        else:
            a.legend(loc='lower right', framealpha=0.95)
    fig.tight_layout()
    out = RES / 'fig2_renormalized_consistency.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


def main():
    d, fine, opt = load()
    make_fig1(d, fine, opt)
    make_fig2(d, fine, opt)


if __name__ == '__main__':
    main()
