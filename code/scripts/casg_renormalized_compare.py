#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重归一化对比分析：16 题 question_weights 归一化到和=1.0 前后，论文基线与方案A 的全套结果。

背景
----
论文实验用 16 道题，但 keypoint_scores.json 的 question_weights 是按 17 题
（含已弃用的 HW4_Q1）归一化的，16 题权重之和为 0.964708，导致总分量纲被压缩。
本脚本把 16 题权重重新归一化（和=1.0），比较：

    基线（量规分值）  ×  {原权重 qw, 重归一化 qw_rn}
    方案A（LLM权重）  ×  {原权重 qw, 重归一化 qw_rn}

复算论文口径下的全套数值：
    Table 2 CASG 行（连续最优 α* 处）、Table 3 α 扫描（8 个报告点）、
    Table 4 损失变体（equal/pdsigma01/prio_P/prio_D/prio_sigma 的相对式）。

损失函数（全部相对偏差式，对应 loss_configs 的 relative 家族）：
    equal      L = |P-.2|/.2 + |D-.3|/.3 + |σ-10|/10
    pdsigma01  L = |P-.2|/.2 + |D-.3|/.3 + 0.1|σ-10|/10   （论文 3.4 正式式）
    prio_P     L = 10|P-.2|/.2 + |D-.3|/.3 + |σ-10|/10
    prio_D     L = |P-.2|/.2 + 10|D-.3|/.3 + |σ-10|/10
    prio_sigma L = |P-.2|/.2 + |D-.3|/.3 + 10|σ-10|/10

产物
----
data/results/casg_experiment/renormalized/
    renormalized_compare.json / .xlsx / 对比表_*.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()


ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / 'scripts'))

from alpha_sweep_from_xlsx import (  # noqa: E402
    parse_xlsx, load_deps, compute_casg, metrics_for,
)

W_FILE = REPO_ROOT / 'data' / 'raw' / 'keypoint_weights.json'
OUTDIR = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'renormalized'

TARGET_P, TARGET_D, TARGET_SIGMA = 0.20, 0.30, 10.0
REPORT_ALPHAS = [0.1, 0.3, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]  # 论文 Table 3 报告点

# 论文 Table 2 中不随 question_weights 变化的两行（供对照）
TABLE2_FIXED = {
    'HUM (n=37)': dict(P=0.238, D=0.149, sigma=6.41, r='—', qwk10='—', mae='—'),
    'FEW-LLM': dict(P=0.263, D=0.209, sigma=8.84, r=0.613, qwk10=0.524, mae=5.76),
}

# 相对偏差损失族（w 为相对偏差前的乘子）
LOSS_TERMS = {
    'equal':      {'P': 1.0, 'D': 1.0, 'sigma': 1.0},
    'pdsigma01':  {'P': 1.0, 'D': 1.0, 'sigma': 0.1},
    'prio_P':     {'P': 10.0, 'D': 1.0, 'sigma': 1.0},
    'prio_D':     {'P': 1.0, 'D': 10.0, 'sigma': 1.0},
    'prio_sigma': {'P': 1.0, 'D': 1.0, 'sigma': 10.0},
}


def loss_rel(m, name):
    t = LOSS_TERMS[name]
    return (t['P'] * abs(m['P'] - TARGET_P) / TARGET_P
            + t['D'] * abs(m['D'] - TARGET_D) / TARGET_D
            + t['sigma'] * abs(m['sigma'] - TARGET_SIGMA) / TARGET_SIGMA)


def build_norm_llm(weights, qids):
    out = {}
    for q in qids:
        wq = weights[q]
        s = float(sum(wq.values()))
        assert s > 0, q
        out[q] = {str(k): float(v) * 100.0 / s for k, v in wq.items()}
    return out


def metrics_at(data, norm, qw, human, alpha, loss='pdsigma01'):
    scores, _ = compute_casg(data, norm, qw, float(alpha))
    m = metrics_for(scores, human)
    m['L_' + loss] = loss_rel(m, loss)
    return m


def solve(data, norm, qw, human, loss='pdsigma01'):
    """在 [0.1, 2.0] 上连续优化给定损失，返回 (α*, 指标dict)。"""
    r = minimize_scalar(
        lambda a: metrics_at(data, norm, qw, human, a, loss)['L_' + loss],
        bounds=(0.1, 2.0), method='bounded', options={'xatol': 1e-6})
    m = metrics_at(data, norm, qw, human, r.x, loss)
    m['alpha_star'] = round(float(r.x), 6)
    return m


def run_variant(data, norm, qw, human):
    """跑一个变体：报告点扫描 + 5 个损失变体各自的连续最优。"""
    grid = {a: metrics_at(data, norm, qw, human, a) for a in REPORT_ALPHAS}
    opt = {name: solve(data, norm, qw, human, name) for name in LOSS_TERMS}
    # 细网格（画图用）
    fine = {}
    for a in np.arange(0.1, 2.0001, 0.02):
        fine[round(float(a), 3)] = metrics_at(data, norm, qw, human, a)
    return {'grid': grid, 'opt': opt, 'fine': fine}


def pick(m, keys):
    return {k: (round(float(m[k]), 6) if k != 'qid' else m[k]) for k in keys}


METRIC_KEYS = ['mean', 'P', 'D', 'sigma', 'pearson_r', 'spearman_rho',
               'qwk_10', 'qwk_5', 'mae', 'rmse']


def main():
    data = parse_xlsx()
    norm, qw, human = load_deps()
    W = json.load(open(W_FILE, encoding='utf-8'))
    qids = sorted(data.keys())
    n16 = sum(qw[q] for q in qids)
    qw_rn = {q: qw[q] / n16 for q in qids}
    norm_llm = build_norm_llm(W, qids)

    print(f'题目 {len(qids)}，学生 {len(human)}')
    print(f'16 题原权重之和 = {n16:.6f}，重归一化系数 = 1/{n16:.6f} = {1/n16:.6f}')

    OUTDIR.mkdir(parents=True, exist_ok=True)

    res = {
        'meta': {
            'note': ('16 题 question_weights 重归一化（和=1.0）前后的对比；'
                     '评分框架 compute_casg 完全复用，仅权重来源与题目权重不同'),
            'qids': qids,
            'n_questions': len(qids),
            'n_students': len(human),
            'orig_weight_sum_16q': round(n16, 6),
            'renorm_factor': round(1 / n16, 6),
            'losses_relative': LOSS_TERMS,
            'table2_fixed_rows': TABLE2_FIXED,
        },
        'variants': {},
    }

    variants = {
        'base_orig': ('基线·量规分值·原权重', norm, qw),
        'base_rn': ('基线·量规分值·重归一化', norm, qw_rn),
        'llm_orig': ('方案A·LLM权重·原权重', norm_llm, qw),
        'llm_rn': ('方案A·LLM权重·重归一化', norm_llm, qw_rn),
    }

    for key, (label, n, q) in variants.items():
        v = run_variant(data, n, q, human)
        res['variants'][key] = {
            'label': label,
            'grid_report': {str(a): pick(v['grid'][a], METRIC_KEYS + ['L_pdsigma01'])
                            for a in REPORT_ALPHAS},
            'opt': {name: {**pick(v['opt'][name], METRIC_KEYS + ['alpha_star']),
                           'L_' + name: round(float(loss_rel(v['opt'][name], name)), 6)}
                    for name in LOSS_TERMS},
            'fine': {str(a): pick(v['fine'][a], METRIC_KEYS + ['L_pdsigma01'])
                     for a in sorted(v['fine'].keys())},
        }
        o = v['opt']['pdsigma01']
        print(f"\n[{label}] α*(pdsigma01)={o['alpha_star']:.4f}  "
              f"L={loss_rel(o,'pdsigma01'):.4f}  P={o['P']:.4f}  D={o['D']:.4f}  "
              f"σ={o['sigma']:.3f}  r={o['pearson_r']:.4f}  "
              f"QWK10={o['qwk_10']:.4f}  MAE={o['mae']:.3f}")

    with open(OUTDIR / 'renormalized_compare.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    # 生成 xlsx：报告点扫描 + 损失变体最优 + 细网格
    frames = []
    for key, v in res['variants'].items():
        df = pd.DataFrame([{**{'α': a}, **r} for a, r in v['grid_report'].items()])
        df.insert(0, '变体', v['label'])
        frames.append(df)
    opt_rows = []
    for key, v in res['variants'].items():
        for name, o in v['opt'].items():
            opt_rows.append({'变体': v['label'], '损失': name,
                             'α*': o['alpha_star'], **{k: o[k] for k in METRIC_KEYS},
                             'L': o['L_' + name]})
    fine_rows = []
    for key, v in res['variants'].items():
        for a, r in v['fine'].items():
            fine_rows.append({'变体': v['label'], 'α': float(a), **r})
    with pd.ExcelWriter(OUTDIR / 'renormalized_compare.xlsx', engine='openpyxl') as w:
        pd.concat(frames, ignore_index=True).to_excel(w, sheet_name='报告点扫描', index=False)
        pd.DataFrame(opt_rows).to_excel(w, sheet_name='损失变体最优', index=False)
        pd.DataFrame(fine_rows).to_excel(w, sheet_name='细网格(画图)', index=False)

    print('\n已保存 JSON/XLSX 于', OUTDIR)


if __name__ == '__main__':
    main()
