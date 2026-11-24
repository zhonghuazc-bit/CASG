



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


ROOT = Path(r'C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育')
sys.path.insert(0, str(ROOT / 'scripts'))

from alpha_sweep_from_xlsx import (  
    parse_xlsx, load_deps, compute_casg, metrics_for,
)

OUTDIR = REPO_ROOT / 'data' / 'results' / 'casg_experiment'

TARGET_P, TARGET_D, TARGET_SIGMA = 0.20, 0.30, 10.0
REPORT_ALPHAS = [0.1, 0.3, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]  
ALPHA_BOUNDS = (0.0, 2.0)


TABLE2_FIXED = {
    'HUM (n=37)': dict(P=0.238, D=0.149, sigma=6.41, r='—', qwk10='—', mae='—'),
    'FEW-LLM': dict(P=0.263, D=0.209, sigma=8.84, r=0.613, qwk10=0.524, mae=5.76),
}


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


def metrics_at(data, norm, qw, human, alpha, loss='pdsigma01'):
    scores, _ = compute_casg(data, norm, qw, float(alpha))
    m = metrics_for(scores, human)
    m['L_' + loss] = loss_rel(m, loss)
    return m


def solve(data, norm, qw, human, loss='pdsigma01'):
    
    r = minimize_scalar(
        lambda a: metrics_at(data, norm, qw, human, a, loss)['L_' + loss],
        bounds=ALPHA_BOUNDS, method='bounded', options={'xatol': 1e-6})
    m = metrics_at(data, norm, qw, human, r.x, loss)
    m['alpha_star'] = round(float(r.x), 6)
    return m


def run_variant(data, norm, qw, human):
    
    grid = {a: metrics_at(data, norm, qw, human, a) for a in REPORT_ALPHAS}
    opt = {name: solve(data, norm, qw, human, name) for name in LOSS_TERMS}
    
    fine = {}
    for a in np.arange(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1] + 1e-4, 0.02):
        fine[round(float(a), 3)] = metrics_at(data, norm, qw, human, a)
    return {'grid': grid, 'opt': opt, 'fine': fine}


def pick(m, keys):
    return {k: (round(float(m[k]), 6) if k != 'qid' else m[k]) for k in keys}


METRIC_KEYS = ['mean', 'P', 'D', 'sigma', 'pearson_r', 'spearman_rho',
               'qwk_10', 'qwk_5', 'mae', 'rmse']


def main():
    data = parse_xlsx()
    norm, qw, human = load_deps()
    qids = sorted(data.keys())

    
    n16 = sum(qw[q] for q in qids)
    qw16 = {q: qw[q] / n16 for q in qids}

    print(f'题目 {len(qids)}，学生 {len(human)}；对 16 题直接分配权重，权重和=1，系数 1/{n16:.6f}')

    OUTDIR.mkdir(parents=True, exist_ok=True)

    v = run_variant(data, norm, qw16, human)
    res = {
        'meta': {
            'note': ('16 题 question_weights 直接分配权重使和=1.0（满分 100 分制），'
                     '与人工总分同量纲；评分框架 compute_casg 见 alpha_sweep_from_xlsx.py'),
            'qids': qids,
            'n_questions': len(qids),
            'n_students': len(human),
            'weight_sum_16q': round(n16, 6),
            'losses_relative': LOSS_TERMS,
            'table2_fixed_rows': TABLE2_FIXED,
            'alpha_bounds': list(ALPHA_BOUNDS),
        },
        'results': {
            'label': '16 题整卷 CASG（对 16 题直接分配权重，和=1，满分 100）',
            'grid_report': {str(a): pick(v['grid'][a], METRIC_KEYS + ['L_pdsigma01'])
                            for a in REPORT_ALPHAS},
            'opt': {name: {**pick(v['opt'][name], METRIC_KEYS + ['alpha_star']),
                           'L_' + name: round(float(loss_rel(v['opt'][name], name)), 6)}
                    for name in LOSS_TERMS},
            'fine': {str(a): pick(v['fine'][a], METRIC_KEYS + ['L_pdsigma01'])
                     for a in sorted(v['fine'].keys())},
        },
    }

    o = v['opt']['pdsigma01']
    print(f"\n[pdsigma01 连续最优] α*={o['alpha_star']:.4f}  "
          f"L={loss_rel(o, 'pdsigma01'):.4f}  P={o['P']:.4f}  D={o['D']:.4f}  "
          f"σ={o['sigma']:.3f}  r={o['pearson_r']:.4f}  "
          f"QWK10={o['qwk_10']:.4f}  MAE={o['mae']:.3f}")

    with open(OUTDIR / 'casg_table234.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    
    df = pd.DataFrame([{**{'α': a}, **r} for a, r in v['grid'].items()])
    opt_rows = [{**{'损失': name}, 'α*': o['alpha_star'],
                 **{k: o[k] for k in METRIC_KEYS}, 'L': o['L_' + name]}
                for name, o in v['opt'].items()]
    fine_rows = [{'α': float(a), **r} for a, r in v['fine'].items()]
    with pd.ExcelWriter(OUTDIR / 'casg_table234.xlsx', engine='openpyxl') as w:
        df.to_excel(w, sheet_name='报告点扫描', index=False)
        pd.DataFrame(opt_rows).to_excel(w, sheet_name='损失变体最优', index=False)
        pd.DataFrame(fine_rows).to_excel(w, sheet_name='细网格(画图)', index=False)

    print('\n已保存 JSON/XLSX 于', OUTDIR)


if __name__ == '__main__':
    main()