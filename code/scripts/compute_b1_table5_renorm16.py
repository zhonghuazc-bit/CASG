#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1 跨班 Table 5 重归一化口径复算（16 题 question_weights 归一化到和=1.0）。

背景
----
compute_b1_table5_v2_35scope.py 用原始 17 题归一化的 question_weights 计算 16 题，
导致 max_score = 96.4708。本脚本将其除以 16 题权重之和（重归一化，和=1.0），
max_score = 100，CASG 分数与人工总分（满分 100）同一量纲，其余逻辑完全一致：
  - Human v2（补分后）行不变
  - transferred @ α*_B2(renorm)：从 renormalized_compare.json 取重归一化后的开发班最优
  - recalibrated α*_B1：仍用 B1 脚本原损失配置 'paper'
  - 采样标定 n ∈ {5,10,15,20}
产物: data/results/b1_class/b1_table5_renorm16.json
"""

import json
import sys
from pathlib import Path

import numpy as np
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

from alpha_sweep_from_xlsx import compute_casg  # noqa: E402
from alpha_per_question_analysis import qwk_safe, full_metrics  # noqa: E402
from casg_losses import LOSS_FUNCTIONS  # noqa: E402

ALL_QIDS = ['HW1_Q1', 'HW1_Q2', 'HW1_Q3',
            'HW2_Q1', 'HW2_Q2', 'HW2_Q3',
            'HW3_Q1', 'HW3_Q2', 'HW3_Q3',
            'HW4_Q1', 'HW4_Q2', 'HW4_Q3', 'HW4_Q4', 'HW4_Q5',
            'HW5_Q1', 'HW5_Q2', 'HW5_Q3']
QIDS16 = [q for q in ALL_QIDS if q != 'HW4_Q1']

BASE = REPO_ROOT / 'data' / 'results' / 'b1_class'
EVAL_FILE = BASE / 'casg_b1_evaluations_v2.json'
V2_FILE = BASE / 'b1_human_scores_penalty_corrected.json'
KP_FILE = REPO_ROOT / 'data' / 'raw' / 'keypoint_scores.json'
RENORM_RES = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'renormalized' / 'renormalized_compare.json'
OUT_JSON = BASE / 'b1_table5_renorm16.json'

LOSS_NAME = 'paper'   # 与 B1 原脚本一致
N_REPEATS = 20
SEED = 42


def main():
    evals = json.load(open(EVAL_FILE, encoding='utf-8'))
    v2 = json.load(open(V2_FILE, encoding='utf-8'))
    kp = json.load(open(KP_FILE, encoding='utf-8'))
    rn = json.load(open(RENORM_RES, encoding='utf-8'))

    norm = kp['normalized_100']
    qw_raw = kp['question_weights']
    wsum = sum(qw_raw[q] for q in QIDS16)
    qw_rn = {q: qw_raw[q] / wsum for q in QIDS16}
    a_b2 = rn['variants']['base_rn']['opt']['pdsigma01']['alpha_star']
    print(f'重归一化: 16 题权重之和 {wsum:.6f} -> 1.0；α*_B2(renorm) = {a_b2}')

    # 16 题数据
    data = {}
    for qid in QIDS16:
        data[qid] = {}
        for sid, ev in evals.get(qid, {}).items():
            if ev is None:
                continue
            kps = sorted(ev['keypoints'], key=lambda k: k.get('id', 0))
            data[qid][sid] = [
                dict(status=k.get('status', 'missing'),
                     coverage=k.get('coverage_ratio', 0.0),
                     band=k.get('quality_band', None))
                for k in kps
            ]

    htot = v2['human_total']
    sids = sorted({s for q in QIDS16 for s in data[q]
                   if all(s in data[q2] for q2 in QIDS16)})
    print(f'全覆盖子集: {len(sids)} 人 × 16 题')
    h = np.array([htot[s] for s in sids], float)
    max_score = sum(qw_rn.values()) * 100.0
    print(f'max_score = {max_score:.4f}')

    def casg_total(alpha):
        _, per_q = compute_casg(data, norm, qw_rn, float(alpha))
        return np.array([sum(per_q[q][s] * qw_rn[q] for q in QIDS16)
                         for s in sids], float)

    # 1) Human v2（补分后）——与重归一化无关
    m_human = full_metrics(h, h)
    print(f"\nHuman v2: P={m_human['P']:.3f} D={m_human['D']:.3f} σ={m_human['sigma']:.2f} mean={h.mean():.2f}")

    # 2) transferred @ α*_B2(renorm)
    c_fix = casg_total(a_b2)
    m_fix = full_metrics(c_fix, h, max_score=max_score)
    print(f"transferred @ {a_b2:.4f}: P={m_fix['P']:.3f} D={m_fix['D']:.3f} σ={m_fix['sigma']:.2f} "
          f"mean={c_fix.mean():.2f} r={m_fix['pearson_r']:.3f} QWK10={m_fix['qwk_10']:.3f} MAE={m_fix['mae']:.2f}")

    # 3) recalibrated α*_B1（loss 'paper'，同原脚本）
    func = LOSS_FUNCTIONS[LOSS_NAME]['func']

    def obj(alpha):
        c = casg_total(alpha)
        m = full_metrics(c, h, max_score=max_score)
        return float(func(m))

    res = minimize_scalar(obj, bounds=(0.1, 2.0), method='bounded',
                          options={'xatol': 1e-6})
    a_b1 = float(res.x)
    c_best = casg_total(a_b1)
    m_best = full_metrics(c_best, h, max_score=max_score)
    print(f"recalibrated @ {a_b1:.4f}: P={m_best['P']:.3f} D={m_best['D']:.3f} σ={m_best['sigma']:.2f} "
          f"mean={c_best.mean():.2f} r={m_best['pearson_r']:.3f} QWK10={m_best['qwk_10']:.3f} MAE={m_best['mae']:.2f}")
    print(f"Δα = |{a_b1:.4f} - {a_b2:.4f}| = {abs(a_b1-a_b2):.4f}")

    # 4) 采样标定
    rng = np.random.default_rng(SEED)
    sampling = {}
    for n in [5, 10, 15, 20]:
        alphas, rs, sigmas, ps = [], [], [], []
        for _ in range(N_REPEATS):
            sample = rng.choice(sids, size=min(n, len(sids)), replace=False)
            mask = {s: True for s in sample}
            data_s = {q: {s: v for s, v in data[q].items() if mask.get(s)}
                      for q in QIDS16}
            h_s = np.array([htot[s] for s in sample], float)

            def obj_s(alpha):
                _, per_q_s = compute_casg(data_s, norm, qw_rn, float(alpha))
                c = np.array([sum(per_q_s[q][s] * qw_rn[q] for q in QIDS16)
                              for s in sample], float)
                m = full_metrics(c, h_s, max_score=max_score)
                return float(func(m))

            r2 = minimize_scalar(obj_s, bounds=(0.1, 2.0), method='bounded',
                                 options={'xatol': 1e-6})
            c_all = casg_total(r2.x)
            m_all = full_metrics(c_all, h, max_score=max_score)
            alphas.append(r2.x)
            rs.append(m_all['pearson_r'])
            sigmas.append(m_all['sigma'])
            ps.append(m_all['P'])
        a_arr = np.array(alphas)
        row = dict(sample_size=n, alpha_mean=round(float(a_arr.mean()), 4),
                   alpha_std=round(float(a_arr.std()), 4),
                   alpha_min=round(float(a_arr.min()), 4),
                   alpha_max=round(float(a_arr.max()), 4),
                   r_mean=round(float(np.mean(rs)), 4),
                   sigma_mean=round(float(np.mean(sigmas)), 4),
                   P_mean=round(float(np.mean(ps)), 4))
        sampling[n] = row
        print(f"  n={n}: α={row['alpha_mean']:.4f}±{row['alpha_std']:.3f} "
              f"r={row['r_mean']:.3f} σ={row['sigma_mean']:.2f} P={row['P_mean']:.3f}")

    out = dict(
        scope='B1 35人×16题, Human=v2(补分后), question_weights 重归一化(和=1.0), '
              f'α*_B2={a_b2:.4f} transferred / α*_B1 recalibrated',
        n=len(sids), max_score=round(max_score, 4),
        orig_weight_sum_16q=round(wsum, 6),
        alpha_star_b2=round(a_b2, 6), alpha_star_b1=round(a_b1, 4),
        delta_alpha=round(abs(a_b1 - a_b2), 4),
        human_v2=dict(P=round(m_human['P'], 3), D=round(m_human['D'], 3),
                      sigma=round(m_human['sigma'], 2), mean=round(float(h.mean()), 2)),
        transferred=dict(P=round(m_fix['P'], 3), D=round(m_fix['D'], 3),
                         sigma=round(m_fix['sigma'], 2), mean=round(float(c_fix.mean()), 2),
                         r=round(m_fix['pearson_r'], 3), qwk10=round(m_fix['qwk_10'], 3),
                         mae=round(m_fix['mae'], 2)),
        recalibrated=dict(alpha=round(a_b1, 4), loss=round(float(func(m_best)), 6),
                          P=round(m_best['P'], 3), D=round(m_best['D'], 3),
                          sigma=round(m_best['sigma'], 2), mean=round(float(c_best.mean()), 2),
                          r=round(m_best['pearson_r'], 3), qwk10=round(m_best['qwk_10'], 3),
                          mae=round(m_best['mae'], 2)),
        sampling={str(k): v for k, v in sampling.items()},
    )
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n已输出: {OUT_JSON}')


if __name__ == '__main__':
    main()
