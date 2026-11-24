#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1 班 FEW-LLM 基线的指标计算与对比。

输入
----
data/results/b1_class/b1_llm_v2_scores.json   （run_b1_llm_v2_baseline.py 的产物）
data/results/b1_class/b1_human_scores_penalty_corrected.json （B1 人工 v2）

口径
----
- 学生：16 道共有题完整覆盖子样本（35 人，与 Table 5 / 跨班实验一致）
- LLM 逐题分：每题 0-100（少样本 LLM v2 整体评分）
- 总分（主口径，100 分制，与重归一化后的 CASG/人工一致）：
      total = Σ_q  LLM分_q × qw_rn_q ，qw_rn 为 16 题重归一化权重（和=1）
- 另算旧口径 total_old = Σ_q LLM分_q × qw_q（17 题归一化，16 题和=0.9647）备查
- 指标用 full_metrics（P/D/σ 对满分 100；r/QWK10/QWK5/MAE 对 B1 人工 v2）

产物
----
data/results/b1_class/b1_llm_v2_baseline_metrics.json
"""

import json
import sys
from pathlib import Path

import numpy as np

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
from alpha_per_question_analysis import full_metrics  # noqa: E402

Q16 = ['HW1_Q1', 'HW1_Q2', 'HW1_Q3', 'HW2_Q1', 'HW2_Q2', 'HW2_Q3',
       'HW3_Q1', 'HW3_Q2', 'HW3_Q3', 'HW4_Q2', 'HW4_Q3', 'HW4_Q4', 'HW4_Q5',
       'HW5_Q1', 'HW5_Q2', 'HW5_Q3']
BASE = REPO_ROOT / 'data' / 'results' / 'b1_class'
SCORE_FILE = BASE / 'b1_llm_v2_scores.json'
HUMAN_FILE = BASE / 'b1_human_scores_penalty_corrected.json'
KP_FILE = REPO_ROOT / 'data' / 'raw' / 'keypoint_scores.json'
RENORM_FILE = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'renormalized' / 'renormalized_compare.json'
OUT_JSON = BASE / 'b1_llm_v2_baseline_metrics.json'


def load_human():
    v2 = json.load(open(HUMAN_FILE, encoding='utf-8'))
    return v2['human_total']


def main():
    kp = json.load(open(KP_FILE, encoding='utf-8'))
    qw = kp['question_weights']
    s16 = sum(qw[q] for q in Q16)
    qw_rn = {q: qw[q] / s16 for q in Q16}

    evals = json.load(open(BASE / 'casg_b1_evaluations_v2.json', encoding='utf-8'))
    sids35 = sorted({s for q in Q16 for s in evals[q]
                     if all(s in evals[q2] for q2 in Q16)})
    print(f'子样本: {len(sids35)} 人 × {len(Q16)} 题')

    scores = json.load(open(SCORE_FILE, encoding='utf-8'))
    n_done = sum(len(v) for v in scores.values())
    n_scored = sum(1 for q in Q16 for s in sids35
                   if scores.get(q, {}).get(s, {}).get('score') is not None)
    print(f'评分文件: 记录 {n_done}，35 人子样本内有效分 {n_scored}/{len(sids35)*len(Q16)}')
    if n_scored < len(sids35) * len(Q16):
        print('!! 评分未完成，继续等待后重跑')

    htot = load_human()
    totals_rn, totals_old = {}, {}
    perq_out = {}
    for s in sids35:
        tr, to = 0.0, 0.0
        perq_out[s] = {}
        for q in Q16:
            sc = scores.get(q, {}).get(s, {}).get('score')
            sc = float(sc) if sc is not None else np.nan
            perq_out[s][q] = None if np.isnan(sc) else round(sc, 1)
            if not np.isnan(sc):
                tr += sc * qw_rn[q]
                to += sc * qw[q]
        totals_rn[s] = tr
        totals_old[s] = to

    arr = np.array([totals_rn[s] for s in sids35], float)
    harr = np.array([htot[s] for s in sids35], float)

    m100 = full_metrics(arr, harr, max_score=100.0)
    arr_old = np.array([totals_old[s] for s in sids35], float)
    m96 = full_metrics(arr_old, harr, max_score=s16 * 100.0)

    print('\n=== B1 FEW-LLM 基线（重归一化 100 分制，n=35）===')
    print(f"mean={m100['mean']:.2f} P={m100['P']:.3f} D={m100['D']:.3f} σ={m100['sigma']:.2f} "
          f"r={m100['pearson_r']:.3f} QWK10={m100['qwk_10']:.3f} QWK5={m100['qwk_5']:.3f} MAE={m100['mae']:.2f}")
    print('\n=== 对照（重归一化 100 分制）===')
    ref = json.load(open(RENORM_FILE, encoding='utf-8'))['variants']
    rows = {
        'b1_human_v2': dict(P=0.186, D=0.157, sigma=6.64, mean=81.42),
        'b1_fewllm_new': {k: m100[k] for k in
                          ('P', 'D', 'sigma', 'mean', 'pearson_r', 'qwk_10', 'qwk_5', 'mae')},
        'b1_casg_transfer_rn': ref and None,  # 由 b1_table5_renorm16.json 提供
    }
    b1tab = json.load(open(BASE / 'b1_table5_renorm16.json', encoding='utf-8'))
    print(f"B1 CASG 迁移: {b1tab['transferred']}")
    print(f"B1 CASG 重标定: {b1tab['recalibrated']}")
    print(f"B1 人工: {b1tab['human_v2']}")

    out = dict(
        scope='B1 FEW-LLM (LLM v2 少样本) 基线; 16题全覆盖 35 人; 人工为 v2(补分后)',
        n=len(sids35),
        model_note='deepseek-v4-flash, temp 0.15, 生成上限 3000; 提示词同 grade_v2_improved(B2 基线同款)',
        total_formula='main: Σ LLM_q*qw_rn_q (16题重归一化, 满分100); alt: Σ LLM_q*qw_q (17题归一化, 满分96.47)',
        weight_sum_16q=round(s16, 6),
        metrics_100scale={k: (round(float(m100[k]), 4) if k != 'n' else m100[k])
                          for k in ('n', 'mean', 'P', 'D', 'sigma', 'pearson_r',
                                    'spearman_rho', 'qwk_10', 'qwk_5', 'mae', 'rmse')},
        metrics_96scale={k: (round(float(m96[k]), 4) if k != 'n' else m96[k])
                         for k in ('n', 'mean', 'P', 'D', 'sigma', 'pearson_r',
                                   'spearman_rho', 'qwk_10', 'qwk_5', 'mae', 'rmse')},
        per_student=dict(renorm100={s: round(float(v), 2) for s, v in totals_rn.items()},
                         old96={s: round(float(v), 2) for s, v in totals_old.items()}),
        per_question_score=perq_out,
        comparison={},
    )
    # 与 B1 CASG / 人工并排（100 分制）
    out['comparison'] = {
        'b1_human_v2': b1tab['human_v2'],
        'b1_fewllm': {k: round(float(m100[k]), 3) for k in
                      ('P', 'D', 'sigma', 'mean', 'pearson_r', 'qwk_10', 'mae')},
        'b1_casg_transfer@B2alpha': {k: round(float(b1tab['transferred'][k]), 3)
                                     for k in ('P', 'D', 'sigma', 'mean',
                                               'r', 'qwk10', 'mae')},
        'b1_casg_recalibrated': {k: round(float(b1tab['recalibrated'][k]), 3)
                                 for k in ('P', 'D', 'sigma', 'mean',
                                           'r', 'qwk10', 'mae')},
        'b2_reference': {
            'fewllm_b2_paper': dict(P=0.263, D=0.209, sigma=8.84, mean=73.66,
                                     r=0.613, qwk10=0.524, mae=5.76),
            'casg_b2_renorm': {k: round(float(ref['base_rn']['opt']['pdsigma01'][k]), 3)
                               for k in ('P', 'D', 'sigma', 'mean', 'pearson_r',
                                         'qwk_10', 'mae')},
        },
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n已保存: {OUT_JSON}')


if __name__ == '__main__':
    main()
