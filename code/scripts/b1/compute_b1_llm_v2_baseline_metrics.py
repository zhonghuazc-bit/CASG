



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


ROOT = Path(r'C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育')
sys.path.insert(0, str(ROOT / 'scripts'))

from alpha_per_question_analysis import full_metrics  

Q16 = ['HW1_Q1', 'HW1_Q2', 'HW1_Q3', 'HW2_Q1', 'HW2_Q2', 'HW2_Q3',
       'HW3_Q1', 'HW3_Q2', 'HW3_Q3', 'HW4_Q2', 'HW4_Q3', 'HW4_Q4', 'HW4_Q5',
       'HW5_Q1', 'HW5_Q2', 'HW5_Q3']
BASE = REPO_ROOT / 'data' / 'results' / 'b1_class'
SCORE_FILE = BASE / 'b1_llm_v2_scores.json'
HUMAN_FILE = BASE / 'b1_human_scores.json'
KP_FILE = REPO_ROOT / 'data' / 'raw' / 'keypoint_scores.json'
OUT_JSON = BASE / 'b1_llm_v2_baseline_metrics.json'


def load_human():
    human = json.load(open(HUMAN_FILE, encoding='utf-8'))
    return human['human_total']


def main():
    kp = json.load(open(KP_FILE, encoding='utf-8'))
    qw = kp['question_weights']
    s16 = sum(qw[q] for q in Q16)
    qw16 = {q: qw[q] / s16 for q in Q16}

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
    totals = {}
    perq_out = {}
    for s in sids35:
        t = 0.0
        perq_out[s] = {}
        for q in Q16:
            sc = scores.get(q, {}).get(s, {}).get('score')
            sc = float(sc) if sc is not None else np.nan
            perq_out[s][q] = None if np.isnan(sc) else round(sc, 1)
            if not np.isnan(sc):
                t += sc * qw16[q]
        totals[s] = t

    arr = np.array([totals[s] for s in sids35], float)
    harr = np.array([htot[s] for s in sids35], float)
    m100 = full_metrics(arr, harr, max_score=100.0)

    print('\n=== B1 FEW-LLM 基线（100 分制，n=35）===')
    print(f"mean={m100['mean']:.2f} P={m100['P']:.3f} D={m100['D']:.3f} σ={m100['sigma']:.2f} "
          f"r={m100['pearson_r']:.3f} QWK10={m100['qwk_10']:.3f} QWK5={m100['qwk_5']:.3f} MAE={m100['mae']:.2f}")

    out = dict(
        scope='B1 FEW-LLM (LLM v2 少样本) 基线; 16题全覆盖 35 人',
        n=len(sids35),
        model_note='deepseek-v4-flash, temp 0.15, 生成上限 3000; 提示词同 grade_v2_improved(B2 基线同款)',
        total_formula='total = Σ_q LLM_q × w_q （对 16 题直接分配权重，权重和 = 1，满分 100）',
        weight_sum_16q=round(s16, 6),
        metrics_100scale={k: (round(float(m100[k]), 4) if k != 'n' else m100[k])
                          for k in ('n', 'mean', 'P', 'D', 'sigma', 'pearson_r',
                                    'spearman_rho', 'qwk_10', 'qwk_5', 'mae', 'rmse')},
        per_student={s: round(float(v), 2) for s, v in totals.items()},
        per_question_score=perq_out,
    )
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n已保存: {OUT_JSON}')


if __name__ == '__main__':
    main()