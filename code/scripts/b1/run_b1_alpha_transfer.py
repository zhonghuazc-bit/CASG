



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

from alpha_sweep_from_xlsx import compute_casg  
from alpha_per_question_analysis import full_metrics, align, qwk_safe  
from casg_losses import LOSS_FUNCTIONS  

ALPHA_STAR_B2 = 0.825      
LOSS_NAME = 'pdsigma01'    
SEED = 42
SAMPLE_SIZES = [5, 10, 15, 20]
N_REPEATS = 20

QIDS16 = ['HW1_Q1', 'HW1_Q2', 'HW1_Q3',
          'HW2_Q1', 'HW2_Q2', 'HW2_Q3',
          'HW3_Q1', 'HW3_Q2', 'HW3_Q3',
          'HW4_Q2', 'HW4_Q3', 'HW4_Q4', 'HW4_Q5',
          'HW5_Q1', 'HW5_Q2', 'HW5_Q3']

EVAL_FILE = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'casg_b1_evaluations_v2.json'
HUMAN_FILE = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'b1_human_scores.json'
KP_FILE = REPO_ROOT / 'data' / 'raw' / 'keypoint_scores.json'
OUT_JSON = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'casg_b1_alpha_transfer.json'
OUT_XLSX = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'casg_b1_alpha_transfer.xlsx'


def evals_to_data(evals):
    data = {}
    for qid, students in evals.items():
        data[qid] = {}
        for sid, ev in students.items():
            if ev is None:  
                continue
            kps = sorted(ev['keypoints'], key=lambda k: k.get('id', 0))
            data[qid][sid] = [
                dict(status=k.get('status', 'missing'),
                     coverage=k.get('coverage_ratio', 0.0),
                     band=k.get('quality_band', None))
                for k in kps
            ]
    return data


def filter_data_by_students(data, sids):
    
    return {qid: {sid: kps for sid, kps in students.items() if sid in sids}
            for qid, students in data.items()}


def solve_alpha(data, norm, qw, human, loss_name=LOSS_NAME):
    
    func = LOSS_FUNCTIONS[loss_name]['func']
    max_score = sum(qw.values()) * 100.0

    def obj(alpha):
        scores, _ = compute_casg(data, norm, qw, alpha)
        sids = [s for s in scores if s in human]
        if len(sids) < 3:
            return 1e6
        arr = np.array([scores[s] for s in sids], float)
        harr = np.array([human[s] for s in sids], float)
        m = full_metrics(arr, harr, max_score=max_score)
        return float(func(m))

    res = minimize_scalar(obj, bounds=(0.0, 2.0), method='bounded',
                          options={'xatol': 1e-6})
    return float(res.x)


def eval_alpha_on_all(data_all, norm, qw, human, alpha):
    
    scores, _ = compute_casg(data_all, norm, qw, alpha)
    qids = list(data_all.keys())
    full_sids = sorted({sid for q in qids for sid in data_all[q]
                        if all(sid in data_all[q2] for q2 in qids)})
    sids = [s for s in full_sids if s in scores and s in human]
    max_score = sum(qw.values()) * 100.0
    arr = np.array([scores[s] for s in sids], float)
    harr = np.array([human[s] for s in sids], float)
    m = full_metrics(arr, harr, max_score=max_score)
    return m


def main():
    evals = json.load(open(EVAL_FILE, encoding='utf-8'))
    human = json.load(open(HUMAN_FILE, encoding='utf-8'))
    human_tot = {s: v for s, v in human['human_total'].items() if v is not None}
    kp = json.load(open(KP_FILE, encoding='utf-8'))
    norm, qw = kp['normalized_100'], kp['question_weights']

    
    data_all = {q: evals_to_data({q: evals[q]})[q] for q in QIDS16 if q in evals}
    wsum = sum(qw[q] for q in QIDS16)
    qw16 = {q: qw[q] / wsum for q in QIDS16}
    qids = list(data_all.keys())
    full_sids = sorted({sid for q in qids for sid in data_all[q]
                        if all(sid in data_all[q2] for q2 in qids)})
    print(f'B1 评估学生: {sum(len(v) for v in data_all.values())} 人次, '
          f'16 题全覆盖 {len(full_sids)} 人')

    
    alpha_b1 = solve_alpha(filter_data_by_students(data_all, full_sids), norm, qw16, human_tot)
    print(f'B. B1 全样本最优 α*_B1 = {alpha_b1:.4f}  (B2 为 {ALPHA_STAR_B2:.4f})')
    m_b1 = eval_alpha_on_all(data_all, norm, qw16, human_tot, alpha_b1)
    print(f'   其指标: P={m_b1["P"]:.3f}, D={m_b1["D"]:.3f}, σ={m_b1["sigma"]:.2f}, r={m_b1["pearson_r"]:.3f}')

    
    m_transfer = eval_alpha_on_all(data_all, norm, qw16, human_tot, ALPHA_STAR_B2)
    print(f'A. B2 α*={ALPHA_STAR_B2} 直接用于 B1: P={m_transfer["P"]:.3f}, D={m_transfer["D"]:.3f}, '
          f'σ={m_transfer["sigma"]:.2f}, r={m_transfer["pearson_r"]:.3f}')
    delta_alpha_b2b1 = abs(alpha_b1 - ALPHA_STAR_B2)
    print(f'   Δα = |{alpha_b1:.4f} - {ALPHA_STAR_B2:.4f}| = {delta_alpha_b2b1:.4f}')

    
    rng = np.random.default_rng(SEED)
    sampling_results = []
    for n in SAMPLE_SIZES:
        alphas_sampled = []
        perf_sampled = []  
        for rep in range(N_REPEATS):
            sample = rng.choice(full_sids, size=min(n, len(full_sids)), replace=False)
            data_s = filter_data_by_students(data_all, sample)
            human_s = {s: human_tot[s] for s in sample if s in human_tot}
            if len(human_s) < 3:
                continue
            a = solve_alpha(data_s, norm, qw16, human_s)
            alphas_sampled.append(a)
            m = eval_alpha_on_all(data_all, norm, qw16, human_tot, a)
            perf_sampled.append(dict(alpha=a, P=m['P'], D=m['D'], sigma=m['sigma'],
                                     r=m['pearson_r'], qwk10=m['qwk_10'], mae=m['mae']))
        alphas = np.array(alphas_sampled)
        ps = pd.DataFrame(perf_sampled)
        row = dict(
            sample_size=n, n_repeats=len(alphas),
            alpha_mean=round(float(alphas.mean()), 4), alpha_std=round(float(alphas.std()), 4),
            alpha_min=round(float(alphas.min()), 4), alpha_max=round(float(alphas.max()), 4),
            delta_alpha_vs_b1=round(float(abs(alphas.mean() - alpha_b1)), 4),
            delta_alpha_vs_b2=round(float(abs(alphas.mean() - ALPHA_STAR_B2)), 4),
            P_mean=round(float(ps['P'].mean()), 4), P_std=round(float(ps['P'].std()), 4),
            D_mean=round(float(ps['D'].mean()), 4), sigma_mean=round(float(ps['sigma'].mean()), 2),
            r_mean=round(float(ps['r'].mean()), 4), r_std=round(float(ps['r'].std()), 4),
            qwk10_mean=round(float(ps['qwk10'].mean()), 4), mae_mean=round(float(ps['mae'].mean()), 3),
        )
        sampling_results.append(row)
        print(f'C. 采样 n={n}: α_mean={row["alpha_mean"]:.4f}±{row["alpha_std"]:.4f} '
              f'(ΔvsB1={row["delta_alpha_vs_b1"]:.4f}, ΔvsB2={row["delta_alpha_vs_b2"]:.4f}), '
              f'r_mean={row["r_mean"]:.3f}, P_mean={row["P_mean"]:.3f}')

    
    conclusions = []
    conclusions.append(f'B1 全样本最优 α*_B1 = {alpha_b1:.4f}, B2 为 {ALPHA_STAR_B2:.4f}, Δα = {delta_alpha_b2b1:.4f}')
    conclusions.append(f'B2 α 直接迁移: P={m_transfer["P"]:.3f} (目标0.20, 差{abs(m_transfer["P"]-0.20):.3f}), '
                       f'r={m_transfer["pearson_r"]:.3f}, QWK10={m_transfer["qwk_10"]:.3f}, MAE={m_transfer["mae"]:.2f}')
    ok_transfer = delta_alpha_b2b1 <= 0.1 and abs(m_transfer['P'] - 0.20) <= 0.05 and m_transfer['pearson_r'] >= 0.55
    conclusions.append(f'α 跨班可迁移性判定: {"成立 (Δα≤0.1 且 P 距目标≤0.05 且 r≥0.55)" if ok_transfer else "不成立"}')
    if sampling_results:
        n20 = sampling_results[-1]
        ok_sampling = n20['delta_alpha_vs_b1'] <= 0.15 and n20['r_mean'] >= 0.55
        conclusions.append(f'采样标定可行性: n=20 时 α_mean={n20["alpha_mean"]}±{n20["alpha_std"]}, '
                           f'r={n20["r_mean"]}, {"可行" if ok_sampling else "仍需更大样本"}')
        conclusions.append('样本-代价权衡: n=5→20 的 α 标准差: '
                           + ', '.join(f'{r["sample_size"]}人: σα={r["alpha_std"]:.3f}'
                                       for r in sampling_results))
        conclusions.append(f'经济性: 采样{n20["sample_size"]}人仅需 {n20["sample_size"]*len(qids)} 次LLM评估 '
                           f'(vs 全样本 {len(full_sids)*len(qids)} 次), 节省 '
                           f'{(1 - n20["sample_size"]/max(len(full_sids),1))*100:.0f}% 评估量')

    print('\n===== 结论 =====')
    for c in conclusions:
        print(' -', c)

    out = dict(
        alpha_star_b2=ALPHA_STAR_B2, alpha_star_b1=round(alpha_b1, 4),
        delta_alpha_b2b1=round(delta_alpha_b2b1, 4),
        n_complete_subset=len(full_sids),
        scope='B1 16 题整卷, 对 16 题直接分配权重(和=1.0), 满分 100; α ∈ [0,2]; pdsigma01 损失',
        transfer_metrics={k: m_transfer[k] for k in ['P', 'D', 'sigma', 'pearson_r', 'qwk_10', 'qwk_5', 'mae']},
        b1_full_metrics={k: m_b1[k] for k in ['P', 'D', 'sigma', 'pearson_r', 'qwk_10', 'qwk_5', 'mae']},
        sampling_results=sampling_results,
        conclusions=conclusions,
    )
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        pd.DataFrame(sampling_results).to_excel(w, sheet_name='采样标定', index=False)
        pd.DataFrame({'内容': conclusions}).to_excel(w, sheet_name='结论', index=False)
    print('\n已输出:', OUT_JSON)
    print('        ', OUT_XLSX)


if __name__ == '__main__':
    main()