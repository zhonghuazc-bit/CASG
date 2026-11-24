



import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


ALPHA_STAR = 0.825
T_P, T_D, T_S = 0.20, 0.30, 10.0


QIDS16 = ['HW1_Q1', 'HW1_Q2', 'HW1_Q3',
          'HW2_Q1', 'HW2_Q2', 'HW2_Q3',
          'HW3_Q1', 'HW3_Q2', 'HW3_Q3',
          'HW4_Q2', 'HW4_Q3', 'HW4_Q4', 'HW4_Q5',
          'HW5_Q1', 'HW5_Q2', 'HW5_Q3']

EVAL_FILE = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'casg_b1_evaluations_v2.json'
HUMAN_FILE = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'b1_human_scores.json'
B2_RESULT = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'casg_table234.json'
OUT_JSON = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'casg_b1_scoring_report.json'
OUT_XLSX = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'casg_b1_scoring_report.xlsx'


def load_deps():
    kp = json.load(open(ROOT / 'normalin' / 'data' / 'processed' / 'keypoint_scores.json', encoding='utf-8'))
    norm, qw = kp['normalized_100'], kp['question_weights']
    human = json.load(open(HUMAN_FILE, encoding='utf-8'))
    b1_human_tot = human['human_total']
    b1_human_pq = human['per_question_human']
    return norm, qw, b1_human_tot, b1_human_pq


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


def metrics_block(name, arr, harr=None, max_score=100.0):
    
    arr = np.asarray(arr, float)
    n = len(arr)
    if n < 3:
        return dict(name=name, n=n, P=None, D=None, sigma=None, mean=None,
                    pearson_r=None, qwk_10=None, qwk_5=None, mae=None)
    P = 1 - arr.mean() / max_score
    sd = np.sort(arr)
    k = max(1, int(n * 0.27))
    D = (sd[-k:].mean() - sd[:k].mean()) / max_score
    sigma = float(arr.std())
    out = dict(name=name, n=n, P=round(P, 4), D=round(D, 4), sigma=round(sigma, 4),
               mean=round(float(arr.mean()), 2))
    if harr is not None:
        harr = np.asarray(harr, float)
        
        if len(arr) == len(harr) and arr.std() > 1e-9 and harr.std() > 1e-9:
            from scipy.stats import pearsonr
            r, _ = pearsonr(arr, harr)
            q10 = qwk_safe(harr, arr, 10)
            q5 = qwk_safe(harr, arr, 5)
            mae = float(np.abs(arr - harr).mean())
            out.update(pearson_r=round(float(r), 4), qwk_10=round(q10, 4),
                       qwk_5=round(q5, 4), mae=round(mae, 3))
        else:
            out.update(pearson_r=None, qwk_10=None, qwk_5=None, mae=None)
    return out


def main():
    evals = json.load(open(EVAL_FILE, encoding='utf-8'))
    norm, qw, human_tot, human_pq = load_deps()
    b2 = json.load(open(B2_RESULT, encoding='utf-8'))
    b2_a = b2['results']['opt']['pdsigma01']['alpha_star']

    n_ev = sum(len(v) for v in evals.values())
    print(f'加载评估: {n_ev} 条, 人工: {len([v for v in human_tot.values() if v is not None])} 人')

    
    data = {q: evals_to_data({q: evals[q]})[q] for q in QIDS16 if q in evals}
    wsum = sum(qw[q] for q in QIDS16)
    qw16 = {q: qw[q] / wsum for q in QIDS16}

    scores, per_q = compute_casg(data, norm, qw16, ALPHA_STAR)
    print(f'CASG 打分: {len(scores)} 名学生, 使用 α*={ALPHA_STAR}（B2 最优，见 casg_table234.py）')

    
    rows = []
    for qid in QIDS16:
        if qid not in per_q or not per_q[qid]:
            rows.append(dict(name=f'B1_{qid}', n=0, P=None, D=None, sigma=None,
                             pearson_r=None, qwk_10=None, qwk_5=None, mae=None))
            continue
        arr = np.array([per_q[qid][s] for s in per_q[qid]], float)
        sid_both = [s for s in per_q[qid]
                    if human_pq.get(s, {}).get(qid) is not None]
        h_arr = np.array([human_pq[s][qid] for s in sid_both], float) if sid_both else None
        arr_both = np.array([per_q[qid][s] for s in sid_both], float) if sid_both else None
        m = metrics_block(f'B1_{qid}', arr_both, harr=h_arr)
        m['n_human_align'] = len(sid_both)
        rows.append(m)

    
    full_sids = sorted({sid for q in data for sid in data[q]
                        if all(sid in data[q2] for q2 in data)})
    b1_max = sum(qw16.values()) * 100.0
    sid_hum = [s for s in full_sids if s in scores and human_tot.get(s) is not None]
    arr_tot = np.array([scores[s] for s in sid_hum], float)
    h_tot = np.array([human_tot[s] for s in sid_hum], float)
    tot_row = metrics_block('B1_整卷(16题,全覆盖子样本)', arr_tot, harr=h_tot, max_score=b1_max)
    rows.append(tot_row)
    print(f'整卷子样本: 16 题全覆盖且有人工分 {len(sid_hum)} 人')

    df = pd.DataFrame(rows)
    for col in ['P', 'D', 'sigma']:
        target = T_P if col == 'P' else T_D if col == 'D' else T_S
        df[f'{col}_gap_target'] = df[col].apply(
            lambda v: round(abs(v - target), 4) if v is not None else None)

    
    conclusions = []
    conclusions.append(f'B1 整卷(16题, n={tot_row["n"]}) CASG@α*={ALPHA_STAR}（B2 最优 {b2_a:.4f} 迁移）: '
                       f'P={tot_row["P"]} (目标0.20, 差{abs(tot_row["P"] - T_P):.3f}), '
                       f'D={tot_row["D"]} (目标0.30), σ={tot_row["sigma"]} (目标10, 差{abs(tot_row["sigma"] - T_S):.2f})')
    if tot_row.get('pearson_r') is not None:
        conclusions.append(f'B1 CASG vs 人工: Pearson r={tot_row["pearson_r"]}, QWK10={tot_row["qwk_10"]}, '
                           f'QWK5={tot_row["qwk_5"]}, MAE={tot_row["mae"]} (n={tot_row["n"]})')
        ok_r = tot_row['pearson_r'] is not None and tot_row['pearson_r'] >= 0.60
        ok_q = tot_row['qwk_10'] is not None and tot_row['qwk_10'] >= 0.50
        conclusions.append(f'人工一致性判定: r {"达标≥0.60" if ok_r else "未达标<0.60"}, '
                           f'QWK10 {"达标≥0.50" if ok_q else "未达标<0.50"}')
        print('\n'.join(conclusions))
    else:
        conclusions.append('CASG 与人工一致性无法计算（数据集为空）')

    out = dict(
        alpha_star=ALPHA_STAR,
        alpha_star_b2=b2_a,
        n_students_scored=len(scores),
        n_evaluations=n_ev,
        n_complete_subset=len(sid_hum),
        scope='B1 16 题整卷, 对 16 题直接分配权重(和=1.0), 满分 100, 与论文表 5 口径一致',
        b1_metrics=rows,
        conclusions=conclusions,
    )
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='B1逐题+整卷', index=False)
        pd.DataFrame({'内容': conclusions}).to_excel(w, sheet_name='结论', index=False)
    print('\n已输出:')
    print(' ', OUT_JSON)
    print(' ', OUT_XLSX)
    print('\n===== 逐题汇总 =====')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()