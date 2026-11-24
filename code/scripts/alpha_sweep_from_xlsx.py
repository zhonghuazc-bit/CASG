



import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()


ROOT = Path(r'C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育')
XLSX = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'keypoint_evaluations_detail.xlsx'
KP = REPO_ROOT / 'data' / 'raw' / 'keypoint_scores.json'
BASE = REPO_ROOT / 'data' / 'raw' / '基准评分数据_人工与LLM_v2.xlsx'
OUTDIR = REPO_ROOT / 'data' / 'results' / 'casg_experiment'

TH = (1.0, 1.0, 1.0)  


def parse_xlsx():
    
    raw = pd.read_excel(XLSX, sheet_name=None, header=None)
    data = {}
    for qid, df in raw.items():
        if df.shape[0] < 3:
            continue
        
        starts = [ci for ci in range(df.shape[1])
                  if isinstance(df.iloc[0, ci], str) and df.iloc[0, ci].startswith('要点')]
        if not starts:
            continue
        recs = {}
        for ri in range(2, df.shape[0]):
            sidv = df.iloc[ri, 0]
            if pd.isna(sidv):
                continue
            try:
                sid = str(int(sidv)) if float(sidv).is_integer() else str(sidv)
            except (TypeError, ValueError):
                sid = str(sidv).strip()
            kps = []
            for s in starts:
                st = df.iloc[ri, s]
                cov = df.iloc[ri, s + 1]
                band = df.iloc[ri, s + 2]
                status = 'missing' if pd.isna(st) else str(st).strip().lower()
                if status in ('完全覆盖', '满分', 'fullcover'):
                    status = 'full'
                elif status in ('部分覆盖', '部分', 'part', 'partialcover'):
                    status = 'partial'
                elif status in ('未覆盖', '缺失', 'missing', 'none'):
                    status = 'missing'
                coverage = 0.0 if pd.isna(cov) else float(cov)
                b = None if pd.isna(band) else str(band).strip().lower()
                kps.append({'status': status, 'coverage': coverage, 'band': b})
            recs[sid] = kps
        data[qid] = recs
    return data


def load_deps():
    kp = json.load(open(KP, encoding='utf-8'))
    norm = kp['normalized_100']
    qw = kp['question_weights']
    base = pd.read_excel(BASE)
    base['学生学号'] = base['学生学号'].astype(str)
    human = {str(s): float(h) for s, h in zip(base['学生学号'], base['人工总分'])}
    return norm, qw, human


def compute_casg(data, norm, qw, alpha):
    
    th, tm, tl = TH
    per_q = {}
    for qid, recs in data.items():
        ns = norm.get(qid, {})
        n_kp = max((len(kps) for kps in recs.values()), default=0)
        max_list = [ns.get(str(i + 1), 0.0) for i in range(n_kp)]
        sq = {}
        for sid, kps in recs.items():
            tot = 0.0
            for kp_i in range(n_kp):
                kp = kps[kp_i] if kp_i < len(kps) else {'status': 'missing', 'coverage': 0.0, 'band': None}
                ms = max_list[kp_i] if kp_i < len(max_list) else 0.0
                st = kp['status']
                cov = kp['coverage']
                if st == 'full':
                    sc = ms * th
                elif st == 'partial':
                    sc = ms * (cov ** alpha) * tm
                else:
                    sc = ms * 0.05 * tl
                tot += sc
            sq[sid] = tot
        per_q[qid] = sq

    scores = {}
    for qid, sq in per_q.items():
        w = qw.get(qid, 0.0)
        for sid, v in sq.items():
            scores[sid] = scores.get(sid, 0.0) + v * w
    return scores, per_q


def qwk(yt, yp, bins=10):
    yt = np.asarray(yt, float)
    yp = np.asarray(yp, float)
    edges = np.linspace(0, 100, bins + 1)
    a = np.clip(np.digitize(yt, edges) - 1, 0, bins - 1)
    b = np.clip(np.digitize(yp, edges) - 1, 0, bins - 1)
    w = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(bins):
            w[i, j] = ((i - j) ** 2) / ((bins - 1) ** 2)
    O = np.zeros((bins, bins))
    for x, y in zip(a, b):
        O[x, y] += 1
    if O.sum() == 0:
        return 0.0
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    num = (w * O).sum()
    den = (w * E).sum()
    return float(1 - num / den) if den > 0 else 1.0


def metrics_for(scores, human):
    sids = [s for s in scores if s in human and human[s] > 0]
    arr = np.array([scores[s] for s in sids], float)
    harr = np.array([human[s] for s in sids], float)
    n = len(arr)
    mean = arr.mean()
    sigma = arr.std()
    P = 1 - mean / 100
    sd = np.sort(arr)
    t = max(1, int(n * 0.27))
    bt = max(1, int(n * 0.27))
    D = (sd[-t:].mean() - sd[:bt].mean()) / 100
    r, _ = pearsonr(arr, harr)
    rho, _ = spearmanr(arr, harr)
    q10 = qwk(harr, arr, 10)
    q5 = qwk(harr, arr, 5)
    mae = np.abs(arr - harr).mean()
    rmse = np.sqrt(((arr - harr) ** 2).mean())
    return dict(n=int(n), mean=float(mean), P=float(P), D=float(D), sigma=float(sigma),
                pearson_r=float(r), spearman_rho=float(rho),
                qwk_10=float(q10), qwk_5=float(q5), mae=float(mae), rmse=float(rmse))


def main():
    print('解析 xlsx ...')
    data = parse_xlsx()
    print('题数:', len(data))
    norm, qw, human = load_deps()
    for qid in data:
        xlsx_n = max((len(k) for k in data[qid].values()), default=0)
        print(f'  {qid}: 学生数={len(data[qid])}, xlsx要点数={xlsx_n}, norm要点数={len(norm.get(qid, {}))}')

    wsum = sum(qw.get(q, 0.0) for q in data.keys())
    print('对 16 题直接分配权重(权重和=1):', round(wsum, 6))
    print('人工学生数:', len(human))

    alphas = list(np.arange(0.0, 2.001, 0.1))
    rows = []
    student_by_alpha = {}
    for alpha in alphas:
        scores, _ = compute_casg(data, norm, qw, alpha)
        m = metrics_for(scores, human)
        row = dict(alpha=round(float(alpha), 3), **m)
        rows.append(row)
        student_by_alpha[round(float(alpha), 3)] = {s: round(float(v), 4) for s, v in scores.items()}
        print(f"α={alpha:4.2f} | P={m['P']:.3f} D={m['D']:.3f} σ={m['sigma']:5.2f} | "
              f"r={m['pearson_r']:.3f} ρ={m['spearman_rho']:.3f} | "
              f"QWK10={m['qwk_10']:.3f} QWK5={m['qwk_5']:.3f} | MAE={m['mae']:.2f} RMSE={m['rmse']:.2f}")

    from casg_losses import LOSS_FUNCTIONS, resolve_name
    
    main_loss = resolve_name('pdsigma01')
    func = LOSS_FUNCTIONS[main_loss]['func']

    def obj(a):
        sc, _ = compute_casg(data, norm, qw, a)
        mm = metrics_for(sc, human)
        return float(func(mm))

    res = minimize_scalar(obj, bounds=(0.0, 2.0), method='bounded',
                          options={'maxiter': 100, 'xatol': 1e-6})
    sc_opt, _ = compute_casg(data, norm, qw, res.x)
    m_opt = metrics_for(sc_opt, human)
    m_opt['L'] = float(func(m_opt))
    opt_alpha = round(float(res.x), 4)
    opt_row = dict(alpha=opt_alpha, **m_opt, loss=main_loss, note='continuous_optimal (pdsigma01, 论文 3.4)')
    student_by_alpha[opt_alpha] = {s: round(float(v), 4) for s, v in sc_opt.items()}
    print(f"\n连续最优 α={res.x:.4f} (loss={main_loss}) | L={m_opt['L']:.4f} | "
          f"P={m_opt['P']:.3f} D={m_opt['D']:.3f} "
          f"σ={m_opt['sigma']:.2f} r={m_opt['pearson_r']:.3f} ρ={m_opt['spearman_rho']:.3f} "
          f"QWK10={m_opt['qwk_10']:.3f} MAE={m_opt['mae']:.2f}")

    out = {
        'source': 'keypoint_evaluations_detail.xlsx',
        'strategy': 'power (coverage**alpha)',
        'theta': list(TH),
        'loss': main_loss,
        'grid_alphas': [round(float(a), 3) for a in alphas],
        'sweep': rows,
        'continuous_optimal': opt_row,
        'student_scores_by_alpha': student_by_alpha,
    }
    (OUTDIR / 'alpha_sweep_results.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUTDIR / 'alpha_sweep_results.csv', index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(OUTDIR / 'alpha_sweep_results.xlsx') as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name='alpha_sweep', index=False)
        pd.DataFrame([opt_row]).to_excel(xw, sheet_name='continuous_optimal', index=False)
        sdf = pd.DataFrame(student_by_alpha).T.round(4)
        sdf.index.name = 'alpha'
        sdf.to_excel(xw, sheet_name='student_scores_by_alpha')
    print('\n已保存: alpha_sweep_results.json / .csv / .xlsx')


if __name__ == '__main__':
    main()
