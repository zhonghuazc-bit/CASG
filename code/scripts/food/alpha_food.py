


import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score  

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()


ROOT = Path(r"C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育")
EVAL_DIR = REPO_ROOT / 'data' / 'food_class' / 'coverage_evaluations'
RUBRICS = REPO_ROOT / 'data' / 'food_class' / 'rubrics_食品34要点.json'
ANSWERS = REPO_ROOT / 'data' / 'food_class' / 'student_answers.json'
OUTDIR = REPO_ROOT / 'data' / 'food_class'
QIDS = ["1", "2", "3", "4", "5"]

TH = (1.0, 1.0, 1.0)  
MISSING_FACTOR = 0.05


def load_all():
    
    rub = json.load(open(RUBRICS, encoding="utf-8"))["rubrics"]
    norm = {}
    for qid in QIDS:
        norm[qid] = {str(kp["kp_id"]): float(kp["kp_score"]) for kp in rub[qid]["keypoints"]}

    
    evals = {}
    for qid in QIDS:
        p = EVAL_DIR / f"{qid}_evaluations.json"
        if not p.exists():
            raise FileNotFoundError(f"缺少 {p}")
        evals[qid] = json.load(open(p, encoding="utf-8"))

    
    ans = json.load(open(ANSWERS, encoding="utf-8"))
    human = {s["student_id"]: float(s["human_total_100"]) for s in ans["students"]}
    return norm, evals, human


def compute_casg_all(evals, norm, alpha):
    
    th, tm, tl = TH
    scores, per_q = {}, {}
    for qid in QIDS:
        sq = {}
        for sid, ev in evals[qid].items():
            if ev.get("parse_error"):
                
                sq[sid] = None
                continue
            tot = 0.0
            for kp in ev.get("keypoints", []):
                ms = float(kp.get("max_score", 0))
                st = kp.get("status", "missing")
                cov = float(kp.get("coverage_ratio", 0.0))
                if st == "full":
                    tot += ms * th
                elif st == "partial":
                    
                    if cov <= 0:
                        tot += 0.0
                    else:
                        tot += ms * (cov ** alpha) * tm
                else:
                    tot += ms * MISSING_FACTOR * tl
            sq[sid] = tot
        per_q[qid] = sq

    
    all_sids = set()
    for qid in QIDS:
        all_sids.update(evals[qid].keys())
    for sid in all_sids:
        v = 0.0
        ok = True
        for qid in QIDS:
            x = per_q[qid].get(sid)
            if x is None:
                ok = False
                break
            v += x
        if ok:
            scores[sid] = v
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
    sids = [s for s in scores if s in human]
    arr = np.array([scores[s] for s in sids], float)
    harr = np.array([human[s] for s in sids], float)
    n = len(arr)
    if n == 0:
        return None
    mean = arr.mean()
    sigma = arr.std()
    P = 1 - mean / 100
    sd = np.sort(arr)
    t = max(1, int(n * 0.27))
    D = (sd[-t:].mean() - sd[:t].mean()) / 100
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
    norm, evals, human = load_all()
    print(f"加载: {len(evals)} 题覆盖评估, 人工{len(human)}人")

    
    h = np.array(list(human.values()), float)
    print(f"人工: mean={h.mean():.2f} σ={h.std():.2f} P={1-h.mean()/100:.3f}")
    sd = np.sort(h); t = max(1, int(len(h) * 0.27))
    print(f"       D={(sd[-t:].mean()-sd[:t].mean())/100:.3f}")

    
    alphas = list(np.arange(0.0, 2.001, 0.05))
    rows, student_by_alpha = [], {}
    for alpha in alphas:
        scores, _ = compute_casg_all(evals, norm, alpha)
        m = metrics_for(scores, human)
        row = dict(alpha=round(float(alpha), 3), **m)
        rows.append(row)
        student_by_alpha[round(float(alpha), 3)] = {s: round(float(v), 4) for s, v in scores.items()}
        print(f"α={alpha:4.2f} | P={m['P']:.3f} D={m['D']:.3f} σ={m['sigma']:5.2f} | "
              f"r={m['pearson_r']:.3f} ρ={m['spearman_rho']:.3f} | "
              f"QWK10={m['qwk_10']:.3f} QWK5={m['qwk_5']:.3f} | MAE={m['mae']:.2f}")

    
    def loss_pdsigma01(m):
        return (abs(m['P'] - 0.20) / 0.20 + abs(m['D'] - 0.30) / 0.30 + 0.1 * abs(m['sigma'] - 10) / 10)

    def loss_equal(m):
        return (abs(m['P'] - 0.20) / 0.20 + abs(m['D'] - 0.30) / 0.30 + abs(m['sigma'] - 10) / 10)

    def obj_pdsigma01(a):
        sc, _ = compute_casg_all(evals, norm, a)
        mm = metrics_for(sc, human)
        return loss_pdsigma01(mm) if mm else 1e9

    res = minimize_scalar(obj_pdsigma01, bounds=(0.0, 2.0), method='bounded',
                          options={'maxiter': 80, 'xatol': 0.001})
    sc_opt, px = compute_casg_all(evals, norm, res.x)
    m_opt = metrics_for(sc_opt, human)
    opt_alpha = round(float(res.x), 4)
    opt_row = dict(alpha=opt_alpha, **m_opt, loss_pdsigma01=loss_pdsigma01(m_opt),
                   loss_equal=loss_equal(m_opt), note='continuous_optimal_pdsigma01')
    student_by_alpha[opt_alpha] = {s: round(float(v), 4) for s, v in sc_opt.items()}
    print(f"\n连续最优 α*={opt_alpha} | P={m_opt['P']:.3f} D={m_opt['D']:.3f} σ={m_opt['sigma']:.2f} "
          f"r={m_opt['pearson_r']:.3f} QWK10={m_opt['qwk_10']:.3f} MAE={m_opt['mae']:.2f} "
          f"loss={loss_pdsigma01(m_opt):.4f}")

    
    best = min(rows, key=lambda r: loss_pdsigma01(r))
    print(f"网格最优 α={best['alpha']} loss={loss_pdsigma01(best):.4f}")

    out = {
        'source': 'fooddata coverage_evaluations',
        'strategy': 'power (coverage**alpha), theta=(1,1,1), missing*0.05',
        'loss': 'pdsigma01(论文口径 relative), equal 对照',
        'grid_alphas': [round(float(a), 3) for a in alphas],
        'sweep': rows,
        'continuous_optimal': opt_row,
        'grid_optimal': {**best, 'loss_pdsigma01': loss_pdsigma01(best), 'loss_equal': loss_equal(best)},
        'student_scores_by_alpha': student_by_alpha,
    }
    (OUTDIR / 'casg_food_alpha_sweep.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUTDIR / 'casg_food_alpha_sweep.csv', index=False, encoding='utf-8-sig')
    with pd.ExcelWriter(OUTDIR / 'casg_food_alpha_sweep.xlsx') as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name='sweep', index=False)
        pd.DataFrame([opt_row]).to_excel(xw, sheet_name='optimal', index=False)
    print(f"\n写盘: casg_food_alpha_sweep.json/.csv/.xlsx")


if __name__ == "__main__":
    main()