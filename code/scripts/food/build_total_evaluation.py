


import json, shutil
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
import openpyxl

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()


ROOT = Path(r"C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育")
FD = REPO_ROOT / 'data' / 'food_class'


def new_to_orig(n):
    if n <= 4: return n
    if 5 <= n <= 29: return n + 10
    return n - 25


def qwk(yt, yp, bins=10):
    yt = np.asarray(yt, float); yp = np.asarray(yp, float)
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
    if O.sum() == 0: return 0.0
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    num = (w * O).sum(); den = (w * E).sum()
    return float(1 - num / den) if den > 0 else 1.0

def metrics(arr, harr):
    arr = np.asarray(arr, float); harr = np.asarray(harr, float)
    n = len(arr); mean = arr.mean(); sigma = arr.std()
    P = 1 - mean / 100.0
    sd = np.sort(arr); t = max(1, int(n * 0.27))
    D = (sd[-t:].mean() - sd[:t].mean()) / 100.0
    r, _ = pearsonr(arr, harr)
    rho, _ = spearmanr(arr, harr)
    mae = np.abs(arr - harr).mean()
    rmse = np.sqrt(((arr - harr) ** 2).mean())
    return dict(n=int(n), mean=float(mean), sigma=float(sigma), P=float(P), D=float(D),
                pearson_r=float(r), spearman_rho=float(rho),
                qwk_10=float(qwk(harr, arr, 10)), qwk_5=float(qwk(harr, arr, 5)),
                mae=float(mae), rmse=float(rmse))

def loss(m):
    return (abs(m['P'] - 0.20)/0.20 + abs(m['D'] - 0.30)/0.30 + 0.1*abs(m['sigma'] - 10)/10)


tmp = str(FD / "_csv_tmp.xlsx")
shutil.copy(FD / "期末成绩_学生对应表.csv", tmp)
wc = openpyxl.load_workbook(tmp, data_only=True)["sheet1"]
new_ids = []
human_total, short_auth, human_other = {}, {}, {}
for r in range(2, wc.max_row + 1):
    sid = wc.cell(r, 1).value
    if sid is None: continue
    new_ids.append(sid)
    human_total[sid] = float(wc.cell(r, 4).value)
    short_auth[sid] = float(wc.cell(r, 5).value)
    human_other[sid] = human_total[sid] - short_auth[sid]
new_ids = sorted(new_ids, key=lambda x: int(x.replace("学生", "")))


fs = json.load(open(FD / "llm_direct_scores_fewshot.json", encoding="utf-8"))
fewshot_short20 = {nsid: float(fs[f"学生{new_to_orig(int(nsid.replace('学生','')))}"]["total_20"])
                   for nsid in new_ids}


cs = json.load(open(FD / "casg_food_alpha_sweep.json", encoding="utf-8"))
alphas = list(cs["student_scores_by_alpha"].keys())
casg_short20_by_alpha = {}
for a in alphas:
    d = cs["student_scores_by_alpha"][a]
    casg_short20_by_alpha[a] = {
        nsid: float(d[f"学生{new_to_orig(int(nsid.replace('学生','')))}"]) / 5.0
        for nsid in new_ids}


fewshot_total = {nsid: fewshot_short20[nsid] + human_other[nsid] for nsid in new_ids}
casg_total_by_alpha = {
    a: {nsid: casg_short20_by_alpha[a][nsid] + human_other[nsid] for nsid in new_ids}
    for a in alphas}

H = np.array([human_total[s] for s in new_ids], float)


m_fs = metrics(np.array([fewshot_total[s] for s in new_ids]), H)


casg_sweep_rows = []
for a in alphas:
    arr = np.array([casg_total_by_alpha[a][s] for s in new_ids])
    m = metrics(arr, H)
    m["alpha"] = float(a)
    m["loss_pdsigma01"] = loss(m)
    casg_sweep_rows.append(m)
best = min(casg_sweep_rows, key=lambda r: r["loss_pdsigma01"])
alpha_star = best["alpha"]

best_key = next(k for k in casg_total_by_alpha if abs(float(k) - alpha_star) < 1e-9)
arr_best = np.array([casg_total_by_alpha[best_key][s] for s in new_ids])
m_casg_raw = metrics(arr_best, H)


out = {
    "construction": {
        "scale": "100分制总分 = 20分制简答 + 人类已批其他题总分",
        "human_other_def": "human_other = 期末成绩(整卷100) - 简答题总分(20_校准后)",
        "note": "CASG 统一用总分层面求解的 α*_total，不做额外的分布校准；few-shot 与人工基线均不校准。",
        "few_shot_calibrated": False,
        "casg_calibrated": False,
        "human_calibrated": False,
    },
    "alpha_resweep_total": {
        "alpha_star": float(alpha_star),
        "best_loss": float(best["loss_pdsigma01"]),
        "best_metrics": {k: best[k] for k in ["n","mean","P","D","sigma","pearson_r","spearman_rho","qwk_10","qwk_5","mae","rmse"]},
        "sweep": [{k: r[k] for k in ["alpha","mean","P","D","sigma","pearson_r","spearman_rho","qwk_10","mae","loss_pdsigma01"]} for r in casg_sweep_rows],
    },
    "human": metrics(H, H),
    "few_shot_total": m_fs,
    "casg_total": m_casg_raw,
    "per_student": {
        nsid: {
            "human_total_100": human_total[nsid],
            "short_auth_20": short_auth[nsid],
            "human_other": human_other[nsid],
            "fewshot_short_20": fewshot_short20[nsid],
            "fewshot_total_100": fewshot_total[nsid],
            "casg_short_20_alpha_star": casg_short20_by_alpha[best_key][nsid],
            "casg_total_100_alpha_star": float(arr_best[new_ids.index(nsid)]),
        } for nsid in new_ids
    },
}
(FD / "comparison_metrics_total.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def show(name, m):
    print(f"{name:24} mean={m['mean']:6.2f} σ={m['sigma']:6.2f} P={m['P']:.3f} D={m['D']:.3f} "
          f"r={m['pearson_r']:.3f} ρ={m['spearman_rho']:.3f} QWK10={m['qwk_10']:.3f} MAE={m['mae']:.2f}")

print("\n===== 总分(100分制) 评估结果 =====")
show("人工基线(human)", out["human"])
show("few-shot(不校准)", m_fs)
show("CASG 总分(α*_total)", m_casg_raw)
print(f"\nCASG 总分层面 α* = {alpha_star}  (loss={best['loss_pdsigma01']:.4f}, best_key={best_key})")
print("注: 仅求解 α*_total，未做额外分布校准。写盘: comparison_metrics_total.json")
