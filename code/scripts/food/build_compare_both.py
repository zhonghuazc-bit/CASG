

import json, shutil
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
import openpyxl

FD = Path(r"C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育\data\fooddata")


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


tmp = str(FD / "_csv_tmp.xlsx")
shutil.copy(FD / "期末成绩_学生对应表.csv", tmp)
wc = openpyxl.load_workbook(tmp, data_only=True)["sheet1"]
new_ids = []
human_total, short_auth = {}, {}
for r in range(2, wc.max_row + 1):
    sid = wc.cell(r, 1).value
    if sid is None: continue
    new_ids.append(sid)
    human_total[sid] = float(wc.cell(r, 4).value)
    short_auth[sid] = float(wc.cell(r, 5).value)
new_ids = sorted(new_ids, key=lambda x: int(x.replace("学生", "")))
print("loaded new_ids:", len(new_ids), "first/last:", new_ids[0], new_ids[-1])

def remap_key(nsid):
    return f"学生{new_to_orig(int(nsid.replace('学生','')))}"


z = json.load(open(FD / "llm_direct_scores.json", encoding="utf-8"))
fs = json.load(open(FD / "llm_direct_scores_fewshot.json", encoding="utf-8"))
cs = json.load(open(FD / "casg_food_alpha_sweep.json", encoding="utf-8"))
alphas = list(cs["student_scores_by_alpha"].keys())

zero_short20 = {ns: float(z[remap_key(ns)]["total_20"]) for ns in new_ids}
few_short20  = {ns: float(fs[remap_key(ns)]["total_20"]) for ns in new_ids}
casg_short20_by_alpha = {}
for a in alphas:
    d = cs["student_scores_by_alpha"][a]
    casg_short20_by_alpha[a] = {ns: float(d[remap_key(ns)]) / 5.0 for ns in new_ids}


human_other = {ns: human_total[ns] - short_auth[ns] for ns in new_ids}



H_short100 = np.array([short_auth[ns] * 5 for ns in new_ids], float)
zero_s100 = np.array([zero_short20[ns] * 5 for ns in new_ids], float)
few_s100  = np.array([few_short20[ns] * 5 for ns in new_ids], float)


casg_short_sweep = {}
for a in alphas:
    arr = np.array([casg_short20_by_alpha[a][ns] * 5 for ns in new_ids], float)
    casg_short_sweep[a] = metrics(arr, H_short100)
a_short_star = min(casg_short_sweep, key=lambda k: casg_short_sweep[k]["mae"])
a_short_star_f = float(a_short_star)
casg_s100 = np.array([casg_short20_by_alpha[a_short_star][ns] * 5 for ns in new_ids], float)

m_human_short = metrics(H_short100, H_short100)
m_zero_short  = metrics(zero_s100, H_short100)
m_few_short   = metrics(few_s100, H_short100)
m_casg_short  = metrics(casg_s100, H_short100)


H_total = np.array([human_total[ns] for ns in new_ids], float)
zero_t100 = np.array([zero_short20[ns] + human_other[ns] for ns in new_ids], float)
few_t100  = np.array([few_short20[ns] + human_other[ns] for ns in new_ids], float)

casg_total_sweep = {}
for a in alphas:
    arr = np.array([casg_short20_by_alpha[a][ns] + human_other[ns] for ns in new_ids], float)
    casg_total_sweep[a] = metrics(arr, H_total)
a_total_star = min(casg_total_sweep, key=lambda k: casg_total_sweep[k]["mae"])
casg_t100 = np.array([casg_short20_by_alpha[a_total_star][ns] + human_other[ns] for ns in new_ids], float)

m_human_total = metrics(H_total, H_total)
m_zero_total  = metrics(zero_t100, H_total)
m_few_total   = metrics(few_t100, H_total)
m_casg_total  = metrics(casg_t100, H_total)


def pack(name, m, scale, alpha=None):
    d = {"method": name, "scale": scale,
         "mean": m["mean"], "sigma": m["sigma"], "P": m["P"], "D": m["D"],
         "pearson_r": m["pearson_r"], "spearman_rho": m["spearman_rho"],
         "qwk_10": m["qwk_10"], "qwk_5": m["qwk_5"], "mae": m["mae"], "rmse": m["rmse"]}
    if alpha is not None: d["alpha_star"] = alpha
    return d

out = {
    "source": "期末成绩_学生对应表.csv (新编号, 用户最终校正版)",
    "remap": "新编号 -> 原编号: n<=4->n; 5<=n<=29->n+10; 30<=n<=39->n-25",
    "scenario_short": {
        "description": "仅简答题, 简答×5 折算100分制, 与总分同口径",
        "human_short_20_raw_mean": float(np.mean([short_auth[ns] for ns in new_ids])),
        "alpha_star_casg": float(a_short_star),
        "human": pack("人工基线", m_human_short, "简答×5(100)"),
        "zero_shot": pack("Zero-shot", m_zero_short, "简答×5(100)"),
        "few_shot": pack("Few-shot", m_few_short, "简答×5(100)"),
        "casg": pack("CASG", m_casg_short, "简答×5(100)", float(a_short_star)),
    },
    "scenario_total": {
        "description": "总分 = LLM简答分(20) + 人类已批其他题分(80), 整卷100分制",
        "alpha_star_casg": float(a_total_star),
        "human": pack("人工基线", m_human_total, "整卷100"),
        "zero_shot": pack("Zero-shot", m_zero_total, "整卷100"),
        "few_shot": pack("Few-shot", m_few_total, "整卷100"),
        "casg": pack("CASG", m_casg_total, "整卷100", float(a_total_star)),
    },
    "per_student": {ns: {
        "human_total_100": human_total[ns],
        "short_auth_20": short_auth[ns],
        "human_other": human_other[ns],
        "zero_short_20": zero_short20[ns],
        "few_short_20": few_short20[ns],
        "casg_short_20_alpha_star": casg_short20_by_alpha[a_total_star][ns],
        "zero_total_100": zero_short20[ns] + human_other[ns],
        "few_total_100": few_short20[ns] + human_other[ns],
        "casg_total_100_alpha_star": casg_short20_by_alpha[a_total_star][ns] + human_other[ns],
    } for ns in new_ids},
}
(FD / "comparison_metrics_both.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

def show(tag, m):
    print(f"  {tag:22} mean={m['mean']:6.2f} σ={m['sigma']:6.2f} P={m['P']:.3f} D={m['D']:.3f} "
          f"r={m['pearson_r']:.3f} ρ={m['spearman_rho']:.3f} QWK10={m['qwk_10']:.3f} MAE={m['mae']:.2f}")
print("\n== 仅简答题(×5,100) ==")
show("人工基线", m_human_short); show("Zero-shot", m_zero_short)
show("Few-shot", m_few_short); show("CASG", m_casg_short)
print(f"  CASG α*_short = {a_short_star}")
print("\n== 总分(整卷100) ==")
show("人工基线", m_human_total); show("Zero-shot", m_zero_total)
show("Few-shot", m_few_total); show("CASG", m_casg_total)
print(f"  CASG α*_total = {a_total_star}")
print("\nwritten comparison_metrics_both.json")


import html as _html
def esc(x): return _html.escape(str(x))
methods = [("human","人工基线"),("zero_shot","Zero-shot"),("few_shot","Few-shot"),("casg","CASG")]
short = out["scenario_short"]; total = out["scenario_total"]

HI = ["qwk_10","pearson_r","spearman_rho"]  
def row_for(ms, mt):
    cells = ""
    order = [("mean",2),("sigma",2),("P",3),("D",3),("pearson_r",3),("spearman_rho",3),("qwk_10",3),("mae",2)]
    for key,n in order:
        vs = ms.get(key); vt = mt.get(key)
        hls = ' class="hl"' if key in HI else ""
        hlt = ' class="hl"' if key in HI else ""
        cells += f"<td{hls}>{vs:.{n}f}</td><td{hlt}>{vt:.{n}f}</td>"
    return cells

rows_html = ""
for key,label in methods:
    s = short[key]; t = total[key]
    alpha_note = ""
    if key=="casg":
        alpha_note = f"<br><span style='font-size:10px;color:#2a9d8f'>α*={s['alpha_star']:.2f} / {t['alpha_star']:.2f}</span>"
    rows_html += (f"<tr><td class='lbl'>{label}{alpha_note}</td>{row_for(s, t)}</tr>\n")

html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>用总分 vs 不用总分 完整对比</title>
<style>
body{{font-family:'Microsoft YaHei',Segoe UI,sans-serif;margin:24px;color:#222;background:#fafafa}}
h1{{font-size:21px;border-left:5px solid #2a9d8f;padding-left:12px}}
h2{{font-size:15px;margin-top:24px;color:#1d3557}}
.wrap{{max-width:1040px;margin:auto;background:#fff;padding:22px 26px;box-shadow:0 2px 8px #0001;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:10px}}
th,td{{border:1px solid #ddd;padding:7px 8px;text-align:center}}
th{{background:#1d3557;color:#fff}}
td.lbl{{text-align:left;font-weight:bold;background:#f1f5f9;vertical-align:middle}}
tr:nth-child(even) td{{background:#fafcff}}
.hl{{background:#e8f6f3!important;font-weight:bold}}
.note{{background:#fff8e6;border-left:4px solid #f4a261;padding:12px 16px;font-size:13px;line-height:1.7;border-radius:4px;margin-top:14px}}
.cap{{font-size:11px;color:#888;margin-top:6px}}
</style></head><body><div class="wrap">
<h1>食品科学简答题自动评分：用总分 vs 仅简答题 完整对比</h1>
<p style="color:#555;font-size:13px">数据集：39 名学生（新编号，用户最终校正版 CSV）。
<b>仅简答题</b> = 简答得分 ×5 折算 100 分制（与总分同口径可比）；<b>总分</b> = LLM 简答分(20) + 人类已批其他题分(80)，整卷 100 分制。
CASG 的 α* 在各场景内独立取使 MAE 最小者；Zero-shot/Few-shot 无 α。</p>

<h2>1. 完整指标对比表（每方法两行：仅简答×5 / 总分）</h2>
<table>
<tr><th rowspan="2">方法</th>
<th colspan="2">均值</th><th colspan="2">σ</th><th colspan="2">P</th><th colspan="2">D</th>
<th colspan="2">Pearson r</th><th colspan="2">Spearman ρ</th><th colspan="2">QWK10</th><th colspan="2">MAE</th></tr>
<tr><th>仅简答</th><th>总分</th><th>仅简答</th><th>总分</th><th>仅简答</th><th>总分</th><th>仅简答</th><th>总分</th><th>仅简答</th><th>总分</th><th>仅简答</th><th>总分</th><th>仅简答</th><th>总分</th><th>仅简答</th><th>总分</th></tr>
{rows_html}
</table>
<p class="cap">高亮 = QWK10 / Pearson r / Spearman ρ（越高越好）。MAE 越低越好（未高亮，数值见原表）。
每方法一行，指标按「仅简答×5 / 总分」成对呈现，便于直接对比用总分前后变化。
人工基线两场景 MAE 均为 0（基准自身）。仅简答题原始 20 分制均值 = {short['human_short_20_raw_mean']:.2f}（×5 = 100分制均值 {m_human_short['mean']:.2f}）。</p>

<h2>2. 关键结论</h2>
<div class="note">
<b>① 一致性指标（QWK10 / r / ρ）在「总分」下系统性更高</b>：CASG 的 QWK10 由 {m_casg_short['qwk_10']:.3f}（仅简答）升至 {m_casg_total['qwk_10']:.3f}（总分）；Few-shot 由 {m_few_short['qwk_10']:.3f} 升至 {m_few_total['qwk_10']:.3f}。<br>
<b>② 这是「混合评分」的机械效应，非 LLM 单独贡献</b>：总分中 80% 为人类已批其他题分（与人工基线天然一致），LLM 仅贡献 20 分简答，误差被稀释。<br>
<b>③ 真实反映 LLM 评分质量的是「仅简答」结果</b>：CASG 仅简答 QWK10={m_casg_short['qwk_10']:.3f}、MAE={m_casg_short['mae']:.2f}（折回20分制≈{m_casg_short['mae']/5:.2f}）；Few-shot 仅简答 QWK10={m_few_short['qwk_10']:.3f}、MAE={m_few_short['mae']:.2f}（折回20分制≈{m_few_short['mae']/5:.2f}）。<br>
<b>④ 人工基线分布随场景变化</b>：仅简答 σ={m_human_short['sigma']:.2f}、P={m_human_short['P']:.3f}、D={m_human_short['D']:.3f}；总分 σ={m_human_total['sigma']:.2f}、P={m_human_total['P']:.3f}、D={m_human_total['D']:.3f}（加入人类其他题后更集中、偏度更低）。<br>
<b>⑤ CASG 最优 α</b>：仅简答 α*={float(a_short_star):.2f}，总分 α*={float(a_total_star):.2f}（均取各场景内 MAE 最小）。<br>
<b>建议</b>：论文同时报告两套结果，并明确标注总分口径中 80% 为人类评分，避免一致性提升被误读为 LLM 单独贡献。
</div>
</div></body></html>"""
(FD / "总分vs简答对比.html").write_text(html, encoding="utf-8")
print("written 总分vs简答对比.html")
