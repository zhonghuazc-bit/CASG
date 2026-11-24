

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
    return dict(mean=float(mean), sigma=float(sigma), P=float(P), D=float(D),
                pearson_r=float(r), spearman_rho=float(rho),
                qwk_10=float(qwk(harr, arr, 10)), mae=float(mae), rmse=float(np.sqrt(((arr-harr)**2).mean())))

def loss_pdsigma01(m):
    return abs(m["P"]-0.2)/0.2 + abs(m["D"]-0.3)/0.3 + 0.1*abs(m["sigma"]-10)/10
def loss_equal(m):
    return abs(m["P"]-0.2)/0.2 + abs(m["D"]-0.3)/0.3 + abs(m["sigma"]-10)/10


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
def remap_key(nsid): return f"学生{new_to_orig(int(nsid.replace('学生','')))}"
human_other = {ns: human_total[ns] - short_auth[ns] for ns in new_ids}


cs = json.load(open(FD / "casg_food_alpha_sweep.json", encoding="utf-8"))
sba = cs["student_scores_by_alpha"]
alphas = sorted(sba.keys(), key=lambda a: float(a))


def scenario_vecs(alpha):
    d = sba[alpha]
    short100 = np.array([float(d[remap_key(ns)]) for ns in new_ids], float)         
    total100 = np.array([float(d[remap_key(ns)]) / 5.0 + human_other[ns] for ns in new_ids], float)
    return short100, total100

H_short = np.array([short_auth[ns] * 5 for ns in new_ids], float)
H_total = np.array([human_total[ns] for ns in new_ids], float)

rows_short, rows_total = [], []
for a in alphas:
    s100, t100 = scenario_vecs(a)
    ms = metrics(s100, H_short)
    mt = metrics(t100, H_total)
    rows_short.append(dict(alpha=float(a), **ms,
                           loss_pdsigma01=loss_pdsigma01(ms), loss_equal=loss_equal(ms)))
    rows_total.append(dict(alpha=float(a), **mt,
                           loss_pdsigma01=loss_pdsigma01(mt), loss_equal=loss_equal(mt)))


def best(rows, key):
    return min(rows, key=lambda r: r[key])
opt = {
    "short_loss_p": best(rows_short, "loss_pdsigma01")["alpha"],
    "short_loss_e": best(rows_short, "loss_equal")["alpha"],
    "short_mae":    best(rows_short, "mae")["alpha"],
    "total_loss_p": best(rows_total, "loss_pdsigma01")["alpha"],
    "total_loss_e": best(rows_total, "loss_equal")["alpha"],
    "total_mae":    best(rows_total, "mae")["alpha"],
}


chk = cs["sweep"][0]
rs0 = rows_short[alphas.index("0.0")]
assert abs(rs0["P"]-chk["P"])<1e-6 and abs(rs0["mae"]-chk["mae"])<1e-6, "short 校验不符"
print("validate short α=0: P=%.4f mae=%.4f (file sweep P=%.4f mae=%.4f) OK" %
      (rs0["P"], rs0["mae"], chk["P"], chk["mae"]))
print("最优α -> 仅简答: loss_p=%.4f loss_e=%.4f mae=%.4f | 总分: loss_p=%.4f loss_e=%.4f mae=%.4f"
      % (opt["short_loss_p"], opt["short_loss_e"], opt["short_mae"],
         opt["total_loss_p"], opt["total_loss_e"], opt["total_mae"]))


out = {"loss_def": {
            "loss_pdsigma01": "|P-0.2|/0.2 + |D-0.3|/0.3 + 0.1|σ-10|/10",
            "loss_equal": "|P-0.2|/0.2 + |D-0.3|/0.3 + |σ-10|/10"},
       "optimal_alpha": opt,
       "short_sweep": rows_short, "total_sweep": rows_total}
(FD / "casg_alpha_sweep_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def line_chart_key(title, xlabel, key, y_dec=3):
    xs = [r["alpha"] for r in rows_short]
    xmin, xmax = 0.0, 2.0
    allv = [r[key] for r in rows_short] + [r[key] for r in rows_total]
    ymin = min(allv); ymax = max(allv)
    pad = (ymax - ymin) * 0.08 + 1e-9
    ymin, ymax = ymin - pad, ymax + pad
    W, H = 660, 250
    pl, pr, pt, pb = 60, 20, 28, 38
    def X(v): return pl + (v - xmin) / (xmax - xmin) * (W - pl - pr)
    def Y(v): return H - pb - (v - ymin) / (ymax - ymin) * (H - pt - pb)
    svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" role="img">']
    svg.append(f'<text x="{pl}" y="18" font-size="13" font-weight="500" fill="#1d3557">{title}</text>')
    svg.append(f'<line x1="{pl}" y1="{H-pb}" x2="{W-pr}" y2="{H-pb}" stroke="#5F5E5A" stroke-width="1"/>')
    svg.append(f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{H-pb}" stroke="#5F5E5A" stroke-width="1"/>')
    for f in [0, 0.5, 1.0]:
        yv = ymin + f * (ymax - ymin); yy = Y(yv)
        svg.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{W-pr}" y2="{yy:.1f}" stroke="#E6E6E6" stroke-width="0.5"/>')
        svg.append(f'<text x="{pl-6}" y="{yy+4:.1f}" font-size="11" fill="#888" text-anchor="end">{yv:.{y_dec}f}</text>')
    for xv in [0, 0.5, 1.0, 1.5, 2.0]:
        xx = X(xv)
        svg.append(f'<text x="{xx:.1f}" y="{H-pb+16}" font-size="11" fill="#888" text-anchor="middle">{xv:.1f}</text>')
    svg.append(f'<text x="{(pl+W-pr)/2:.0f}" y="{H-4}" font-size="11" fill="#555" text-anchor="middle">{xlabel}</text>')
    bx = X(opt["short_mae"] if "MAE" in title else opt["short_loss_p"])
    svg.append(f'<line x1="{bx:.1f}" y1="{pt}" x2="{bx:.1f}" y2="{H-pb}" stroke="#F4A261" stroke-width="1" stroke-dasharray="4 3"/>')
    svg.append(f'<text x="{bx+4:.1f}" y="{pt+10}" font-size="10" fill="#F4A261">α*≈0</text>')
    for color, name, rows in [("#A32D2D","短-仅简答",rows_short),("#185FA5","总-整卷",rows_total)]:
        pts = " ".join(f"{X(r['alpha']):.1f},{Y(r[key]):.1f}" for r in rows)
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        last = rows[-1]
        svg.append(f'<circle cx="{X(last["alpha"]):.1f}" cy="{Y(last[key]):.1f}" r="3" fill="{color}"/>')
        svg.append(f'<line x1="{pl+10}" y1="{pt+6}" x2="{pl+30}" y2="{pt+6}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{pl+36}" y="{pt+10}" font-size="11" fill="#444">{name}</text>')
    svg.append('</svg>')
    return "".join(svg)

chart_loss = line_chart_key("α 优化轨迹 · 损失 loss（越低越好）", "α (幂指数)", "loss_pdsigma01", y_dec=2)
chart_mae = line_chart_key("α 优化轨迹 · MAE（越低越好）", "α (幂指数)", "mae", y_dec=2)


def tbl(rows, opt_alpha):
    body = ""
    for r in rows:
        hl = ' class="hl"' if abs(r["alpha"]-opt_alpha) < 1e-9 else ""
        body += (f"<tr{hl}><td>{r['alpha']:.4f}</td><td>{r['loss_pdsigma01']:.4f}</td>"
                 f"<td>{r['P']:.4f}</td><td>{r['D']:.4f}</td><td>{r['sigma']:.3f}</td>"
                 f"<td>{r['qwk_10']:.4f}</td><td>{r['pearson_r']:.4f}</td><td>{r['mae']:.3f}</td></tr>\n")
    return (f'<div class="scroll"><table><tr><th>α</th><th>loss<br>(pdsigma01)</th><th>P</th>'
            f'<th>D</th><th>σ</th><th>QWK10</th><th>Pearson r</th><th>MAE</th></tr>\n{body}</table></div>')

html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>CASG α 优化过程数据</title>
<style>
body{{font-family:'Microsoft YaHei',Segoe UI,sans-serif;margin:20px;color:#222;background:#fafafa}}
h1{{font-size:20px;border-left:5px solid #2a9d8f;padding-left:12px}}
h2{{font-size:15px;margin-top:22px;color:#1d3557}}
.wrap{{max-width:1080px;margin:auto;background:#fff;padding:20px 26px;box-shadow:0 2px 8px #0001;border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}}
th,td{{border:1px solid #ddd;padding:5px 7px;text-align:center}}
th{{background:#1d3557;color:#fff}}
tr:nth-child(even) td{{background:#fafcff}}
.hl{{background:#e8f6f3!important;font-weight:bold}}
.scroll{{max-height:420px;overflow:auto;border:1px solid #eee}}
.note{{background:#fff8e6;border-left:4px solid #f4a261;padding:10px 14px;font-size:12px;line-height:1.7;border-radius:4px;margin-top:10px}}
.tag{{display:inline-block;background:#e8f6f3;color:#0f6e56;border:1px solid #9FE1CB;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:6px}}
.chartbox{{border:1px solid #eee;border-radius:6px;padding:6px 10px;margin-top:10px;background:#fff}}
</style></head><body><div class="wrap">
<h1>食品班 CASG 评分：α 优化全过程数据</h1>
<p style="color:#555;font-size:13px">数据集：39 名学生（新编号，用户最终校正版 CSV）。
α 仅作用于 <b>部分覆盖</b> 要点：得分比例 = 覆盖率^α。扫描 42 个 α∈[0,2]，对每个 α 计算与人工基线的各项指标。
两场景：<b>仅简答</b>（简答×5，100分制）与 <b>总分</b>（LLM简答+人类其他题，100分制）。
最优 α 在两类口径下均为 <span class="tag">α*≈0</span>（grid 精确为 0.0；连续优化为 0.0006）。</p>

<h2>1. 损失函数定义</h2>
<div class="note">
<b>loss_pdsigma01</b>（论文口径 relative，σ 权重 0.1）：<br>
&nbsp;&nbsp;L = |P−0.2|/0.2 + |D−0.3|/0.3 + 0.1·|σ−10|/10<br>
<b>loss_equal</b>（对照，σ 权重 1.0）：<br>
&nbsp;&nbsp;L = |P−0.2|/0.2 + |D−0.3|/0.3 + |σ−10|/10<br>
目标分布取论文设定 P*=0.2、D*=0.3、σ*=10（注意：本班真实分布 P≈0.36–0.43、D≈0.41–0.60、σ≈16–24，与目标差距较大，故损失绝对值偏高属正常）。
</div>

<h2>2. 优化轨迹折线图</h2>
<div class="chartbox">{chart_loss}</div>
<div class="chartbox">{chart_mae}</div>
<p style="font-size:11px;color:#888">损失与 MAE 均随 α 增大而单调上升，最低点落在 α=0，橙色虚线标出 α*。</p>

<h2>3. 仅简答场景（×5，100分制）· 完整 α 扫描</h2>
{tbl(rows_short, opt["short_loss_p"])}
<h2>4. 总分场景（整卷100分制）· 完整 α 扫描</h2>
{tbl(rows_total, opt["total_loss_p"])}

<h2>5. 最优 α 汇总</h2>
<table style="width:auto">
<tr><th>场景</th><th>最优(按 loss_pdsigma01)</th><th>最优(按 loss_equal)</th><th>最优(按 MAE)</th></tr>
<tr><td>仅简答</td><td>{opt['short_loss_p']:.4f}</td><td>{opt['short_loss_e']:.4f}</td><td>{opt['short_mae']:.4f}</td></tr>
<tr><td>总分</td><td>{opt['total_loss_p']:.4f}</td><td>{opt['total_loss_e']:.4f}</td><td>{opt['total_mae']:.4f}</td></tr>
</table>
<p style="font-size:11px;color:#888">三种判据均指向 α≈0，说明在本数据上 α 越大（部分覆盖折扣越重）越偏离人工基线，故最优为不折扣（α=0）。</p>
</div></body></html>"""
(FD / "casg_alpha_sweep.html").write_text(html, encoding="utf-8")
print("written casg_alpha_sweep.html + casg_alpha_sweep_table.json")
