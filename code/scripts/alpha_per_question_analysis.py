



import argparse
import json
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from alpha_sweep_from_xlsx import parse_xlsx, compute_casg  

import casg_losses  
from casg_losses import LOSS_FUNCTIONS, compute_loss_values, resolve_name  

TH = (1.0, 1.0, 1.0)
ALPHAS = np.arange(0.0, 2.001, 0.1)


def qwk_safe(yt, yp, bins=10):
    
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
        return np.nan
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    den = (w * E).sum()
    if den <= 0:
        return np.nan
    num = (w * O).sum()
    return float(1 - num / den)


def align(scores, human):
    
    sids = [s for s in scores if s in human]
    arr = np.array([scores[s] for s in sids], float)
    harr = np.array([human[s] for s in sids], float)
    return arr, harr


def full_metrics(arr, harr, max_score=100.0):
    
    n = int(len(arr))
    mean = float(arr.mean())
    P = 1 - mean / max_score
    sd = np.sort(arr)
    k = max(1, int(n * 0.27))
    D = (sd[-k:].mean() - sd[:k].mean()) / max_score
    sigma = float(arr.std())
    if arr.std() > 1e-9 and harr.std() > 1e-9:
        r, _ = pearsonr(arr, harr)
        rho, _ = spearmanr(arr, harr)
    else:
        r, rho = np.nan, np.nan
    q10 = qwk_safe(harr, arr, 10)
    q5 = qwk_safe(harr, arr, 5)
    mae = float(np.abs(arr - harr).mean())
    rmse = float(np.sqrt(((arr - harr) ** 2).mean()))
    return dict(n=n, mean=mean, P=float(P), D=float(D), sigma=sigma,
                pearson_r=float(r), spearman_rho=float(rho),
                qwk_10=q10, qwk_5=q5, mae=mae, rmse=rmse)


def solve_optimal(data, norm, qw, human, qid=None, loss='equal'):
    
    if loss not in LOSS_FUNCTIONS:
        raise ValueError(f'未知损失名: {loss}; 可用: {list(LOSS_FUNCTIONS.keys())}')
    func = LOSS_FUNCTIONS[loss]['func']

    def obj(alpha):
        if qid is None:
            scores, _ = compute_casg(data, norm, qw, alpha)
            arr, harr = align(scores, human)
        else:
            _, per_q = compute_casg(data, norm, qw, alpha)
            arr, harr = align(per_q[qid], human)
        m = full_metrics(arr, harr)
        return float(func(m))

    res = minimize_scalar(obj, bounds=(0.0, 2.0), method='bounded',
                          options={'maxiter': 100, 'xatol': 1e-6})
    alpha = float(res.x)
    if qid is None:
        scores, _ = compute_casg(data, norm, qw, alpha)
        arr, harr = align(scores, human)
    else:
        _, per_q = compute_casg(data, norm, qw, alpha)
        arr, harr = align(per_q[qid], human)
    m = full_metrics(arr, harr)
    m['alpha'] = alpha
    m['loss_used'] = loss
    return m


def loss_cols(m):
    
    return {f'loss_{name}': round(v, 4) for name, v in compute_loss_values(m).items()}


def main():
    parser = argparse.ArgumentParser(description='逐题 α 分析与配置文件驱动的损失求解实验')
    parser.add_argument('--config', default=casg_losses.DEFAULT_MAIN_LOSS,
                        help=f'主损失配置名或json路径, 默认 {casg_losses.DEFAULT_MAIN_LOSS}')
    parser.add_argument('--configs', default=','.join(LOSS_FUNCTIONS.keys()),
                        help='参与求解与对比的损失配置(逗号分隔, 名或json路径), 默认全部')
    args = parser.parse_args()

    main_loss = resolve_name(args.config)
    if main_loss not in LOSS_FUNCTIONS:
        raise ValueError(f'主损失 {main_loss} 未注册; 可用: {list(LOSS_FUNCTIONS.keys())}')
    loss_names = []
    for n in args.configs.split(','):
        if not n.strip():
            continue
        rn = resolve_name(n)
        if rn not in loss_names:
            loss_names.append(rn)
    if not loss_names:
        raise ValueError('--configs 未匹配到任何已注册损失')
    print(f'主损失: {main_loss}  ({LOSS_FUNCTIONS[main_loss]["label"]})')
    print(f'参与对比的损失: {loss_names}')

    print('解析 xlsx ...')
    data = parse_xlsx()
    kp = json.load(open(KP, encoding='utf-8'))
    norm, qw = kp['normalized_100'], kp['question_weights']
    df = pd.read_excel(BASE)
    df['学生学号'] = df['学生学号'].astype(str)
    sids_all = df['学生学号'].tolist()

    
    qids = sorted(data.keys())
    qcol_h = {q: q + '_人工分' for q in qids}
    qcol_l = {q: q + '_LLM_v2分' for q in qids}
    human_pq, llm_pq = {}, {}
    for q in qids:
        human_pq[q] = {s: float(v) for s, v in zip(sids_all, df[qcol_h[q]]) if pd.notna(v)}
        llm_pq[q] = {s: float(v) for s, v in zip(sids_all, df[qcol_l[q]]) if pd.notna(v)}
    human_tot = {s: float(v) for s, v in zip(sids_all, df['人工总分']) if pd.notna(v)}
    llm_tot = {s: float(v) for s, v in zip(sids_all, df['LLM_v2总分']) if pd.notna(v)}

    print('题目数:', len(qids), qids)

    
    sweep_rows = []
    for qid in qids:
        for alpha in ALPHAS:
            _, per_q = compute_casg(data, norm, qw, float(alpha))
            arr, harr = align(per_q[qid], human_pq[qid])
            m = full_metrics(arr, harr)
            sweep_rows.append(dict(qid=qid, alpha=round(float(alpha), 3), **m, **loss_cols(m)))
    sweep_df = pd.DataFrame(sweep_rows)

    
    base_rows = []
    for qid in qids:
        h_arr = np.array(list(human_pq[qid].values()), float)
        l_arr = np.array(list(llm_pq[qid].values()), float)
        
        hm = full_metrics(h_arr, h_arr)
        
        lm = full_metrics(l_arr, h_arr)
        base_rows.append(dict(qid=qid,
                              human_P=hm['P'], human_D=hm['D'], human_sigma=hm['sigma'], human_mean=hm['mean'],
                              llm_P=lm['P'], llm_D=lm['D'], llm_sigma=lm['sigma'], llm_mean=lm['mean'],
                              llm_r=lm['pearson_r'], llm_rho=lm['spearman_rho'],
                              llm_qwk10=lm['qwk_10'], llm_mae=lm['mae'], n_human=hm['n']))
    base_df = pd.DataFrame(base_rows)

    
    summ_rows = []
    var_rows = []   
    for qid in qids:
        opts = {name: solve_optimal(data, norm, qw, human_pq[qid], qid=qid, loss=name)
                for name in loss_names}
        opt_main = opts[main_loss]
        b = base_df[base_df.qid == qid].iloc[0]
        row = dict(
            qid=qid, n=opt_main['n'],
            alpha_star=round(opt_main['alpha'], 4),
            P=round(opt_main['P'], 4), D=round(opt_main['D'], 4), sigma=round(opt_main['sigma'], 4),
            r=round(opt_main['pearson_r'], 4), rho=round(opt_main['spearman_rho'], 4),
            qwk10=round(opt_main['qwk_10'], 4), mae=round(opt_main['mae'], 4),
            
            llm_r=round(b['llm_r'], 4), llm_rho=round(b['llm_rho'], 4),
            llm_qwk10=round(b['llm_qwk10'], 4), llm_mae=round(b['llm_mae'], 4),
            llm_P=round(b['llm_P'], 4), llm_D=round(b['llm_D'], 4), llm_sigma=round(b['llm_sigma'], 4),
            human_P=round(b['human_P'], 4), human_D=round(b['human_D'], 4), human_sigma=round(b['human_sigma'], 4),
            
            d_r=round(opt_main['pearson_r'] - b['llm_r'], 4),
            d_rho=round(opt_main['spearman_rho'] - b['llm_rho'], 4),
            d_qwk10=round((opt_main['qwk_10'] - b['llm_qwk10']) if opt_main['qwk_10'] == opt_main['qwk_10'] else np.nan, 4),
            d_mae=round(opt_main['mae'] - b['llm_mae'], 4),
            better_r='✓' if (opt_main['pearson_r'] > b['llm_r']) else '✗',
            better_rho='✓' if (opt_main['spearman_rho'] > b['llm_rho']) else '✗',
            better_qwk10='✓' if (opt_main['qwk_10'] > b['llm_qwk10']) else '✗',
            better_mae='✓' if (opt_main['mae'] < b['llm_mae']) else '✗',
        )
        
        row.update(loss_cols(opt_main))
        for name in loss_names:
            if name != main_loss:
                row[f'alpha_star_{name}'] = round(opts[name]['alpha'], 4)
        summ_rows.append(row)

        for name in loss_names:
            opt = opts[name]
            var_rows.append(dict(
                qid=qid, loss=name, loss_label=LOSS_FUNCTIONS[name]['label'],
                alpha=round(opt['alpha'], 4),
                P=round(opt['P'], 4), D=round(opt['D'], 4), sigma=round(opt['sigma'], 4),
                r=round(opt['pearson_r'], 4), rho=round(opt['spearman_rho'], 4),
                qwk10=round(opt['qwk_10'], 4), mae=round(opt['mae'], 4),
                **loss_cols(opt),
            ))
    summ_df = pd.DataFrame(summ_rows)

    
    exam_rows = []
    for alpha in ALPHAS:
        scores, _ = compute_casg(data, norm, qw, float(alpha))
        arr, harr = align(scores, human_tot)
        m = full_metrics(arr, harr)
        exam_rows.append(dict(alpha=round(float(alpha), 3), **m, **loss_cols(m)))
    exam_df = pd.DataFrame(exam_rows)

    exam_opts = {name: solve_optimal(data, norm, qw, human_tot, qid=None, loss=name)
                 for name in loss_names}
    exam_eq = exam_opts[main_loss]
    lm_exam = full_metrics(np.array(list(llm_tot.values()), float),
                           np.array([human_tot[s] for s in llm_tot if s in human_tot], float))
    hm_exam = full_metrics(np.array(list(human_tot.values()), float),
                           np.array(list(human_tot.values()), float))
    exam_row = dict(
        qid='整卷', n=exam_eq['n'],
        alpha_star=round(exam_eq['alpha'], 4),
        P=round(exam_eq['P'], 4), D=round(exam_eq['D'], 4), sigma=round(exam_eq['sigma'], 4),
        r=round(exam_eq['pearson_r'], 4), rho=round(exam_eq['spearman_rho'], 4),
        qwk10=round(exam_eq['qwk_10'], 4), mae=round(exam_eq['mae'], 4),
        llm_r=round(lm_exam['pearson_r'], 4), llm_rho=round(lm_exam['spearman_rho'], 4),
        llm_qwk10=round(lm_exam['qwk_10'], 4), llm_mae=round(lm_exam['mae'], 4),
        llm_P=round(lm_exam['P'], 4), llm_D=round(lm_exam['D'], 4), llm_sigma=round(lm_exam['sigma'], 4),
        human_P=round(hm_exam['P'], 4), human_D=round(hm_exam['D'], 4), human_sigma=round(hm_exam['sigma'], 4),
        d_r=round(exam_eq['pearson_r'] - lm_exam['pearson_r'], 4),
        d_rho=round(exam_eq['spearman_rho'] - lm_exam['spearman_rho'], 4),
        d_qwk10=round((exam_eq['qwk_10'] - lm_exam['qwk_10']) if exam_eq['qwk_10'] == exam_eq['qwk_10'] else np.nan, 4),
        d_mae=round(exam_eq['mae'] - lm_exam['mae'], 4),
        better_r='✓' if (exam_eq['pearson_r'] > lm_exam['pearson_r']) else '✗',
        better_rho='✓' if (exam_eq['spearman_rho'] > lm_exam['spearman_rho']) else '✗',
        better_qwk10='✓' if (exam_eq['qwk_10'] > lm_exam['qwk_10']) else '✗',
        better_mae='✓' if (exam_eq['mae'] < lm_exam['mae']) else '✗',
    )
    exam_row.update(loss_cols(exam_eq))
    for name in loss_names:
        if name != main_loss:
            exam_row[f'alpha_star_{name}'] = round(exam_opts[name]['alpha'], 4)
    for name in loss_names:
        opt = exam_opts[name]
        var_rows.append(dict(
            qid='整卷', loss=name, loss_label=LOSS_FUNCTIONS[name]['label'],
            alpha=round(opt['alpha'], 4),
            P=round(opt['P'], 4), D=round(opt['D'], 4), sigma=round(opt['sigma'], 4),
            r=round(opt['pearson_r'], 4), rho=round(opt['spearman_rho'], 4),
            qwk10=round(opt['qwk_10'], 4), mae=round(opt['mae'], 4),
            **loss_cols(opt),
        ))
    var_df = pd.DataFrame(var_rows)

    
    n_better = {k: int((summ_df[k] == '✓').sum()) for k in ['better_r', 'better_rho', 'better_qwk10', 'better_mae']}
    print('\n=== 16 题中 CASG@α*(主损失 %s) 优于 LLM_v2 的题数 ===' % main_loss)
    print(f"  Pearson r 提升: {n_better['better_r']}/16")
    print(f"  Spearman ρ 提升: {n_better['better_rho']}/16")
    print(f"  QWK10 提升: {n_better['better_qwk10']}/16")
    print(f"  MAE 降低: {n_better['better_mae']}/16")
    print(f"\n整卷: α*={exam_row['alpha_star']}, r={exam_row['r']}(LLM {exam_row['llm_r']}), "
          f"ρ={exam_row['rho']}({exam_row['llm_rho']}), QWK10={exam_row['qwk10']}({exam_row['llm_qwk10']}), "
          f"MAE={exam_row['mae']}({exam_row['llm_mae']})")
    print(f"  整卷 P={exam_row['P']}/D={exam_row['D']}/σ={exam_row['sigma']} "
          f"(LLM {exam_row['llm_P']}/{exam_row['llm_D']}/{exam_row['llm_sigma']}, 人工 {exam_row['human_P']}/{exam_row['human_D']}/{exam_row['human_sigma']})")
    print('  各损失 α*: ' + ', '.join(f'{name}={exam_row.get(f"alpha_star_{name}", exam_row["alpha_star"])}'
                                     for name in loss_names))

    
    note = pd.DataFrame([
        ['数据源', 'keypoint_evaluations_detail.xlsx (16题要点覆盖) + 基准评分数据_人工与LLM_v2.xlsx (逐题0-100)'],
        ['CASG 逐题分', 'power 策略: partial 给 max*(coverage**α), full 给 max, missing 给 max*0.05; θ=(1,1,1); 每题满分 100'],
        ['P/D/σ 定义', 'P=1-mean/100; D=(top27%-bot27%)/100; σ=总体标准差 ddof=0'],
        ['相关/误差指标', '与人工逐题分比较: Pearson r, Spearman ρ, QWK(10/5箱, 0-100等距), MAE, RMSE'],
        ['损失配置', 'JSON 配置文件 scripts/loss_configs/*.json; 修改/新增损失=编辑或复制配置文件, 代码无需改动'],
        ['主损失配置', f'{main_loss}: {LOSS_FUNCTIONS[main_loss]["label"]}'],
        ['配置来源', LOSS_FUNCTIONS[main_loss]['source']],
        ['α 求解范围', 'scipy.optimize.minimize_scalar bounded, α∈[0, 2], xatol=0.001'],
        ['逐题 QWK 注意', '每题 n≈37 且分数区间窄, QWK10 分箱稀疏易退化(返回 nan); 逐题比较以 r/ρ/MAE 为主, QWK10 以整卷为准'],
        ['better 标记', f'CASG@α*(主损失 {main_loss}) 的指标相对 LLM_v2 基准提升: r/ρ/QWK10 增大、MAE 减小记为 ✓'],
        ['扫描表损失值', '「逐题α扫描」「整卷α扫描」每行含全部已注册损失的 loss_<name> 列(相对偏差, 各配置按配置文件计算)'],
    ], columns=['项目', '说明'])
    note_df = note

    with pd.ExcelWriter(OUTDIR / 'alpha_per_question_results.xlsx', engine='openpyxl') as w:
        note_df.to_excel(w, sheet_name='说明', index=False)
        summ_df.to_excel(w, sheet_name='逐题最优汇总', index=False)
        base_df.to_excel(w, sheet_name='逐题基准', index=False)
        sweep_df.to_excel(w, sheet_name='逐题α扫描', index=False)
        exam_df.to_excel(w, sheet_name='整卷α扫描', index=False)
        pd.DataFrame([exam_row]).to_excel(w, sheet_name='整卷最优', index=False)
        var_df.to_excel(w, sheet_name='损失变体对比', index=False)

    out = dict(
        meta=dict(data_sources=[str(XLSX), str(KP), str(BASE)],
                  main_loss=main_loss,
                  losses={name: LOSS_FUNCTIONS[name]['label'] for name in loss_names},
                  loss_configs={name: LOSS_FUNCTIONS[name]['config'] for name in loss_names},
                  alpha_range=[0, 2]),
        per_question_baselines=base_df.to_dict(orient='records'),
        per_question_optimal=summ_df.to_dict(orient='records'),
        per_question_sweep=sweep_df.to_dict(orient='records'),
        exam_sweep=exam_df.to_dict(orient='records'),
        exam_optimal=exam_row,
        loss_variant_comparison=var_df.to_dict(orient='records'),
        n_better_vs_llm_v2=n_better,
    )

    
    out_xlsx_main = OUTDIR / f'alpha_per_question_results_{main_loss}.xlsx'
    out_json_main = OUTDIR / f'alpha_per_question_results_{main_loss}.json'
    out_xlsx_fixed = OUTDIR / 'alpha_per_question_results.xlsx'
    out_json_fixed = OUTDIR / 'alpha_per_question_results.json'

    with pd.ExcelWriter(out_xlsx_main, engine='openpyxl') as w:
        note_df.to_excel(w, sheet_name='说明', index=False)
        summ_df.to_excel(w, sheet_name='逐题最优汇总', index=False)
        base_df.to_excel(w, sheet_name='逐题基准', index=False)
        sweep_df.to_excel(w, sheet_name='逐题α扫描', index=False)
        exam_df.to_excel(w, sheet_name='整卷α扫描', index=False)
        pd.DataFrame([exam_row]).to_excel(w, sheet_name='整卷最优', index=False)
        var_df.to_excel(w, sheet_name='损失变体对比', index=False)
    with open(out_json_main, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    import shutil
    shutil.copyfile(out_xlsx_main, out_xlsx_fixed)
    shutil.copyfile(out_json_main, out_json_fixed)

    print('\n已输出:')
    print('  ', out_xlsx_main, ' (带配置名)')
    print('  ', out_xlsx_fixed, ' (固定名=最近一次运行)')
    print('  ', out_json_main)
    print('  ', out_json_fixed)


if __name__ == '__main__':
    main()
