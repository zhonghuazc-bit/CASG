#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1 班 FEW-LLM（LLM v2，少样本）评分：得到 B1 班自身的 FEW-LLM 基线。

口径与 B2 班基线（llm_scores_v2.json，由 code/scripts/grade_v2_improved.py 产生）完全一致：
  - 同一提示词（system + 3 个 few-shot 示例 + 当前题面/量规/学生答案）
  - 同一配置（experiment_config.yaml 的 llm.api_mode：deepseek-v4-flash、temperature 0.15、max_tokens 800）
  - 每题整体给 0-100 整数分；输出【评分分析】与【分数】

评分范围：论文 B1 跨班口径 —— 16 道共有题 × 35 人完整覆盖子样本（共 560 条）。
支持断点续跑：已完成的 (qid,sid) 直接跳过。

产物：data/results/b1_class/b1_llm_v2_scores.json
      （结构：{"HW1_Q1": {"学号": {"score": int, "raw": "...", "ts": ...}}, ...}）
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()


NORMALIN = REPO_ROOT / 'normalin'
sys.path.insert(0, str(NORMALIN / 'scripts'))

from grade_v2_improved import (  # noqa: E402
    build_system_prompt, build_user_prompt, parse_score_v2,
    FEWSHOT_HIGH, FEWSHOT_MEDIUM, FEWSHOT_LOW,  # noqa: F401  (确保示例内容被加载)
)

ROOT = REPO_ROOT
OUT_FILE = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'b1_llm_v2_scores.json'

Q16 = ['HW1_Q1', 'HW1_Q2', 'HW1_Q3', 'HW2_Q1', 'HW2_Q2', 'HW2_Q3',
       'HW3_Q1', 'HW3_Q2', 'HW3_Q3', 'HW4_Q2', 'HW4_Q3', 'HW4_Q4', 'HW4_Q5',
       'HW5_Q1', 'HW5_Q2', 'HW5_Q3']

DELAY = 0.4          # 每次调用之间的间隔
MAX_RETRIES = 5
SAVE_EVERY = 10
# deepseek-v4-flash 为推理模型：reasoning_content 会占用 max_tokens 预算，
# 原配置 800 常导致 content 为空或被截断。此处仅提高生成上限（不改变提示词与
# temperature=0.15），使分析与【分数】能完整输出；评分口径与 B2 基线一致。
MAX_TOKENS = 3000


def load_config():
    cfg = yaml.safe_load(open(NORMALIN / 'configs' / 'experiment_config.yaml', encoding='utf-8'))
    return cfg['llm']['api_mode']


def load_sources():
    questions = {q['question_id']: q
                 for q in json.load(open(NORMALIN / 'data' / 'processed' / 'questions.json', encoding='utf-8'))}
    rubrics = json.load(open(NORMALIN / 'data' / 'processed' / 'rubrics.json', encoding='utf-8'))
    essays = json.load(open(ROOT / 'normalin' / 'results' / 'b1_class' / 'b1_essay_answers.json', encoding='utf-8'))['data']
    evals = json.load(open(ROOT / 'normalin' / 'results' / 'b1_class' / 'casg_b1_evaluations_v2.json', encoding='utf-8'))
    sids35 = sorted({s for q in Q16 for s in evals[q]
                     if all(s in evals[q2] for q2 in Q16)})
    print(f'完整覆盖子样本: {len(sids35)} 人 × {len(Q16)} 题 = {len(sids35)*len(Q16)} 条', flush=True)
    return questions, rubrics, essays, sids35


def rubric_to_text(rubric):
    if isinstance(rubric, str):
        return rubric
    if isinstance(rubric, dict):
        parts = []
        for i, point in enumerate(rubric.get('key_points', []), 1):
            parts.append(f"{i}. {point.get('description', '')}（最高 {point.get('max_points', '')} 分）")
        if rubric.get('total_points'):
            parts.append(f"\n满分：{rubric['total_points']} 分")
        return "\n".join(parts) if parts else str(rubric)
    return str(rubric)


def build_client(am):
    from openai import OpenAI
    client = OpenAI(
        api_key=am['api_key'],
        base_url=am.get('base_url', 'https://api.deepseek.com/v1'),
        timeout=am.get('timeout', 120),
    )
    return client


def grade_one(client, model, system_prompt, question, answer, rubric, max_tokens=MAX_TOKENS):
    prompt = build_user_prompt(question, answer, rubric_to_text(rubric))
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{'role': 'system', 'content': system_prompt},
                          {'role': 'user', 'content': prompt}],
                temperature=0.15,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            finish = resp.choices[0].finish_reason
            raw = msg.content or ''
            score = parse_score_v2(raw)
            if score is None or not raw.strip():
                # 空输出或解析失败（常见于推理占用预算导致截断），休息后重试
                time.sleep(2.0)
                print(f'    [warn] 空/不可解析输出(finish={finish})，重试 {attempt+1}', flush=True)
                continue
            return score, raw, finish
        except Exception as e:
            wait = DELAY * (2 ** attempt)
            print(f'    [api err {type(e).__name__}] 重试 {attempt+1}: 等 {wait:.1f}s', flush=True)
            time.sleep(wait)
    return None, '', None


def main():
    am = load_config()
    questions, rubrics, essays, sids35 = load_sources()

    model = am.get('model', 'deepseek-chat')
    print(f'model={model}  temperature=0.15  max_tokens(生成上限)={MAX_TOKENS}', flush=True)

    system_prompt = build_system_prompt()

    # 已存在结果 -> 断点续跑
    done = {}
    if OUT_FILE.exists():
        done = json.load(open(OUT_FILE, encoding='utf-8'))
        n_done = sum(len(v) for v in done.values())
        print(f'发现已有结果 {n_done} 条，将跳过继续。', flush=True)

    client = build_client(am)
    jobs = [(q, s) for q in Q16 for s in sids35]
    todo = [(q, s) for q, s in jobs
            if done.get(q, {}).get(s, {}).get('score') is None]
    print(f'待评分: {len(todo)} 条', flush=True)

    counter = 0
    t0 = time.time()
    for qid, sid in todo:
        rec = essays.get(qid, {}).get(sid, {})
        answer = (rec.get('student_answer') or '').strip()
        question = (questions.get(qid) or {}).get('text') or rec.get('q_text', '') or '[题目缺失]'
        rubric = rubrics.get(qid, '[评分标准缺失]')

        score, raw, finish = grade_one(client, model, system_prompt, question, answer, rubric)
        done.setdefault(qid, {})
        done[qid][sid] = {'score': score, 'raw': raw, 'finish': finish,
                          'ts': time.strftime('%Y-%m-%d %H:%M:%S')}
        counter += 1
        if counter % 5 == 0 or counter == len(todo):
            el = time.time() - t0
            eta = el / counter * (len(todo) - counter) / 60 if counter else 0
            ok_n = sum(1 for q in done.values() for v in q.values() if v.get('score') is not None)
            print(f'[{counter}/{len(todo)}] {sid}/{qid} score={score} '
                  f'累计成功 {ok_n} 预计还需 {eta:.1f} 分钟', flush=True)
            OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            json.dump(done, open(OUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        time.sleep(DELAY)

    json.dump(done, open(OUT_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    ok_n = sum(1 for q in done.values() for v in q.values() if v.get('score') is not None)
    fail_n = sum(1 for q in done.values() for v in q.values() if v.get('score') is None)
    print(f'\n完成: 成功 {ok_n}，失败 {fail_n}，文件 {OUT_FILE}', flush=True)


if __name__ == '__main__':
    main()
