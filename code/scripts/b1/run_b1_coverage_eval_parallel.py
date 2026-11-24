



import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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

ALL_QIDS = ['HW1_Q1', 'HW1_Q2', 'HW1_Q3',
            'HW2_Q1', 'HW2_Q2', 'HW2_Q3',
            'HW3_Q1', 'HW3_Q2', 'HW3_Q3',
            'HW4_Q1', 'HW4_Q2', 'HW4_Q3', 'HW4_Q4', 'HW4_Q5',
            'HW5_Q1', 'HW5_Q2', 'HW5_Q3']

B1_ANSWERS_JSON = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'b1_essay_answers.json'
EVAL_OUT = REPO_ROOT / 'data' / 'results' / 'b1_class' / 'casg_b1_evaluations_v2.json'


def load_b1_answers():
    d = json.load(open(B1_ANSWERS_JSON, encoding='utf-8'))['data']
    ans = {}
    for qid in ALL_QIDS:
        ans[qid] = {}
        for sid, rec in d.get(qid, {}).items():
            a = rec['student_answer'].strip()
            if len(a) >= 5:
                ans[qid][sid] = a
    return ans


def load_rubric_and_scores():
    rubrics = json.load(open(ROOT / 'normalin' / 'data' / 'processed' / 'rubrics.json', encoding='utf-8'))
    kp = json.load(open(ROOT / 'normalin' / 'data' / 'processed' / 'keypoint_scores.json', encoding='utf-8'))
    norm = kp['normalized_100']
    questions = json.load(open(ROOT / 'normalin' / 'data' / 'processed' / 'questions.json', encoding='utf-8'))
    qtext = {q['question_id']: q.get('text', q.get('question_text', '')) for q in questions}
    return rubrics, norm, qtext


def normalize_eval(ev):
    kps = ev.get('keypoints', [])
    for k in kps:
        try:
            k['coverage_ratio'] = float(k.get('coverage_ratio', 0.0))
        except (TypeError, ValueError):
            k['coverage_ratio'] = 0.0
        st = str(k.get('status', 'missing')).strip().lower()
        if st in ('完全覆盖', '满分', 'full', 'fullcover'):
            k['status'] = 'full'
        elif st in ('部分覆盖', '部分', 'part', 'partial', 'partialcover'):
            k['status'] = 'partial'
        else:
            k['status'] = 'missing'
    return ev


def _work(qid, sid, ans, qt, rubric, kp_scores):
    
    import json as _json
    import re as _re
    import time as _t
    from casg_keypoint_evaluation import client, MODEL
    from casg_prompt_template import build_casg_system_prompt, build_casg_user_prompt

    sys_prompt = build_casg_system_prompt()
    user_prompt = build_casg_user_prompt(qt, ans, rubric, qid,
                                         keypoint_scores=kp_scores, score_source='fixed')
    max_retry = 6
    for attempt in range(1, max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'system', 'content': sys_prompt},
                          {'role': 'user', 'content': user_prompt}],
                temperature=0.15,
                max_tokens=16384,  
                timeout=180,      
            )
            raw = (resp.choices[0].message.content or '').strip()
            if not raw:
                
                if attempt < max_retry:
                    _t.sleep(2 * attempt)
                continue
            
            json_match = None
            for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'(\{.*\})']:
                m = _re.search(pattern, raw, _re.DOTALL)
                if m:
                    json_match = m.group(1)
                    break
            eval_data = _json.loads(json_match or raw)
            if not isinstance(eval_data.get('keypoints'), list):
                raise ValueError('缺少 keypoints 列表')
            return qid, sid, normalize_eval(eval_data)
        except Exception as e:
            print(f'  [RETRY] {qid}/{sid} 第{attempt}次: {type(e).__name__}: {str(e)[:80]}', flush=True)
            if attempt < max_retry:
                _t.sleep(2 * attempt)
    print(f'  [FAIL] {qid}/{sid}: 重试{max_retry}次仍失败', flush=True)
    return qid, sid, None


def run_parallel(answers, rubrics, norm, qtext, workers, only_qid=None):
    evals = {}
    if EVAL_OUT.exists():
        evals = json.load(open(EVAL_OUT, encoding='utf-8'))
        n_ok = sum(1 for v in evals.values() for val in v.values() if val is not None)
        n_fail = sum(1 for v in evals.values() for val in v.values() if val is None)
        print(f'加载已有评估: {n_ok} 成功, {n_fail} 失败占位', flush=True)

    qids = [only_qid] if only_qid else ALL_QIDS
    tasks = []
    for qid in qids:
        if qid not in ALL_QIDS:
            print(f'未知题目 {qid}, 跳过'); continue
        evals.setdefault(qid, {})
        rubric, kp_scores, qt = rubrics[qid], norm[qid], qtext.get(qid, qid)
        items = sorted(answers[qid].items())
        
        done = set(k for k, v in evals[qid].items() if v is not None)
        todo = [(s, a) for s, a in items if s not in done]
        
        evals[qid] = {k: v for k, v in evals[qid].items() if v is not None}
        print(f'[{qid}] 待评估 {len(todo)} 条', flush=True)
        for sid, ans in todo:
            tasks.append((qid, sid, ans, qt, rubric, kp_scores))

    print(f'共 {len(tasks)} 条待评估, workers={workers}', flush=True)
    t0 = time.time()
    ok = 0
    fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_work, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            qid, sid, ev = fut.result()
            if ev:
                evals[qid][sid] = ev
                ok += 1
            else:
                
                evals[qid].setdefault(sid, None)
                fail += 1
            if i % 20 == 0 or i == len(futs):
                json.dump(evals, open(EVAL_OUT, 'w', encoding='utf-8'), ensure_ascii=False)
                speed = i / max(time.time() - t0, 1)
                print(f'  [{i}/{len(futs)}] 成功 {ok}, 失败 {fail}, {speed:.1f} 条/s, 已保存断点', flush=True)
    json.dump(evals, open(EVAL_OUT, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'完成: {ok} 成功, {fail} 失败, 用时 {time.time()-t0:.0f}s', flush=True)
    return evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qid', default=None)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    rubrics, norm, qtext = load_rubric_and_scores()
    answers = load_b1_answers()
    evals = run_parallel(answers, rubrics, norm, qtext, args.workers, args.qid)
    print('各题成功数:', {q: sum(1 for v in evals.get(q, {}).values() if v is not None) for q in ALL_QIDS}, flush=True)
    n_fail = sum(1 for v in evals.values() for val in v.values() if val is None)
    if n_fail:
        print(f'⚠ 仍有 {n_fail} 条失败待处理, 可重跑脚本自动重试', flush=True)


if __name__ == '__main__':
    main()