


import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()


ROOT = Path(r"C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育")
sys.path.insert(0, str(ROOT / "scripts"))

from casg_prompt_template import build_casg_system_prompt
from casg_pipeline import CASGGrader

ANSWERS = REPO_ROOT / 'data' / 'food_class' / 'student_answers.json'
RUBRICS = REPO_ROOT / 'data' / 'food_class' / 'rubrics_食品34要点.json'
OUTDIR = REPO_ROOT / 'data' / 'food_class' / 'coverage_evaluations'
QIDS = ["1", "2", "3", "4", "5"]

MODEL = "deepseek-v4-flash"
MAX_RETRIES = 5
RETRY_DELAYS = [3, 8, 15, 30, 60]
WORKERS = 4
WRITE_LOCK = threading.Lock()


def load_config():
    with open(ROOT / "normalin" / "configs" / "experiment_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["llm"]["api_mode"]


def build_user_prompt(question_text, answer_text, rubric_keypoints, qid, keypoint_scores):
    
    lines = [f"请对以下学生答案进行逐要点覆盖度评估。\n", f"=== 题目信息 ===\n题目ID: {qid}\n",
             "【题目内容】\n" + question_text + "\n", "【评分标准】"]
    for kp in rubric_keypoints:
        lines.append(f"\n要点 {kp['kp_id']}：{kp['kp_name']}")
        lines.append(f"  【评分要点】{kp['desc']}")
        lines.append(f"  【详细答案】{kp['answer']}")
        sub = kp.get("subpoints", [])
        if sub:
            lines.append("  【子要点】(逐一判定覆盖, coverage_ratio=覆盖子要点数/总子要点数)")
            for i, sp in enumerate(sub, 1):
                lines.append(f"    - 子要点{i}: {sp}")
    if keypoint_scores:
        lines.append("\n【各要点分值分配】(必须使用这些固定分值作 max_score, 不要自行分配)")
        for kp_id in sorted(keypoint_scores, key=lambda x: int(x)):
            lines.append(f"  - 要点 {kp_id}: {keypoint_scores[kp_id]:.2f} 分")
    lines.append(f"\n【学生答案】\n{answer_text}")
    lines.append("\n请严格按照 JSON 格式输出评估结果，不要包含其他内容。")
    return "\n".join(lines)


def call_evaluate(question_text, answer_text, rubric_keypoints, qid, keypoint_scores):
    
    client = globals()["_client"]
    system_prompt = build_casg_system_prompt()
    user_prompt = build_user_prompt(question_text, answer_text, rubric_keypoints, qid, keypoint_scores)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.15,
                max_tokens=8192,
                timeout=180,
            )
            raw = resp.choices[0].message.content
            if not raw or not raw.strip():
                raise ValueError(f"空返回(attempt {attempt})")
            return raw
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
    raise RuntimeError(f"评估调用失败: {last_err}")


def _persist(outpath, results):
    
    with WRITE_LOCK:
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)


def _eval_one(qid, qtext, answer, keypoints, fixed_scores, grader):
    
    try:
        raw = call_evaluate(qtext, answer, keypoints, qid, fixed_scores)
    except Exception as e:
        return {"parse_error": True, "raw": "", "error": str(e)[:200]}
    grader_tmp = CASGGrader(fixed_max_scores=fixed_scores)
    eval_data = grader_tmp.grade_answer(raw, qid)
    if eval_data is None:
        try:
            raw = call_evaluate(qtext, answer, keypoints, qid, fixed_scores)
        except Exception as e:
            raw = ""
        eval_data = grader_tmp.grade_answer(raw, qid)
    if eval_data is None:
        return {"parse_error": True, "raw": raw[:500]}
    for kp in eval_data.get("keypoints", []):
        if kp.get("status") == "partial":
            cov = kp.get("coverage_ratio", 0.0)
            kp["quality_band"] = "high" if cov >= 0.8 else ("medium" if cov >= 0.5 else "low")
    return eval_data


def evaluate_question(qid, qtext, answers_for_q, rubric_keypoints, fixed_scores, grader):
    
    outpath = OUTDIR / f"{qid}_evaluations.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if outpath.exists():
        with open(outpath, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  [断点续跑] 已评估 {len(results)} 人")

    pending = [(sid, ans) for sid, ans in answers_for_q if sid not in results]
    if not pending:
        return outpath, results
    print(f"  待评估 {len(pending)} 人 (workers={WORKERS})")

    done_cnt = [0]
    done_lock = threading.Lock()

    def _worker(item):
        sid, ans = item
        res = _eval_one(qid, qtext, ans, rubric_keypoints, fixed_scores, grader)
        with done_lock:
            done_cnt[0] += 1
            n = done_cnt[0]
        status = "OK" if not res.get("parse_error") else "ERR"
        print(f"  [{n}/{len(pending)}] {sid} {status}", flush=True)
        return sid, res

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_worker, item): item[0] for item in pending}
        for fut in as_completed(futs):
            sid, res = fut.result()
            results[sid] = res
            _persist(outpath, results)
    return outpath, results


def main():
    global _client
    llm_cfg = load_config()
    _client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"), timeout=120)

    answers_data = json.load(open(ANSWERS, encoding="utf-8"))
    rubrics_data = json.load(open(RUBRICS, encoding="utf-8"))

    
    answers_for_q = {q: [] for q in QIDS}
    for stu in answers_data["students"]:
        sid = stu["student_id"]
        for a in stu["answers"]:
            answers_for_q[a["qid"]].append((sid, a["answer_text"]))

    meta = {}
    total_ok = 0
    for qid in QIDS:
        qtext = rubrics_data["rubrics"][qid]["question_text"]
        kps = rubrics_data["rubrics"][qid]["keypoints"]
        fixed_scores = {int(kp["kp_id"]): float(kp["kp_score"]) for kp in kps}
        print(f"\n=== 评估 Q{qid}: {qtext[:20]}... ({len(answers_for_q[qid])}人, {len(kps)}要点) ===")
        grader = CASGGrader(fixed_max_scores=fixed_scores)
        outpath, results = evaluate_question(qid, qtext, answers_for_q[qid], kps, fixed_scores, grader)
        n_ok = sum(1 for v in results.values() if not v.get("parse_error"))
        total_ok += n_ok
        meta[qid] = {"n": len(results), "n_ok": n_ok, "n_fail": len(results) - n_ok,
                     "out": str(outpath)}
        print(f"  Q{qid} 完成: 成功={n_ok}, 失败={len(results) - n_ok}")

    with open(OUTDIR / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n总成功评估: {total_ok} / 195")
    print(f"结果目录: {OUTDIR}")


if __name__ == "__main__":
    main()