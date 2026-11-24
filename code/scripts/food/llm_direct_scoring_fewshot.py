


import json
import sys
import time
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
ANSWERS = REPO_ROOT / 'data' / 'food_class' / 'student_answers.json'
RUBRICS = REPO_ROOT / 'data' / 'food_class' / 'rubrics_食品34要点.json'
FEWSHOT = REPO_ROOT / 'data' / 'food_class' / 'few_shot_examples.json'
OUT = REPO_ROOT / 'data' / 'food_class' / 'llm_direct_scores_fewshot.json'
QIDS = ["1", "2", "3", "4", "5"]
MODEL = "deepseek-v4-flash"
WORKERS = 4
MAX_RETRIES = 8
RETRY_DELAYS = [3, 8, 15, 30, 30, 45, 45, 60]

SYSTEM = """你是一位食品工程原理课程的评分教师。请根据题目和评分标准，对学生答案直接给出 0-4 分的整数分（允许 0.5 分步进）。
你会看到同题的三个档位示例：高分作答、中分作答、低分作答（均来自教师真实评分），请以此为尺度校准你的评分标准，使你的打分严苛程度与教师一致。
只输出一个 JSON 对象：{"score": 分数, "reason": "简要评分理由"}
不要输出其他内容。"""


def build_prompt(question_text, rubric_keypoints, answer_text, examples):
    lines = [f"题目：{question_text}", "评分标准："]
    for kp in rubric_keypoints:
        lines.append(f"- 要点{kp['kp_id']}（{kp['kp_score']/20*4:.1f} 分）: {kp['kp_name']} | 判分: {kp['desc']}")
        lines.append(f"  标准: {kp['answer']}")
    lines.append("\n【同题评分示例（教师真实评分，用于校准尺度）】")
    for ex in examples:
        lines.append(f"\n示例{ex['band']}（{ex['human_text']}）：")
        lines.append(f"学生答案：{ex['answer_text']}")
        lines.append(f"教师评分：{ex['human_text']}")
    lines.append(f"\n【待评分学生答案】（0-4分制，满分4分）：\n{answer_text}")
    return "\n".join(lines)


def load_config():
    with open(ROOT / "normalin" / "configs" / "experiment_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["llm"]["api_mode"]


def score_one(client, qid, question_text, keypoints, answer_text, examples):
    prompt = build_prompt(question_text, keypoints, answer_text, examples)
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.15, max_tokens=8192, timeout=180,
            )
            raw = (resp.choices[0].message.content or "").strip()
            if not raw:
                raise ValueError("空返回")
            m = raw[raw.find("{"): raw.rfind("}") + 1]
            data = json.loads(m)
            score = float(data.get("score"))
            if score < 0 or score > 4:
                raise ValueError(f"分数越界: {score}")
            return score, str(data.get("reason", ""))[:100]
        except Exception as e:
            last = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
    raise RuntimeError(f"Q{qid} 直接评分失败: {last}")


def main():
    cfg = load_config()
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url", "https://api.deepseek.com/v1"), timeout=180)

    answers_data = json.load(open(ANSWERS, encoding="utf-8"))
    rubrics_data = json.load(open(RUBRICS, encoding="utf-8"))["rubrics"]
    fewshot = json.load(open(FEWSHOT, encoding="utf-8"))

    results = {}
    if OUT.exists():
        results = json.load(open(OUT, encoding="utf-8"))
        print(f"已有 {len(results)} 名学生直接评分（断点恢复）")

    tasks = []
    for stu in answers_data["students"]:
        sid = stu["student_id"]
        for a in stu["answers"]:
            qid = a["qid"]
            if sid in results and qid in results[sid]:
                continue
            tasks.append((sid, qid, a["answer_text"]))

    print(f"待评分 {len(tasks)} 条 (workers={WORKERS})")
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {}
        for sid, qid, ans in tasks:
            qtext = rubrics_data[qid]["question_text"]
            kps = rubrics_data[qid]["keypoints"]
            exs = [fewshot[qid][b] for b in ["high", "medium", "low"]]
            fut = ex.submit(score_one, client, qid, qtext, kps, ans, exs)
            futs[fut] = (sid, qid)
        for fut in as_completed(futs):
            sid, qid = futs[fut]
            try:
                score, reason = fut.result()
                results.setdefault(sid, {})[qid] = {"score": score, "reason": reason}
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(tasks)}", flush=True)
            except Exception as e:
                print(f"  {sid} Q{qid} 失败: {str(e)[:50]}", flush=True)
                results.setdefault(sid, {})[qid] = {"score": None, "reason": str(e)[:100]}
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    final = {}
    for sid, qs in results.items():
        vals = [qs[q].get("score") for q in QIDS]
        if None in vals:
            final[sid] = {"per_q": qs, "total_20": None, "total_100": None, "incomplete": True}
            continue
        t20 = sum(vals)
        final[sid] = {"per_q": qs, "total_20": round(t20, 1), "total_100": round(t20 / 20 * 100, 2), "incomplete": False}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    nok = sum(1 for v in final.values() if not v.get("incomplete"))
    print(f"\n完成: 完整学生={nok}/{len(final)}")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()