


import json
import os
import re
import sys
import time
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

ANSWERS = REPO_ROOT / 'data' / 'food_class' / 'student_answers.json'
OUT = REPO_ROOT / 'data' / 'food_class' / 'rubrics_食品34要点.json'

MAX_FULL_SAMPLES = 3  
PER_Q_WEIGHT = 20.0  

QIDS = ["1", "2", "3", "4", "5"]

with open(ROOT / "normalin" / "configs" / "experiment_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
llm_cfg = config["llm"]["api_mode"]
client = OpenAI(
    api_key=llm_cfg["api_key"],
    base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
    timeout=120,
)
MODEL = "deepseek-v4-flash"

BUILD_PROMPT = """你是一位食品工程原理课程的资深命题教师。请根据以下 3 份【学生满分作答】（代表该题最优、完整、稳定的标准表述），提炼该简答题的评分要点量规。

要求：
1. 只纳入 3 份满分作答中共有、稳定出现的评分要点；剔除个别作答中偶发、含糊、错误或偏离主题的表述。
2. 要点数量少而精（2-4 个），宁缺毋滥，覆盖题干所问的全部核心环节（概念/表达式/物理意义/分类/区别/机理步骤等）。
3. 每个要点包含：
   - kp_name: 要点名称（短，如 "离心泵工作原理"）
   - kp_score: 该要点分值（100分制，∑=20）
   - desc: 评分要点描述（判分依据，回答须涉及什么）
   - answer: 详细标准答案（综合满分作答的准确完整表述，含关键专业术语、公式、数值、步骤）
   - subpoints: 该要点的子要点列表（2-5 个，用列表形式，如 ["叶轮高速旋转", "离心力甩出液体", "从排出口排出"]）。子要点用于逐项判定该要点的部分覆盖程度，应是该要点(answer)中可独立判定存在与否的最小概念单元，互不重叠。
4. 分值分配 kp_score（本题总分为 20 分，按各要点重要程度分配，20=各要点 kp_score 之和，允许 .5 分）：
   —— 注意这里 20 分是"该题在整卷100分制中的权重"，不是卷面4分；要点分值须加总恰好等于 20。
5. 严格以 JSON 数组输出，不要其他内容：
[{"kp_id": 1, "kp_name": "...", "kp_score": 8.0, "desc": "...", "answer": "...", "subpoints": ["...", "..."]}]

题目：__QUESTION__
【满分作答1】
__ANS1__
【满分作答2】
__ANS2__
【满分作答3】
__ANS3__
"""


def parse_kps(raw):
    
    text = (raw or "").strip()
    
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        obj = json.loads(m.group(0))
    else:
        obj = json.loads(text)
    if isinstance(obj, dict):
        for k in ("keypoints", "points", "rubric"):
            if k in obj and isinstance(obj[k], list):
                obj = obj[k]
                break
    if not isinstance(obj, list):
        raise ValueError("非数组输出")
    
    out = []
    for i, kp in enumerate(obj):
        sub = kp.get("subpoints")
        if not isinstance(sub, list):
            sub = []
        out.append({
            "kp_id": int(str(kp.get("kp_id", i + 1)).replace("kp", "")),
            "kp_name": str(kp.get("kp_name", f"要点{i+1}")),
            "kp_score": float(kp.get("kp_score", 0)),
            "desc": str(kp.get("desc", "")),
            "answer": str(kp.get("answer", "")),
            "subpoints": [str(x) for x in sub],
        })
    return out


def call_llm(question_text, samples, qid):
    user = (BUILD_PROMPT
            .replace("__QUESTION__", question_text)
            .replace("__ANS1__", samples[0])
            .replace("__ANS2__", samples[1] if len(samples) > 1 else "")
            .replace("__ANS3__", samples[2] if len(samples) > 2 else ""))
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你输出严格 JSON。"},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=8192,
                timeout=180,
            )
            raw = resp.choices[0].message.content
            
            dbg_dir = REPO_ROOT / 'data' / 'food_class' / 'rubric_build_raw'
            dbg_dir.mkdir(parents=True, exist_ok=True)
            with open(dbg_dir / f"Q{qid}_raw.txt", "w", encoding="utf-8") as f:
                f.write(raw or "")
            if not raw or not raw.strip():
                raise ValueError("模型返回空内容")
            kps = parse_kps(raw)
            
            total = sum(float(k["kp_score"]) for k in kps)
            if abs(total - PER_Q_WEIGHT) > 1e-6 and kps:
                scale = PER_Q_WEIGHT / total
                for k in kps:
                    k["kp_score"] = round(float(k["kp_score"]) * scale, 1)
            return kps
        except Exception as e:
            last_err = e
            print(f"  Q{qid} 尝试{attempt+1}失败: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(5)
    raise RuntimeError(f"Q{qid} 量规生成失败: {last_err}")


def load_answers():
    with open(ANSWERS, encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_full_samples(data, qid, n=MAX_FULL_SAMPLES):
    samples = []
    for s in data["students"]:
        for a in s["answers"]:
            if a["qid"] == qid and abs(a["human_score_4"] - 4.0) < 1e-6:
                samples.append(a["answer_text"])
                break
    
    seen, out = set(), []
    for t in samples:
        t = (t or "").strip()[:800]
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= n:
            break
    return out


def main():
    data = load_answers()
    
    rubrics = {}
    if OUT.exists():
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            rubrics = prev.get("rubrics", {})
            print(f"加载已有 {len(rubrics)} 题量规")
        except Exception:
            rubrics = {}
    for qid in QIDS:
        if qid in rubrics and rubrics[qid].get("keypoints"):
            print(f"Q{qid}: 已有量规({len(rubrics[qid]['keypoints'])}个要点), 跳过")
            continue
        qtext = data["questions"][qid]["text"]
        samples = get_full_samples(data, qid)
        print(f"Q{qid}: 满分作答样本数={len(samples)}")
        if not samples:
            raise SystemExit(f"Q{qid} 无满分作答, 无法构建量规")
        kps = call_llm(qtext, samples, qid)
        rubrics[qid] = {
            "question_text": qtext,
            "q_total": PER_Q_WEIGHT,
            "full_samples_used": samples,
            "keypoints": kps,
        }
        print(f"  -> {len(kps)} 个要点, 分值总和={sum(float(k['kp_score']) for k in kps)}")
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"metadata": {"source": ANSWERS.name, "model": MODEL,
                                    "q_total_per_question_100": PER_Q_WEIGHT,
                                    "note": "由每题满分作答经 LLM 生成，待人工复核。复核后各要点 kp_score 之和应为 20。"},
                       "rubrics": rubrics}, f, ensure_ascii=False, indent=2)
        time.sleep(1)

    print(f"写盘完成: {OUT}")


if __name__ == "__main__":
    main()