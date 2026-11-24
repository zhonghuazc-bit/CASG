


import json
import time
import csv
import os
import re
from pathlib import Path
from collections import defaultdict
import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

import numpy as np
from openai import OpenAI, APIError, RateLimitError






FEWSHOT_HIGH = {
    "question": "远程监督数据构造与 RL 策略选择器降噪。知识库中已知三元组<乔布斯, 联合创始人, 苹果公司>。从新闻中抓取三句话：(1)乔布斯是苹果公司的联合创始人。(2)乔布斯重返苹果公司担任CEO。(3)苹果公司的现任CEO库克非常敬佩乔布斯。问：(1)按远程监督基本假设，系统会给这三句话自动打什么关系标签？(2)哪几句是噪声？为什么？(3)简述如何利用强化学习训练策略选择器来缓解噪声问题？",
    "rubric": "(1)根据远程监督基本假设，只要句子包含该实体对，都会被自动标注为知识库中的关系，即三句话均标注为\"联合创始人\"。(2)句子2和句子3是噪声。因为句子2表达\"担任CEO\"，句子3表达\"敬佩\"，均不表达\"联合创始人\"关系。(3)利用策略选择器先选择部分样本，基于这些样本训练关系分类模型。若选出子集使分类模型在训练集取得较高准确率，则用模型预测置信度作为奖励分数返回更新策略选择器，不断循环迭代直至稳定。",
    "answer": "(1)自动标签：三句话均标注为联合创始人。远程监督假设：若句子包含知识库三元组的实体对，则该句子表达此三元组关系。(2)噪声句子：句子2、句子3。原因：句子2表达CEO任职关系，句子3表达敬佩关系，均非\"联合创始人\"关系，属于错误标注噪声。(3)RL策略选择器降噪思路：1.将样本筛选建模为决策任务，策略选择器输出\"保留/丢弃\"样本的动作；2.以模型抽取准确率为奖励信号，奖励高质量样本、惩罚噪声样本；3.通过强化学习迭代优化策略，自动筛选出真实表达目标关系的句子，剔除噪声；4.最终仅用高质量样本训练关系抽取模型，降低远程监督噪声影响。",
    "score": 83,
    "analysis": "答案结构清晰，分点作答，正确识别了三句话均被标注为联合创始人，准确判定了噪声句子及原因，并完整描述了策略选择器的四步工作流程。扣分原因：策略选择器的强化学习反馈机制描述可以更精准。"
}

FEWSHOT_MEDIUM = {
    "question": "请简述目前实现 SPARQL 查询的两种核心底层方法，并指明哪一种方法能够更直接地反映图谱的网状特征及其背后的本质含义。",
    "rubric": "评分要点1：两种核心实现方法——基于关系型数据库查询的方法和基于图数据库（子图）查询的方法。评分要点2：关系型转换的劣势——将三元组存为关系表并转换为SQL语句时，SPARQL查询经常包含多个三元组，会导致生成的SQL语句中包含多个自连接操作，执行效率低下。评分要点3：子图匹配的本质——子图查询方法比转换SQL更直接。在这种机制下，一个SPARQL查询在本质上被看作是一个\"小规模的查询图\"。引擎在底层图谱中寻找与该查询图同构的数据片段，完美契合图数据的网状拓扑本质。",
    "answer": "SPARQL查询的两种核心底层实现方法：基于三元组匹配的迭代法——将SPARQL拆解为三元组模式，逐一对知识图谱的三元组库进行匹配、筛选、连接，本质是表查询/匹配思路，依赖大量的查找、合并、JOIN操作。基于图遍历的索引法——直接将SPARQL翻译为图节点-关系的导航遍历，从起点实体出发，沿着关系边跳转访问关联节点，完全贴合图的网状结构。能直接反映图谱网状特征的方法：基于图遍历的索引法更直接反映知识图谱的网状结构与本质含义，它以实体为节点、关系为边进行物理导航，完美匹配图的拓扑模型，而非模拟二维表匹配。",
    "score": 60,
    "analysis": "答案正确识别了两种方法但表述不够学术化，将\"子图匹配/图数据库查询\"换成了\"图遍历索引法\"（概念混淆）。关键失分点：未能明确指出关系型方案中\"自连接(Self-join)会导致效率低下\"这一核心缺陷，只是在概念层面描述而缺少技术层面的深入分析。"
}

FEWSHOT_LOW = {
    "question": "在公安系统构建\"家庭关系图谱推理引擎\"时，需要推导\"daughter(X,Y)\"这样的关系。请根据以下背景知识，用一阶逻辑规则的形式写出推导daughter关系的规则。背景知识：parent(eve, tom)、parent(eve, ann)、parent(tom, mary)、parent(ann, jim)、female(eve)、female(ann)、female(mary)。",
    "rubric": "规则应正确覆盖正例(mary, ann)和(eve, tom)，排除反例(如tom—不是女性、eve-ann不存在parent(ann,eve))。规则形式应为：daughter(X,Y) :- female(X), parent(Y,X)。满分要求：零错误、零遗漏、可解释、可验证。",
    "answer": "这条规则：覆盖全部2个正例：(mary, ann)、(eve, tom)；排除全部2个反例：tom因不是女性被排除，eve-ann因不存在parent(ann, eve)被排除。零错误、零遗漏、可解释、可验证——完全符合公安系统对严谨公理的要求",
    "score": 27,
    "analysis": "答案严重缺失：没有写出实际的一阶逻辑规则表达式（如 daughter(X,Y) :- female(X), parent(Y,X)）。虽声称\"零错误、零遗漏\"，但完全未提供规则本身——这是题目的核心要求。只做了分析性的口头描述，缺少形式化的规则定义。"
}


def build_few_shot_block():
    
    examples = [
        ("【高分示例 - 83/100】", FEWSHOT_HIGH),
        ("【中等分示例 - 60/100】", FEWSHOT_MEDIUM),
        ("【低分示例 - 27/100】", FEWSHOT_LOW),
    ]
    parts = []
    for label, ex in examples:
        parts.append(f"""{label}
题目：{ex['question']}
评分标准：{ex['rubric']}
学生答案：{ex['answer']}
得分：{ex['score']}/100
评分分析：{ex['analysis']}""")
    return "\n\n".join(parts)


def load_config():
    config_path = ROOT / "configs" / "experiment_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data():
    processed = ROOT / "data" / "processed"
    with open(processed / "questions.json", "r", encoding="utf-8") as f:
        questions = {q["question_id"]: q for q in json.load(f)}
    with open(processed / "answers.json", "r", encoding="utf-8") as f:
        answers = json.load(f)
    with open(processed / "rubrics.json", "r", encoding="utf-8") as f:
        rubrics = json.load(f)
    return questions, answers, rubrics


def rubric_to_text(rubric):
    if isinstance(rubric, str):
        return rubric
    if isinstance(rubric, dict):
        parts = []
        if "key_points" in rubric:
            for i, point in enumerate(rubric["key_points"], 1):
                desc = point.get("description", "")
                max_p = point.get("max_points", "")
                parts.append(f"{i}. {desc}（最高 {max_p} 分）")
        if "total_points" in rubric:
            parts.append(f"\n满分：{rubric['total_points']} 分")
        return "\n".join(parts) if parts else str(rubric)
    return str(rubric)


def build_system_prompt():
    return """你是一位严格但公正的大学课程助教。你的任务是对学生的简答题答案进行评分（0-100分）。

评分原则：
1. 严格对照评分标准逐条审核，答案覆盖哪条给对应分数
2. 答非所问或完全偏离题目 → 0-20分
3. 部分正确但不完整或缺少关键点 → 30-60分  
4. 基本正确，覆盖大部分要点但表述有瑕疵 → 60-85分
5. 完全正确，覆盖所有要点且表述精准 → 85-100分

具体步骤：
第一步：判断答案是否与题目相关。如果不相关，直接给低分。
第二步：逐条比对评分标准中的每个要点，标记"已覆盖"/"部分覆盖"/"未覆盖"。
第三步：根据覆盖情况和答案质量综合定分。
第四步：输出评分分析，然后输出最终分数。"""


def build_user_prompt(question, answer, rubric_text):
    answer = (answer or "").strip()[:3000]
    
    few_shot = build_few_shot_block()
    
    prompt = f"""以下是你需要批改的题目。请先阅读下面的3个评分示例，理解评分标准，再对当前题目进行评分。

=== 评分示例 ===
{few_shot}

=== 当前题目 ===
【题目】
{question}

【评分标准】
{rubric_text}

【学生答案】
{answer}

请按以下格式输出你的评分：
【评分分析】
（简要分析：答案是否相关、覆盖了哪些要点、缺失了哪些要点、总体评价）

【分数】
X（仅输出0-100的整数）"""
    return prompt


def parse_score_v2(text):
    
    text = text.strip()
    
    
    m = re.search(r'【分数】\s*\n?\s*(\d+)', text)
    if m:
        score = int(m.group(1))
        if 0 <= score <= 100:
            return score
    
    
    m = re.search(r'分数[：:]\s*(\d+)', text)
    if m:
        score = int(m.group(1))
        if 0 <= score <= 100:
            return score
    
    
    nums = re.findall(r'\b(\d+)\b', text)
    for raw in reversed(nums):
        score = int(raw)
        if 0 <= score <= 100:
            return score
    
    return None


class ImprovedDeepSeekGrader:
    def __init__(self, config):
        llm_cfg = config["llm"]["api_mode"]
        self.client = OpenAI(
            api_key=llm_cfg["api_key"],
            base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
            timeout=120,
        )
        self.model = llm_cfg.get("model", "deepseek-chat")
        self.system_prompt = build_system_prompt()

    def grade(self, question, answer, rubric, temperature=0.15, max_retries=3, delay=3):
        prompt = build_user_prompt(question, answer, rubric_to_text(rubric))
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=800,  
                )
                raw = response.choices[0].message.content
                score = parse_score_v2(raw)
                if score is None:
                    print(f"    WARNING: Could not parse score, raw: {raw[:200]}")
                    
                    continue
                return {
                    "score": score,
                    "raw_response": raw,
                    "tokens": response.usage.total_tokens if response.usage else None,
                }
            except (APIError, RateLimitError) as e:
                wait = delay * (2 ** attempt)
                print(f"    API error (attempt {attempt+1}/{max_retries}): {type(e).__name__}. Retry in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"    Error (attempt {attempt+1}/{max_retries}): {e}. Retry in {delay}s...")
                time.sleep(delay)
        return {"score": None, "raw_response": "", "tokens": None, "error": "max_retries"}


def main():
    config = load_config()
    questions, answers, rubrics = load_data()
    grader = ImprovedDeepSeekGrader(config)

    n_total = len(answers)
    print("=" * 60)
    print("DeepSeek LLM 评分 v2 — 带 Few-Shot 示例")
    print(f"模型: {grader.model}")
    print(f"待评分答案: {n_total} 条")
    print("=" * 60)

    results = []
    fail_count = 0

    for idx, rec in enumerate(answers, 1):
        qid = rec["question_id"]
        sid = rec["student_id"]
        q = questions.get(qid, {})
        question_text = q.get("text", "[题目缺失]")
        rubric = rubrics.get(qid, "[评分标准缺失]")
        answer_text = rec.get("answer", "")

        print(f"[{idx}/{n_total}] {sid} / {qid} ...", end=" ", flush=True)
        res = grader.grade(question_text, answer_text, rubric)

        if res["score"] is None:
            fail_count += 1
            print(f"FAILED")
        else:
            print(f"score={res['score']}")

        results.append({
            "student_id": sid,
            "question_id": qid,
            "score": res["score"],
            "raw_response": res.get("raw_response", ""),
            "tokens": res.get("tokens"),
            "error": res.get("error"),
        })

        
        if idx % 30 == 0 or idx == n_total:
            out_path = ROOT / "data" / "processed" / "llm_scores_v2_partial.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  -> 已保存评分结果 ({idx}/{n_total})")

        time.sleep(0.5)

    
    out_path = ROOT / "data" / "processed" / "llm_scores_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n最终评分结果已保存: {out_path}")
    print(f"成功: {n_total - fail_count}/{n_total}, 失败: {fail_count}/{n_total}")

    build_aggregated_csv(results, questions)


def build_aggregated_csv(results, questions):
    scores_by_student = defaultdict(dict)
    for r in results:
        if r["score"] is not None:
            scores_by_student[r["student_id"]][r["question_id"]] = r["score"]

    qids = sorted(questions.keys())
    students = sorted(scores_by_student.keys())

    csv_path = ROOT / "data" / "processed" / "llm_scores_v2_matrix.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["student_id"] + qids)
        for sid in students:
            row = [sid]
            for qid in qids:
                row.append(scores_by_student[sid].get(qid, ""))
            writer.writerow(row)
    print(f"分数矩阵 CSV: {csv_path}")

    totals = []
    for sid, qscores in scores_by_student.items():
        vals = [v for v in qscores.values() if v is not None]
        if vals:
            totals.append({"student_id": sid, "total": sum(vals), "answered": len(vals)})

    total_csv = ROOT / "data" / "processed" / "llm_total_scores_v2.csv"
    with open(total_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["student_id", "answered", "total"])
        writer.writeheader()
        for row in sorted(totals, key=lambda x: x["total"], reverse=True):
            writer.writerow(row)
    print(f"总分 CSV: {total_csv}")

    if totals:
        totals_arr = np.array([t["total"] for t in totals])
        print(f"\n=== 原始 LLM 总分分布 (v2) ===")
        print(f"  人数: {len(totals_arr)}")
        print(f"  均值: {totals_arr.mean():.2f}")
        print(f"  标准差: {totals_arr.std():.2f}")
        print(f"  最小: {totals_arr.min():.2f}")
        print(f"  最大: {totals_arr.max():.2f}")


if __name__ == "__main__":
    main()
