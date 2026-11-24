


def build_casg_system_prompt():
    
    return """你是一位严谨的评分分析师。你的任务是对每个评分要点进行精确的覆盖度评估。

== 两层评判标准 ==

每个要点包含：
- 【评分要点】：核心主题，判断学生是否涉及该要点
- 【详细答案】：完整标准答案，判断学生是否全面、准确

评判逻辑：
1. 对照【评分要点】判断是否涉及主题
2. 严格对照【详细答案】逐一检查是否覆盖所有关键概念
3. 遗漏详细答案中任何关键概念 → 判为partial

== 严格性要求 ==

**关键规则：不能因为"核心思想等价"就判定为full**
- 详细答案中的每个专业术语、具体数字、关键概念都必须被覆盖
- "等价表述"指完全不同的词但表达完全相同的概念（如"自连接"和"Self-join"）
- 以下情况不是等价表述：
  - 只提到现象但没提到原因/机制
  - 只提到部分要素但遗漏其他要素
  - 用模糊描述替代具体专业术语

== 覆盖状态判定 ==

- "full"：完整覆盖【详细答案】中所有关键概念
- "partial"：涉及主题但遗漏一个或多个关键概念
- "missing"：完全未涉及该要点主题

== coverage_ratio 计算 ==

对partial要点：
- coverage_ratio = 覆盖的关键概念数 / 详细答案中的关键概念总数
- 必须引用学生答案原文作为依据

== 输出格式 ==

以JSON格式输出：
{
  "question_id": "题目ID",
  "total_keypoints": 要点总数,
  "keypoints": [
    {
      "id": 1,
      "name": "要点名称",
      "max_score": 分值,
      "status": "full" | "partial" | "missing",
      "coverage_ratio": 0.0-1.0,
      "quality_band": "high" | "medium" | "low" | null,
      "evidence": "引用学生答案原文",
      "covered_subpoints": ["覆盖的概念1", ...],
      "missing_subpoints": ["遗漏的概念1", ...],
      "reasoning": "与详细答案的对照说明"
    }
  ],
  "summary": {
    "n_full": 完全覆盖数,
    "n_partial": 部分覆盖数,
    "n_missing": 未覆盖数,
    "overall_coverage": 总覆盖率
  }
}

== 注意事项 ==

1. 只评估覆盖情况，不计算最终分数
2. 严禁以"核心思想等价"为由忽略具体概念缺失
3. evidence必须引用学生答案原文
4. 有疑问时判为partial而非full"""


def build_casg_user_prompt(question_text, answer_text, rubric_data, question_id, keypoint_scores=None, score_source=None):
    
    answer_text = (answer_text or "").strip()[:3000]
    
    
    if isinstance(rubric_data, dict) and 'keypoints' in rubric_data:
        
        rubric_lines = []
        for kp in rubric_data['keypoints']:
            kp_id = kp['kp_id']
            kp_name = kp['kp_name']
            desc = kp['desc']
            answer = kp['answer']
            
            rubric_lines.append(f"要点 {kp_id}：{kp_name}")
            rubric_lines.append(f"  【评分要点】{desc}")
            rubric_lines.append(f"  【详细答案】{answer}")
            rubric_lines.append("")
        
        rubric_text = "\n".join(rubric_lines).strip()
    else:
        
        rubric_text = rubric_data if isinstance(rubric_data, str) else str(rubric_data)
    
    
    if keypoint_scores:
        kp_score_lines = []
        for kp_id in sorted(keypoint_scores.keys()):
            score = keypoint_scores[kp_id]
            kp_score_lines.append(f"  - 要点 {kp_id}: {score:.2f} 分")
        
        kp_scores_text = "\n".join(kp_score_lines)
        source_note = f"（分值来源: {score_source}）" if score_source else ""
        
        scores_section = f"""
【各要点分值分配】{source_note}
{kp_scores_text}

**重要**: 你在输出 JSON 时，必须使用上述固定分值作为每个要点的 max_score，不要自行分配分值。
"""
    else:
        scores_section = """
【注意】未提供要点分值分配。请根据评分标准合理分配各要点分值，确保总分 = 100 分。
"""
    
    return f"""请对以下学生答案进行逐要点覆盖度评估。

=== 题目信息 ===
题目ID: {question_id}

【题目内容】
{question_text}

【评分标准】
{rubric_text}
{scores_section}
【学生答案】
{answer_text}

请严格按照 JSON 格式输出评估结果，不要包含其他内容。"""
