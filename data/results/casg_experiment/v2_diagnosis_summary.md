# CASG v2 诊断报告

## 核心发现

### 1. LLM 评估数据质量问题

检查 HW1_Q3 的评估数据后发现严重问题：

**典型错误案例**（学生 S001，要点 1）：
- `evidence`: "学生答案明确列出了两种方法：子图匹配法和SQL转换法，与评分要点要求一致，完全覆盖。"
- `covered_subpoints`: ["提出子图匹配法", "提出SQL转换法"]
- `missing_subpoints`: []
- `reasoning`: "学生答案明确列出了两种方法..."
- **但 `status` = "missing"** ← 错误！

**根因分析**：
1. LLM 在 `` 标签中进行了详细推理，正确判断学生覆盖了要点
2. 但在生成 JSON 时，`status` 字段给出了错误的判断
3. 这可能是由于：
   - Prompt 设计问题：没有明确要求 LLM 根据推理结果正确设置 status
   - LLM 理解偏差：可能将 "evidence" 和 "status" 混淆
   - 输出格式问题：JSON 结构复杂，LLM 在生成时出错

### 2. 类型匹配问题（已修复）

**问题**：`keypoint_scores.json` 中的 key 是字符串（如 "1"），但 pipeline 用 int 查找

**影响**：导致所有 keypoint 的 `max_score` 被强制设为 0，最终所有分数为 0

**修复**：统一转换为字符串进行匹配（已完成）

### 3. 评估数据统计

修复数据后，LLM 的评估分布：
- full: 379 (41.2%)
- partial: 84 (9.1%)
- **missing: 456 (49.6%)** ← 近一半！

这个比例明显不合理，说明 LLM 在 Stage 1/2 的评估存在系统性偏差。

## 方案对比结果

| 方案 | P值 | QWK | Spearman ρ |
|------|-----|-----|------------|
| v1 (θ仅partial) | 0.5415 | 0.0337 | 0.2550 |
| **v2 (θ影响full+missing基础分)** | **0.4094** | 0.0729 | 0.2718 |
| v2b (θ影响full,missing=0) | 0.5341 | 0.0369 | 0.2767 |
| LLM v2 基线 | 0.2274 | **0.4503** | **0.3931** |

**结论**：
1. v2 方案确实改善了 P 值（0.54→0.41），更接近目标
2. 但所有 CASG 方案的 QWK 和 ρ 仍远低于 LLM v2 基线
3. **根本问题**：Stage 1/2 的评估质量太差，即使 Stage 3 再怎么优化 θ 也无法弥补

## 解决方案

### 方案 A：修复 Stage 1/2 评估质量（推荐）

**核心思路**：在 Stage 1/2 层面修复数据质量问题，而非仅在 Stage 3 调整

**具体措施**：

1. **优化 Prompt 设计**：
   - 明确要求 LLM 根据 evidence 和 reasoning 正确设置 status
   - 添加约束：如果 covered_subpoints 非空且 missing_subpoints 为空，则 status 必须是 "full"
   - 添加示例，引导 LLM 正确输出

2. **后处理修正**：
   - 在 Stage 1/2 完成后，根据 evidence、covered_subpoints、missing_subpoints 自动修正 status
   - 例如：如果 covered_subpoints 包含所有子要点，且 missing_subpoints 为空，则 status = "full"

3. **重新运行 Stage 1/2**：
   - 使用优化后的 prompt 重新调用 LLM
   - 或者使用后处理修正现有数据

### 方案 B：跳过 Stage 1/2，直接使用 LLM v2 分数

**核心思路**：如果 Stage 1/2 质量无法保证，可以考虑跳过

**具体措施**：
- 直接使用 LLM v2 的评分结果
- 在 Stage 3 使用 θ 进行分布调整
- 这实际上就是 Normalin 框架的原始设计

### 方案 C：混合方案

**核心思路**：结合 Stage 1/2 和 LLM v2 的优势

**具体措施**：
- 对于 Stage 1/2 评估质量高的题目（如 full 比例合理的），使用 CASG 方案
- 对于评估质量低的题目，直接使用 LLM v2 分数

## 建议

**推荐方案 A**，因为：
1. CASG 的核心价值在于通过结构化评估（覆盖度分析）来提供更细粒度的评分
2. 如果 Stage 1/2 质量无法保证，CASG 的优势就无法发挥
3. 优化 prompt 或后处理修正是相对低成本的工作

**具体实施步骤**：

1. **短期**：使用后处理修正现有数据
   - 根据 evidence、covered_subpoints、missing_subpoints 自动修正 status
   - 重新计算分数，验证效果

2. **中期**：优化 prompt 设计
   - 添加更明确的约束和示例
   - 重新运行 Stage 1/2

3. **长期**：评估 CASG 与 LLM v2 的优劣
   - 对比两种方案的 QWK、ρ 等指标
   - 确定最终方案

## 下一步行动

1. 实现后处理修正逻辑
2. 重新计算分数，验证效果
3. 如果效果改善，继续优化 prompt
4. 如果效果仍不理想，考虑方案 B 或 C
