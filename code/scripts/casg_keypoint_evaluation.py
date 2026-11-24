


import sys
import os
import json
import time
import re
import traceback
from pathlib import Path
from datetime import datetime

from openai import OpenAI
import yaml

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()



ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from casg_prompt_template import build_casg_system_prompt, build_casg_user_prompt


with open(ROOT / "normalin" / "configs" / "experiment_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

llm_cfg = config["llm"]["api_mode"]
client = OpenAI(
    api_key=llm_cfg["api_key"],
    base_url=llm_cfg.get("base_url", "https://api.deepseek.com/v1"),
    timeout=120,
)
MODEL = "deepseek-v4-flash"


def load_all_data():
    
    print("加载数据...")
    
    
    with open(ROOT / "normalin" / "data" / "processed" / "questions.json", "r", encoding="utf-8") as f:
        questions_list = json.load(f)
    questions = {q["question_id"]: q for q in questions_list}
    
    
    with open(ROOT / "normalin" / "data" / "processed" / "answers.json", "r", encoding="utf-8") as f:
        answers_list = json.load(f)
    
    answers = {}
    for item in answers_list:
        qid = item["question_id"]
        sid = item["student_id"]
        if qid not in answers:
            answers[qid] = {}
        answers[qid][sid] = item["answer"]
    
    
    with open(ROOT / "normalin" / "data" / "processed" / "rubrics.json", "r", encoding="utf-8") as f:
        rubrics = json.load(f)
    
    
    with open(ROOT / "normalin" / "data" / "processed" / "keypoint_scores.json", "r", encoding="utf-8") as f:
        keypoint_data = json.load(f)
    
    print(f"  题目数: {len(questions)}")
    print(f"  学生答案记录数: {len(answers_list)}")
    print(f"  评分要点题目数: {len(keypoint_data['normalized_100'])}")
    
    return questions, answers, rubrics, keypoint_data


def evaluate_single_student_answer(
    question_id: str,
    student_id: str,
    answer_text: str,
    question_text: str,
    rubric_data: dict,  
    keypoint_scores: dict,  
    verbose: bool = False
) -> dict:
    
    
    system_prompt = build_casg_system_prompt()
    user_prompt = build_casg_user_prompt(
        question_text, answer_text, rubric_data, question_id,
        keypoint_scores=keypoint_scores,
        score_source="fixed"
    )
    
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=4096,
        )
        raw_text = response.choices[0].message.content
    except Exception as e:
        if verbose:
            print(f"    LLM调用失败: {e}")
        return None
    
    
    try:
        
        json_match = None
        for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```', r'(\{.*\})']:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                json_match = match.group(1)
                break
        
        if not json_match:
            json_match = raw_text
        
        eval_data = json.loads(json_match)
        
        
        if "keypoints" not in eval_data or not isinstance(eval_data["keypoints"], list):
            if verbose:
                print(f"    JSON结构错误: 缺少keypoints")
            return None
        
        return eval_data
        
    except Exception as e:
        if verbose:
            print(f"    JSON解析失败: {e}")
            if verbose > 1:
                print(f"    原始响应: {raw_text[:500]}")
        return None


def run_evaluation():
    
    print("=" * 80)
    print("基于最新评分要点的逐要点覆盖度评估")
    print("=" * 80)
    
    
    questions, answers, rubrics, keypoint_data = load_all_data()
    
    
    output_dir = REPO_ROOT / 'data' / 'results' / 'casg_experiment' / 'keypoint_evaluations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    total_students = 0
    total_success = 0
    total_failed = 0
    
    
    for qid in sorted(answers.keys()):
        if qid not in questions:
            print(f"\n跳过 {qid}: 题目信息缺失")
            continue
        
        if qid not in rubrics:
            print(f"\n跳过 {qid}: 评分标准缺失")
            continue
        
        if qid not in keypoint_data['normalized_100']:
            print(f"\n跳过 {qid}: 评分要点缺失")
            continue
        
        question = questions[qid]
        rubric_data = rubrics[qid]  
        keypoint_scores = keypoint_data['normalized_100'][qid]
        
        print(f"\n{'='*60}")
        print(f"处理题目: {qid}")
        print(f"  题目: {question['text'][:100]}...")
        print(f"  评分要点数: {len(keypoint_scores)}")
        print(f"  学生数: {len(answers[qid])}")
        print(f"{'='*60}")
        
        
        question_results = {}
        
        
        for sid, answer_text in answers[qid].items():
            total_students += 1
            
            if not answer_text or not answer_text.strip():
                print(f"  学生 {sid}: 答案为空，跳过")
                continue
            
            print(f"  评估学生 {sid}...", end=" ")
            
            
            eval_data = evaluate_single_student_answer(
                question_id=qid,
                student_id=sid,
                answer_text=answer_text,
                question_text=question['text'],
                rubric_data=rubric_data,
                keypoint_scores=keypoint_scores,
                verbose=True
            )
            
            if eval_data is None:
                print("❌ 失败")
                total_failed += 1
                continue
            
            
            question_results[sid] = eval_data
            total_success += 1
            
            
            keypoints = eval_data.get('keypoints', [])
            n_full = sum(1 for kp in keypoints if kp.get('status') == 'full')
            n_partial = sum(1 for kp in keypoints if kp.get('status') == 'partial')
            n_missing = sum(1 for kp in keypoints if kp.get('status') == 'missing')
            print(f"✅ full={n_full}, partial={n_partial}, missing={n_missing}")
            
            
            time.sleep(0.5)
        
        
        output_file = output_dir / f"{qid}_evaluations.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(question_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n已保存 {qid} 评估结果: {len(question_results)} 学生")
    
    
    print(f"\n{'='*80}")
    print("评估完成")
    print(f"{'='*80}")
    print(f"总学生-题目组合数: {total_students}")
    print(f"成功: {total_success}")
    print(f"失败: {total_failed}")
    print(f"成功率: {total_success/total_students*100:.1f}%")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
