
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path

def extract_keypoint_scores_from_rubric(rubric_text, question_max_score=15):
    
    
    patterns = [
        r'评分要点\s*(\d+)[^（(]*?[（(](\d+(?:\.\d+)?)\s*分[）)]',
        r'第\s*(\d+)\s*问[^\d]*?(\d+(?:\.\d+)?)\s*分',
        r'[（(](\d+(?:\.\+)?)\s*分[）)]',  
    ]
    
    scores = {}
    for pattern in patterns:
        matches = re.findall(pattern, rubric_text)
        for match in matches:
            if isinstance(match, tuple):
                kp_id, score = match
                try:
                    scores[int(kp_id)] = float(score)
                except:
                    pass
            else:
                
                try:
                    score = float(match)
                    
                    kp_id = len(scores) + 1
                    scores[kp_id] = score
                except:
                    pass
    
    
    if scores:
        total = sum(scores.values())
        if abs(total - question_max_score) > 0.1:
            
            scale = question_max_score / total
            scores = {k: v * scale for k, v in scores.items()}
        return scores
    
    return None


def count_keypoints_in_rubric(rubric_text):
    
    
    kp_matches = re.findall(r'评分要点\s*\d+|第\s*\d+\s*问', rubric_text)
    return len(kp_matches) if kp_matches else 1


def allocate_scores_evenly(n_keypoints, total_score=15):
    
    base_score = total_score / n_keypoints
    return {i+1: base_score for i in range(n_keypoints)}


def infer_scores_from_human_scores(question_id, rubric_text, human_scores_df, question_max_score=15):
    
    n_keypoints = count_keypoints_in_rubric(rubric_text)
    
    
    scores = allocate_scores_evenly(n_keypoints, question_max_score)
    
    
    return scores, "LLM_inferred"


def load_keypoint_scores(rubrics_path, questions_path, human_data_path=None):
    
    with open(rubrics_path, 'r', encoding='utf-8') as f:
        rubrics = json.load(f)
    
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = {q['question_id']: q for q in json.load(f)}
    
    
    human_df = None
    if human_data_path and Path(human_data_path).exists():
        human_df = pd.read_excel(human_data_path, sheet_name='基准评分数据')
    
    keypoint_scores = {}
    score_sources = {}
    
    for qid, rubric in rubrics.items():
        q_max = questions.get(qid, {}).get('max_score', 15)
        
        
        scores = extract_keypoint_scores_from_rubric(rubric, q_max)
        
        if scores:
            keypoint_scores[qid] = scores
            score_sources[qid] = "rubric_extracted"
        else:
            
            human_col = f"{qid}_人工分" if human_df is not None else None
            human_scores = human_df[human_col] if human_col and human_col in human_df.columns else None
            
            scores, source = infer_scores_from_human_scores(
                qid, rubric, human_scores, q_max
            )
            keypoint_scores[qid] = scores
            score_sources[qid] = source
    
    return keypoint_scores, score_sources


def normalize_to_100(keypoint_scores, target_total=100):
    
    normalized = {}
    
    for qid, scores in keypoint_scores.items():
        original_total = sum(scores.values())
        if original_total > 0:
            scale = target_total / original_total
            normalized[qid] = {kp_id: score * scale for kp_id, score in scores.items()}
        else:
            normalized[qid] = scores
    
    return normalized


if __name__ == "__main__":
    
    rubrics_path = "C:/Users/zc/Documents/周翀/研究课题/LLM辅助教育/data/raw/rubrics.json"
    questions_path = "C:/Users/zc/Documents/周翀/研究课题/LLM辅助教育/data/raw/questions.json"
    human_data_path = "C:/Users/zc/Documents/周翀/研究课题/LLM辅助教育/data/raw/基准评分数据_人工与LLM_v2.xlsx"
    
    keypoint_scores, sources = load_keypoint_scores(rubrics_path, questions_path, human_data_path)
    
    print("=" * 80)
    print("要点分值分配结果")
    print("=" * 80)
    
    for qid in sorted(keypoint_scores.keys()):
        scores = keypoint_scores[qid]
        source = sources[qid]
        total = sum(scores.values())
        
        print(f"\n{qid} (来源: {source}):")
        for kp_id, score in sorted(scores.items()):
            print(f"  要点 {kp_id}: {score:.2f} 分")
        print(f"  总分: {total:.2f}")
    
    
    normalized = normalize_to_100(keypoint_scores, target_total=100)
    
    print("\n" + "=" * 80)
    print("归一化到 100 分后")
    print("=" * 80)
    
    for qid in sorted(normalized.keys()):
        scores = normalized[qid]
        total = sum(scores.values())
        print(f"\n{qid}:")
        for kp_id, score in sorted(scores.items()):
            print(f"  要点 {kp_id}: {score:.2f} 分")
        print(f"  总分: {total:.2f}")
    
    
    output_path = "C:/Users/zc/Documents/周翀/研究课题/LLM辅助教育/data/raw/keypoint_scores.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'original': keypoint_scores,
            'normalized_100': normalized,
            'sources': sources
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已保存到: {output_path}")
