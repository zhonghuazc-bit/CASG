

import json
import re
import csv
import os
from pathlib import Path
from bs4 import BeautifulSoup
from html import unescape


def clean_html(text):
    
    if not text:
        return ""
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = unescape(text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_homework_html(filepath):
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    
    title_elem = soup.find('h2', class_='lookTit')
    homework_name = title_elem.get_text(strip=True) if title_elem else Path(filepath).stem
    
    
    hw_names = [
        "知识图谱存储与查询",
        "知识图谱推理",
        "知识抽取作业",
        "实体和关系抽取",
        "知识表示",
    ]
    homework_seq = 1
    for i, name in enumerate(hw_names):
        if name in homework_name or name in str(filepath):
            homework_seq = i + 1
            break
    
    results = {
        "homework_id": f"HW{homework_seq}",
        "homework_name": homework_name,
        "homework_seq": homework_seq,
        "subjective_questions": [],
    }
    
    
    type_divs = soup.find_all('div', class_='typeDiv')
    
    for type_div in type_divs:
        
        stem = type_div.find('div', class_='stem_con')
        if not stem:
            continue
        
        
        answer_div = type_div.find('div', class_='answerDiv')
        if not answer_div:
            continue
        
        
        answer_p = answer_div.find('div', class_='p_wid1000')
        answer_text = clean_html(str(answer_p)) if answer_p else clean_html(answer_div.get_text())
        
        
        is_subjective = '评分要点' in answer_text or '评分标准' in answer_text or len(answer_text) > 80
        
        
        if re.match(r'^答案[：:]\s*[对错ABCD]', answer_text[:20]):
            is_subjective = False
        
        if not is_subjective:
            continue
        
        
        stem_text = stem.get_text(strip=True)
        q_num_match = re.match(r'(\d+)\.', stem_text)
        q_num = int(q_num_match.group(1)) if q_num_match else len(results["subjective_questions"]) + 1
        
        question_id = f"HW{homework_seq}_Q{q_num}"
        
        
        question_text = re.sub(r'^\d+\.\s*', '', stem_text)
        question_text = re.sub(r'（\d+(?:\.\d+)?分\)\s*$', '', question_text).strip()
        
        
        score_match = re.search(r'（(\d+(?:\.\d+)?)分\)', stem_text)
        max_score = float(score_match.group(1)) if score_match else 15.0
        
        results["subjective_questions"].append({
            "question_id": question_id,
            "text": question_text,
            "max_score": max_score,
            "rubric": answer_text,
        })
    
    return results


def main():
    homework_files = [
        "C:/Users/zc/Downloads/知识图谱存储与查询.html",
        "C:/Users/zc/Downloads/知识图谱推理.html",
        "C:/Users/zc/Downloads/知识抽取作业.html",
        "C:/Users/zc/Downloads/实体和关系抽取.html",
        "C:/Users/zc/Downloads/知识表示(纳入最终考核）.html",
    ]
    
    output_dir = "C:/Users/zc/WorkBuddy/2026-05-12-task-13/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    all_questions = []
    all_rubrics = {}
    
    total_subj = 0
    total_all = 0
    
    for filepath in homework_files:
        if not os.path.exists(filepath):
            print(f"  [SKIP] Not found: {filepath}")
            continue
        
        print(f"\n  Parsing: {Path(filepath).name}")
        hw_info = parse_homework_html(filepath)
        n_subj = len(hw_info["subjective_questions"])
        total_subj += n_subj
        
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        total_all += len(re.findall(r'class="stem_con"', content))
        
        print(f"    Total questions: {total_all}")
        print(f"    Subjective (with rubric): {n_subj}")
        
        for q in hw_info["subjective_questions"]:
            all_questions.append({
                "question_id": q["question_id"],
                "text": q["text"],
                "max_score": q["max_score"],
                "topic": hw_info["homework_name"],
            })
            all_rubrics[q["question_id"]] = q["rubric"]
    
    print(f"\n  === Summary ===")
    print(f"  Total subjective questions extracted: {total_subj}")
    
    
    questions_path = os.path.join(output_dir, "questions.json")
    with open(questions_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {questions_path}")
    
    
    rubrics_path = os.path.join(output_dir, "rubrics.json")
    with open(rubrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_rubrics, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {rubrics_path}")
    
    
    scores_path = os.path.join(output_dir, "scores_template.csv")
    with open(scores_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['student_id', 'question_id', 'score'])
        for q in all_questions:
            writer.writerow(['S001', q['question_id'], ''])
    print(f"  Saved: {scores_path} (fill in actual student scores)")
    
    
    print(f"\n  === Extracted Subjective Questions ({len(all_questions)}) ===")
    for q in all_questions:
        rubric_preview = all_rubrics[q['question_id']][:60]
        print(f"    {q['question_id']}: {q['text'][:50]}... (max={q['max_score']})")
        print(f"      Rubric: {rubric_preview}...")
        print()


if __name__ == "__main__":
    main()
