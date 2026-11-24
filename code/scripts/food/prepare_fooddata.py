


import json
import os
from collections import Counter
from pathlib import Path

import openpyxl

def _repo_root(start=None):
    """Walk up until the directory containing both code/ and data/ is found."""
    p = Path(start or __file__).resolve()
    for cand in [p.parent] + list(p.parents):
        if (cand / "code").is_dir() and (cand / "data").is_dir():
            return cand
    return p.parent


REPO_ROOT = _repo_root()


ROOT = Path(r"C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育")
SRC = REPO_ROOT / 'data' / 'food_class' / '食品工程原理简答题批阅39学生.xlsx'
OUT = REPO_ROOT / 'data' / 'food_class' / 'student_answers.json'

QMAX = 4.0        
PAPER_MAX = 20.0  


def load_rows(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["批改识别结果"]
    rows = []
    for r in ws.iter_rows(values_only=True):
        if r[0] is None or str(r[0]).strip() == "学生编号":
            continue
        s0 = str(r[0]).strip()
        if "合计" in s0 or "AI生成" in s0 or "仅供参考" in s0:
            continue
        
        rows.append({
            "sid": s0,
            "qid": str(r[1]).strip(),
            "qtext": str(r[2]).strip() if r[2] else "",
            "answer": str(r[3]).strip() if r[3] else "",
            "point_detail": str(r[4]).strip() if r[4] else "",
            "score": float(r[5]) if r[5] is not None else None,
            "paper_total": float(r[6]) if r[6] is not None else None,
        })
    return rows


def main():
    rows = load_rows(SRC)
    assert len(rows) == 39 * 5, f"行数异常: {len(rows)} (期望 195)"

    
    questions = {}
    for r in rows:
        if r["qid"] not in questions and r["qtext"]:
            questions[r["qid"]] = r["qtext"]
    assert set(questions) == {"1", "2", "3", "4", "5"}, f"题号异常: {list(questions)}"

    
    students = {}
    for r in rows:
        stu = students.setdefault(r["sid"], {"answers": {}})
        stu["answers"][r["qid"]] = {
            "answer_text": r["answer"],
            "point_detail": r["point_detail"],
            "human_score_4": r["score"],
        }

    
    student_list = []
    bad = []
    for sid in sorted(students, key=lambda x: int(x.replace("学生", ""))):
        d = students[sid]
        ans = d["answers"]
        if len(ans) != 5:
            bad.append(f"{sid}: 题数={len(ans)}")
            continue
        scores4 = [ans[q]["human_score_4"] for q in ["1", "2", "3", "4", "5"]]
        if any(s is None for s in scores4):
            bad.append(f"{sid}: 存在 None 分")
            continue
        total20 = sum(scores4)
        student_list.append({
            "student_id": sid,
            "answers": [
                {"qid": q,
                 "answer_text": ans[q]["answer_text"],
                 "point_detail": ans[q]["point_detail"],
                 "human_score_4": ans[q]["human_score_4"]}
                for q in ["1", "2", "3", "4", "5"]
            ],
            "human_total_20": round(total20, 2),
            "human_total_100": round(total20 / PAPER_MAX * 100.0, 2),
        })

    if bad:
        raise SystemExit("数据异常:\n" + "\n".join(bad))

    
    mismatch = []
    for r in rows:
        s5 = sum(a["human_score_4"] for a in students[r["sid"]]["answers"].values())
        if r["paper_total"] is not None and abs(s5 - r["paper_total"]) > 1e-6:
            mismatch.append((r["sid"], round(s5, 2), r["paper_total"]))
    if mismatch:
        print("[提示] 卷面总分列与5题和不等(以5题和为准):")
        for sid, s5, pt in mismatch:
            print(f"   {sid}: 5题和={s5}, 卷面列={pt}")

    
    question_weights_100 = {q: round(100.0 / 5, 4) for q in ["1", "2", "3", "4", "5"]}

    payload = {
        "metadata": {
            "n_students": len(student_list),
            "n_questions": 5,
            "question_max_score": QMAX,
            "paper_max_score": PAPER_MAX,
            "source": SRC.name,
            "note": "human_total_20 以5题得分之和为准",
        },
        "questions": {q: {"text": questions[q]} for q in ["1", "2", "3", "4", "5"]},
        "students": student_list,
        "question_weights_100": question_weights_100,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    
    dist = Counter()
    for s in student_list:
        for a in s["answers"]:
            dist[a["human_score_4"]] += 1
    total = [s["human_total_20"] for s in student_list]
    print(f"写盘: {OUT}")
    print(f"学生数={len(student_list)}, 题数=5, 每生答案数=5")
    print(f"4分制分数分布(逐题): {dict(sorted(dist.items(), reverse=True))}")
    print(f"卷面20分: 均值={sum(total)/len(total):.2f}, 范围={min(total)}-{max(total)}")
    print(f"100分制均值: {sum(s['human_total_100'] for s in student_list)/len(student_list):.2f}")


if __name__ == "__main__":
    main()