


import json
from pathlib import Path

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
OUT = REPO_ROOT / 'data' / 'food_class' / 'few_shot_examples.json'

QIDS = ["1", "2", "3", "4", "5"]
ANSWER_MAX_CHARS = 600  


def pick_examples(records):
    
    
    high = max(records, key=lambda r: r["human_score_4"])
    
    low = min(records, key=lambda r: r["human_score_4"])
    
    rest = [r for r in records if r["student_id"] not in (high["student_id"], low["student_id"])]
    medium = min(rest, key=lambda r: abs(r["human_score_4"] - 2.0))
    return {"high": high, "medium": medium, "low": low}


def main():
    data = json.load(open(ANSWERS, encoding="utf-8"))
    
    by_q = {q: [] for q in QIDS}
    for stu in data["students"]:
        sid = stu["student_id"]
        for a in stu["answers"]:
            qid = a["qid"]
            by_q[qid].append({
                "student_id": sid,
                "qid": qid,
                "answer_text": (a["answer_text"] or "").strip(),
                "human_score_4": float(a.get("human_score_4", 0.0)),
            })

    out = {}
    print(f"{'题':<4}{'档位':<8}{'学生':<10}{'人工分':>8}  示例答案前80字")
    print("-" * 90)
    for qid in QIDS:
        recs = by_q[qid]
        picked = pick_examples(recs)
        out[qid] = {}
        for band in ["high", "medium", "low"]:
            r = picked[band]
            ex = {
                "band": band,
                "student_id": r["student_id"],
                "qid": qid,
                "answer_text": r["answer_text"][:ANSWER_MAX_CHARS],
                "human_score_4": r["human_score_4"],
                "human_text": f"{r['human_score_4']:.1f}/4.0",
            }
            out[qid][band] = ex
            preview = r["answer_text"][:80].replace("\n", " ")
            print(f"Q{qid:<3}{band:<8}{r['student_id']:<10}{r['human_score_4']:>8.1f}  {preview}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入: {OUT}")


if __name__ == "__main__":
    main()