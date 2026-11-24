
import json
import re
from typing import Optional, Dict, Any


class CASGGrader:
    

    valid_statuses = {"full", "partial", "missing"}

    def __init__(self, fixed_max_scores: Optional[Dict[int, float]] = None):
        
        self.fixed_max_scores = fixed_max_scores

    def grade_answer(self, raw_text: str, question_id: Optional[str] = None) -> Optional[Dict]:
        
        eval_data = self.parse_evaluation(raw_text)
        if eval_data is None:
            return None

        
        eval_data = self._validate_evaluation(eval_data, question_id)
        return eval_data

    def parse_evaluation(self, raw_text: str) -> Optional[Dict]:
        
        if not raw_text or not isinstance(raw_text, str):
            return None

        
        cleaned = raw_text
        if "```" in cleaned:
            cleaned = "\n".join(line for line in cleaned.split("\n") if not line.startswith("```"))

        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            
            pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
            matches = re.findall(pattern, cleaned, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

            
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end+1])
                except json.JSONDecodeError:
                    pass

            return None

    def _validate_evaluation(self, eval_data: Dict, question_id: Optional[str] = None) -> Optional[Dict]:
        
        if not isinstance(eval_data, dict):
            return None

        
        if "keypoints" not in eval_data:
            return None

        keypoints = eval_data["keypoints"]
        if not isinstance(keypoints, list):
            return None

        
        n_full = 0
        n_partial = 0
        n_missing = 0

        for i, kp in enumerate(keypoints):
            if not isinstance(kp, dict):
                continue

            kp_id = kp.get("id", i + 1)

            
            status = kp.get("status", "missing")
            if status not in self.valid_statuses:
                kp["status"] = "missing"

            
            if kp["status"] == "full":
                n_full += 1
                kp["coverage_ratio"] = 1.0
            elif kp["status"] == "missing":
                n_missing += 1
                kp["coverage_ratio"] = 0.0
            else:  
                n_partial += 1
                
                cr = kp.get("coverage_ratio", 0.5)
                try:
                    cr = float(cr)
                    cr = max(0.0, min(1.0, cr))
                except (TypeError, ValueError):
                    cr = 0.5
                kp["coverage_ratio"] = cr

            
            if self.fixed_max_scores:
                
                
                kp_id_str = str(kp_id)
                fixed_keys_str = {str(k): k for k in self.fixed_max_scores.keys()}
                
                if kp_id_str in fixed_keys_str:
                    
                    original_key = fixed_keys_str[kp_id_str]
                    kp["max_score"] = self.fixed_max_scores[original_key]
                    kp["max_score_source"] = "fixed"
                else:
                    
                    kp["max_score"] = 0
                    kp["max_score_source"] = "invalid"
                    kp["status"] = "missing"
                    kp["coverage_ratio"] = 0.0
            else:
                
                ms = kp.get("max_score", 0)
                try:
                    kp["max_score"] = float(ms)
                except (TypeError, ValueError):
                    kp["max_score"] = 0
                kp["max_score_source"] = "LLM_assigned"

        
        if self.fixed_max_scores:
            existing_ids = {kp.get("id", i+1) for i, kp in enumerate(keypoints) if isinstance(kp, dict)}
            for kp_id, score in self.fixed_max_scores.items():
                if kp_id not in existing_ids:
                    keypoints.append({
                        "id": kp_id,
                        "name": f"要点{kp_id}",
                        "max_score": score,
                        "max_score_source": "fixed",
                        "status": "missing",
                        "coverage_ratio": 0.0,
                        "quality_band": None,
                        "evidence": "",
                        "covered_subpoints": [],
                        "missing_subpoints": [],
                        "reasoning": "LLM未评估此要点，默认标记为missing"
                    })
                    n_missing += 1

        
        if "summary" not in eval_data:
            eval_data["summary"] = {}
        eval_data["summary"]["n_full"] = n_full
        eval_data["summary"]["n_partial"] = n_partial
        eval_data["summary"]["n_missing"] = n_missing

        total = n_full + n_partial + n_missing
        if total > 0:
            eval_data["summary"]["overall_coverage"] = (n_full + n_partial * 0.5) / total
        else:
            eval_data["summary"]["overall_coverage"] = 0.0

        
        eval_data["total_max_score"] = sum(kp.get("max_score", 0) for kp in keypoints)

        return eval_data
