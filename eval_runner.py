from typing import Dict, Any
from utils import extract_json
from eval_metrics import (
    overinterpretation_score,
    unsupported_claim_rate,
    contradiction_rate,
    uncertainty_quality
)

def evaluate_one(verified_output: Dict[str, Any], final_output_text: str) -> Dict[str, Any]:
    final_json = extract_json(final_output_text)

    revised = verified_output.get("revised_components", {})
    scores = {k: overinterpretation_score(v) for k, v in revised.items() if isinstance(v, dict)}

    return {
        "overinterpretation_per_component": scores,
        "unsupported_claim_rate": unsupported_claim_rate(verified_output),
        "contradiction_rate": contradiction_rate(verified_output),
        "uncertainty_quality": uncertainty_quality(final_json),
        "overall_confidence": final_json.get("confidence", {}).get("overall", None)
    }
