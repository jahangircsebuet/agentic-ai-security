import re
from typing import Dict, Any, List

OVER_SPECIFIC_PATTERNS = [
    r"\bon (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\b\d{4}\b",                     # explicit year
    r"\baccording to (bbc|cnn|reuters)\b",
    r"\bthis confirms\b",
    r"\bdefinitely\b|\bcertainly\b",
]

HEDGE_PATTERNS = [
    r"\bmay\b|\bmight\b|\bpossibly\b|\bunclear\b|\binsufficient evidence\b",
    r"\bcannot determine\b|\bnot enough context\b"
]

def _count_matches(text: str, patterns: List[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text.lower()))

def overinterpretation_score(component_json: Dict[str, Any]) -> float:
    """
    Heuristic: penalize over-specific language if uncertainty is high or evidence_used is weak.
    """
    summary = (component_json.get("summary") or "")
    uncertainty = (component_json.get("uncertainty") or "medium").lower()
    evidence_used = component_json.get("evidence_used") or []

    over_specific = _count_matches(summary, OVER_SPECIFIC_PATTERNS)
    hedges = _count_matches(summary, HEDGE_PATTERNS)

    evidence_strength = min(1.0, len(evidence_used) / 3.0)  # 0..1
    uncertainty_penalty = {"low": 0.2, "medium": 0.6, "high": 1.0}.get(uncertainty, 0.6)

    # Over-specific with low evidence and high uncertainty -> high score (bad)
    raw = (over_specific * uncertainty_penalty) * (1.0 - evidence_strength)
    # hedging reduces score
    raw = max(0.0, raw - 0.5 * hedges)
    return float(min(5.0, raw))  # cap

def unsupported_claim_rate(verifier_output: Dict[str, Any]) -> float:
    flags = verifier_output.get("flags", [])
    if not flags:
        return 0.0
    unsupported = [f for f in flags if f.get("type") in ("unsupported", "over_specific")]
    return len(unsupported) / max(1, len(flags))

def contradiction_rate(verifier_output: Dict[str, Any]) -> float:
    flags = verifier_output.get("flags", [])
    if not flags:
        return 0.0
    contradictions = [f for f in flags if f.get("type") == "contradiction"]
    return len(contradictions) / max(1, len(flags))

def uncertainty_quality(synth_json: Dict[str, Any]) -> float:
    """
    Simple: reward explicit limits when confidence is low.
    """
    conf = (synth_json.get("confidence", {}) or {}).get("overall", 0.0)
    limits = synth_json.get("limits") or []
    if conf < 0.6:
        return 1.0 if len(limits) >= 2 else 0.5
    return 1.0
