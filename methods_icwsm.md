## Methods

### Task Definition: Contextual Interpretation
Given a social media post p (text plus optional image/video captions and metadata), we construct a contextual interpretation C_p consisting of five dimensions:
WHAT (explicit content), WHO (poster background/behavior), WHEN (contemporary events), WHERE (posting surface and norms), and HOW (audience engagement).
The output includes a structured context object and a short integrated summary, along with uncertainty and confidence signals.

### Agentic Context Construction
We implement context construction as a multi-agent pipeline in which each contextual dimension is handled by a specialized LLM-based agent. A Planner agent first selects which contextual dimensions to compute (agentic routing) based on post ambiguity, topical cues, and the expected reliability of available evidence. The selected agents then generate component outputs in a standardized structured JSON format (summary, key claims, evidence used, and uncertainty level).

### Verification and Uncertainty Modeling
To reduce over-interpretation and hallucinated context, a Verifier/Auditor agent audits the component outputs for unsupported claims, over-specific assertions, cross-component contradictions, and privacy risks. The verifier produces revised component outputs and assigns confidence scores per dimension. A Synthesizer agent then generates the final contextual interpretation using only verifier-approved content, explicitly acknowledging unknowns and missing context.

### Synthetic Context for Development
To enable scalable development without privacy risk, we initially use synthetic user profiles and engagement packets for WHO and HOW (persona + post history + interaction history; reaction aggregates and comment emotion distributions). These are later replaceable with real extraction modules under an IRB-compliant data collection protocol.

### Outputs
The system produces:
(1) a context object containing WHAT/WHO/WHEN/WHERE/HOW summaries,
(2) an integrated situational interpretation,
(3) confidence (overall and per component),
and (4) a list of limitations/uncertainties.



### Algorithm: Agentic Contextual Interpretation

**Input:** post p (text + captions + metadata), context packets X (WHO/WHEN/WHERE/HOW evidence)  
**Output:** context object C_p, final summary S_p, confidence scores

1. **Plan:**  
   Use PlannerAgent to output a route plan R = {run, skip, rationale}  
   Ensure WHAT ∈ run

2. **Construct Components (routed):**  
   For each component c ∈ R.run:  
   - If c == WHAT: run WhatAgent(p) → O_c  
   - Else: run Agent_c(X_c) → O_c  
   Each O_c is structured JSON (summary, claims, evidence, uncertainty)

3. **Verify:**  
   V_in ← {p, R, {O_c}}  
   Run VerifierAgent(V_in) → V_out  
   Extract revised components O'_c and confidence conf_c

4. **Synthesize:**  
   S_in ← {p, {O'_c}, conf_c, verifier_notes}  
   Run SynthesizerAgent(S_in) → final JSON containing:
   - context_object (WHAT/WHO/WHEN/WHERE/HOW)
   - final_summary
   - confidence overall + per component
   - limits/uncertainties

5. **Return** final JSON
