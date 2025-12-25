# Evaluation (ICWSM-ready)

## Goals
We evaluate whether agentic context construction improves contextual interpretation of social media posts by:
(1) increasing alignment with human contextual summaries,
(2) reducing over-interpretation/hallucinated context,
(3) improving efficiency via selective context routing,
and (4) generalizing beyond Facebook to 1–2 additional platforms.

## Datasets
- Primary: Facebook posts (text + optional image/video captions), with synthetic WHO/HOW for development.
- Human-labeled subset: N≈250 posts with human contextual summaries and ratings (Relevance, Clarity, Accuracy, Completeness).
- Optional generalization: smaller set from Reddit or X/Twitter (10–20% of total).

## Systems Compared (Baselines)
B1. WHAT-only: multimodal summary without context.
B2. Static-All: always compute WHAT+WHO+WHEN+WHERE+HOW.
B3. Agentic-Routed: Planner selects a subset of context components.
B4. Agentic-Routed + Verifier: adds verification/audit before synthesis.

## Metrics

### Automatic Metrics (alignment)
- Semantic similarity to human contextual summaries (embedding-based similarity; report mean/median/std).
- Optional: ROUGE-L / BLEU as secondary descriptive metrics (ICWSM typically prefers human eval).

### Human Evaluation (primary)
Annotators rate outputs on:
- Relevance (1–5)
- Clarity (1–5)
- Accuracy (1–5)
- Completeness (1–5)
Also label comparative preference:
- System Better / Ground Truth Better / Parity
Report inter-annotator agreement (pairwise Cohen’s κ) and average.

### Safety/Robustness Metrics (ICWSM-friendly)
- Over-interpretation rate: fraction of outputs asserting specific events/persona claims without evidence.
- Unsupported-claim rate: number of verifier-flagged unsupported assertions per post.
- Contradiction rate: fraction of posts with internal inconsistencies (e.g., WHO vs WHAT).
- Uncertainty quality: % of cases where system correctly says “unclear” when evidence is weak.

### Efficiency Metrics
- Average #components executed per post (routing efficiency).
- Inference cost proxy: #LLM calls or tokens per post.
- Latency per post (optional).

## Experiments

### E1: Context Necessity Analysis
Stratify posts by ambiguity/topic and measure performance drops by removing each component:
- -WHO, -WHEN, -WHERE, -HOW
Compare to agentic selection to show “conditional necessity” of context.

### E2: Routing vs Static Context
Compare B2 vs B3 under a fixed budget:
- max K components per post (e.g., K=3)
Report quality vs cost curves.

### E3: Verification Impact
Compare B3 vs B4:
- reduction in unsupported/over-specific claims
- improved human-rated accuracy
- cases where verifier forces uncertainty instead of hallucination

### E4: Cross-platform Generalization (lightweight)
Train/prompts unchanged; evaluate on Reddit/X subset.
Goal: demonstrate conceptual transfer of context components rather than maximizing scores.

## Tables to Report

Table A: Overall performance (human scores)
| System | Rel | Clar | Acc | Comp | κ | Preference(%) |
|---|---:|---:|---:|---:|---:|---:|

Table B: Safety/robustness
| System | Over-interpret% | Unsupported claims/post | Contradiction% |
|---|---:|---:|---:|

Table C: Efficiency
| System | Avg components/run | Avg LLM calls | Tokens/post |
|---|---:|---:|---:|

## Qualitative Analysis (Required at ICWSM)
Provide 6–10 case studies:
- ambiguous short post where WHO/HOW resolves intent
- group/page post where WHERE shifts interpretation
- event-related post where WHEN helps (and where it fails)
- verifier catches an over-specific claim and rewrites it safely
