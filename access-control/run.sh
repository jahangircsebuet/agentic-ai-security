# ✅ How to use it incrementally (your requested workflow)
# A) Start from “basic” (only planner + baseline vs cap, minimal gates)

# model = Qwen/Qwen2.5-7B-Instruct
# export PYTORCH_ALLOC_CONF=expandable_segments:True
# exec -a run  python agentic_ac_modular.py \
#   --planner_backend hf \
#   --planner_model Qwen/Qwen2.5-7B-Instruct \
#   --kb_path finance_kb.json \
#   --outdir runs/step0_basic \
#   --enable_context_eval 0 --enable_label_gate 0 --enable_cap_provenance 0 \
#   --enable_two_key 0 --enable_taint 0 --enable_attack_suite 0 \
#   --enable_output_dlp 0 --enable_memory_gate 0 --enable_tool_gate 0


# model = meta-llama//Meta-Llama-3-8B-Instruct
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec -a run  python agentic_ac_modular.py \
  --planner_backend hf \
  --planner_model meta-llama//Meta-Llama-3-8B-Instruct \
  --kb_path finance_kb.json \
  --outdir runs/step0_basic \
  --enable_context_eval 0 --enable_label_gate 0 --enable_cap_provenance 0 \
  --enable_two_key 0 --enable_taint 0 --enable_attack_suite 0 \
  --enable_output_dlp 0 --enable_memory_gate 0 --enable_tool_gate 0


# B) Turn on only label-aware retrieval gate (fix your earlier issue)
# python3 agentic_ac_modular.py ... \
#   --outdir runs/step1_label_gate \
#   --enable_label_gate 1

# C) Add context evaluator
# python3 agentic_ac_modular.py ... \
#   --outdir runs/step2_context \
#   --enable_label_gate 1 --enable_context_eval 1

# D) Add capability provenance (no escalation)
# python3 agentic_ac_modular.py ... \
#   --outdir runs/step3_provenance \
#   --enable_label_gate 1 --enable_context_eval 1 --enable_cap_provenance 1

# E) Add two-key execution
# python3 agentic_ac_modular.py ... \
#   --outdir runs/step4_two_key \
#   --enable_label_gate 1 --enable_context_eval 1 --enable_cap_provenance 1 --enable_two_key 1


# F) Add taint tracking + attack suite
# python3 agentic_ac_modular.py ... \
#   --outdir runs/step5_full \
#   --enable_label_gate 1 --enable_context_eval 1 --enable_cap_provenance 1 \
#   --enable_two_key 1 --enable_taint 1 --enable_attack_suite 1


# ✅ Multi-model run for your planned families (Llama/Qwen/Mistral/Gemma/Vicuna)
# If you run each via vLLM on ports 8001–8005:
# python3 agentic_ac_modular.py \
#   --planner_backend vllm --planner_model dummy \
#   --kb_path finance_kb.json --outdir runs/multimodel_full \
#   --enable_label_gate 1 --enable_context_eval 1 --enable_cap_provenance 1 \
#   --enable_two_key 1 --enable_taint 1 --enable_attack_suite 1 \
#   --models_json '[
#     {"name":"Llama-3.1-8B-Instruct","backend":"vllm","model":"meta-llama/Llama-3.1-8B-Instruct","base_url":"http://localhost:8001/v1"},
#     {"name":"Qwen2.5-7B-Instruct","backend":"vllm","model":"Qwen/Qwen2.5-7B-Instruct","base_url":"http://localhost:8002/v1"},
#     {"name":"Mistral-7B-Instruct-v0.3","backend":"vllm","model":"mistralai/Mistral-7B-Instruct-v0.3","base_url":"http://localhost:8003/v1"},
#     {"name":"Gemma-3-12B-IT","backend":"vllm","model":"google/gemma-3-12b-it","base_url":"http://localhost:8004/v1"},
#     {"name":"Vicuna-7B-v1.5","backend":"vllm","model":"lmsys/vicuna-7b-v1.5","base_url":"http://localhost:8005/v1"}
#   ]'


