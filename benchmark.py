from agents import get_agents
from tasks import get_tasks
from evaluator import ResearchEvaluator
from crewai import Crew, Process
import time

# 1. Models to benchmark
managers = {
    "GPT-4o": "gpt-4o",
    "Claude-3.7": "anthropic/claude-3-7-sonnet",
    "Gemini-2.0": "google/gemini-2.0-pro"
}

evaluator = ResearchEvaluator()
benchmark_results = []

for name, model_id in managers.items():
    print(f"🚀 Testing Manager: {name}")
    start_time = time.time()
    
    # Initialize Crew with specific Manager
    crew = Crew(
        agents=get_agents(),
        tasks=get_tasks(),
        manager_llm=model_id,
        process=Process.hierarchical
    )
    
    output = crew.kickoff()
    execution_time = time.time() - start_time
    
    # 2. Evaluate the quality
    score_report = evaluator.evaluate_result(output.raw)
    
    benchmark_results.append({
        "Manager": name,
        "Execution_Time": execution_time,
        "Summary": output.raw,
        "Judge_Report": score_report
    })

# 3. Save to Research Report
df_results = pd.DataFrame(benchmark_results)
df_results.to_csv("results/benchmark_report.csv", index=False)
print("✅ Benchmarking Complete. Results saved to results/benchmark_report.csv")