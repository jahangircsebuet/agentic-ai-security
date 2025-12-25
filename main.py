# Purpose: The Orchestrator/Heart of the 5W1H Analysis Crew
# Key Contents: Orchestrates the crew, triggers the kickoff, and saves the final result.


from crewai import Crew, Process, Task
from agents import who_agent, what_agent, where_agent, when_agent, how_agent, coordinator_agent
from tasks import who_task, what_task, where_task, when_task, how_task
from data_gen import generate_social_dataset

# 1. Get your data
df = generate_social_dataset(num_users=1)
current_post = df.iloc[0].to_dict()

# 2. Define the Manager's Task
master_task = Task(
    description=f"Analyze this post: {current_post}. Coordinate specialists to create a 5W1H summary.",
    expected_output="A cohesive master summary integrating all 5 modules.",
    agent=None # The manager_agent will handle this
)

# 3. Setup the Crew
# research_crew = Crew(
#     agents=[who_agent], # Add all specialist agents here
#     tasks=[master_task],
#     process=Process.hierarchical,
#     manager_llm="gpt-4o", # Manager needs a strong LLM to delegate
#     verbose=True
# )

# The 'Brain' that brings it all together
master_summary_crew = Crew(
    agents=[who_agent, what_agent, where_agent, when_agent, how_agent],
    tasks=[who_task, what_task, where_task, when_task, how_task],
    process=Process.hierarchical,
    # You can provide a specific LLM to act as the Manager/Coordinator
    manager_llm="gpt-4", 
    verbose=True
)

# 4. Run and Save
result = master_summary_crew.kickoff()
with open("output/results.txt", "w") as f:
    f.write(result)


from crew import kickoff, run_verifier, build_verifier_input
from crew import build_synthesis_input, run_synthesizer
from data_examples import example_post, example_context


# Step 1: Context agents
component_outputs, plan = kickoff(example_post, example_context)

# Step 2: Verification
verifier_input = build_verifier_input(example_post, plan, component_outputs)
verified_output = run_verifier(verifier_input)

# Step 3: Synthesis
synthesis_input = build_synthesis_input(example_post, verified_output)
final_summary = run_synthesizer(synthesis_input)

print("\nFINAL VERIFIED CONTEXTUAL SUMMARY:\n")
print(final_summary)