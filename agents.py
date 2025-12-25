# Purpose: The Brains behind the 5W1H Analysis Crew
# Key Contents: Definitions for the 5 Specialist Agents (WHO, WHAT, WHERE, WHEN, HOW).

from crewai import Agent, Task, Crew, Process
from tools import SocialMediaTools

from crewai import LLM, Agent
from crewai_tools import SerperDevTool
from crewai import Agent
from prompts import PLANNER_PROMPT, VERIFIER_PROMPT, VERIFIER_PROMPT_ICWSM, SYNTHESIZER_PROMPT_ICWSM

# Define different models for different 'cognitive' costs
smart_model = LLM(model="gpt-4o", temperature=0.2)
fast_model = LLM(model="gpt-4o-mini", temperature=0.5)
reasoning_model = LLM(model="anthropic/claude-3-7-sonnet", temperature=0) # Great for WHO
llama_model = LLM(model="meta-llama/Meta-Llama-3.1-70B-Instruct") #larger model for complex reasoning
smaller_llama_model = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct") # smaller model for complex reasoning
mixtral_model = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.3)

# ✅ Open weights
# ✅ Strong instruction tuning
# ✅ Good for noisy social media text
# ⚠️ Less strong global reasoning
google_model = LLM(model="google/gemma-2-9b-it") #Multilingual / social text friendly
reasoning_model = mixtral_model


# Suggested hybrid setup (best practice) ------> Try later
# For your agentic system, I strongly recommend this mix:
# planner_model     = LLM("gpt-4o-mini")                     # routing accuracy
# what_model        = LLM("Mixtral-8x7B-Instruct")           # open
# who_model         = LLM("claude-3-7-sonnet")               # strongest reasoning
# verifier_model    = LLM("Llama-3.1-70B-Instruct")          # open + robust
# synthesizer_model = LLM("Llama-3.1-70B-Instruct")          # open + safe

# And report results as:
# proprietary-only
# open-source-only
# mixed

# We evaluate our framework using both proprietary (GPT-4o, Claude-3.7) and open-weight models (LLaMA-3.1, Mixtral) to assess robustness and reproducibility across model families.

# WHO Module
who_agent = Agent(
    role='Persona Profiler',
    goal='Infer user background and behavioral context from history. Identify if the user is a Student, Educator, or Politician based on this inference.',
    backstory='Expert in digital sociology and linguistic patterns. You analyze user-generated content to build detailed personas.',
    allow_delegation=False, # Specialists focus only on their task
    llm=reasoning_model,
    verbose=True
)

# WHAT Module
what_agent = Agent(
    role='Content Semanticist',
    goal='Identify the core topics, themes, and underlying message of the user post.',
    backstory='You specialize in natural language understanding and thematic extraction.',
    llm=reasoning_model,
    verbose=True
)

# WHERE Module
where_agent = Agent(
    role='Platform Contextualizer',
    goal='Analyze the platform environment (Group/Page) to determine audience norms. Infer platform and audience context',
    backstory='You are a digital anthropologist who understands platform-specific behaviors. You analyze where the post was shared and audience norms.',
    llm=reasoning_model,
    verbose=True
)

# WHEN Module
# Initialize the search tool
search_tool = SerperDevTool()

when_agent = Agent(
    role='Temporal Correlation Specialist',
    goal='Link the post timestamp to contemporary global or local events/news. Verify real-world historical events and news related to the post timestamp.',
    backstory="""You are an expert Digital Historian. Your job is to use the internet 
    to find what major events were happening on the exact date and location of a post. 
    You never guess; you always verify using your search tool.""",
    tools=[search_tool], # The agent can now use Google Search!
    llm=reasoning_model,
    verbose=True
)

# HOW Module
how_agent = Agent(
    role='Interaction Sentiment Analyst',
    goal='Summarize audience reaction by analyzing comments and engagement sentiment.',
    backstory='You are an expert in affective computing and social feedback loops. You interpret reactions, comments, and emotional trends.',
    llm=reasoning_model,
    verbose=True
)

# Planner Agent to decide which contexts to gather
planner_agent = Agent(
    role="Context Planner",
    goal="Decide which contextual dimensions are needed to interpret the post",
    backstory=(
        "You decide which social contexts (WHO, WHEN, WHERE, HOW) "
        "are necessary to understand a social media post to minimize speculation and cost."
    ),
    llm=reasoning_model,
    system_message=PLANNER_PROMPT,
    verbose=True
)

verifier_agent = Agent(
    role="Verification/Auditor",
    goal="Audit contextual components for grounding, contradictions, and privacy risks",
    backstory="You reduce hallucination and overclaiming in context construction.",
    llm=reasoning_model,
    system_message=VERIFIER_PROMPT_ICWSM,
    verbose=True
)

# The Coordinator/Synthesizer (Manager) Agent
synthesizer_agent = Agent(
    role="Context Synthesizer",
    goal="Synthesize 5W1H reports into a single, cohesive contextual summary/interpretation.",
    backstory="A master editor who excels at finding connections between different data points. You integrate all context components into a final summary.",
    llm=reasoning_model,
    system_message=SYNTHESIZER_PROMPT_ICWSM,
    verbose=True
)