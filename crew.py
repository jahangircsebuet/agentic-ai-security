import json
from crewai import Crew
from utils import extract_json
from tasks import synthesis_task
from agents import synthesizer_agent

from agents import (
    planner_agent, verifier_agent,
    what_agent, who_agent, when_agent, where_agent, how_agent, synthesizer_agent
)
from tasks import (
    planning_task, what_task, who_task, when_task, where_task, how_task,
    verifier_task, synthesis_task
)

COMP_TO_AGENT = {
    "WHAT": what_agent,
    "WHO": who_agent,
    "WHEN": when_agent,
    "WHERE": where_agent,
    "HOW": how_agent,
}

def run_planner(post_dict):
    crew = Crew(
        agents=[planner_agent],
        tasks=[planning_task(planner_agent, json.dumps(post_dict, ensure_ascii=False, indent=2))],
        verbose=True
    )
    raw = crew.kickoff()
    return raw

def parse_plan(raw_plan: str):
    # Crew outputs can include extra text; try to locate JSON block
    start = raw_plan.find("{")
    end = raw_plan.rfind("}")
    plan = json.loads(raw_plan[start:end+1])
    run = plan.get("run", [])
    if "WHAT" not in run:
        run.insert(0, "WHAT")
    return plan

def build_routed_crew(post_dict, context_packets, plan):
    run_components = plan["run"]

    routed_tasks = []
    # WHAT uses raw post
    if "WHAT" in run_components:
        routed_tasks.append(what_task(COMP_TO_AGENT["WHAT"], json.dumps(post_dict, indent=2)))

    # context packets (synthetic now)
    if "WHO" in run_components:
        routed_tasks.append(who_task(COMP_TO_AGENT["WHO"], json.dumps(context_packets["who"], indent=2)))
    if "WHEN" in run_components:
        routed_tasks.append(when_task(COMP_TO_AGENT["WHEN"], json.dumps(context_packets["when"], indent=2)))
    if "WHERE" in run_components:
        routed_tasks.append(where_task(COMP_TO_AGENT["WHERE"], json.dumps(context_packets["where"], indent=2)))
    if "HOW" in run_components:
        routed_tasks.append(how_task(COMP_TO_AGENT["HOW"], json.dumps(context_packets["how"], indent=2)))

    # Verifier input: includes post, component outputs will be available after kickoff,
    # but for a first scaffold we pass what we *intend* to produce + evidence packets.
    # In the next iteration, we’ll wire real outputs into this.
    verification_input = {
        "post": post_dict,
        "plan": plan,
        "evidence_packets": context_packets,
        "components": {
            "WHAT": "<<filled after run>>",
            "WHO": "<<filled after run>>",
            "WHEN": "<<filled after run>>",
            "WHERE": "<<filled after run>>",
            "HOW": "<<filled after run>>",
        }
    }
    routed_tasks.append(verifier_task(verifier_agent, json.dumps(verification_input, indent=2)))

    # Synthesis (uses bundle; we’ll pass placeholder now, then you’ll refine)
    context_bundle = {
        "plan": plan,
        "post": post_dict,
        "components": "<<use verified revised_components in next iteration>>",
        "evidence_packets": context_packets
    }
    routed_tasks.append(synthesis_task(synthesizer_agent, json.dumps(context_bundle, indent=2)))

    routed_agents = [COMP_TO_AGENT[c] for c in run_components if c in COMP_TO_AGENT]
    routed_agents += [verifier_agent, synthesizer_agent]

    return Crew(agents=routed_agents, tasks=routed_tasks, verbose=True)


def kickoff(post_dict, context_packets):
    # 1) Run planner
    raw_plan = run_planner(post_dict)
    plan = parse_plan(str(raw_plan))

    # 2) Run routed context agents
    crew = build_routed_crew(post_dict, context_packets, plan)
    raw_outputs = crew.kickoff()

    # 3) Collect component outputs
    component_outputs = {}

    for output in raw_outputs:
        try:
            parsed = extract_json(str(output))
            component = parsed.get("component")
            if component:
                component_outputs[component] = parsed
        except Exception:
            continue  # ignore non-JSON outputs

    return component_outputs, plan


def build_verifier_input(post_dict, plan, component_outputs):
    return {
        "post": post_dict,
        "routing_plan": plan,
        "components": component_outputs,
        "instruction": (
            "Audit each component for unsupported claims, "
            "over-specific assertions, contradictions, and privacy risks."
        )
    }

def run_verifier(verifier_input):
    """Run the verifier agent on the given input and return parsed JSON output.
        {
            "flags": [...],
            "component_confidence": {...},
            "revised_components": {
                "WHAT": {...},
                "WHO": {...}
            },
            "global_notes": "..."
            }
    """
    crew = Crew(
        agents=[verifier_agent],
        tasks=[verifier_task(verifier_agent, verifier_input)],
        verbose=True
    )
    raw = crew.kickoff()
    return extract_json(str(raw))

def build_synthesis_input(post_dict, verified_output):
    return {
        "post": post_dict,
        "verified_components": verified_output["revised_components"],
        "component_confidence": verified_output["component_confidence"],
        "verifier_notes": verified_output["global_notes"]
    }

def run_synthesizer(synthesis_input):
    crew = Crew(
        agents=[synthesizer_agent],
        tasks=[synthesis_task(synthesizer_agent, synthesis_input)],
        verbose=True
    )
    return crew.kickoff()