# Purpose: The Assignments for the 5W1H Analysis Crew
# Key Contents: Detailed instructions for each agent and their expected outputs.

from crewai import Task

# WHAT Module Task
def what_task(agent, post):
    return Task(
    description="""
        Generate the WHAT summary.
        Focus only on explicit content (text, image captions, video captions) of the post:

        Post:
        {post}

        Identify the main topic and core themes of the social media post. 
        Focus on answering: What is the user talking about? What are the 
        primary entities (people, places, events etc.) mentioned?
    """,
    expected_output="""
        A concise summary (2-3 sentences) of the post's content and a 
        bulleted list of key themes.
    """,
    agent=agent
)


# WHO Module Task
def who_task(agent, user_context):
    return Task(
        description=f"""
        Generate the WHO summary.
        Infer user background and behavioral context from history:

        Data:
        {user_context}

        Identify the main topic and core themes of the social media post. 
        Focus on answering: What is the user talking about? What are the 
        primary entities (people, places, events etc.) mentioned?
    """,
    expected_output="""
        A concise summary (2-3 sentences) of the post's content and a 
        bulleted list of key themes.
    """,
    agent=agent
)

# WHEN Module Task
def when_task(agent, temporal_context):
    return Task(
        description=f"""
            Using the post timestamp, research if any significant real-world events, 
            political unrest, or cultural moments were happening at that time. 
            Contextualize the post within this temporal window.

            Data:
            {temporal_context}
        """,
        expected_output="""
            A summary explaining the historical or news-related context of the 
            post's creation date.
        """,
        agent=agent
    )

# WHERE Module Task
def where_task(agent, surface_context):
    return Task(
        description=f"""
            Analyze the platform information (e.g., Facebook group name, 
            Twitter handle, Telegram channel). Determine the likely 'vibe' 
            and purpose of this digital space and who its typical audience is.

            Data:
            {surface_context}
        """,
        expected_output="""
            A brief description of the digital environment and how it might 
            influence the way the post was written or received.
        """,
        agent=agent
    )

# HOW Module Task
def how_task(agent, engagement_context):
    return Task(
        description=f"""
            Analyze the comment history associated with the post. 
            Identify the overall sentiment (Positive, Negative, Neutral) 
            and summarize the general audience reaction.

            Data:
            {engagement_context}
        """,
        expected_output="""
            A sentiment analysis summary and a brief description of the 
            interactions between the user and their audience.
        """,
        agent=agent
    )

# Planner Module Task
def planning_task(agent, post):
    """Decide which contextual dimensions (WHO, WHEN, WHERE, HOW) are required. Planner outputs a strict JSON route plan:
        {
        "run": ["WHAT", "WHO", "HOW", "WHERE"],
        "skip": ["WHEN"],
        "rationale": {"WHEN":"no temporal cues"},
        "confidence": 0.78
        }
    """
    return Task(
        description=f"""
            Given the post below, decide which contextual dimensions
            (WHO, WHEN, WHERE, HOW) are required.

            Post:
            {post}
        """,
        agent=agent,
        expected_output="List of required context components"
    )

# Synthesis Module Task
def synthesis_task(agent, context_bundle):
    return Task(
        description=f"""
            Combine WHAT, WHO, WHEN, WHERE, HOW into a final contextual interpretation.

            Context:
            {context_bundle}
        """,
        agent=agent,
        expected_output="Final contextual summary"
    )