import random
from datetime import datetime, timedelta

PERSONAS = [
    "political", "businessperson", "entrepreneur", "educator", "religious",
    "artist", "athlete", "technologist", "scientist", "healthcare_professional",
    "traveler", "musician"
]

TOPICS = [
    "politics", "sports", "religion", "education", "business", "health",
    "technology", "travel", "entertainment", "family", "memes", "local_news",
    "career", "science", "finance", "food", "relationships", "social_issues", "other"
]

SURFACES = ["Profile", "Page", "Group"]
GROUP_TOPICS = ["Mental Health Support", "Local Community", "University Students", "Job Seekers", "Religious Discussion"]

def dirichlet_dist(n, alpha=0.6):
    # simple dirichlet-like using gamma draws
    xs = [random.gammavariate(alpha, 1.0) for _ in range(n)]
    s = sum(xs)
    return [x/s for x in xs]

def make_topic_mix():
    probs = dirichlet_dist(len(TOPICS), alpha=random.uniform(0.3, 1.2))
    return {t: round(p, 3) for t, p in zip(TOPICS, probs)}

def make_reactions():
    base = random.randint(10, 500)
    like = int(base * random.uniform(0.5, 0.9))
    love = int(base * random.uniform(0.05, 0.25))
    angry = int(base * random.uniform(0.0, 0.15))
    sad = int(base * random.uniform(0.0, 0.15))
    haha = max(0, base - (like + love + angry + sad))
    return {"like": like, "love": love, "haha": haha, "sad": sad, "angry": angry}

def make_comment_emotions():
    # match your HOW idea: aggregated trends
    supportive = round(random.uniform(0.0, 0.8), 2)
    toxic = round(random.uniform(0.0, 0.5), 2)
    neutral = round(max(0.0, 1.0 - supportive - toxic), 2)
    # normalize if needed
    s = supportive + toxic + neutral
    return {
        "supportive": round(supportive/s, 2),
        "toxic": round(toxic/s, 2),
        "neutral": round(neutral/s, 2),
    }

def generate_user_context(user_id: str):
    persona = random.choice(PERSONAS)
    post_history = make_topic_mix()
    interaction_history = make_topic_mix()

    return {
        "user_id": user_id,
        "persona": persona,
        "post_history": post_history,
        "interaction_history": interaction_history
    }

def generate_when_context(post_text: str, timestamp: str):
    # synthetic event candidates; later replace with retrieval
    keywords = [w.strip("#.,!?").lower() for w in post_text.split()[:6]]
    candidate_events = [
        f"Possible related discussion around '{k}' trending near posting time"
        for k in keywords[:2]
        if k
    ] or ["No strong event signal detected"]
    return {"keywords": keywords[:5], "events": candidate_events, "timestamp": timestamp}

def generate_where_context(platform="Facebook"):
    surface = random.choice(SURFACES)
    group_topic = random.choice(GROUP_TOPICS) if surface == "Group" else None
    return {"platform": platform, "surface": surface, "group_topic": group_topic}

def generate_how_context():
    return {
        "reactions": make_reactions(),
        "comment_emotions": make_comment_emotions(),
        "engagement_note": "Synthetic engagement snapshot (replace with real extraction later)"
    }

def generate_context_packets(post: dict, user_id: str):
    who = generate_user_context(user_id)
    when = generate_when_context(post.get("text",""), post.get("timestamp",""))
    where = generate_where_context(post.get("platform","Facebook"))
    how = generate_how_context()
    return {"who": who, "when": when, "where": where, "how": how}
