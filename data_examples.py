from synth_profiles import generate_context_packets

example_post = {
    "post_id": "p001",
    "text": "Feeling frustrated about the quota situation... students deserve fairness. #protest",
    "image_captions": ["A crowd of people holding signs in a street"],
    "video_captions": [],
    "timestamp": "2025-12-20T18:10:00",
    "platform": "Facebook",
    "surface": "Group",
    "group_topic": "University Students"
}

example_context = generate_context_packets(example_post, user_id="u123")
