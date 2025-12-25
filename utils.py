import json

def extract_json(text: str):
    """
    Safely extract JSON object from LLM output.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in output")
    return json.loads(text[start:end+1])
