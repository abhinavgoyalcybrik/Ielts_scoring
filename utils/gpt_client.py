import os
import json
import re
from openai import OpenAI

_client = None

def get_client():
    global _client

    if _client is None:

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        _client = OpenAI(api_key=api_key)

    return _client


def _strip_code_fence(content: str) -> str:
    """Strip ```json ... ``` / ``` ... ``` fences some models wrap JSON in."""
    stripped = content.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _extract_json_object(content: str) -> str:
    """Fall back to the outermost {...} span if the model added stray prose."""
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start:end + 1]
    return content


def call_gpt(prompt):
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty GPT response")

    candidate = _strip_code_fence(content)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    candidate = _extract_json_object(candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from GPT:\n{content}")
