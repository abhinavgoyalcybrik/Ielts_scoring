import os
import json
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


def call_gpt(prompt):
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty GPT response")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from GPT:\n{content}")
