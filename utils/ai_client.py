import os
import json
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables and read the OpenAI key from OPENAI_API_KEY.
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=API_KEY)


def _call_gpt(prompt: str, system_msg: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Empty GPT response")

    return content.strip()


def _parse_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from GPT:\n{content}")


def call_gpt_writing(prompt: str) -> dict:
    content = _call_gpt(
        prompt,
        system_msg="You are a certified IELTS Writing examiner. Respond ONLY in valid JSON."
    )
    return _parse_json(content)


def call_gpt_refine_answer(question: str, answer: str, target_band: int = 8) -> str:
    prompt = (
        f"Improve the following IELTS Writing answer to Band {target_band}.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Return ONLY the improved answer."
    )

    return _call_gpt(
        prompt,
        system_msg="You are an IELTS Writing tutor."
    )
