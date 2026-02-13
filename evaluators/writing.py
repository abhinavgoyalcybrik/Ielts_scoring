from pathlib import Path
from utils.band import round_band
from utils.ai_client import call_gpt_writing, call_gpt_refine_answer
from utils.cefr_mapper import map_ielts_to_cefr
from utils.vocabulary_feedback import get_writing_vocabulary_reference, analyze_vocabulary


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE_DIR / "prompts"


def clamp(score):
    try:
        return max(0.0, min(9.0, float(score)))
    except Exception:
        return 5.0


def count_words(text: str) -> int:
    return len(text.split())


def get_vocabulary_to_learn(essay: str, task_type: str, band: float) -> list:
    """
    Extract vocabulary for the user to learn based on their essay and band level.
    Returns 12-20 words/phrases that are relevant and slightly above current level.
    """
    # Get topic-relevant vocabulary for the task
    vocab_reference = get_writing_vocabulary_reference(task_type)
    
    # Analyze what the user already used
    vocab_analysis = analyze_vocabulary(essay)
    good_usage = set(vocab_analysis.get("good_usage", []))
    
    # Filter vocabulary: prefer words not yet used by student
    vocab_to_learn = []
    for vocab_item in vocab_reference:
        word = vocab_item.get("word", "").lower()
        hint = vocab_item.get("usage_hint", "")
        
        # Skip if student already used this word
        if word not in good_usage and word.lower() not in " ".join(good_usage).lower():
            vocab_to_learn.append({
                "word": vocab_item["word"],
                "usage_hint": hint
            })
    
    # If we have too few, add suggested improvements
    if len(vocab_to_learn) < 12:
        suggested = vocab_analysis.get("suggested_improvements", [])
        for suggestion in suggested:
            if len(vocab_to_learn) >= 20:
                break
            # Extract words from suggestion
            import re
            words = re.findall(r"'([^']+)'", suggestion)
            for word in words:
                if len(vocab_to_learn) < 20:
                    vocab_to_learn.append({
                        "word": word,
                        "usage_hint": f"Advanced alternative"
                    })
    
    # Return 12-20 words
    return vocab_to_learn[0:20]  # Max 20



def validate_word_count(task_type: str, essay: str):
    """
    Count words directly from essay text.
    NO HARD VALIDATION - evaluation proceeds regardless of word count.
    Low word count impacts Task Response and band scoring naturally.
    """
    wc = count_words(essay)
    return wc


def apply_coherence_penalty_cap(mistakes: list) -> list:
    """
    Cap repetition-related coherence errors to maximum 2 per answer.
    If copy-paste is identified, flag it once only.
    """
    coherence_repetition_errors = [
        m for m in mistakes 
        if m.get("error_type") == "coherence" and "repetition" in m.get("explanation", "").lower()
    ]
    
    if len(coherence_repetition_errors) > 2:
        # Keep only first 2 repetition errors, remove the rest
        repetition_count = 0
        filtered_mistakes = []
        for m in mistakes:
            is_repetition = (
                m.get("error_type") == "coherence" and 
                "repetition" in m.get("explanation", "").lower()
            )
            if is_repetition:
                if repetition_count < 2:
                    filtered_mistakes.append(m)
                    repetition_count += 1
            else:
                filtered_mistakes.append(m)
        return filtered_mistakes
    
    return mistakes


def apply_fair_band_scoring(overall_band: float, task_response_score: float, task_type: str) -> float:
    """
    Apply IELTS fair scoring rules:
    - Do NOT reduce overall band below 5 if task is fully addressed and meaning is clear
    - Use examiner judgment, not strict penalization
    
    Rule: If task_response >= 6 (task is addressed), minimum overall band is 5
    """
    if task_response_score >= 6.0 and overall_band < 5.0:
        return 5.0
    return overall_band



def evaluate_writing(data: dict):

    metadata = data.get("metadata", {})
    question = metadata.get("question", "").strip()
    essay = data.get("user_answers", {}).get("text", "").strip()

    if not essay:
        raise ValueError("Essay text missing")

    task_type = "task_1" if metadata.get("task_type") in ("task1", "task_1") else "task_2"
    word_count = validate_word_count(task_type, essay)

    prompt_file = (
        PROMPTS_DIR / "writing_task1_prompt.txt"
        if task_type == "task_1"
        else PROMPTS_DIR / "writing_task2_prompt.txt"
    )

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = (
        prompt_template
        .replace("<<<QUESTION>>>", question)
        .replace("<<<ESSAY_TEXT>>>", essay)
        .replace("<<<WORD_COUNT>>>", str(word_count))
        .replace("<<<TASK_TYPE>>>", task_type)
    )

    ai = call_gpt_writing(prompt)

    tr = clamp(ai.get("task_response", 5))
    cc = clamp(ai.get("coherence_cohesion", 5))
    lr = clamp(ai.get("lexical_resource", 5))
    gr = clamp(ai.get("grammar_accuracy", 5))

    if task_type == "task_1":
        overall = tr * 0.3 + cc * 0.25 + lr * 0.25 + gr * 0.2
    else:
        overall = tr * 0.4 + cc * 0.3 + lr * 0.2 + gr * 0.1

    # Apply fair band scoring rule: don't reduce below 5 if task is addressed
    overall = apply_fair_band_scoring(overall, tr, task_type)
    band = round_band(overall)

    refined = call_gpt_refine_answer(question, essay, 8)
    
    # Apply coherence penalty cap: max 2 repetition-related errors
    mistakes = apply_coherence_penalty_cap(ai.get("mistakes", []))
    
    # Map to CEFR level
    cefr = map_ielts_to_cefr(band)
    
    # Get vocabulary to learn
    vocab_list = get_vocabulary_to_learn(essay, task_type, band)

    return {
        "overall_band": band,
        "cefr_level": cefr,
        "criteria_scores": {
            "task_response": tr,
            "coherence_cohesion": cc,
            "lexical_resource": lr,
            "grammar_accuracy": gr
        },
        "mistakes": mistakes,
        "refined_answer": refined,
        "word_count": word_count,
        "vocabulary_to_learn": vocab_list
    }
