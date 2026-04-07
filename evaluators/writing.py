from pathlib import Path
from utils.band import round_band
from utils.ai_client import call_gpt_writing, call_gpt_text
from utils.cefr_mapper import map_ielts_to_cefr
from utils.vocabulary_feedback import analyze_vocabulary, generate_topic_vocabulary
from utils.safety import safe_gpt_call, safe_output, normalize_feedback


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE_DIR / "prompts"


def clamp(score):
    try:
        return max(0.0, min(9.0, float(score)))
    except Exception:
        return 5.0


def count_words(text: str) -> int:
    return len(text.split())


def get_vocabulary_to_learn(essay: str, task_type: str, band: float, question: str = "") -> list:
    """
    Extract TASK-SPECIFIC vocabulary for the user to learn.
    
    STRICT RULES:
    - Task 1: ONLY chart/data terminology (NO argumentation or topic-specific words)
    - Task 2: ONLY argumentation + topic-specific vocabulary (NO chart/data terminology)
    - ZERO overlap between Task 1 and Task 2 vocabulary lists
    - Returns 12-20 words/phrases specific to the task and topic
    
    Args:
        essay: The user's writing
        task_type: "task_1" (chart/diagram) or "task_2" (opinion/discussion)
        band: The user's IELTS band score
    
    Returns:
        List of task-specific vocabulary to learn (12-20 items)
    """
    # Normalize task_type
    if task_type in ("task_1", "task1"):
        task_type_normalized = "task_1"
    elif task_type == "general_task_1":
        task_type_normalized = "general_task_1"
    else:
        task_type_normalized = "task_2"
    
    # Dynamically build vocabulary from question/topic
    vocab_reference = generate_topic_vocabulary(question or "", essay, task_type_normalized)
    
    # Analyze what the user already used
    vocab_analysis = analyze_vocabulary(essay)
    good_usage = set(word.lower() for word in vocab_analysis.get("good_usage", []))
    
    # Filter vocabulary: prefer words not yet used by student
    vocab_to_learn = []
    for vocab_item in vocab_reference:
        word = vocab_item.get("word", "").lower()
        hint = vocab_item.get("usage_hint", "")
        item_task_type = vocab_item.get("task_type", task_type_normalized)
        
        # STRICT RULE: Only include vocabulary marked for this task type
        if item_task_type != task_type_normalized:
            continue
        
        # Skip if student already used this word/phrase
        if word in good_usage or any(w in " ".join(good_usage) for w in word.split()):
            continue
        
        # All suggested vocabulary should be B2+ (IELTS Band 6+)
        vocab_to_learn.append({
            "word": vocab_item["word"],
            "usage_hint": hint,
            "task_specific": True,
            "task_type": task_type_normalized  # Explicitly mark task type
        })
    
    # If insufficient vocabulary from reference, add more from the same task type
    if len(vocab_to_learn) < 12:
        # Get ALL vocabulary of this task type and fill gaps
        all_task_vocab = generate_topic_vocabulary(question or "", essay, task_type_normalized)
        for item in all_task_vocab:
            if len(vocab_to_learn) >= 20:
                break
            word_lower = item["word"].lower()
            # Only add if not already in the list
            if not any(v["word"].lower() == word_lower for v in vocab_to_learn):
                vocab_to_learn.append({
                    "word": item["word"],
                    "usage_hint": item.get("usage_hint", ""),
                    "task_specific": True,
                    "task_type": task_type_normalized
                })
    
    # Cap at 20, ensure all items are unique (remove duplicates while preserving order)
    seen = set()
    final_vocab = []
    for item in vocab_to_learn:
        word_lower = item["word"].lower()
        # STRICT: Verify task type matches before including
        if item.get("task_type") != task_type_normalized:
            continue
        if word_lower not in seen:
            seen.add(word_lower)
            final_vocab.append(item)
        if len(final_vocab) >= 20:
            break
    
    return final_vocab[0:15]  # Return max 15 items



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
        if (m.get("error_type") == "coherence" or m.get("type") == "coherence")
        and "repetition" in m.get("explanation", "").lower()
    ]
    
    if len(coherence_repetition_errors) > 2:
        # Keep only first 2 repetition errors, remove the rest
        repetition_count = 0
        filtered_mistakes = []
        for m in mistakes:
            is_repetition = (
                (m.get("error_type") == "coherence" or m.get("type") == "coherence")
                and "repetition" in m.get("explanation", "").lower()
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

    default_ai = {
        "task_response": 5,
        "coherence_cohesion": 5,
        "lexical_resource": 5,
        "grammar_accuracy": 5,
        "mistakes": [],
        "strengths": [],
        "examiner_response": ""
    }

    ai = safe_gpt_call(prompt, fallback=default_ai, caller=call_gpt_writing) or default_ai

    tr = clamp(ai.get("task_response", 5))
    cc = clamp(ai.get("coherence_cohesion", 5))
    lr = clamp(ai.get("lexical_resource", 5))
    gr = clamp(ai.get("grammar_accuracy", ai.get("grammar", 5)))

    if task_type == "task_1":
        overall = tr * 0.3 + cc * 0.25 + lr * 0.25 + gr * 0.2
    else:
        overall = tr * 0.4 + cc * 0.3 + lr * 0.2 + gr * 0.1

    # Apply fair band scoring rule: don't reduce below 5 if task is addressed
    overall = apply_fair_band_scoring(overall, tr, task_type)
    band = round_band(overall)

    # =========================
    # STRICT IELTS WORD COUNT BAND CAP
    # =========================
    if task_type == "task_1":
        if word_count < 50:
            band = min(band, 4.5)
        elif word_count < 100:
            band = min(band, 5.5)

    if task_type == "task_2":
        if word_count < 80:
            band = min(band, 4.5)
        elif word_count < 150:
            band = min(band, 5.5)

    # Controlled refinement with fallback and length guard
    refine_prompt = (
        f"Rewrite this IELTS Task {'1' if task_type == 'task_1' else '2'} answer to Band 7 level. "
        f"Keep it clear, not too long:\n{essay}"
    )
    refined = safe_gpt_call(
        refine_prompt,
        fallback=essay,
        caller=lambda p: call_gpt_text(p, system_msg="You are an IELTS Writing tutor.")
    )
    refined = safe_output(refined, essay)
    if isinstance(refined, str) and len(refined.split()) > 180:
        refined = " ".join(refined.split()[:180])
    
    # Apply coherence penalty cap: max 2 repetition-related errors
    raw_mistakes = ai.get("mistakes", [])
    if isinstance(raw_mistakes, list):
        filtered_mistakes = []
        for m in raw_mistakes:
            exp = (m.get("explanation", "") or "")
            if "no error" in exp.lower():
                continue
            filtered_mistakes.append(m)
        raw_mistakes = filtered_mistakes
    mistakes = apply_coherence_penalty_cap(raw_mistakes if isinstance(raw_mistakes, list) else [])
    
    # Map to CEFR level
    cefr = map_ielts_to_cefr(band)
    
    # Get vocabulary to learn
    vocab_list = generate_topic_vocabulary(question, essay, task_type)
    topic_words = [w for w in vocab_list if w.get("task_specific", True)]
    connectors = [w for w in vocab_list if not w.get("task_specific", True)]
    connectors = connectors[:2]
    vocab_list = (topic_words + connectors)[:10]

    feedback_text = ai.get("examiner_response", "") or ""
    feedback = normalize_feedback(feedback_text) or "Clear and concise answer; focus on stronger linking."
    improvement_text = ""
    if isinstance(ai.get("feedback"), dict):
        improvement_text = ai.get("feedback", {}).get("improvements", "")
    improvement = normalize_feedback(improvement_text) or "Improve coherence with clearer transitions."

    result = {
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
        "word_count": word_count
    }

    # Consistent top-level shape for downstream UI
    result["band_score"] = result.get("overall_band", band)
    result["feedback"] = safe_output(feedback, "Provide clearer structure and examples.")
    result["improvement"] = safe_output(improvement, "Use varied vocabulary and clearer linking words.")
    result["vocabulary"] = vocab_list

    return result
