from evaluators.writing import evaluate_writing
from evaluators.speaking import evaluate_speaking

# SAFE analysis-only imports
from utils.cefr_mapper import map_ielts_to_cefr
from utils.wpm import calculate_writing_wpm, calculate_speaking_wpm
from utils.vocabulary_feedback import analyze_vocabulary, get_writing_vocabulary_reference
import re
import json


# Minimal, conservative cleaning for user inputs that contain typing noise
def _clean_text_minimal(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    s = text
    # Ensure a space after a period when missing
    s = re.sub(r"\.([A-Za-z0-9])", r". \1", s)
    # Remove very long letter-only tokens (likely typing noise)
    s = re.sub(r"\b[a-zA-Z]{12,}\b", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _map_grammar_score_to_cefr(score: float) -> str:
    # Conservative mapping using grammar accuracy score (0-9)
    try:
        s = float(score)
    except Exception:
        return "B2"
    if s >= 8.0:
        return "C1"
    if s >= 7.0:
        return "B2+"
    return "B2"


def _coherence_summary_for_tasks(tasks: dict) -> str:
    parts = []
    for tname, tdata in sorted(tasks.items()):
        coh = None
        # Try to read possible keys where cohesion score may appear
        if isinstance(tdata, dict):
            criteria = tdata.get("criteria_scores", {}) or {}
            coh = criteria.get("coherence_cohesion") or criteria.get("coherence_and_cohesion") or tdata.get("coherence_and_cohesion")
        # Fallback
        if coh is None:
            parts.append(f"{tname}: coherence not scored")
            continue
        try:
            c = float(coh)
        except Exception:
            parts.append(f"{tname}: coherence unclear")
            continue
        if c >= 8:
            parts.append(f"{tname}: clear organisation and effective progression")
        elif c >= 7:
            parts.append(f"{tname}: generally logical but minor cohesion issues")
        else:
            parts.append(f"{tname}: cohesion weaknesses evident")
    return "; ".join(parts)


def _get_criterion_reason(criterion_name: str, score: float, task_type: str) -> str:
    """Generate a 1-line reason for a criterion score."""
    score = float(score)
    
    # Task Response / Task Achievement reasons
    if criterion_name in ("task_response", "task_achievement"):
        if score >= 8:
            return "Addresses all task requirements with clear position and relevant support"
        elif score >= 7:
            return "Addresses task with clear position and adequate examples"
        elif score >= 6:
            return "Addresses task with some clarity but lacks fully developed ideas"
        elif score >= 5:
            return "Attempts to address task but ideas are underdeveloped"
        else:
            return "Does not adequately address task requirements"
    
    # Coherence & Cohesion reasons
    elif criterion_name in ("coherence_cohesion", "coherence_and_cohesion"):
        if score >= 8:
            return "Well-organised with clear progression and effective linking devices"
        elif score >= 7:
            return "Generally logical organisation with mostly clear progression"
        elif score >= 6:
            return "Adequate organisation but some lack of clarity in progression"
        elif score >= 5:
            return "Inconsistent organisation with limited linking devices"
        else:
            return "Poor organisation and lack of clear progression"
    
    # Lexical Resource / Vocabulary reasons
    elif criterion_name in ("lexical_resource",):
        if score >= 8:
            return "Sophisticated vocabulary with precise word choice and variety"
        elif score >= 7:
            return "Good range of vocabulary with appropriate word choice"
        elif score >= 6:
            return "Adequate vocabulary but with some repetition"
        elif score >= 5:
            return "Limited vocabulary with frequent repetition"
        else:
            return "Poor vocabulary range limiting communication"
    
    # Grammar Accuracy / Grammar Range and Accuracy reasons
    elif criterion_name in ("grammar_accuracy", "grammatical_range_and_accuracy"):
        if score >= 8:
            return "Confident use of complex structures with minimal errors"
        elif score >= 7:
            return "Mostly accurate with good range of complex structures"
        elif score >= 6:
            return "Accurate in simple structures with inconsistent complex use"
        elif score >= 5:
            return "Basic accuracy with frequent grammatical errors"
        else:
            return "Poor grammatical accuracy restricting communication"
    
    return "Score reflects task completion and language control"


def _format_strict_writing_output(task_1_result: dict, task_2_result: dict) -> dict:
    """
    Format writing evaluation output according to strict specification:
    Section A: OVERALL RESULT
    Section B: CRITERIA BREAKDOWN (mandatory for each task)
    Section C: ERRORS FOUND
    Section D: REFINED ANSWER
    Section E: USEFUL VOCABULARY (exact title required)
    Section F: HOW THESE WORDS IMPROVE THE ANSWER (exact title required)
    """
    
    # Extract bands
    try:
        task_1_band = float(task_1_result.get("overall_band", 5.0))
        task_2_band = float(task_2_result.get("overall_band", 5.0))
    except (ValueError, TypeError):
        task_1_band = 5.0
        task_2_band = 5.0
    
    # Calculate overall writing band (Task 1 = 1/3, Task 2 = 2/3)
    from utils.band import round_band
    overall_writing_band = round_band((task_1_band * 1 / 3) + (task_2_band * 2 / 3))
    
    # ========== SECTION A: OVERALL RESULT ==========
    section_a = {
        "section": "A. OVERALL RESULT",
        "task_1_band": round(task_1_band, 1),
        "task_2_band": round(task_2_band, 1),
        "overall_writing_band": round(overall_writing_band, 1)
    }
    
    # ========== SECTION B: CRITERIA BREAKDOWN ==========
    section_b = {
        "section": "B. CRITERIA BREAKDOWN",
        "task_1": {},
        "task_2": {}
    }
    
    # Task 1 criteria with reasons
    task_1_criteria = task_1_result.get("criteria_scores", {})
    section_b["task_1"]["Task Response"] = {
        "band": round(float(task_1_criteria.get("task_response", 5.0)), 1),
        "reason": _get_criterion_reason("task_response", task_1_criteria.get("task_response", 5.0), "task_1")
    }
    section_b["task_1"]["Coherence & Cohesion"] = {
        "band": round(float(task_1_criteria.get("coherence_cohesion", 5.0)), 1),
        "reason": _get_criterion_reason("coherence_cohesion", task_1_criteria.get("coherence_cohesion", 5.0), "task_1")
    }
    section_b["task_1"]["Lexical Resource"] = {
        "band": round(float(task_1_criteria.get("lexical_resource", 5.0)), 1),
        "reason": _get_criterion_reason("lexical_resource", task_1_criteria.get("lexical_resource", 5.0), "task_1")
    }
    section_b["task_1"]["Grammar Accuracy"] = {
        "band": round(float(task_1_criteria.get("grammar_accuracy", 5.0)), 1),
        "reason": _get_criterion_reason("grammar_accuracy", task_1_criteria.get("grammar_accuracy", 5.0), "task_1")
    }
    
    # Task 2 criteria with reasons
    task_2_criteria = task_2_result.get("criteria_scores", {})
    section_b["task_2"]["Task Response"] = {
        "band": round(float(task_2_criteria.get("task_response", 5.0)), 1),
        "reason": _get_criterion_reason("task_response", task_2_criteria.get("task_response", 5.0), "task_2")
    }
    section_b["task_2"]["Coherence & Cohesion"] = {
        "band": round(float(task_2_criteria.get("coherence_cohesion", 5.0)), 1),
        "reason": _get_criterion_reason("coherence_cohesion", task_2_criteria.get("coherence_cohesion", 5.0), "task_2")
    }
    section_b["task_2"]["Lexical Resource"] = {
        "band": round(float(task_2_criteria.get("lexical_resource", 5.0)), 1),
        "reason": _get_criterion_reason("lexical_resource", task_2_criteria.get("lexical_resource", 5.0), "task_2")
    }
    section_b["task_2"]["Grammar Accuracy"] = {
        "band": round(float(task_2_criteria.get("grammar_accuracy", 5.0)), 1),
        "reason": _get_criterion_reason("grammar_accuracy", task_2_criteria.get("grammar_accuracy", 5.0), "task_2")
    }
    
    # ========== SECTION C: ERRORS FOUND ==========
    section_c = {
        "section": "C. ERRORS FOUND",
        "task_1": [],
        "task_2": []
    }
    
    # Task 1 errors
    task_1_mistakes = task_1_result.get("mistakes", [])
    for i, error in enumerate(task_1_mistakes[:10]):  # Max 10 errors per task
        if isinstance(error, dict):
            section_c["task_1"].append({
                "index": i + 1,
                "sentence": error.get("sentence", ""),
                "error_type": error.get("error_type", "other"),
                "problem": error.get("why_problem", error.get("explanation", "Unclear error")),
                "correction": error.get("correction", "Review text carefully")
            })
    
    # Task 2 errors
    task_2_mistakes = task_2_result.get("mistakes", [])
    for i, error in enumerate(task_2_mistakes[:10]):  # Max 10 errors per task
        if isinstance(error, dict):
            section_c["task_2"].append({
                "index": i + 1,
                "sentence": error.get("sentence", ""),
                "error_type": error.get("error_type", "other"),
                "problem": error.get("why_problem", error.get("explanation", "Unclear error")),
                "correction": error.get("correction", "Review text carefully")
            })
    
    # ========== SECTION D: REFINED ANSWER ==========
    section_d = {
        "section": "D. REFINED ANSWER",
        "task_1": task_1_result.get("refined_answer", ""),
        "task_2": task_2_result.get("refined_answer", "")
    }
    
    # ========== SECTION E: USEFUL VOCABULARY ==========
    # Get vocabulary reference from task-specific lists
    vocab_ref_task_1 = get_writing_vocabulary_reference("task_1")
    vocab_ref_task_2 = get_writing_vocabulary_reference("task_2")
    
    # Combine and ensure 12-20 words total
    useful_vocab = []
    for word_dict in vocab_ref_task_1[:10]:
        if isinstance(word_dict, dict):
            useful_vocab.append({
                "word": word_dict.get("word", ""),
                "usage_hint": word_dict.get("usage_hint", "")
            })
    for word_dict in vocab_ref_task_2[:10]:
        if isinstance(word_dict, dict):
            useful_vocab.append({
                "word": word_dict.get("word", ""),
                "usage_hint": word_dict.get("usage_hint", "")
            })
    
    # Ensure we have 12-20 words
    useful_vocab = useful_vocab[:20]  # Cap at 20
    
    section_e = {
        "section": "Useful Vocabulary",  # EXACT TITLE
        "vocabulary_list": useful_vocab
    }
    
    # ========== SECTION F: HOW THESE WORDS IMPROVE THE ANSWER ==========
    section_f = {
        "section": "How These Words Improve the Answer",  # EXACT TITLE
        "improvements": [
            {
                "bullet": 1,
                "focus": "Lexical Resource",
                "description": "These words expand your vocabulary range from basic to advanced level, allowing you to express ideas with precision and variety across different topics."
            },
            {
                "bullet": 2,
                "focus": "Task Response",
                "description": "Topic-specific vocabulary enables you to address task requirements more directly and demonstrate clear understanding of the question."
            },
            {
                "bullet": 3,
                "focus": "Coherence",
                "description": "Using varied and accurate vocabulary reduces repetition and improves overall readability, making your arguments flow more naturally."
            }
        ]
    }
    
    # ========== COMBINE ALL SECTIONS ==========
    formatted_output = {
        "status": "valid",
        "strict_format": True,
        "A_OVERALL_RESULT": section_a,
        "B_CRITERIA_BREAKDOWN": section_b,
        "C_ERRORS_FOUND": section_c,
        "D_REFINED_ANSWER": section_d,
        "E_USEFUL_VOCABULARY": section_e,
        "F_HOW_WORDS_IMPROVE": section_f
    }
    
    return formatted_output



def evaluate_attempt(data):
    """
    Central evaluator dispatcher
    ENABLED: Writing + Speaking
    SCORING LOGIC IS NOT TOUCHED
    """

    test_type = data.get("test_type")

    # =========================
    # WRITING - SINGLE TASK EVALUATION (with post-processing)
    # =========================
    if test_type == "writing":
        result = evaluate_writing(data)   # 🔒 DO NOT TOUCH INTERNAL LOGIC

        # -------- SAFE POST-PROCESSING --------
        # Accept either legacy 'overall_writing_band' or 'overall_band'
        overall_band = result.get("overall_writing_band") or result.get("overall_band")

        # Determine word count from possible return shapes
        word_count = 0
        if "word_count" in result:
            word_count = result.get("word_count", 0)
        elif "tasks" in result and "task_2" in result["tasks"]:
            word_count = result["tasks"]["task_2"].get("word_count", 0)

        if overall_band:
            # CEFR must be strictly derived from the overall band only
            result["cefr_level"] = map_ielts_to_cefr(overall_band)

        # Time may be provided as 'time_taken_minutes' or 'time_minutes'
        time_minutes = data.get("time_taken_minutes") or data.get("time_minutes") or 0
        if time_minutes and isinstance(time_minutes, (int, float)) and time_minutes > 0:
            result["wpm"] = calculate_writing_wpm(word_count, time_minutes)
        else:
            # Per requirement: do not guess when time missing
            result["wpm"] = 0.0

        # Extract answer text from common locations for vocabulary analysis
        answer_text = ""
        if isinstance(data.get("user_answers"), dict):
            answer_text = data.get("user_answers", {}).get("text", "")
        if not answer_text:
            answer_text = data.get("task_2", {}).get("answer", "")

        # Minimal cleaning applied only for analysis outputs and refined_answer text
        cleaned_answer_text = _clean_text_minimal(answer_text)
        result["vocabulary_feedback"] = analyze_vocabulary(cleaned_answer_text)

        # Build performance_analysis object
        tasks = result.get("tasks", {}) if isinstance(result.get("tasks", {}), dict) else {}

        # Determine per-task word counts (fall back to counting refined_answer)
        task_word_counts = {}
        for tname, tdata in tasks.items():
            wc = 0
            if isinstance(tdata, dict):
                wc = tdata.get("word_count") or 0
                if not wc:
                    # try counting refined_answer
                    text = tdata.get("refined_answer") or ""
                    if text:
                        wc = len(_clean_text_minimal(text).split())
            task_word_counts[tname] = int(wc or 0)

        # Time assumptions: accept dict under data['time_taken_minutes'] else default Task1=20 Task2=40
        default_times = {"task_1": 20, "task_2": 40}
        times = default_times.copy()
        user_times = data.get("time_taken_minutes")
        if isinstance(user_times, dict):
            for k in ("task_1", "task_2"):
                if k in user_times and isinstance(user_times[k], (int, float)) and user_times[k] > 0:
                    times[k] = float(user_times[k])

        # Compute WPM per task
        wpm_tasks = {}
        total_words = 0
        total_minutes = 0
        for k in ("task_1", "task_2"):
            wc = task_word_counts.get(k, 0)
            mins = times.get(k, default_times[k])
            w = 0.0
            if mins and mins > 0:
                w = round(float(wc) / float(mins), 1)
            wpm_tasks[k] = w
            total_words += wc
            total_minutes += mins

        overall_wpm = round(float(total_words) / float(total_minutes), 1) if total_minutes > 0 else 0.0

        # Overall CEFR writing level derived strictly from overall band
        overall_cefr = map_ielts_to_cefr(overall_band) if overall_band else None

        # Grammar CEFR: derive from tasks' grammar accuracy scores if present
        grammar_scores = []
        for tname, tdata in tasks.items():
            if isinstance(tdata, dict):
                crit = tdata.get("criteria_scores") or {}
                g = crit.get("grammar_accuracy") or crit.get("grammatical_range_and_accuracy") or tdata.get("grammar_accuracy")
                if g is not None:
                    try:
                        grammar_scores.append(float(g))
                    except Exception:
                        pass
        avg_grammar = (sum(grammar_scores) / len(grammar_scores)) if grammar_scores else None
        grammar_cefr = _map_grammar_score_to_cefr(avg_grammar if avg_grammar is not None else 7.0)

        # Coherence summary
        coherence_summary = _coherence_summary_for_tasks(tasks) if tasks else "No task coherence data"

        # Improvement suggestions (minimal, aligned with visible mistakes)
        improvement = {"task_1": [], "task_2": []}
        for k in ("task_1", "task_2"):
            t = tasks.get(k, {})
            mistakes = []
            if isinstance(t, dict):
                mistakes = t.get("mistakes", []) or []
            if mistakes:
                # Inspect mistake types conservatively
                for m in mistakes:
                    et = m.get("error_type", "")
                    if et == "coherence":
                        improvement[k].append("Avoid repetition and rephrase repeated clauses")
                    elif et == "grammar":
                        improvement[k].append("Proofread punctuation and sentence boundaries")
                    else:
                        improvement[k].append("Review sentence clarity and remove extraneous text")
                # Deduplicate suggestions
                improvement[k] = list(dict.fromkeys(improvement[k]))
            else:
                # Conservative, minimal suggestions
                if k == "task_1":
                    improvement[k] = ["Vary sentence openings and linking phrases", "Make overview comparisons explicit"]
                else:
                    improvement[k] = ["Avoid verbatim repetition", "Proofread punctuation and spacing"]

        # Exam readiness (conservative)
        exam_ready = {"writing_module": "No", "task_1": "No", "task_2": "No"}
        try:
            ob = float(overall_band) if overall_band is not None else 0.0
            if ob >= 7.0:
                exam_ready["writing_module"] = "Yes"
            t1b = float(tasks.get("task_1", {}).get("overall_band", 0)) if tasks.get("task_1") else 0
            t2b = float(tasks.get("task_2", {}).get("overall_band", 0)) if tasks.get("task_2") else 0
            exam_ready["task_1"] = "Yes" if t1b >= 7.0 else "No"
            exam_ready["task_2"] = "Yes" if t2b >= 7.0 else "No"
        except Exception:
            pass

        # Clean refined answers minimally (fix obvious typing noise and spacing)
        for tname, tdata in tasks.items():
            if isinstance(tdata, dict):
                ra = tdata.get("refined_answer")
                if isinstance(ra, str) and ra:
                    cleaned = _clean_text_minimal(ra)
                    # Only replace if changed
                    if cleaned != ra:
                        tdata["refined_answer"] = cleaned

        # Attach performance_analysis
        result["performance_analysis"] = {
            "time_taken_minutes": {"task_1": times.get("task_1"), "task_2": times.get("task_2")},
            "wpm": {"task_1": wpm_tasks.get("task_1"), "task_2": wpm_tasks.get("task_2"), "overall": overall_wpm},
            "overall_cefr_writing_level": overall_cefr,
            "grammar_cefr_level": grammar_cefr,
            "coherence_summary": coherence_summary,
            "improvement_suggestions": improvement,
            "exam_readiness": exam_ready
        }

        # Add vocabulary reference for learning (per task)
        result["vocabulary_reference"] = {
            "task_1": get_writing_vocabulary_reference("task_1"),
            "task_2": get_writing_vocabulary_reference("task_2")
        }

        return result

    # =========================
    # SPEAKING (TEXT ONLY) - PART-WISE ASSESSMENT
    # =========================
    if test_type == "speaking":
        # Check if multi-part format (part_1, part_2, part_3)
        if any(k in data for k in ["part_1", "part_2", "part_3"]):
            # Multi-part format - call evaluate_speaking with full data
            result = evaluate_speaking(data)
        else:
            # Legacy single-part format - has "part", "transcript", "audio_metrics"
            result = evaluate_speaking(data)

        return result

    # =========================
    raise ValueError("Only writing and speaking evaluation are enabled right now")


# =========================
# PUBLIC API: Format writing results to strict A-F format
# =========================
def format_writing_strict(task_1_result: dict, task_2_result: dict) -> dict:
    """
    PUBLIC FUNCTION: Format two separate task evaluations into strict A-F format.
    
    Usage:
        task_1_result = evaluate_attempt(task_1_data)
        task_2_result = evaluate_attempt(task_2_data)
        formatted = format_writing_strict(task_1_result, task_2_result)
    
    Returns:
        Dictionary with all mandatory sections A-F:
        - A_OVERALL_RESULT: Task bands and overall writing band
        - B_CRITERIA_BREAKDOWN: All 4 criteria with bands and 1-line reasons
        - C_ERRORS_FOUND: List of errors from both tasks
        - D_REFINED_ANSWER: Refined answers for both tasks
        - E_USEFUL_VOCABULARY: Section with exact title "Useful Vocabulary"
        - F_HOW_WORDS_IMPROVE: Section with exact title "How These Words Improve the Answer"
    """
    return _format_strict_writing_output(task_1_result, task_2_result)
