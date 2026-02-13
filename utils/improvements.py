# ==============================
# READING IMPROVEMENTS
# ==============================
def reading_improvements(correct: int):
    if correct >= 35:
        return [
            "Work on speed and handling complex inference questions.",
            "Continue practising full-length reading tests under timed conditions."
        ]
    elif correct >= 27:
        return [
            "Improve skimming and scanning techniques.",
            "Focus on identifying paraphrased information in passages."
        ]
    elif correct >= 20:
        return [
            "Practise understanding main ideas and supporting details.",
            "Work on common IELTS question types such as True/False/Not Given."
        ]
    else:
        return [
            "Focus on basic reading comprehension skills.",
            "Build vocabulary and practise reading short passages regularly."
        ]


# ==============================
# LISTENING IMPROVEMENTS
# ==============================
def listening_improvements(correct: int):
    if correct >= 35:
        return [
            "Maintain accuracy while improving speed.",
            "Practise listening to a range of English accents."
        ]
    elif correct >= 27:
        return [
            "Focus on identifying key words and distractors.",
            "Practise note-completion and form-completion questions."
        ]
    elif correct >= 20:
        return [
            "Improve ability to follow conversations and instructions.",
            "Practise listening for specific information."
        ]
    else:
        return [
            "Practise basic listening skills with short audio clips.",
            "Build vocabulary commonly used in everyday situations."
        ]


# ==============================
# SPEAKING IMPROVEMENTS
# ==============================
def speaking_improvements(criteria_scores: dict):
    improvements = []

    fluency = criteria_scores.get("fluency_coherence", 6)
    lexical = criteria_scores.get("lexical_resource", 6)
    grammar = criteria_scores.get("grammar_accuracy", 6)
    pronunciation = criteria_scores.get("pronunciation", 6)

    if fluency < 6:
        improvements.append(
            "Work on speaking more smoothly with fewer pauses and repetitions."
        )

    if lexical < 6:
        improvements.append(
            "Try to use a wider range of vocabulary and more precise expressions."
        )

    if grammar < 6:
        improvements.append(
            "Practise using a mix of simple and complex sentence structures accurately."
        )

    if pronunciation < 6:
        improvements.append(
            "Focus on clearer pronunciation, word stress, and intonation."
        )

    if not improvements:
        improvements.append(
            "Maintain your current speaking level through regular speaking practice."
        )

    return improvements


# ==============================
# WRITING IMPROVEMENTS
# ==============================
def writing_improvements(criteria_scores: dict, task_type: str):
    improvements = []

    tr = criteria_scores.get("task_response", 6)
    cc = criteria_scores.get("coherence_cohesion", 6)
    lr = criteria_scores.get("lexical_resource", 6)
    gr = criteria_scores.get("grammar_accuracy", 6)

    if tr < 6:
        if task_type == "task_1":
            improvements.append(
                "Ensure you include a clear overview summarising the main trends or features."
            )
        else:
            improvements.append(
                "Make your position clear and fully address all parts of the question."
            )

    if cc < 6:
        improvements.append(
            "Improve paragraphing and use linking words more effectively to organise ideas."
        )

    if lr < 6:
        improvements.append(
            "Expand your vocabulary and avoid repeating the same words and phrases."
        )

    if gr < 6:
        improvements.append(
            "Practise using a wider range of complex sentence structures with fewer errors."
        )

    if not improvements:
        improvements.append(
            "To achieve a higher band, focus on more precise vocabulary, varied sentence structures, and clearer development of ideas."
        )

    return improvements
