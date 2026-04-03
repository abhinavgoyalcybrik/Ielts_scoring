import re
from typing import List, Dict


# Deterministic helpers and small local lexicons for topic-aware suggestions.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were",
    "i", "me", "my", "we", "you", "they", "he", "she", "it", "to", "in",
    "on", "for", "of", "with", "that", "this", "there", "here"
}

BASIC_REPLACEMENTS = {
    "big": ["substantial", "considerable"],
    "good": ["beneficial", "advantageous"],
    "many": ["numerous", "a large number of"],
    "very": ["highly", "particularly"],
}

TOPIC_ALTERNATIVES = {
    "city": ["urban centre", "metropolitan area"],
    "hometown": ["place of origin", "residential area"],
    "park": ["public green space", "recreational area"],
    "job": ["employment", "career opportunities"],
    "education": ["educational opportunities", "academic provision"],
    "music": ["musical tradition", "musical genre"],
    "travel": ["international travel", "tourism industry"],
    "study": ["pursue further study", "undertake academic study"],
    "stress": ["psychological strain", "pressure"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+'?\w*|\w+", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def analyze_vocabulary(text: str) -> Dict[str, List[str]]:
    """
    Return transcript-aware vocabulary feedback with two keys:
      - good_usage: 2–5 items/phrases actually used by the candidate
      - suggested_improvements: concrete higher-band alternatives (topic-relevant)

    Heuristics (deterministic):
    - If text is empty or very short, return empty lists.
    - good_usage: prefer meaningful bigrams/trigrams and advanced single words found in text.
    - suggested_improvements: map basic words and topic keywords to higher-band alternatives.
    """
    if not text or not isinstance(text, str):
        return {"good_usage": [], "suggested_improvements": []}

    tokens = _tokenize(text)
    if len(tokens) < 6:
        return {"good_usage": [], "suggested_improvements": []}

    lowered = text.lower()
    suggested: List[str] = []
    good_usage: List[str] = []

    # --- Good usage: collect meaningful bigrams/trigrams present in text ---
    bigrams = _ngrams(tokens, 2)
    trigrams = _ngrams(tokens, 3)

    def meaningful(phrase: str) -> bool:
        parts = phrase.split()
        return all(p not in STOPWORDS and len(p) > 2 for p in parts)

    # prefer trigrams, then bigrams, then single nouns (heuristic)
    for ng in (trigrams + bigrams):
        if meaningful(ng) and ng not in good_usage:
            good_usage.append(ng)
        if len(good_usage) >= 5:
            break

    # if still short, pick some strong single-word content tokens
    if len(good_usage) < 2:
        freq = {}
        for t in tokens:
            if t not in STOPWORDS and len(t) > 3:
                freq[t] = freq.get(t, 0) + 1
        # sort by frequency then length
        for w in sorted(freq.keys(), key=lambda x: (-freq[x], -len(x))):
            if w not in good_usage:
                good_usage.append(w)
            if len(good_usage) >= 5:
                break

    # --- Suggested improvements: basic replacements for weak words ---
    seen = set()
    for bw, alts in BASIC_REPLACEMENTS.items():
        if re.search(r"\b" + re.escape(bw) + r"\b", lowered):
            suggestion = f"Consider '{alts[0]}' or '{alts[1]}' instead of '{bw}'"
            suggested.append(suggestion)
            seen.add(suggestion)

    # Topic-aware alternatives (only suggest if topic word used)
    for topic, alts in TOPIC_ALTERNATIVES.items():
        if re.search(r"\b" + re.escape(topic) + r"\b", lowered):
            suggestion = f"Consider '{alts[0]}' or '{alts[1]}' instead of '{topic}'"
            if suggestion not in seen:
                suggested.append(suggestion)
                seen.add(suggestion)

    # If no targeted suggestions found, but the text contains some lower-level phrases,
    # provide a couple of generic phrase upgrades if applicable
    if not suggested:
        for phrase, alts in {"big city": ["metropolitan area", "urban centre"],
                             "in my opinion": ["I believe", "I would argue that"]}.items():
            if phrase in lowered:
                suggested.append(f"Consider '{alts[0]}' or '{alts[1]}' instead of '{phrase}'")

    # Deduplicate good_usage and keep up to 5 items
    good_usage = list(dict.fromkeys(good_usage))[:5]

    # --- Post-process suggested items into a clean, non-overlapping list ---
    # Extract candidate phrases from suggestion strings (quoted phrases preferred)
    candidates: List[str] = []
    for s in suggested:
        quotes = re.findall(r"'([^']+)'", s)
        if quotes:
            for q in quotes:
                candidates.append(q.strip())
        else:
            # fallback: split on ' or ' or commas
            parts = re.split(r"\bor\b|,", s)
            for p in parts:
                p = p.strip().strip("'\"")
                if p:
                    candidates.append(p)

    # Normalize and deduplicate while merging overlapping/duplicate ideas
    normalized = []
    for c in candidates:
        low = c.lower()
        toks = set(re.findall(r"\w+", low))
        if not toks:
            continue
        replaced = False
        # If an existing phrase is subset of this one, replace it with the more advanced (longer) phrase
        for i, exist in enumerate(list(normalized)):
            exist_low = exist.lower()
            exist_toks = set(re.findall(r"\w+", exist_low))
            if exist_toks <= toks and len(toks) > len(exist_toks):
                normalized[i] = c
                replaced = True
                break
            if toks <= exist_toks:
                replaced = True
                break
        if not replaced:
            normalized.append(c)

    # Final dedupe while preserving order and uniqueness
    seen = set()
    final_suggestions = []
    for item in normalized:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        final_suggestions.append(item)
        if len(final_suggestions) >= 10:
            break

    return {"good_usage": good_usage, "suggested_improvements": final_suggestions}


def detect_essay_topic(text: str) -> str:
    """
    Detect the essay topic from text keywords.
    Returns: topic string (e.g., 'education', 'technology', 'environment', 'work', 'travel')
    """
    text_lower = text.lower()
    
    topic_keywords = {
        "technology": ["technology", "internet", "online", "digital", "software", "artificial", "ai", "robot", "innovation"],
        "education": ["education", "school", "university", "study", "learning", "student", "academic", "degree"],
        "work": ["work", "job", "career", "employment", "employer", "workplace", "professional"],
        "environment": ["environment", "climate", "pollution", "sustainable", "green", "energy", "carbon", "emissions"],
        "health": ["health", "exercise", "fitness", "disease", "diet", "sport", "wellness", "medical"],
        "travel": ["travel", "tourism", "destination", "tourism industry", "culture", "tourist", "trip"],
        "transport": ["transport", "transportation", "traffic", "public transport", "vehicle", "driving", "roads"],
        "social": ["social", "family", "relationship", "community", "society", "people", "culture"],
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return topic
    
    return "general"


def generate_topic_vocabulary(question: str, essay: str, task_type: str) -> List[Dict[str, str]]:
    """
    Generate task-aware vocabulary dynamically from question/topic.
    Returns 10-15 items with usage hints and task_type tagging.
    """
    question_lower = (question or "").lower()
    essay_lower = (essay or "").lower()
    combined_text = f"{question_lower} {essay_lower}".strip()
    topic = detect_essay_topic(combined_text)

    def build(items: List[Dict[str, str]], task_label: str, limit: int = 15) -> List[Dict[str, str]]:
        seen = set()
        vocab = []
        for item in items:
            word = item.get("word", "").strip()
            if not word or word.lower() in seen:
                continue
            task_specific = item.get("task_specific", True)
            seen.add(word.lower())
            vocab.append({
                "word": word,
                "usage_hint": item.get("usage_hint", ""),
                "task_type": task_label,
                "task_specific": task_specific
            })
            if len(vocab) >= limit:
                break
        return vocab

    # General Training Task 1 (letter)
    is_letter = (
        task_type == "general_task_1"
        or "letter" in question_lower
        or question_lower.startswith("write a letter")
        or "dear" in essay_lower
    )
    if is_letter:
        letter_vocab = [
            {"word": "invite", "usage_hint": "ask someone to come"},
            {"word": "visit", "usage_hint": "go to see someone"},
            {"word": "stay", "usage_hint": "live temporarily"},
            {"word": "arrange", "usage_hint": "make plans or organise"},
            {"word": "convenient", "usage_hint": "easy and suitable time/place"},
            {"word": "nearby", "usage_hint": "close in distance"},
            {"word": "comfortable", "usage_hint": "pleasant and relaxing"},
            {"word": "appreciate", "usage_hint": "express gratitude politely"},
            {"word": "accommodate", "usage_hint": "provide lodging or space"},
            {"word": "suggest", "usage_hint": "propose an idea"},
            {"word": "clarify", "usage_hint": "make something clear"},
            {"word": "arrangements", "usage_hint": "plans you have made"},
        ]
        return build(letter_vocab, "general_task_1", limit=12)

    # Academic Task 1 (charts/data/process with strict 50/50 balance)
    if task_type in ("task_1", "task1"):
        base_words = [
            "illustrates",
            "proportion",
            "trend",
            "increase",
            "decline"
        ][:5]

        if "transport" in question_lower or "travel" in question_lower:
            topic_words_raw = ["commute", "traffic", "vehicle", "passengers", "public transport"]
        elif "population" in question_lower:
            topic_words_raw = ["residents", "demographics", "growth rate", "urban", "rural"]
        elif "education" in question_lower or "students" in question_lower:
            topic_words_raw = ["students", "enrollment", "graduates", "institutions", "literacy"]
        elif "sales" in question_lower or "company" in question_lower:
            topic_words_raw = ["revenue", "profit", "consumers", "market share", "growth"]
        else:
            topic_words_raw = ["category", "data", "figures", "comparison", "segment"]

        topic_words_raw = topic_words_raw[:5]
        final_words = base_words + topic_words_raw  # exactly 10 (5 base + 5 topic)

        vocab_structured = [
            {"word": w, "usage_hint": "context-based usage", "task_type": "task_1", "task_specific": True}
            for w in final_words
        ]
        return build(vocab_structured, "task_1", limit=10)

    # Task 2 (argument + topic-specific)
    topic_vocab = {
        "technology": [
            {"word": "innovation", "usage_hint": "new ideas or methods"},
            {"word": "automation", "usage_hint": "machines handling tasks"},
            {"word": "digitalization", "usage_hint": "converting to digital form"},
            {"word": "efficiency", "usage_hint": "doing tasks with less waste"},
            {"word": "dependency", "usage_hint": "reliance on technology"},
        ],
        "education": [
            {"word": "curriculum", "usage_hint": "course of study"},
            {"word": "pedagogy", "usage_hint": "teaching methodology"},
            {"word": "literacy", "usage_hint": "ability to read and write"},
            {"word": "assessment", "usage_hint": "evaluation of learning"},
            {"word": "scholarship", "usage_hint": "financial aid for study"},
        ],
        "environment": [
            {"word": "sustainability", "usage_hint": "meeting needs without depletion"},
            {"word": "emissions", "usage_hint": "release of gases"},
            {"word": "renewable", "usage_hint": "naturally replenished resources"},
            {"word": "conservation", "usage_hint": "protecting natural resources"},
            {"word": "biodiversity", "usage_hint": "variety of life forms"},
        ],
        "health": [
            {"word": "prevention", "usage_hint": "avoiding illness"},
            {"word": "nutrition", "usage_hint": "quality of diet"},
            {"word": "sedentary", "usage_hint": "inactive lifestyle"},
            {"word": "wellbeing", "usage_hint": "state of health and happiness"},
            {"word": "vaccination", "usage_hint": "immunisation against disease"},
        ],
        "transport": [
            {"word": "congestion", "usage_hint": "traffic crowding"},
            {"word": "infrastructure", "usage_hint": "roads, rail, facilities"},
            {"word": "commute", "usage_hint": "daily travel to work"},
            {"word": "public transit", "usage_hint": "buses, trains, metros"},
            {"word": "sustainability", "usage_hint": "environmentally friendly travel"},
        ],
        "work": [
            {"word": "productivity", "usage_hint": "output per time"},
            {"word": "flexibility", "usage_hint": "adaptable working patterns"},
            {"word": "collaboration", "usage_hint": "working together"},
            {"word": "burnout", "usage_hint": "exhaustion from stress"},
            {"word": "remote work", "usage_hint": "working away from office"},
        ],
        "social": [
            {"word": "inequality", "usage_hint": "unequal distribution"},
            {"word": "cohesion", "usage_hint": "unity in a community"},
            {"word": "diversity", "usage_hint": "variety of backgrounds"},
            {"word": "engagement", "usage_hint": "active participation"},
            {"word": "marginalized", "usage_hint": "pushed to the edge of society"},
        ],
        "general": [
            {"word": "policy", "usage_hint": "plan of action"},
            {"word": "impact", "usage_hint": "effect or influence"},
            {"word": "evidence", "usage_hint": "supporting facts"},
            {"word": "stakeholders", "usage_hint": "people affected by an issue"},
            {"word": "trade-off", "usage_hint": "balancing competing factors"},
        ],
    }

    argument_vocab = [
        {"word": "furthermore", "usage_hint": "add another argument", "task_specific": False},
        {"word": "however", "usage_hint": "introduce contrast", "task_specific": False},
        {"word": "consequently", "usage_hint": "show result", "task_specific": False},
        {"word": "moreover", "usage_hint": "add emphasis", "task_specific": False},
        {"word": "mitigate", "usage_hint": "make a problem less severe", "task_specific": False},
        {"word": "beneficial", "usage_hint": "helpful or advantageous", "task_specific": False},
        {"word": "notably", "usage_hint": "highlight a key point", "task_specific": False},
        {"word": "for instance", "usage_hint": "introduce an example", "task_specific": False},
        {"word": "on the other hand", "usage_hint": "present an opposing view", "task_specific": False},
        {"word": "conclude", "usage_hint": "finish an argument", "task_specific": False},
    ]

    topic_terms = topic_vocab.get(topic, topic_vocab["general"])

    limit = 15
    topic_target = max(1, round(limit * 0.7))
    connector_target = max(1, limit - topic_target)

    chosen = []
    seen = set()

    # First, add topic-specific terms up to target
    for item in topic_terms:
        word = item.get("word", "").lower()
        if not word or word in seen:
            continue
        chosen.append(item)
        seen.add(word)
        if len(chosen) >= topic_target:
            break

    # Then, add connectors up to target
    connectors_added = 0
    for item in argument_vocab:
        word = item.get("word", "").lower()
        if not word or word in seen:
            continue
        chosen.append(item)
        seen.add(word)
        connectors_added += 1
        if connectors_added >= connector_target:
            break

    # Fill remaining slots prioritizing topic terms, then connectors
    for item in topic_terms:
        if len(chosen) >= limit:
            break
        word = item.get("word", "").lower()
        if word and word not in seen:
            chosen.append(item)
            seen.add(word)

    if len(chosen) < limit:
        for item in argument_vocab:
            if len(chosen) >= limit:
                break
            word = item.get("word", "").lower()
            if word and word not in seen:
                chosen.append(item)
                seen.add(word)

    return build(chosen, "task_2", limit=limit)


def get_writing_vocabulary_reference(task_type: str = None, essay_text: str = "", question: str = "") -> List[Dict[str, str]]:
    """
    Return TASK-SPECIFIC vocabulary list for IELTS Writing tasks.
    Now generated dynamically from the question/topic to avoid fixed lists.
    """
    return generate_topic_vocabulary(question or "", essay_text or "", task_type or "task_2")
