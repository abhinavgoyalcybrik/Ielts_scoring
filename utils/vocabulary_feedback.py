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


def get_writing_vocabulary_reference(task_type: str = None) -> List[Dict[str, str]]:
    """
    Return topic-relevant vocabulary list for IELTS Writing tasks.
    Each item includes word/phrase and a usage hint for learning.
    
    Args:
        task_type: "task_1" (chart/diagram) or "task_2" (opinion/discussion) or None (general)
    
    Returns:
        List of dicts with keys: "word", "usage_hint"
    """
    
    # General academic vocabulary (applies to both tasks)
    general_vocab = [
        {"word": "proportion", "usage_hint": "use with figures and percentages"},
        {"word": "substantial", "usage_hint": "describe large changes or amounts"},
        {"word": "significant", "usage_hint": "emphasize importance of trends"},
        {"word": "decline", "usage_hint": "describe downward movement in data"},
        {"word": "surge", "usage_hint": "describe rapid increase"},
        {"word": "fluctuate", "usage_hint": "show up-and-down changes"},
    ]
    
    # Task 1 specific (charts, data, comparisons)
    task1_vocab = general_vocab + [
        {"word": "depicts", "usage_hint": "describe what a chart shows"},
        {"word": "workforce", "usage_hint": "employed population"},
        {"word": "sector", "usage_hint": "industry area (agriculture, services)"},
        {"word": "structural shift", "usage_hint": "long-term economic change"},
        {"word": "prominence", "usage_hint": "importance or dominance"},
        {"word": "marked", "usage_hint": "clearly noticeable (marked change)"},
        {"word": "predominantly", "usage_hint": "mainly, mostly"},
        {"word": "in contrast", "usage_hint": "show difference between items"},
    ]
    
    # Task 2 specific (opinion, arguments, discussion)
    task2_vocab = general_vocab + [
        {"word": "contend", "usage_hint": "argue or claim (contend that...)"},
        {"word": "alleviate", "usage_hint": "reduce or ease a problem"},
        {"word": "sustainable", "usage_hint": "environmentally and socially viable"},
        {"word": "infrastructure", "usage_hint": "basic systems (roads, transport, facilities)"},
        {"word": "mitigate", "usage_hint": "make less severe (mitigate climate change)"},
        {"word": "encompasses", "usage_hint": "includes or contains"},
        {"word": "furthermore", "usage_hint": "add another strong point"},
        {"word": "inherent", "usage_hint": "natural, built-in quality"},
    ]
    
    # Return based on task type
    if task_type and "task_1" in str(task_type).lower():
        return task1_vocab
    elif task_type and "task_2" in str(task_type).lower():
        return task2_vocab
    else:
        return general_vocab
