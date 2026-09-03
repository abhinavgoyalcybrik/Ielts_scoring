from pathlib import Path
import json
import logging
import os
import re
from utils.band import round_band
from utils.ai_client import call_gpt_writing, call_gpt_text, call_gpt_extract, call_gpt_refine, _looks_like_refusal
from utils.cefr_mapper import map_ielts_to_cefr
from utils.vocabulary_feedback import analyze_vocabulary, generate_topic_vocabulary
from utils.safety import safe_gpt_call, safe_output, normalize_feedback


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = BASE_DIR / "prompts"

RELEVANCE_NOTICE_MESSAGES = {
    "completely_off_topic": "Your answer is not relevant to the topic of the question asked.",
    "partially_off_topic": "Your answer only partly addresses the topic of the question asked.",
}

# Surfaced when the deterministic image_data_accuracy cap (see
# evaluate_writing()) actually reduced Task Achievement, so the candidate
# understands WHY their score was limited despite otherwise strong
# writing - same pattern as RELEVANCE_NOTICE_MESSAGES above.
IMAGE_ACCURACY_NOTICE_MESSAGES = {
    "significantly_inaccurate": "Your description of the chart's data does not match what the image actually shows - this significantly limits your Task Achievement score.",
    "partially_inaccurate": "Some of the figures or trends you described don't fully match the actual chart - this limits your Task Achievement score.",
}

# Only used for Academic Task 1 when the caller actually supplies the
# chart/graph/diagram image - substituted into the
# <<<IMAGE_VERIFICATION_INSTRUCTIONS>>> placeholder in
# writing_task1_common.txt. Left as an empty string (no-op) whenever no
# image is provided, or for General Training's letter / Task 2, neither
# of which has a source image to verify against. Also instructs GPT to
# self-report "image_data_accuracy" in the same checklist call, which
# evaluate_writing() then enforces deterministically as a real Task
# Achievement cap rather than trusting GPT's own checklist judgment to
# have genuinely factored it in.
IMAGE_VERIFICATION_INSTRUCTIONS = (
    "IMAGE VERIFICATION (CRITICAL - DO NOT SKIP): an actual image of the "
    "chart/graph/diagram/table IS attached to this message alongside this "
    "text. You must genuinely look at it and compare it against the "
    "candidate's answer - this is not optional and \"not_applicable\" is "
    "NOT a valid response here, because an image WAS provided.\n"
    "\n"
    "Task Achievement for Task 1 is not a text-only judgment - the image "
    "is evidence, and you are comparing two things against each other:\n"
    "the actual chart/graph/diagram, and what the candidate wrote about "
    "it. Do this analysis of the image FIRST, before judging Task "
    "Achievement, in this order:\n"
    "1. What TYPE of chart/graph/diagram/table is this (pie, bar, line, "
    "table, process diagram, map)?\n"
    "2. What is the title/subject of the data?\n"
    "3. What are ALL the categories/series shown (e.g. every meal type, "
    "every country, every year, every stage of a process)? List them to "
    "yourself - you will need to check every one was addressed.\n"
    "4. What are the units (%, count, currency, time)?\n"
    "5. What are the actual key values and trends shown - the genuine "
    "highs, lows, comparisons, and changes that a summary should cover?\n"
    "Only after this - compare the candidate's answer against what you "
    "just identified, and judge Task Achievement from that comparison, "
    "not from the candidate's text in isolation.\n"
    "\n"
    "Specifically check for, and treat each as a genuine Task Achievement "
    "issue if present (not a presentation/language issue):\n"
    "- Missing key features: an important trend, comparison, high/low "
    "point genuinely visible in the image that the answer never mentions.\n"
    "- Missing overview: no general summary statement of the main "
    "pattern(s) across the whole image.\n"
    "- Incorrect overview: the stated general pattern doesn't match what "
    "the image actually shows overall.\n"
    "- Incorrect comparisons: a stated comparison between two data points "
    "is wrong (e.g. claims A is higher than B when the image shows the "
    "opposite).\n"
    "- Incorrect figures: a specific number/percentage stated doesn't "
    "match the image.\n"
    "- Invented information: a figure, trend, or category mentioned that "
    "does not appear in the image at all.\n"
    "- Wrong trends: a described direction of change (increased/"
    "decreased/fluctuated) that doesn't match the image.\n"
    "- Wrong categories: data attributed to the wrong category/series "
    "(e.g. a figure that belongs to \"lunch\" described as \"dinner\").\n"
    "- Missing coverage: an ENTIRE category, series, chart, or stage that "
    "the image shows is never addressed anywhere in the answer at all - "
    "this is different from omitting a minor detail; check specifically "
    "whether every distinct chart/category/series you identified in step "
    "3 above is addressed at least once, and treat a fully-skipped one as "
    "a genuine Task Achievement gap, not a stylistic choice.\n"
    "- Inappropriate selection: mechanically listing every single data "
    "point with no selection of the genuinely key features, or the "
    "reverse - selecting only trivial details while ignoring the main "
    "pattern.\n"
    "A described trend or figure that contradicts the real image is a "
    "genuine Task Achievement problem (inaccurate or misread data), not "
    "merely a presentation issue - factor this into the Task Achievement "
    "checklist above (e.g. Band 7 requires main trends or differences to "
    "be CORRECTLY identified, not just present; Band 6 requires "
    "information to be appropriately selected and supported using "
    "figures/data that are actually accurate).\n"
    "Do NOT invent a data-accuracy problem when the candidate's figures "
    "genuinely do match the image - only report what you can actually "
    "verify against what is shown.\n"
    "\n"
    "Set \"image_data_accuracy\" to exactly one of (REQUIRED - you MUST "
    "pick one of these three, never \"not_applicable\", since the image "
    "is genuinely attached):\n"
    "- \"accurate\": you compared the candidate's described data against "
    "the image and it genuinely matches.\n"
    "- \"partially_inaccurate\": some but not all of the described "
    "figures/trends are wrong or misread compared to what the image "
    "actually shows.\n"
    "- \"significantly_inaccurate\": the candidate substantially misread "
    "the data compared to the image (e.g. described a trend backwards, "
    "invented figures not in the image, or confused categories)."
)

# Static, non-GPT explanatory text for the "severity" tag attached to each
# mistake - mistakes are shown so the candidate can polish further, but the
# tag itself is the only signal for whether an issue actually affects the
# band. Same wording pattern as evaluators/speaking_audio.py's
# SEVERITY_LEGEND, adapted for a reader (written text) instead of a
# listener (spoken answer).
SEVERITY_LEGEND = {
    "minor": (
        "Noticeable but meaning stays fully clear - a reader would not need to stop, "
        "re-read, or guess. The official IELTS descriptors allow this kind of imperfection "
        "even at the top bands."
    ),
    "significant": (
        "A reader would have to stop, re-read, or guess the meaning, or the same minor "
        "issue type recurs often enough across the answer to become systematic. These are "
        "the issues that genuinely affect your band score."
    ),
}

# Phrases that mean GPT itself concluded, in its own "explanation" text,
# that a flagged "mistake" is not actually an error - a real, confirmed
# case had an explanation literally saying "...however, the original is
# acceptable. This is borderline but not an error, so no correction
# needed," while still emitting a "corrected" field that introduced a NEW
# grammatical error of its own. The prompt now also tells GPT to delete
# such objects itself, but that's a soft instruction - this is the
# deterministic backstop, broadened past the old bare "no error" check
# (which missed exactly this phrasing) to the other common ways GPT
# self-admits a flagged item isn't real.
_NO_GENUINE_ERROR_PHRASES = (
    "no error",
    "not an error",
    "not a genuine error",
    "no correction needed",
    "no correction is needed",
    "does not need correction",
    "doesn't need correction",
    "not incorrect",
    "original is acceptable",
    "borderline but not",
    "not necessary to correct",
    "no need to correct",
    "acceptable as is",
    "acceptable and no correction",
    "this is correct",
    "is actually correct",
    "is grammatically correct",
    "already correct",
    # Item 3 additions - live output has flagged mistakes whose own
    # explanation concedes there's no real error, using phrasing the list
    # above was too narrow to catch: "...this is a minor stylistic point
    # and does not affect meaning", "'I would like to' is less direct" (a
    # preference, not an error), "The phrase 'with different cultures' is
    # correct, but...". Matched case-insensitively on "explanation", same
    # as every phrase above.
    "does not affect meaning",
    "is acceptable",
    "is also acceptable",
    "is also possible",
    "is also correct",
    "can be written",
    "minor stylistic point",
    "stylistic preference",
    "is less direct",
    "could be improved",
    "would be clearer",
    "is not incorrect",
    "is correct, but",
)


# Real, well-known IELTS paragraph structure - applied to the Band 9 model
# answer whether it's built by refining the candidate's own (on-topic)
# essay or written fresh from the question (off-topic case, see below).
PARAGRAPH_STRUCTURE_INSTRUCTIONS = {
    "task_1": (
        "Structure the answer in exactly 4 paragraphs: "
        "(1) Introduction - paraphrase the question/task in your own words. "
        "(2) Overview - 2-3 main trends or features, with no specific figures yet. "
        "(3) Body paragraph 1 - key details and comparisons for one part of the data. "
        "(4) Body paragraph 2 - the remaining key details and comparisons. "
        "Separate each paragraph with one blank line (a real line break, not just "
        "a topic-sentence phrase) - this is REQUIRED, not optional; a response "
        "returned as a single block of text is not acceptable regardless of how "
        "good the content is. "
        "CRITICAL - do not omit any chart/data category given in the question: if "
        "the task describes multiple charts/graphs/tables, every single one of "
        "them must be reported on with its own real figures somewhere in the "
        "answer - never drop an entire category (e.g. writing about two charts "
        "out of three given) even to save space or improve flow."
    ),
    "task_2": (
        "Structure the answer in 4 or 5 paragraphs: an introduction, 2 or 3 body "
        "paragraphs (one per main idea/argument), and a conclusion. Use 3 body "
        "paragraphs only if the argument genuinely needs a third distinct point - "
        "otherwise 2 is normal and expected. Separate each paragraph with one "
        "blank line (a real line break) - this is REQUIRED, not optional; a "
        "response returned as a single block of text is not acceptable regardless "
        "of how good the content is."
    ),
}


# Structural markers of a General Training Task 1 letter that an Academic
# Task 1 chart/graph/diagram report never has - checked in Python
# (deterministic) rather than left to GPT to infer inline mid-prompt, the
# same "don't trust a judgment call to GPT when code can just compute it"
# reasoning used everywhere else in this codebase. This replaces the old
# "FORMAT AWARENESS" prompt instruction that asked GPT to notice the
# format itself and apply the matching inline checklist alternative.
_LETTER_SALUTATION_PATTERN = re.compile(r'^\s*Dear\b', re.IGNORECASE)
_LETTER_SIGNOFF_PATTERN = re.compile(
    r'\b(Yours\s+(sincerely|faithfully|truly)|Best\s+regards|Kind\s+regards|Best\s+wishes)\b',
    re.IGNORECASE,
)
_LETTER_QUESTION_PATTERN = re.compile(r'\bletter\b', re.IGNORECASE)


def _detect_task1_variant(question: str, essay: str) -> str:
    """Deterministic Academic vs General Training detection for Writing
    Task 1 - a letter (General Training) has unmistakable structural
    markers (a "Dear ..." opening salutation, a sign-off like "Yours
    sincerely", or the question itself asking for a letter) that a
    chart/graph/diagram report (Academic) never has. Returns "academic"
    or "general" - used to pick which Task Achievement checklist file to
    load, since that is the ONLY criterion that genuinely differs between
    the two variants (Coherence & Cohesion, Lexical Resource, and
    Grammatical Range & Accuracy stay identical/common for both)."""
    essay_stripped = (essay or "").strip()
    if _LETTER_SALUTATION_PATTERN.match(essay_stripped):
        return "general"
    if _LETTER_SIGNOFF_PATTERN.search(essay_stripped):
        return "general"
    if _LETTER_QUESTION_PATTERN.search(question or ""):
        return "general"
    return "academic"


def clamp(score):
    try:
        return max(0.0, min(9.0, float(score)))
    except Exception:
        return 5.0


def _highest_fully_met_band(band_flags) -> float:
    """Deterministically pick the highest band (9 down to 1) where GPT
    marked EVERY feature of that band's checklist as true, for the Task 1/
    Task 2 conjunctive descriptor checklists - the official descriptors'
    own rule that a candidate must fully fit the positive features of a
    band, applied in code rather than trusted to GPT's own self-reported
    float. Same pattern as evaluators/speaking_audio.py's
    _highest_fully_met_band().

    Convention (declared explicitly to GPT in each prompt's RESPONSE
    FORMAT section - see the "BAND FLAG CONVENTION" instruction):
    band_flags is a CASCADING threshold, not an exclusive single pick.
    band_flags["N"] means "this response's quality is at least band N",
    so a response whose true ceiling is band 7 should have "7" down to
    "1" all true and "8"/"9" false. The highest true band is always the
    correct answer regardless of what the lower flags say, so this
    function's return value doesn't depend on the convention being
    followed - but a gap below the ceiling (some lower band marked
    false) means GPT didn't actually follow the declared convention, so
    that's logged for visibility rather than silently accepted.
    """
    if not isinstance(band_flags, dict):
        return 1.0

    flags = [(band, band_flags.get(str(band)) is True) for band in (9, 8, 7, 6, 5, 4, 3, 2, 1)]
    highest = next((band for band, is_true in flags if is_true), None)
    if highest is None:
        return 1.0

    gap = next((band for band, is_true in flags if band < highest and not is_true), None)
    if gap is not None:
        logging.warning(
            f"[BAND CASCADE GAP] ceiling band={highest} but lower band={gap} "
            f"is false ({band_flags}) - GPT did not follow the declared "
            f"cascading band_flags convention; returning the ceiling band "
            f"regardless, since that's still the correct read either way"
        )

    return float(highest)


def _coherence_paragraph_cap(essay: str, word_count: int) -> float | None:
    """docs/ielts-writing-error-taxonomy.md section 3.1 (band-limiting):
    "No paragraph breaks - single block of text" caps Coherence and
    Cohesion at Band 5, regardless of what GPT's own checklist grid
    reports - this is a structural fact about the text, not a judgment
    call, so it's checked directly rather than trusted to GPT's
    self-report (the same "don't trust prompt compliance alone"
    reasoning behind every other deterministic cap in this function).
    This is a CEILING only - if GPT's own grid already lands at or below
    5, this has no effect; it only pulls an over-generous score down.
    Only applies once the essay is long enough that real paragraphing
    would actually be expected - a short response has nothing to
    paragraph, so there's nothing to penalise."""
    if word_count < 100:
        return None
    blocks = [b for b in re.split(r'\n\s*\n', essay or "") if b.strip()]
    return 5.0 if len(blocks) <= 1 else None


def _grammar_punctuation_cap(essay: str, word_count: int) -> float | None:
    """docs/ielts-writing-error-taxonomy.md section 5.1 (band-limiting):
    "No sentence boundaries / punctuation absent" caps Grammatical Range
    and Accuracy at Band 4-5 - checked directly from terminal-punctuation
    density rather than trusted to GPT's self-report, for the same
    reason as _coherence_paragraph_cap() above. A CEILING only. Genuinely
    zero terminal punctuation across a real answer is the more severe
    case (cap 4, matching Band 4 GRA's "punctuation is often faulty or
    inadequate" alongside its very limited range); punctuation present
    but far too sparse to mark real sentence boundaries is the milder
    case (cap 5)."""
    if word_count < 30:
        return None
    terminal_marks = len(re.findall(r'[.!?]', essay or ""))
    if terminal_marks == 0:
        return 4.0
    if word_count / terminal_marks > 40:
        return 5.0
    return None


def count_words(text: str) -> int:
    return len(text.split())


def _strip_wrapping_quotes(text: str) -> str:
    value = str(text or "").strip()
    prev = None
    while value != prev and len(value) >= 2:
        prev = value
        value = re.sub(r'^[\"\'\u201c\u201d\u2018\u2019]+|[\"\'\u201c\u201d\u2018\u2019]+$', "", value).strip()
    return value


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r'(?<=[.!?])\s+|\n+', str(text)) if part and part.strip()]


def _normalize_mistake_severity_and_category(mistakes: list) -> list:
    """Deterministic backstop for each mistake's self-reported "severity" -
    don't trust GPT's tag on faith, default to "significant" (the safer
    direction) if missing or invalid. Same pattern already used for
    Speaking's per-criterion severity fields in generate_mistakes().

    "category" is only trimmed, NOT lowercased - the taxonomy uses proper-
    case names (e.g. "Article Errors"), and lowercasing here used to
    silently overwrite that in the field actually returned to the
    caller. Escalation grouping below still matches case-insensitively
    via its own internal key, so a stray case difference between two
    occurrences of the same category still groups correctly."""
    normalized = []
    for m in mistakes:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        severity = m.get("severity")
        m["severity"] = severity if severity in ("minor", "significant") else "significant"
        m["category"] = str(m.get("category", "") or "").strip()
        normalized.append(m)
    return normalized


_TRUE_ARTICLES = {"a", "an", "the"}
_COMMON_PREPOSITIONS = {
    "of", "in", "on", "to", "by", "with", "for", "at", "from", "about",
    "into", "onto", "than", "as", "over", "under", "between", "among",
}


def _fix_article_preposition_mislabel(mistakes: list) -> list:
    """A real, recurring miscategorisation a QA review caught: a missing
    "of" (e.g. "29% [of] sodium") gets labeled "Article Errors" even
    though a/an/the were never involved - "of" is a preposition. The
    prompt's own CATEGORY REFERENCE now explicitly distinguishes these
    (see the "ARTICLE vs PREPOSITION" instruction), but that alone
    didn't reliably stop it in practice - the same "prompt-only
    instructions have repeatedly proven unreliable" pattern behind every
    other deterministic backstop in this file. Diffs "original" against
    "corrected" to find the actual inserted word(s); reclassifies to
    "Preposition Errors" only when none of the added words are genuinely
    a/an/the and at least one is a common preposition, so a real article
    fix (e.g. "a" inserted) is left alone."""
    for m in mistakes:
        if m.get("category") != "Article Errors":
            continue
        original_words = set(re.findall(r"[a-zA-Z']+", str(m.get("original") or "").lower()))
        corrected_words = set(re.findall(r"[a-zA-Z']+", str(m.get("corrected") or "").lower()))
        added_words = corrected_words - original_words
        if added_words and added_words.isdisjoint(_TRUE_ARTICLES) and (added_words & _COMMON_PREPOSITIONS):
            m["category"] = "Preposition Errors"
    return mistakes


def _escalate_frequent_minor_mistakes(mistakes: list) -> list:
    """A single 'minor' issue type is invisible to the band score; the SAME
    minor issue type recurring 4+ times across one answer is not - it's a
    systematic gap, and the official descriptors explicitly distinguish
    systematic from non-systematic errors at the higher bands. Escalates
    every occurrence in a (type, category) group to "significant" once
    that group reaches the threshold. Groups case-insensitively (via
    .lower() on the key only) so a stray case difference in GPT's
    category naming doesn't silently split one real pattern into two
    separate groups - the original casing in each mistake's own
    "category" field is left untouched."""
    counts = {}
    for m in mistakes:
        if m.get("severity") == "minor" and m.get("category"):
            key = (m.get("type"), m["category"].lower())
            counts[key] = counts.get(key, 0) + 1

    for m in mistakes:
        if m.get("severity") != "minor" or not m.get("category"):
            continue
        key = (m.get("type"), m["category"].lower())
        if counts.get(key, 0) >= 4:
            m["severity"] = "significant"
            m["escalated_to"] = "systematic"
            m["occurrence_count"] = counts[key]
    return mistakes


def _escalate_error_clusters(mistakes: list, essay: str) -> list:
    """3 or more errors inside the SAME sentence compound each other's
    impact on the reader even if each is individually minor - escalate
    every mistake anchored to a sentence once that sentence's count
    reaches the threshold, regardless of each mistake's own type."""
    sentences = _split_sentences(essay)
    if not sentences:
        return mistakes

    sentence_indices = []
    sentence_counts = {}
    for m in mistakes:
        original = str(m.get("original") or "").strip().lower()
        idx = None
        if original:
            for i, sentence in enumerate(sentences):
                s = sentence.lower()
                if original in s or s in original:
                    idx = i
                    break
        sentence_indices.append(idx)
        if idx is not None:
            sentence_counts[idx] = sentence_counts.get(idx, 0) + 1

    for m, idx in zip(mistakes, sentence_indices):
        if idx is not None and sentence_counts.get(idx, 0) >= 3:
            m["severity"] = "significant"
            m.setdefault("escalated_to", "cluster")
    return mistakes


def _dedupe_mistakes_by_normalized_span(mistakes: list) -> list:
    """Item 5 of the six deterministic fixes. Two mistake objects anchored
    to the EXACT SAME normalized "original" span are the same underlying
    issue reported twice - occasionally once under one type/category and
    once under another. Keeps exactly one per span: whichever copy has the
    higher (post-escalation) severity, or the first-seen copy if severity
    ties. Runs LAST in the mistake pipeline, after severity normalization
    and both escalation passes above, so "higher severity" reflects the
    final, fully-escalated value - not a stale pre-escalation tag. Only
    ever removes a mistake object; never invents or edits the surviving
    one. Same span-normalization convention already used elsewhere in this
    file (whitespace-collapsed, quote-stripped, lowercased - see
    _mistake_correction_relates_to_original's original_norm). A mistake
    with no usable "original" span is left alone - there's nothing to
    dedupe it against."""
    _SEVERITY_RANK = {"significant": 1, "minor": 0}

    def _span_key(m):
        return re.sub(r"\s+", " ", _strip_wrapping_quotes(str(m.get("original") or ""))).strip().lower()

    best_by_span = {}
    for m in mistakes:
        span = _span_key(m)
        if not span:
            continue
        current_best = best_by_span.get(span)
        if current_best is None or _SEVERITY_RANK.get(m.get("severity"), 0) > _SEVERITY_RANK.get(current_best.get("severity"), 0):
            best_by_span[span] = m

    result = []
    emitted_spans = set()
    for m in mistakes:
        span = _span_key(m)
        if not span:
            result.append(m)
            continue
        if span in emitted_spans:
            continue
        emitted_spans.add(span)
        result.append(best_by_span[span])
    return result


def _truncate_to_sentence_boundary(text: str, max_words: int) -> str:
    """Trim text to at most max_words WITHOUT cutting the final sentence in
    half. The previous approach (" ".join(text.split()[:max_words])) sliced
    on a raw word count, which routinely chopped the model's rewritten
    answer off mid-sentence - no conclusion, trailing off wherever the Nth
    word happened to land - since GPT commonly overshoots an "exactly N
    words" instruction. Keeps whole sentences up to the budget; if even the
    first sentence alone exceeds it, keeps that one sentence anyway rather
    than mutilating it (a slightly-over-budget complete sentence reads far
    better than a truncated fragment).

    Paragraph-aware: the previous version ran _split_sentences() (which
    treats a run of newlines as just another sentence boundary) directly
    on the whole text and rejoined every kept sentence with " ".join() -
    silently flattening the entire refined_answer to one block the
    moment truncation actually ran, which is most of the time, since the
    refine step's own target word count routinely gets overshot. A real
    QA review caught the "Band 9 model answer" coming back unparagraphed
    despite the refine prompt explicitly asking for paragraph breaks -
    this was the actual cause, not the prompt being ignored. Splits into
    paragraphs first and keeps the "\\n\\n" boundaries between whichever
    paragraphs survive the budget."""
    text = (text or "").strip()
    if not text or len(text.split()) <= max_words:
        return text

    paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return text

    kept_paragraphs = []
    word_count = 0
    kept_any_sentence = False
    for para in paragraphs:
        kept_sentences = []
        for sentence in _split_sentences(para):
            sentence_words = len(sentence.split())
            if kept_any_sentence and word_count + sentence_words > max_words:
                break
            kept_sentences.append(sentence)
            word_count += sentence_words
            kept_any_sentence = True
        if kept_sentences:
            kept_paragraphs.append(" ".join(kept_sentences))
        if word_count >= max_words:
            break
    return "\n\n".join(kept_paragraphs)


def _best_matching_sentence(original: str, refined_answer: str) -> str:
    source = _strip_wrapping_quotes(original)
    candidates = _split_sentences(refined_answer)
    if not source or not candidates:
        return ""

    source_tokens = set(re.findall(r"[a-z0-9']+", source.lower()))
    if not source_tokens:
        return candidates[0]

    best = ""
    best_score = 0.0
    for candidate in candidates:
        candidate_tokens = set(re.findall(r"[a-z0-9']+", candidate.lower()))
        if not candidate_tokens:
            continue
        overlap = len(source_tokens & candidate_tokens)
        union = len(source_tokens | candidate_tokens) or 1
        score = overlap / union
        if score > best_score:
            best_score = score
            best = candidate
    return best or candidates[0]


def _mistake_original_is_verbatim(original: str, essay: str, require_unique_multiword: bool = False) -> bool:
    """A real, confirmed hallucination a QA review caught: GPT sometimes
    misquotes what the candidate actually wrote while presenting it as
    "original" - e.g. quoting "a student received grade B..." when the
    candidate's real text already reads "a student received a grade
    B...", then "correcting" it back to what was already there. This
    invents an error that never existed, and is worse than a normal
    false positive since the student can't even find the quoted phrase
    in their own essay. The prompt's own "ORIGINAL MUST BE AN EXACT
    QUOTE" instruction didn't reliably stop it in practice - the same
    "prompt-only instructions have repeatedly proven unreliable" pattern
    behind every other deterministic backstop in this file. Checked
    here instead: "original" (normalized only for whitespace and
    wrapping quotes, never reworded) must be an actual substring of the
    real submitted text, or the mistake is dropped entirely.

    require_unique_multiword=True (opt-in, default False so every other
    call site of this function is byte-identical to before): also
    requires "original" to be at least 3 words AND occur EXACTLY ONCE in
    the essay. Live output has flagged mistakes with "original" set to a
    bare word ("features", "consumed", "includes") that occurs five to
    seven times in the essay - the candidate can't tell which occurrence
    is meant, and the plain substring check above passes trivially since
    any common word is a substring of something. A genuine single-word
    error is still reportable in principle, but the detector has to quote
    enough surrounding text to locate it uniquely - this deliberately
    does not attempt to auto-expand a short span into a longer one
    itself; guessing which occurrence was meant is exactly the invention
    this check exists to remove, so a too-short or ambiguous span is
    dropped, not repaired. Only wired to the actual per-mistake filtering
    call site below - NOT to the vocabulary-suggestion or refined-answer-
    leak call sites elsewhere in this file, which check a different kind
    of span for a different purpose."""
    original_norm = re.sub(r"\s+", " ", _strip_wrapping_quotes(original or "")).strip().lower()
    essay_norm = re.sub(r"\s+", " ", essay or "").lower()
    if not original_norm:
        return False
    if not require_unique_multiword:
        return original_norm in essay_norm
    if len(original_norm.split()) < 3:
        return False
    return essay_norm.count(original_norm) == 1


def _mistake_correction_relates_to_original(original: str, corrected: str, refined_answer: str) -> bool:
    """Task 1-only deterministic backstop for a real, confirmed bug: a
    mistake's "corrected" field sometimes turns out to be a long,
    unrelated, generic sentence rather than an actual fix for the
    flagged "original" span - in the confirmed case, the exact sentence
    also appeared in the independently-generated refined_answer rewrite,
    meaning it was never really a targeted correction of what was
    flagged at all. Short, genuine phrase-level corrections (the normal
    case) are never touched by this check - it only rejects long,
    full-sentence "corrections" that don't meaningfully relate to what
    they claim to fix, matching the same word-overlap technique already
    used by _best_matching_sentence() above."""
    corrected_clean = _strip_wrapping_quotes(corrected).strip()
    original_clean = _strip_wrapping_quotes(original).strip()
    if len(corrected_clean.split()) < 10:
        return True
    if refined_answer and corrected_clean.rstrip(".") in refined_answer:
        return False
    orig_tokens = set(re.findall(r"[a-z0-9']+", original_clean.lower()))
    corr_tokens = set(re.findall(r"[a-z0-9']+", corrected_clean.lower()))
    if not orig_tokens or not corr_tokens:
        return True
    overlap = len(orig_tokens & corr_tokens) / len(orig_tokens)
    if overlap < 0.3 and len(corr_tokens) > len(orig_tokens) * 1.5:
        return False
    return True


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


# ---------------------------------------------------------------------------
# v2 refine pipeline (WRITING_INDEPENDENT_MODEL_ANSWER flag, default OFF) -
# refined_answer must independently meet Band 9 rather than inheriting a
# weak/wrong submission's missing content and structure. See the approved
# "refined_answer overhaul" plan. Every function below is additive - the
# flag-OFF path in evaluate_writing() is untouched byte-for-byte.
# ---------------------------------------------------------------------------
def _extract_chart_data(image_url: str, question: str) -> dict | None:
    """Task 1 Academic with an image only. Extracts structured chart data
    BEFORE the candidate's essay is ever read by this step, so the
    extraction can't be biased by what the candidate claimed - the essay is
    never in this call's context at all (see
    prompts/writing_chart_extraction.txt). Runs sequentially, after the
    existing scoring call - this codebase has no concurrency infrastructure
    yet (the approved architecture plan's Step 1, which would add it, has
    not been implemented). Fails open: returns None once safe_gpt_call's
    own retries are exhausted, so the caller falls through to the
    text-derived category fallback rather than blocking the whole
    submission on one bad extraction call."""
    with open(PROMPTS_DIR / "writing_chart_extraction.txt", "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template.replace("<<<QUESTION>>>", question)

    def _call_extract_validated(p):
        result = call_gpt_extract(p, image_url=image_url)
        if not isinstance(result, dict) or "charts" not in result:
            raise ValueError("Chart extraction response missing 'charts' key")
        return result

    extracted = safe_gpt_call(prompt, fallback=None, caller=_call_extract_validated)
    if not isinstance(extracted, dict) or not extracted.get("charts"):
        return None
    return extracted


def _derive_categories_from_question_text(question: str) -> list:
    """Fallback for Academic Task 1 WITHOUT an image (a real, live case -
    this codebase's own eval-harness corpus of 30 questions has none, and
    submissions can genuinely omit the image too). IELTS Task 1 questions
    sometimes name the categories being compared in words even without a
    chart (e.g. "...comparing four types of nutrients..."). Best-effort,
    deterministic, no GPT call - legitimately returns [] when the question
    doesn't spell categories out in a recognisable pattern, in which case
    coverage validation is simply skipped for that submission (see
    _validate_refine_output) rather than faked."""
    if not question:
        return []
    match = re.search(r"\b(?:of|comparing|showing)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3}"
                       r"(?:\s*,\s*[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})+"
                       r"(?:\s*,?\s+and\s+[A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*){0,3})?)", question)
    if not match:
        return []
    parts = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", match.group(1))
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]


def _derive_required_points_from_gt_question(question: str) -> list:
    """General Training Task 1 letters have no image, but real IELTS GT
    letter questions consistently spell out the letter's required content
    points as a clause list introduced by "In your letter" (confirmed
    against this codebase's own 6 GT questions, all of which follow this
    exact pattern). Cheap, deterministic, no GPT call. Best-effort: returns
    [] if the pattern isn't found (real candidate-submitted question text
    won't always match it), in which case coverage validation is simply
    skipped for that submission rather than faked."""
    if not question:
        return []
    match = re.search(r"in your letter[,:]?\s*(.+)$", question, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    tail = match.group(1).strip().rstrip(".")
    points = [p.strip() for p in re.split(r"\.\s+", tail) if p.strip()]
    if len(points) <= 1:
        points = [p.strip() for p in tail.split(",") if p.strip()]
    return [p for p in points if len(p) > 3]


def _format_mistakes_for_prompt(mistakes: list) -> str:
    if not mistakes:
        return "(none flagged)"
    lines = []
    for m in mistakes[:10]:
        original = (m.get("original") or m.get("sentence") or "").strip()
        if original:
            lines.append(f'- "{original}"')
    return "\n".join(lines) if lines else "(none flagged)"


# Off-topic model-answer generation must not receive the candidate's real
# essay text at all - a structural guarantee (nothing to leak) rather than
# a prompt instruction GPT is trusted to follow, the same principle already
# used for chart extraction (which never receives the essay either). This
# fixed placeholder replaces <<<ESSAY>>> only for relevance_status ==
# "off_topic".
_OFF_TOPIC_ESSAY_PLACEHOLDER = (
    "(No candidate material is provided here - this answer was off-topic. "
    "Write entirely from the question below, not from anything the "
    "candidate wrote.)"
)

_PARTIALLY_OFF_TOPIC_INSTRUCTIONS = (
    "RELEVANCE NOTE: parts of the candidate's answer above drift from the "
    "question, or (for Task 1) describe data the chart doesn't show. Keep "
    "the on-topic, correct parts and use them as normal - do not discard "
    "the whole answer over one bad section. Replace ONLY the "
    "drifting/incorrect parts with fresh content that genuinely answers "
    "the question (and, where data is involved, matches the extracted "
    "chart data above, not whatever the candidate claimed)."
)


def _build_refine_prompt_v2(
    task_type: str,
    task1_variant,
    question: str,
    essay: str,
    mistakes: list,
    extracted_data,
    required_points: list,
    word_min: int,
    word_max: int,
    retry_issues=None,
    relevance_status: str = "on_topic",
) -> str:
    """Loads the right template and fills placeholders. Picks the template
    by task type/variant: the chart-report template
    (writing_refine_task1.txt) for Academic Task 1 - used for BOTH the
    real-extraction case and the question-text-derived fallback, since both
    need the identical chart-report structure, just plugging different
    specificity of data into the same EXTRACTED_DATA slot; a dedicated
    letter template (writing_refine_task1_gt.txt) for General Training,
    since a letter's structure (salutation/closing, no "overview
    pattern"/no-conclusion chart language) is genuinely different, not the
    same template with a different data slot; writing_refine_task2.txt for
    Task 2.

    relevance_status branches what gets interpolated for <<<ESSAY>>> and
    <<<RELEVANCE_INSTRUCTIONS>>> - see the "off-topic answers" plan:
    on_topic (default) is unchanged from before this parameter existed;
    partially_off_topic still shows the real essay (needed to know what to
    keep) plus an added instruction to replace only the drifting parts;
    off_topic never shows the real essay at all."""
    if task_type == "task_1" and task1_variant == "general":
        filename = "writing_refine_task1_gt.txt"
    elif task_type == "task_1":
        filename = "writing_refine_task1.txt"
    else:
        filename = "writing_refine_task2.txt"
    with open(PROMPTS_DIR / filename, "r", encoding="utf-8") as f:
        template = f.read()

    data_text = ""
    if task_type == "task_1" and task1_variant != "general":
        if extracted_data and extracted_data.get("charts"):
            data_text = json.dumps(extracted_data, ensure_ascii=False, indent=2)
        elif required_points:
            data_text = (
                "No chart image was provided. Based on the question text, the "
                "categories being compared appear to include: "
                + ", ".join(required_points)
                + ". Use your own general knowledge of what such a chart "
                "typically shows, and be conservative/generic about specific "
                "figures since none were verified."
            )
        else:
            data_text = (
                "No chart image was provided and no specific categories could "
                "be identified from the question text. Write a plausible, "
                "well-structured report based on the question alone, and be "
                "conservative/generic about specific figures since none were "
                "provided or verified."
            )

    mistakes_text = _format_mistakes_for_prompt(mistakes)

    retry_block = ""
    if retry_issues:
        retry_block = (
            "YOUR PREVIOUS ATTEMPT HAD SPECIFIC PROBLEMS - fix ALL of these in "
            "this attempt:\n- " + "\n- ".join(retry_issues)
        )

    essay_text = _OFF_TOPIC_ESSAY_PLACEHOLDER if relevance_status == "off_topic" else essay
    relevance_instructions = _PARTIALLY_OFF_TOPIC_INSTRUCTIONS if relevance_status == "partially_off_topic" else ""

    filled = (
        template
        .replace("<<<QUESTION>>>", question)
        .replace("<<<ESSAY>>>", essay_text)
        .replace("<<<MISTAKES>>>", mistakes_text)
        .replace("<<<WORD_MIN>>>", str(word_min))
        .replace("<<<WORD_MAX>>>", str(word_max))
        .replace("<<<RETRY_ISSUES_BLOCK>>>", retry_block)
    )
    if "<<<RELEVANCE_INSTRUCTIONS>>>" in filled:
        filled = filled.replace("<<<RELEVANCE_INSTRUCTIONS>>>", relevance_instructions)
    if "<<<EXTRACTED_DATA>>>" in filled:
        filled = filled.replace("<<<EXTRACTED_DATA>>>", data_text)
    if "<<<REQUIRED_CATEGORIES>>>" in filled:
        req_text = "\n".join(f"- {p}" for p in required_points) if required_points else "(none identified)"
        filled = filled.replace("<<<REQUIRED_CATEGORIES>>>", req_text)
    return filled


# Common words excluded from the off-topic content-overlap check below -
# connectors/function words that legitimately appear in any essay
# regardless of topic, so counting them would make every pair of essays
# look suspiciously similar.
_COMMON_WORDS_FOR_OVERLAP_CHECK = frozenset((
    "the", "and", "that", "this", "with", "from", "have", "will", "would",
    "could", "should", "about", "which", "their", "there", "these", "those",
    "been", "being", "into", "than", "then", "when", "where", "what", "some",
    "more", "most", "such", "also", "many", "much", "very", "just", "only",
    "over", "after", "before", "because", "while", "each", "other", "same",
    "does", "doing", "here", "your", "they", "them", "were", "example",
    "people", "however", "therefore", "although", "essay", "question",
    "answer", "candidate", "will", "shall", "must", "might", "study",
    "think", "believe", "opinion", "conclusion", "overall", "summary",
))


def _content_word_overlap_ratio(text_a: str, text_b: str) -> float:
    """Fraction of text_a's distinctive content words (4+ letters, common
    words excluded) that also appear in text_b - used by the off_topic
    validation check below to verify a model answer hasn't leaked the
    candidate's actual essay content (their specific examples/named
    entities/topic-specific vocabulary). The prompt structurally can't
    reproduce essay content for an off_topic response (see
    _OFF_TOPIC_ESSAY_PLACEHOLDER - the essay is never shown to GPT at
    all), but this checks that guarantee held rather than trusting it."""
    def _content_words(text):
        words = re.findall(r"[a-zA-Z']{4,}", (text or "").lower())
        return set(w for w in words if w not in _COMMON_WORDS_FOR_OVERLAP_CHECK)
    words_a = _content_words(text_a)
    if not words_a:
        return 0.0
    words_b = _content_words(text_b)
    return len(words_a & words_b) / len(words_a)


def _validate_refine_output(
    parsed,
    extracted_data,
    required_points: list,
    mistakes: list,
    essay: str,
    task_type: str,
    task1_variant,
    word_min: int,
    word_max: int,
    relevance_status: str = "on_topic",
) -> list:
    """ONE unified list of everything wrong with a generated refine
    response, so a single retry can name every problem at once rather than
    needing several sequential retry passes (word budget, then coverage,
    then paragraph structure, ...). Empty list = passed. Deliberately
    checks category/required-point NAMES only, never numeric values -
    genuine Band 9 prose paraphrases figures ("just under half" instead of
    "48%"), so a strict numeric match would false-fail good writing;
    numeric correctness instead relies on the refine call's own
    self-reported data_contradiction_flag (mirrors image_data_accuracy's
    self-report + Python-enforced-consequence pattern used for scoring -
    Python still independently enforces the consequence, never trusts the
    self-report as gospel)."""
    if not isinstance(parsed, dict):
        return ["refine response was not a JSON object"]
    refined = parsed.get("refined_answer")
    if not isinstance(refined, str) or not refined.strip():
        return ["refined_answer field is missing or empty"]

    issues = []
    refined_norm = refined.lower()

    if task_type == "task_1" and task1_variant != "general" and extracted_data:
        for chart in extracted_data.get("charts", []) or []:
            for cat in chart.get("categories", []) or []:
                name = (cat.get("name") or "").strip()
                if name and name.lower() not in refined_norm:
                    issues.append(f"missing coverage of category '{name}'")
    elif task_type == "task_1" and task1_variant == "general" and required_points:
        for point in required_points:
            words = re.findall(r"[a-zA-Z']{4,}", point)
            if words and not any(w.lower() in refined_norm for w in words):
                issues.append(f"missing coverage of required point '{point}'")

    if parsed.get("data_contradiction_flag") is True:
        issues.append(
            "the model itself flagged a data contradiction in its own draft "
            "(data_contradiction_flag=true)"
        )

    blocks = [b for b in re.split(r'\n\s*\n', refined) if b.strip()]
    if task_type == "task_2" and len(blocks) < 4:
        issues.append(
            f"only {len(blocks)} paragraph(s) - Task 2 needs a minimum of 4 "
            f"(intro, body per view, conclusion)"
        )

    word_count = len(refined.split())
    if word_count >= 80 and len(blocks) <= 1:
        issues.append("no paragraph breaks (\\n\\n) survived in the rewrite despite the explicit instruction")

    for m in mistakes or []:
        original = m.get("original") or m.get("sentence") or ""
        if original and _mistake_original_is_verbatim(original, refined):
            issues.append(f'a flagged mistake still appears verbatim in the rewrite: "{original[:80]}"')

    for vs in parsed.get("vocabulary_suggestions") or []:
        if not isinstance(vs, dict):
            continue
        phrase = vs.get("original_phrase") or ""
        if phrase and not _mistake_original_is_verbatim(phrase, essay):
            issues.append(f'vocabulary suggestion quotes a phrase not actually in the candidate\'s essay: "{phrase[:80]}"')

    if word_count > word_max:
        issues.append(f"word_budget_exceeded: {word_count} words, budget was {word_max}")

    if relevance_status == "off_topic":
        overlap = _content_word_overlap_ratio(essay, refined)
        if overlap > 0.25:
            issues.append(
                f"off_topic path may have leaked candidate content: "
                f"{overlap:.0%} of the candidate's distinctive words also "
                f"appear in the model answer"
            )

    return issues


def _generate_refined_answer_v2(
    task_type: str,
    task1_variant,
    question: str,
    essay: str,
    mistakes: list,
    extracted_data,
    required_points: list,
    word_min: int,
    word_max: int,
    image_url,
    relevance_status: str = "on_topic",
) -> dict:
    """Orchestrator for the v2 refine pipeline. Builds the prompt, calls
    call_gpt_refine, validates the result against _validate_refine_output,
    and - if anything failed - rebuilds the prompt naming exactly what was
    wrong and retries ONCE more. Never truncates: a word-budget overshoot is
    just one more entry in the same issues list that drives the single
    retry; if the retry still overshoots, the longer text is kept rather
    than cut, since losing the word cap is cosmetic but losing a whole
    chart/category/content point is not. Bounded at exactly 2
    call_gpt_refine attempts total, matching "retry once" - this needed new
    control flow rather than reusing safe_gpt_call's own retry loop as-is,
    since that loop sends the identical prompt on every attempt and cannot
    carry "attempt 2: here's specifically what was wrong".

    relevance_status is threaded into both prompt-building (which decides
    whether the real essay is shown at all - see
    _build_refine_prompt_v2/_OFF_TOPIC_ESSAY_PLACEHOLDER) and validation
    (the off-topic content-leak check). Does NOT return a "rewrite_basis"/
    "model_answer_source" field - that's computed deterministically by the
    caller (evaluate_writing()) from relevance_status, which it already
    knows before calling this function, rather than asked of GPT here.

    Known limitation: on total GPT failure (every attempt raises), the
    fallback is the candidate's own essay text, same as every other
    fail-open fallback in this file - for relevance_status="off_topic"
    this is a genuinely wrong fallback (an off-topic essay standing in as
    its own "model answer"), but this is an already-rare edge case (GPT
    failing on both attempts) and the alternative (a bare error string)
    would break every downstream consumer expecting a real essay-shaped
    string in refined_answer."""
    attempts_meta = []
    retry_issues = None
    parsed = None

    for attempt in (1, 2):
        prompt = _build_refine_prompt_v2(
            task_type, task1_variant, question, essay, mistakes,
            extracted_data, required_points, word_min, word_max,
            retry_issues=retry_issues, relevance_status=relevance_status,
        )

        def _call_refine_v2_validated(p, _image_url=image_url):
            result = call_gpt_refine(p, image_url=_image_url)
            if not isinstance(result, dict) or not (result.get("refined_answer") or "").strip():
                raise ValueError("v2 refine response missing non-empty refined_answer")
            return result

        parsed = safe_gpt_call(prompt, fallback=None, caller=_call_refine_v2_validated)
        if parsed is None:
            parsed = {
                "refined_answer": essay,
                "vocabulary_suggestions": [],
            }
            attempts_meta.append({"attempt": attempt, "issues": ["gpt_call_failed_after_retries"]})
            break

        issues = _validate_refine_output(
            parsed, extracted_data, required_points, mistakes, essay,
            task_type, task1_variant, word_min, word_max,
            relevance_status=relevance_status,
        )
        attempts_meta.append({"attempt": attempt, "issues": issues})
        if not issues:
            break
        if attempt == 1:
            retry_issues = issues
            continue
        break

    final_issues = attempts_meta[-1]["issues"] if attempts_meta else []
    hard_final_issues = [i for i in final_issues if not i.startswith("word_budget_exceeded")]

    return {
        "refined_answer": (parsed.get("refined_answer") or essay).strip(),
        "vocabulary_suggestions": parsed.get("vocabulary_suggestions") or [],
        "diagnostics": {
            "validation_passed": not hard_final_issues,
            "issues": final_issues,
            "attempts": len(attempts_meta),
        },
    }


def _map_tied_vocabulary_suggestions(suggestions: list, essay: str, task_type: str) -> list:
    """Verbatim-checks each GPT-suggested vocabulary upgrade against the
    candidate's real essay (reusing _mistake_original_is_verbatim's exact
    technique - same "don't trust prompt compliance alone" backstop as
    everywhere else in this file), drops any suggestion whose claimed
    phrase isn't actually there, and maps survivors into the EXISTING
    result["vocabulary"] item shape (word/usage_hint/task_type/
    task_specific) plus two new additive keys (original_phrase/
    placement_context) - same key, not a second parallel vocabulary field,
    so nothing downstream has two lists to reconcile."""
    mapped = []
    for vs in suggestions or []:
        if not isinstance(vs, dict):
            continue
        original_phrase = (vs.get("original_phrase") or "").strip()
        stronger = (vs.get("stronger_alternative") or "").strip()
        placement = (vs.get("placement_context") or "").strip()
        if not original_phrase or not stronger:
            continue
        if not _mistake_original_is_verbatim(original_phrase, essay):
            continue
        usage_hint = f'instead of "{original_phrase}"'
        if placement:
            usage_hint += f" - {placement}"
        mapped.append({
            "word": stronger,
            "usage_hint": usage_hint,
            "task_type": task_type,
            "task_specific": True,
            "original_phrase": original_phrase,
            "placement_context": placement,
        })
    return mapped[:10]



def _fix_literal_newline_escaping(essay: str) -> tuple:
    """Live payloads have arrived with paragraph breaks written as the
    literal two-character sequence "/n" (forward slash + n) instead of a
    real newline - the text then has zero real newlines, the paragraph
    cap fires, and Coherence drops to 5.0 on an essay that genuinely has
    paragraphs; this has cost a full band on real submissions. Fires only
    when the essay contains at least one literal "/n" AND has ZERO real
    newline characters already - if real newlines are present, a "/n"
    elsewhere is left untouched, since it could be genuine prose (e.g. a
    literal discussion of the "/n" escape sequence) sitting alongside real
    paragraph breaks, not a corrupted paragraph break. Does not change the
    paragraph cap itself - the cap is correct, it was being fed corrupted
    input. Returns (possibly-converted essay, diagnostics dict)."""
    literal_count = essay.count("/n")
    if literal_count > 0 and "\n" not in essay:
        return essay.replace("/n", "\n"), {"fired": True, "literals_converted": literal_count}
    return essay, {"fired": False, "literals_converted": 0}


def evaluate_writing(data: dict):

    # Real LLM token usage for this evaluation, summed from the scoring
    # call's own usage metadata (never estimated) - same pattern already
    # used in evaluators/speaking_audio.py's final_response["usage"].
    # Missing/malformed usage on the call is already handled inside
    # record_token_usage() itself (a no-op), so this sum is always safe.
    usage_log = []

    metadata = data.get("metadata", {})
    question = metadata.get("question", "").strip()
    essay = data.get("user_answers", {}).get("text", "").strip()
    essay, literal_newline_diagnostics = _fix_literal_newline_escaping(essay)

    if not essay:
        raise ValueError("Essay text missing")

    task_type = "task_1" if metadata.get("task_type") in ("task1", "task_1") else "task_2"
    word_count = validate_word_count(task_type, essay)

    # Task 1's Task Achievement is the ONLY criterion that genuinely
    # differs between Academic (chart/graph/diagram report) and General
    # Training (letter) - Coherence & Cohesion, Lexical Resource, and
    # Grammatical Range & Accuracy are identical/common for both, so only
    # that one checklist is split into separate files. task1_variant is
    # None for Task 2, which has no such distinction at all.
    task1_variant = None
    if task_type == "task_1":
        task1_variant = _detect_task1_variant(question, essay)
        with open(PROMPTS_DIR / "writing_task1_common.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        criteria_file = (
            PROMPTS_DIR / "writing_task1_academic_criteria.txt"
            if task1_variant == "academic"
            else PROMPTS_DIR / "writing_task1_general_criteria.txt"
        )
        with open(criteria_file, "r", encoding="utf-8") as f:
            task_achievement_block = f.read()
        prompt_template = prompt_template.replace("<<<TASK_ACHIEVEMENT_CHECKLIST>>>", task_achievement_block)
    else:
        with open(PROMPTS_DIR / "writing_task2_prompt.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()

    # The chart/graph/diagram image only ever applies to Academic Task 1 -
    # General Training's letter and Task 2 have nothing to verify against,
    # and even for Academic Task 1 it's optional (a submission without one
    # falls back to today's text-only evaluation). image_verification_text
    # is a no-op empty string whenever there's no image to attach, so this
    # substitution is safe to run unconditionally below.
    image_url = None
    image_verification_text = ""
    if task_type == "task_1" and task1_variant == "academic":
        image_url = metadata.get("image_url")
        if image_url:
            image_verification_text = IMAGE_VERIFICATION_INSTRUCTIONS

    prompt = (
        prompt_template
        .replace("<<<QUESTION>>>", question)
        .replace("<<<ESSAY_TEXT>>>", essay)
        .replace("<<<WORD_COUNT>>>", str(word_count))
        .replace("<<<TASK_TYPE>>>", task_type)
        .replace("<<<IMAGE_VERIFICATION_INSTRUCTIONS>>>", image_verification_text)
    )

    # A genuine GPT/API failure (not "the model found nothing to flag")
    # must default to a NEUTRAL band, not the worst one - the same real
    # bug already found and fixed once in evaluators/speaking_audio.py's
    # generate_scores(), where a true API failure was silently producing
    # a worst-possible score instead of a neutral default. Without a
    # matching "5" entry here, _highest_fully_met_band() would find no
    # band marked true in this fallback and default to 1.0 for every
    # criterion on total failure.
    _neutral_bands = {str(n): (n == 5) for n in range(1, 10)}
    default_ai = {
        "task_achievement_bands": _neutral_bands,
        "task_response_bands": _neutral_bands,
        "coherence_cohesion_bands": _neutral_bands,
        "lexical_resource_bands": _neutral_bands,
        "grammar_bands": _neutral_bands,
        "mistakes": [],
        "strengths": "",
        "improvement": "",
    }

    # Both the Task 1 prompt (writing_task1_common.txt + the variant's
    # criteria file) and writing_task2_prompt.txt use the conjunctive
    # per-band checklist from the official descriptors -
    # GPT reports which features of each band genuinely hold, and the
    # band itself is picked deterministically here, the same "don't trust
    # a self-reported float" reasoning behind every other checklist-driven
    # score in this codebase. Task 1's first criterion is keyed
    # "task_achievement_bands"; Task 2's is "task_response_bands" -
    # everything else is the same key names in both prompts.
    first_criterion_key = "task_achievement_bands" if task_type == "task_1" else "task_response_bands"
    _required_ai_keys = (first_criterion_key, "coherence_cohesion_bands", "lexical_resource_bands", "grammar_bands")

    def _call_gpt_writing_validated(p):
        # A real, confirmed bug: GPT occasionally returns syntactically
        # valid JSON that's missing almost all required fields (observed
        # live: a 2-key response instead of the normal ~8). This used to
        # pass straight through as a "successful" call - safe_gpt_call
        # only checked that the response wasn't empty, not that it
        # actually contained what was asked for - silently producing
        # Band 1 on every criterion (since _highest_fully_met_band()
        # safely defaults missing/non-dict input to 1.0) with no error
        # surfaced anywhere. Raising here makes safe_gpt_call's own
        # retry/fallback treat an incomplete response exactly like a
        # failed one, instead of quietly accepting it as valid.
        result = call_gpt_writing(p, image_url=image_url, usage_log=usage_log)
        missing = [k for k in _required_ai_keys if k not in result]
        if missing:
            raise ValueError(f"GPT response missing required keys: {missing}")
        return result

    # Captures WHY, if safe_gpt_call falls back - see on_failure below.
    # A plain list (not a scalar) so the nested callback can append to it
    # without needing `nonlocal`. Distinguishes an observed live failure
    # mode (gpt-4o outright refusing on an ordinary essay - "I'm sorry, I
    # can't assist with that request.") from every other failure (invalid
    # JSON, missing keys, network error) - both already correctly become
    # ai_evaluation_failed=True either way; this only adds which kind, so
    # a refusal isn't silently lumped in with a generic parse error in
    # the eval log.
    _last_failure = []

    ai = safe_gpt_call(
        prompt, fallback=default_ai, caller=_call_gpt_writing_validated,
        on_failure=lambda e: _last_failure.append(e),
    ) or default_ai
    # safe_gpt_call() returns the SAME `default_ai` object (by identity)
    # only when the real OpenAI call failed on every retry (e.g. a
    # candidate's image_url the API couldn't actually fetch -
    # invalid_image_url - or, now, an incomplete response as above) - a
    # genuine failure was previously indistinguishable from a real, weak
    # evaluation: the neutral fallback (band 5 on every criterion, empty
    # mistakes, empty strengths/improvement) looks exactly like a
    # plausible generic result once feedback/improvement fall back to
    # their own hardcoded generic strings below. Surfaced explicitly so
    # a failed evaluation is never silently mistaken for a real one.
    ai_evaluation_failed = ai is default_ai
    ai_evaluation_failed_reason = None
    if ai_evaluation_failed and _last_failure:
        ai_evaluation_failed_reason = "refusal" if _looks_like_refusal(str(_last_failure[-1])) else "other"

    tr = _highest_fully_met_band(ai.get(first_criterion_key))
    cc = _highest_fully_met_band(ai.get("coherence_cohesion_bands"))
    lr = _highest_fully_met_band(ai.get("lexical_resource_bands"))
    gr = _highest_fully_met_band(ai.get("grammar_bands"))
    # The descriptors' own Band 1 rule ("Responses of 20 words or fewer
    # are rated at Band 1") is a mechanical, unambiguous word count check,
    # stated identically for both Task 1 and Task 2 - enforce it directly
    # rather than relying on GPT to apply it consistently.
    if word_count <= 20:
        tr = cc = lr = gr = 1.0

    # Structural, taxonomy-derived band caps (docs/ielts-writing-error-
    # taxonomy.md sections 3.1/5.1) - computed directly from the text,
    # not from GPT's self-report, and applied as CEILINGS only (they can
    # only lower cc/gr, never raise them). Deliberately narrow: neither
    # cap touches Lexical Resource - a wall-of-text essay's vocabulary
    # can still be genuinely wide and precise regardless of its
    # punctuation/paragraphing, and LR must be judged on the words alone
    # (see the prompt's own CRITERION BOUNDARIES instruction).
    coherence_cap = _coherence_paragraph_cap(essay, word_count)
    if coherence_cap is not None:
        cc = min(cc, coherence_cap)

    grammar_punct_cap = _grammar_punctuation_cap(essay, word_count)
    if grammar_punct_cap is not None:
        gr = min(gr, grammar_punct_cap)

    # Topic relevance - GPT self-reports it in the same checklist call
    # (no extra GPT call needed), Python validates and enforces the cap,
    # the same pattern already proven in
    # evaluators/speaking_audio.py's generate_scores(). All four criteria
    # get capped (unlike Speaking, which excludes pronunciation - Writing
    # has no acoustic-only criterion; every one of these four is judged
    # from the text itself, which is off-topic).
    topic_relevance = str(ai.get("topic_relevance", "on_topic")).strip().lower()
    if topic_relevance not in ("on_topic", "partially_off_topic", "completely_off_topic"):
        topic_relevance = "on_topic"
    relevance_cap = {"completely_off_topic": 5.0, "partially_off_topic": 6.0}.get(topic_relevance)
    if relevance_cap is not None:
        tr = min(tr, relevance_cap)
        cc = min(cc, relevance_cap)
        lr = min(lr, relevance_cap)
        gr = min(gr, relevance_cap)

    # Image data accuracy (Academic Task 1 only, and only when an image
    # was actually attached - see image_url above) - GPT self-reports
    # whether the candidate's stated figures/trends genuinely match the
    # real chart in the SAME checklist call, Python enforces the
    # consequence deterministically rather than trusting GPT's own
    # checklist judgment alone (same pattern as topic_relevance above).
    # Caps ONLY Task Achievement (tr) - Coherence & Cohesion, Lexical
    # Resource, and Grammatical Range are about writing quality, not
    # whether the specific numbers are correct, so they're unaffected by
    # a data-accuracy problem. Mirrors the official descriptors' own
    # rule that misread/backwards data limits Task Achievement.
    image_data_accuracy = "not_applicable"
    # True only if an image was genuinely sent but GPT still reported
    # "not_applicable" (or an invalid value) - the prompt explicitly
    # forbids that response whenever an image is attached, so this means
    # GPT skipped the actual verification task despite being shown the
    # image. Surfaced rather than silently accepted, since prompt-only
    # instructions have repeatedly proven unreliable elsewhere in this
    # codebase and this is worth being able to detect/monitor.
    image_verification_incomplete = False
    if image_url:
        image_data_accuracy = str(ai.get("image_data_accuracy", "not_applicable")).strip().lower()
        if image_data_accuracy not in ("accurate", "partially_inaccurate", "significantly_inaccurate", "not_applicable"):
            image_data_accuracy = "not_applicable"
        if image_data_accuracy == "not_applicable":
            image_verification_incomplete = True
        image_accuracy_cap = {"significantly_inaccurate": 5.0, "partially_inaccurate": 6.0}.get(image_data_accuracy)
        if image_accuracy_cap is not None:
            tr = min(tr, image_accuracy_cap)

    # Official IELTS Writing scoring is a SIMPLE EQUAL AVERAGE of the 4
    # criteria (25% each) for both Task 1 and Task 2 - there is no
    # published weighting that favors Task Achievement/Response over the
    # other three. A weighted average used to be applied here instead
    # (0.3/0.25/0.25/0.2 for Task 1, 0.4/0.3/0.2/0.1 for Task 2), which
    # silently over-weighted TA/TR and under-weighted Grammar relative to
    # real Cambridge/IDP practice.
    overall = (tr + cc + lr + gr) / 4

    # Apply fair band scoring rule: don't reduce below 5 if task is addressed
    overall = apply_fair_band_scoring(overall, tr, task_type)
    band = round_band(overall)

    # NOTE: there used to be an additional hard overall-band cap here for
    # "moderately" underlength answers (e.g. Task 1 < 100 words -> capped
    # at 5.5, Task 2 < 150 -> capped at 5.5) on top of the word_count <= 20
    # rule above. That extra cap is NOT supported by the official IDP band
    # descriptors - the descriptors only state (1) 20 words or fewer is
    # Band 1 (handled above) and (2) Lexical Resource Band 3 may apply "due
    # to the response being significantly underlength" - a criterion-level
    # judgment already made by GPT's own Lexical Resource checklist
    # (_highest_fully_met_band() above), not a separate fixed-threshold
    # overall-band penalty. Removed rather than kept as an undocumented
    # extra deduction with no descriptor basis.

    # relevance_status: the 3-way on_topic/partially_off_topic/off_topic
    # classification the v2 refine pipeline uses to decide how the model
    # answer gets built (see _build_refine_prompt_v2 below). Deliberately
    # NOT a new GPT call or a new self-report - derived entirely from the
    # two signals already computed above for scoring
    # (topic_relevance/image_data_accuracy), which GPT already had to form
    # a judgment on to produce in the first place. relevance_reasons
    # records WHICH signal(s) triggered a non-on_topic status, since Task 1
    # can be off-topic in subject ("topic_drift") or on-topic in subject
    # but describing data the chart doesn't show ("wrong_data") - these
    # need different notice wording and different model-answer-building
    # instructions, not the same generic "off-topic" handling.
    relevance_reasons = []
    if topic_relevance == "completely_off_topic":
        relevance_reasons.append("topic_drift")
    elif topic_relevance == "partially_off_topic":
        relevance_reasons.append("topic_drift")
    if image_data_accuracy in ("partially_inaccurate", "significantly_inaccurate"):
        relevance_reasons.append("wrong_data")
    if topic_relevance == "completely_off_topic":
        relevance_status = "off_topic"
    elif relevance_reasons:
        relevance_status = "partially_off_topic"
    else:
        relevance_status = "on_topic"

    # Controlled refinement with fallback and length guard
    if task_type == "task_1":
        target_min_words = 150
        target_max_words = 170
    else:
        target_min_words = 250
        target_max_words = 260

    use_v2_refine = os.getenv("WRITING_INDEPENDENT_MODEL_ANSWER", "false").strip().lower() == "true"
    refine_result_v2 = None

    if use_v2_refine:
        # v2 pipeline (see the "refined_answer overhaul" and "off-topic
        # answers" plans): image-first chart extraction for Academic Task 1
        # with an image, question-text/GT-bullet fallbacks otherwise, an
        # independently-grounded rewrite with Python-side coverage
        # validation and a single named retry - never a post-hoc
        # truncate-from-the-end. Uses the RAW mistakes list
        # (ai.get("mistakes", [])), not the filtered `mistakes` variable,
        # since filtering happens further below and mistake generation must
        # stay strictly before refine generation in call order either way.
        extracted_chart_data = None
        required_points_v2 = []
        raw_mistakes_for_refine = ai.get("mistakes", [])
        if not isinstance(raw_mistakes_for_refine, list):
            raw_mistakes_for_refine = []

        if task_type == "task_1" and task1_variant == "academic" and image_url:
            extracted_chart_data = _extract_chart_data(image_url, question)
        if task_type == "task_1" and task1_variant == "academic" and not extracted_chart_data:
            required_points_v2 = _derive_categories_from_question_text(question)
        elif task_type == "task_1" and task1_variant == "general":
            required_points_v2 = _derive_required_points_from_gt_question(question)

        # Deterministic, not GPT-self-reported: relevance_status is already
        # known (computed above from signals GPT already produced), so
        # Python states plainly where the model answer came from rather
        # than asking GPT to report something Python already knows for
        # certain. Retires the old Task-2-only self-reported "rewrite_basis"
        # field this replaces.
        if relevance_status == "off_topic":
            if task_type == "task_1" and task1_variant == "academic" and extracted_chart_data:
                model_answer_source = "written from the question and chart"
            else:
                model_answer_source = "written from the question"
        else:
            model_answer_source = "built from your answer"

        refine_result_v2 = _generate_refined_answer_v2(
            task_type, task1_variant, question, essay, raw_mistakes_for_refine,
            extracted_chart_data, required_points_v2,
            target_min_words, target_max_words,
            image_url=(image_url if extracted_chart_data else None),
            relevance_status=relevance_status,
        )
        refine_result_v2["model_answer_source"] = model_answer_source
        refined = refine_result_v2["refined_answer"]
    else:
        paragraph_instructions = PARAGRAPH_STRUCTURE_INSTRUCTIONS[task_type]

        if topic_relevance == "on_topic":
            refine_prompt = (
                f"Rewrite this IELTS Task {'1' if task_type == 'task_1' else '2'} answer to Band 9 level. "
                f"EXPAND it to exactly {target_max_words} words (between {target_min_words}-{target_max_words} words). "
                f"Add more examples, detailed explanations, and sophisticated vocabulary. {paragraph_instructions}\n"
                f"Original answer:\n{essay}"
            )
        else:
            # The candidate's own essay doesn't (or only partly) answer the
            # actual question - refining/expanding it would just produce a
            # more fluent version of the WRONG answer. Write a fresh Band 9
            # answer directly from the question instead, ignoring the
            # candidate's off-topic essay entirely.
            refine_prompt = (
                f"Write a fresh, original IELTS Task {'1' if task_type == 'task_1' else '2'} answer at Band 9 level "
                f"that directly and fully addresses the question below. Do not reference, reuse, or follow the topic "
                f"of any other essay. Write exactly {target_max_words} words (between {target_min_words}-"
                f"{target_max_words} words). Use sophisticated vocabulary and precise, natural examples. "
                f"{paragraph_instructions}\n"
                f"Question:\n{question}"
            )

        def _call_refine_validated(p):
            # The paragraph-structure instruction above has repeatedly been
            # ignored in practice (a real-answer QA review found the "Band 9
            # model answer" returned as a single unparagraphed block despite
            # this exact instruction being present) - the same "don't trust
            # prompt compliance alone" reasoning as
            # _call_gpt_writing_validated() above: raising here makes
            # safe_gpt_call's retry treat a genuinely unparagraphed response
            # as a failed one and try again, instead of silently accepting a
            # model answer that would itself trip _coherence_paragraph_cap()
            # if it were ever submitted back through this same evaluator.
            result = call_gpt_text(p, system_msg="You are an IELTS Writing tutor.")
            text = result if isinstance(result, str) else ""
            text_word_count = len(text.split())
            blocks = [b for b in re.split(r'\n\s*\n', text) if b.strip()]
            if text_word_count >= 100 and len(blocks) <= 1:
                raise ValueError("Refined answer missing paragraph breaks despite explicit instruction")
            return result

        refined = safe_gpt_call(
            refine_prompt,
            fallback=essay,
            caller=_call_refine_validated,
        )
        refined = safe_output(refined, essay)
        max_refined_words = 170 if task_type == "task_1" else 260
        if isinstance(refined, str):
            refined = _truncate_to_sentence_boundary(refined, max_refined_words)

    # Apply coherence penalty cap: max 2 repetition-related errors
    raw_mistakes = ai.get("mistakes", [])
    if isinstance(raw_mistakes, list):
        filtered_mistakes = []
        for m in raw_mistakes:
            exp = (m.get("explanation", "") or "").lower()
            if any(phrase in exp for phrase in _NO_GENUINE_ERROR_PHRASES):
                continue
            # Both task types: drop a mistake whose "original" doesn't
            # actually appear in the candidate's real submitted text, OR
            # whose "original" is too short/ambiguous to be useful (under
            # 3 words, or occurring more than once) - see
            # _mistake_original_is_verbatim() for both confirmed bugs this
            # catches.
            if not _mistake_original_is_verbatim(
                m.get("original") or m.get("sentence") or "",
                essay,
                require_unique_multiword=True,
            ):
                continue
            # Item 4: was Task 1 only (the confirmed bug was first found
            # there). Live Task 2 output has the identical bug - a
            # "Paragraphing Errors" mistake whose "original" is the ENTIRE
            # essay and whose "corrected" is the entire rewritten essay,
            # i.e. refined_answer leaking into the mistakes array, which
            # is exactly what this check exists to prevent. Now applies to
            # both task types; the function's own logic/thresholds are
            # unchanged, only where it's called. This does not affect any
            # band - mistakes never feed into scoring (see tr/cc/lr/gr
            # computation above, which finishes before this block ever
            # runs).
            if not _mistake_correction_relates_to_original(
                m.get("original") or m.get("sentence") or "",
                m.get("corrected") or m.get("correction") or "",
                refined,
            ):
                continue
            filtered_mistakes.append(m)
        raw_mistakes = filtered_mistakes
    mistakes = apply_coherence_penalty_cap(raw_mistakes if isinstance(raw_mistakes, list) else [])
    # Ensure every mistake has a clean, non-empty corrected sentence.
    for m in mistakes:
        try:
            original = _strip_wrapping_quotes(m.get("original") or m.get("sentence") or "")
            corrected = _strip_wrapping_quotes(m.get("corrected") or m.get("correction") or "")

            if not corrected and original:
                corrected = _best_matching_sentence(original, refined)
            if not corrected:
                corrected = original

            if original:
                m["original"] = original
                m["sentence"] = original
            if corrected:
                m["corrected"] = corrected
                m["correction"] = corrected
        except Exception:
            # Defensive: do not let one bad mistake object break evaluation
            continue

    # Severity classification + escalation, per the reader-impact error
    # taxonomy: validate/default each mistake's self-reported severity,
    # then escalate a minor issue type recurring 4+ times (systematic) or
    # 3+ errors landing in the same sentence (cluster) to "significant" -
    # both are genuine reader-impact signals a single isolated mistake
    # object can't carry on its own.
    mistakes = _normalize_mistake_severity_and_category(mistakes)
    mistakes = _fix_article_preposition_mislabel(mistakes)
    mistakes = _escalate_frequent_minor_mistakes(mistakes)
    mistakes = _escalate_error_clusters(mistakes, essay)
    # Item 5 - runs last, after severity is fully finalized (see docstring).
    mistakes = _dedupe_mistakes_by_normalized_span(mistakes)

    # Map to CEFR level
    cefr = map_ielts_to_cefr(band)
    
    # Get vocabulary to learn
    vocab_list = []
    if use_v2_refine and refine_result_v2:
        vocab_list = _map_tied_vocabulary_suggestions(
            refine_result_v2["vocabulary_suggestions"], essay, task_type
        )
    if not vocab_list:
        # Either the flag is OFF, or the v2 refine call didn't return any
        # verbatim-verified suggestions - falls back to the existing static
        # topic-bucket generator so a submission is never left with zero
        # vocabulary feedback.
        vocab_list = generate_topic_vocabulary(question, essay, task_type)
        topic_words = [w for w in vocab_list if w.get("task_specific", True)]
        connectors = [w for w in vocab_list if not w.get("task_specific", True)]
        connectors = connectors[:2]
        vocab_list = (topic_words + connectors)[:10]

    # "strengths" and "improvement" are the actual top-level string keys the
    # prompt asks for and returns (see RESPONSE FORMAT in
    # writing_task1_common.txt / writing_task2_prompt.txt) - this used to
    # read ai.get("examiner_response", ...) and
    # ai.get("feedback", {}).get("improvements", ...), neither of which is
    # a key the prompt ever produces, so feedback_text/improvement_text
    # were ALWAYS empty and every essay silently fell back to the same two
    # hardcoded generic strings regardless of what the model actually said
    # - defeating the prompt's own explicit "no generic/boilerplate praise"
    # instruction.
    feedback_text = ai.get("strengths", "") or ""
    feedback = normalize_feedback(feedback_text) or "Clear and concise answer; focus on stronger linking."
    improvement_text = ai.get("improvement", "") or ""
    improvement = normalize_feedback(improvement_text) or "Improve coherence with clearer transitions."
    # When the AI call genuinely failed (see ai_evaluation_failed above),
    # the generic fallback strings above are actively misleading - they
    # read like real, if unremarkable, feedback on the candidate's
    # writing, when in fact nothing was evaluated at all. State that
    # plainly instead.
    if ai_evaluation_failed:
        feedback = "Evaluation temporarily unavailable - please try submitting this answer again."
        improvement = "This response could not be evaluated due to a temporary error. No feedback is available yet."

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
        # The candidate's own submitted text - previously absent from the
        # response entirely (only the Band 9 rewrite, refined_answer, was
        # returned), which made it impossible to see what "original"/
        # "corrected" snippets in "mistakes" actually referred to without
        # separately tracking the original submission.
        "answer_text": essay,
        "refined_answer": refined,
        "word_count": word_count,
        # Diagnostic only - makes the exact input _coherence_paragraph_cap()/
        # _grammar_punctuation_cap() actually saw visible in the response,
        # instead of having to infer it from the score. Added after a false
        # alarm where two essays both showed coherence_cohesion=5 and it was
        # unclear, from the outside, whether that was the cap firing or a
        # genuine GPT judgment - a live HTTP round-trip test (real "\n\n"
        # paragraph breaks in, same count out) and a direct unit test of
        # both cap functions against that exact text (both returned None,
        # cap did not fire) confirmed the pipeline does not strip newlines
        # and the caps were not misfiring - but that took an ad-hoc script
        # to establish. This field makes it visible without one.
        "newline_diagnostics": {
            "newline_count": (essay or "").count("\n"),
            "paragraph_block_count": len([b for b in re.split(r'\n\s*\n', essay or "") if b.strip()]),
            "literal_newline_escaping": literal_newline_diagnostics,
        },
        # True only when the underlying OpenAI call failed on every retry
        # (e.g. an image_url the API couldn't fetch) and every score/field
        # above is the neutral fallback, not a real evaluation - callers
        # should treat a true value as "please retry", not as a genuine
        # (if weak) result.
        "ai_evaluation_failed": ai_evaluation_failed,
        # None when ai_evaluation_failed is False. Otherwise "refusal"
        # (the model declined outright - observed live, non-reproducible,
        # see _looks_like_refusal) or "other" (invalid JSON, missing
        # keys, network error, etc.) - makes the failure MODE visible in
        # the eval log distinctly, without changing what
        # ai_evaluation_failed itself means or how it's set.
        "ai_evaluation_failed_reason": ai_evaluation_failed_reason,
    }

    # Consistent top-level shape for downstream UI
    result["band_score"] = result.get("overall_band", band)
    result["feedback"] = safe_output(feedback, "Provide clearer structure and examples.")
    result["improvement"] = safe_output(improvement, "Use varied vocabulary and clearer linking words.")
    result["vocabulary"] = vocab_list
    result["topic_relevance"] = topic_relevance
    result["relevance_notice"] = RELEVANCE_NOTICE_MESSAGES.get(topic_relevance)
    result["severity_legend"] = SEVERITY_LEGEND
    if use_v2_refine and refine_result_v2:
        # Additive-only diagnostics from the v2 refine pipeline -
        # refined_answer itself keeps its exact name/type/meaning above
        # either way. Absent entirely when the flag is OFF.
        result["refine_diagnostics"] = refine_result_v2["diagnostics"]
        # relevance_status is a NEW top-level field (on_topic/
        # partially_off_topic/off_topic - see the derivation above) -
        # topic_relevance above is untouched, still its own 3-value field
        # with its own value names, still driving the scoring cap alone.
        # relevance_notice is REPURPOSED under this flag from the flat
        # string set two lines above into the structured object below -
        # see the "off-topic answers" plan for why (the string form and
        # the new object share the same key name by explicit choice, not
        # an accident - the object is a strict superset of the string's
        # job). model_answer_source is required on every response
        # (including on_topic), the other four fields are empty strings
        # when there's nothing to report.
        result["relevance_status"] = relevance_status
        what_you_did = str(ai.get("off_topic_what_you_did", "") or "").strip()
        what_was_asked = str(ai.get("off_topic_what_was_asked", "") or "").strip()
        if relevance_status == "on_topic":
            headline, why_it_matters = "", ""
        elif relevance_status == "off_topic":
            headline = "Your answer does not address the question that was asked."
            criterion_name = "Task Achievement" if task_type == "task_1" else "Task Response"
            why_it_matters = (
                f"Because your answer doesn't address the actual question, "
                f"{criterion_name} is capped and your overall band is "
                f"significantly lower than your writing quality alone would "
                f"suggest."
            )
        else:  # partially_off_topic
            has_wrong_data = "wrong_data" in relevance_reasons
            has_topic_drift = "topic_drift" in relevance_reasons
            if has_wrong_data and not has_topic_drift:
                headline = "Some of the data in your answer doesn't match the chart."
            elif has_topic_drift and not has_wrong_data:
                headline = "Part of your answer drifts away from the question."
            else:
                headline = "Part of your answer drifts from the question, and some data doesn't match the chart."
            criterion_name = "Task Achievement" if task_type == "task_1" else "Task Response"
            why_it_matters = (
                f"This partly limits your {criterion_name} score, even though "
                f"the rest of your answer is on-topic."
            )
        result["relevance_notice"] = {
            "headline": headline,
            "what_you_did": what_you_did,
            "what_was_asked": what_was_asked,
            "why_it_matters": why_it_matters,
            "model_answer_source": refine_result_v2.get("model_answer_source", "built from your answer"),
        }
    if task1_variant is not None:
        result["task1_variant"] = task1_variant
        result["image_verification_used"] = bool(image_url)
        result["image_data_accuracy"] = image_data_accuracy
        result["image_accuracy_notice"] = IMAGE_ACCURACY_NOTICE_MESSAGES.get(image_data_accuracy)
        result["image_verification_incomplete"] = image_verification_incomplete

    result["usage"] = {
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage_log),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage_log),
        "total_tokens": sum(u.get("total_tokens", 0) for u in usage_log),
    }

    return result
