# Perturbation variants for the invariance check (writing_eval_harness
# Step 0). Each transformation below changes something that should be
# invisible to an IELTS evaluator - spelling convention, quote glyph, line-
# ending style, or a proper name - while leaving every actual linguistic
# and content judgement identical. A working evaluator's band scores and
# mistake list must not move on any of these. The curly-quote variant in
# particular is a direct regression test for the verbatim-substring check
# in evaluators/writing.py (_mistake_original_is_verbatim) silently
# dropping real mistakes over typography alone - a gap this session's own
# earlier work flagged as a known risk but never had a targeted test for.

import re

# A reasonably exhaustive, one-directional US -> UK spelling map, restricted
# to words actually likely to appear in this corpus - not a general-purpose
# spelling converter. Applied to clean_corpus.py text, which is written in
# UK spelling throughout, so this produces a genuine US-spelling variant of
# the same essay.
_UK_TO_US_SPELLING = {
    "organisation": "organization",
    "organisations": "organizations",
    "recognise": "recognize",
    "colour": "color",
    "favour": "favor",
    "favours": "favors",
    "labour": "labor",
    "labelling": "labeling",
    "programme": "program",
    "programmes": "programs",
    "centre": "center",
    "centres": "centers",
    "artefacts": "artifacts",
    "behaviour": "behavior",
    "neighbour": "neighbor",
    "travelled": "traveled",
    "modelled": "modeled",
    "fuelled": "fueled",
    "defence": "defense",
    "defences": "defenses",
    "practise": "practice",
    "licence": "license",
    "analyse": "analyze",
    "catalogue": "catalog",
    "dialogue": "dialog",
}


def swap_uk_to_us_spelling(text: str) -> str:
    for uk, us in _UK_TO_US_SPELLING.items():
        text = re.sub(rf"\b{uk}\b", us, text)
        text = re.sub(rf"\b{uk.capitalize()}\b", us.capitalize(), text)
    return text


def swap_straight_to_curly_quotes(text: str) -> str:
    """Converts straight quotes/apostrophes to their curly typographic
    equivalents - the kind of substitution a word processor's autocorrect
    performs silently and a candidate would never notice, but which can
    break a naive string-equality substring check."""
    return _curly_quote_pass(text)


def _curly_quote_pass(text: str) -> str:
    result = []
    prev_char = " "
    quote_stack = []
    for ch in text:
        if ch == "'":
            if prev_char.isalnum():
                result.append("’")  # right single quote (apostrophe)
            else:
                result.append("‘")  # left single quote (opening)
        elif ch == '"':
            if quote_stack and quote_stack[-1] == '"':
                result.append("”")
                quote_stack.pop()
            else:
                result.append("“")
                quote_stack.append('"')
        else:
            result.append(ch)
        prev_char = ch
    return "".join(result)


def reformat_line_breaks(text: str) -> str:
    """Converts \\n\\n paragraph breaks to CRLF (\\r\\n\\r\\n) - a common
    artefact of text pasted from Windows word processors - while leaving
    the actual paragraph structure identical in meaning."""
    return text.replace("\n\n", "\r\n\r\n")


_NAME_SWAP_TARGETS = [
    ("Whitfield", "Okonkwo-Reyes"),
]


def swap_candidate_name(text: str) -> str:
    for original, replacement in _NAME_SWAP_TARGETS:
        text = text.replace(original, replacement)
    return text


def build_perturbations(entry: dict) -> list:
    """Given one clean_corpus.py entry, returns a list of
    {variant, text} - the same essay with exactly one perturbation applied
    each time, plus the untouched baseline for comparison. A name-swap
    variant is only produced for entries that actually contain a name from
    _NAME_SWAP_TARGETS (currently just the clean_t1_06 letter)."""
    base_text = entry["text"]
    variants = [
        {"variant": "baseline", "text": base_text},
        {"variant": "uk_to_us_spelling", "text": swap_uk_to_us_spelling(base_text)},
        {"variant": "curly_quotes", "text": _curly_quote_pass(base_text)},
        {"variant": "crlf_paragraph_breaks", "text": reformat_line_breaks(base_text)},
    ]
    name_swapped = swap_candidate_name(base_text)
    if name_swapped != base_text:
        variants.append({"variant": "candidate_name_swapped", "text": name_swapped})
    return variants
