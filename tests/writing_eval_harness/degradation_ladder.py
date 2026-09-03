# Degradation ladder for the ordering check (writing_eval_harness Step 0).
#
# One essay (clean_t2_01, chosen for its clean 5-paragraph structure:
# intro / view A / view B / opinion / conclusion), degraded through 5
# levels, each hand-constructed to be genuinely, unambiguously worse than
# the one before it - not by any evaluator's judgment, but by construction.
# A working evaluator MUST score these non-increasing on every criterion;
# an inversion means the scoring is not tracking real quality, regardless
# of what any single absolute number says.

from clean_corpus import CLEAN_CORPUS_BY_ID

_BASE = CLEAN_CORPUS_BY_ID["clean_t2_01"]

# Level 1: vocabulary flattened to vague, basic words. Every substitution
# here keeps the sentence grammatical and the meaning roughly intact - only
# precision and range of vocabulary drops, which is exactly what Lexical
# Resource is supposed to detect.
_VOCAB_FLATTEN_MAP = [
    ("eroded young people's ability", "made young people's ability bad"),
    ("largely agree with this view, although I believe the picture is more nuanced than it first appears",
     "agree with this a lot, but I think it is more complicated than it looks"),
    ("removes the spontaneity and vulnerability that characterise face-to-face conversation",
     "removes the good things about talking to someone in person"),
    ("young people report feeling anxious in situations that require them to respond immediately",
     "young people say they feel bad in situations where they have to answer fast"),
    ("this avoidance can weaken the very skills that only practice can build",
     "this not doing it can make the skills that you need practice for bad"),
    ("technology has not eliminated social interaction so much as redirected it",
     "technology has not got rid of talking to people, it just changed it"),
    ("It would therefore be inaccurate to claim that technology has replaced social skills altogether",
     "So it would be wrong to say technology has replaced people skills totally"),
    ("Parents and schools that actively encourage unstructured, in-person time",
     "Parents and schools that make sure kids get normal time"),
    ("treating every screen as inherently harmful oversimplifies a genuinely complex issue",
     "saying every screen is bad makes a hard thing too simple"),
    ("thoughtful guidance from parents and educators can help young people benefit from technology",
     "good help from parents and teachers can help young people get good things from technology"),
]


def _flatten_vocabulary(text: str) -> str:
    for original, flattened in _VOCAB_FLATTEN_MAP:
        if original not in text:
            raise AssertionError(
                f"degradation_ladder: expected phrase not found in clean_t2_01 - "
                f"clean_corpus.py must have changed: {original!r}"
            )
        text = text.replace(original, flattened, 1)
    return text


def _strip_paragraph_breaks(text: str) -> str:
    return " ".join(part.strip() for part in text.split("\n\n") if part.strip())


# Level 3: 8 distinct, hand-verified grammar errors injected into the
# level-2 (flattened + unparagraphed) text. Same "find exact string, no
# automated NLP" discipline as error_injection.py, for the same reason -
# these need to be unambiguously real errors, not guesswork.
_GRAMMAR_ERRORS_L3 = [
    ("It is often claimed that technology has made young people's ability bad",
     "It are often claimed that technology has made young people's ability bad"),  # subject-verb
    ("made young people's ability bad to interact with one another in person",
     "made young people's ability bad to interact with one another on person"),  # preposition
    ("Text messages and social media allow a person to compose",
     "Text messages and social media allow person to compose"),  # missing article
    ("young people say they feel bad in situations",
     "young people says they feel bad in situations"),  # subject-verb
    ("On the other hand, technology has not got rid of talking to people",
     "On the other hand, technology have not got rid of talking to people"),  # subject-verb
    ("Many young people maintain rich friendships that begin online",
     "Many young people maintains rich friendships that begin online"),  # subject-verb
    ("video calls allow people to read facial expressions and tone of voice",
     "video calls allow people to read facial expressions and tone of voices"),  # noun number
    ("In my view, the responsibility lies less with technology itself",
     "In my view, the responsibility lay less with technology itself"),  # tense
]


def _inject_eight_grammar_errors(text: str) -> str:
    applied = 0
    for original, broken in _GRAMMAR_ERRORS_L3:
        if original in text:
            text = text.replace(original, broken, 1)
            applied += 1
    if applied < 8:
        raise AssertionError(
            f"degradation_ladder level 3: only {applied}/8 grammar-error "
            f"injections found their target text - fix the spec, don't "
            f"silently ship fewer errors than claimed."
        )
    return text


# Level 4: one body paragraph replaced with content entirely unrelated to
# the question (weather patterns in coastal cities) - a genuine, blatant
# Task Response violation, not a subtle one.
_OFF_TOPIC_REPLACEMENT_TARGET = (
    "On the other hand, technology have not got rid of talking to people, "
    "it just changed it. Many young people maintains rich friendships that "
    "begin online and later move into the physical world, and video calls "
    "allow people to read facial expressions and tone of voices in ways "
    "that text alone cannot capture. So it would be wrong to say "
    "technology has replaced people skills totally; rather, it has "
    "changed which skills are exercised most often."
)
_OFF_TOPIC_PARAGRAPH = (
    "Coastal cities around the world experience highly variable weather "
    "patterns depending on ocean currents and seasonal wind direction. In "
    "the winter months, cold currents can lower average temperatures by "
    "several degrees, while in summer, warm currents often bring humid "
    "air that increases the likelihood of afternoon storms. Local "
    "governments in these regions frequently invest in flood defences to "
    "manage the resulting rainfall."
)


def build_ladder():
    """Returns a list of 5 dicts: {level, description, text} for
    clean_t2_01, degraded step by step. Each level's text is built from
    the previous level's text where the spec calls for "level N + X" -
    the cumulative construction is what "progressively worse" actually
    means here."""
    level0 = _BASE["text"]

    level1 = _flatten_vocabulary(level0)

    level2 = _strip_paragraph_breaks(level1)

    level3 = _inject_eight_grammar_errors(level2)

    if _OFF_TOPIC_REPLACEMENT_TARGET not in level3:
        raise AssertionError(
            "degradation_ladder level 4: off-topic replacement target not "
            "found in level 3 text - check the level 1/3 transformations "
            "above haven't altered this span."
        )
    level4 = level3.replace(_OFF_TOPIC_REPLACEMENT_TARGET, _OFF_TOPIC_PARAGRAPH, 1)

    return [
        {"level": 0, "description": "untouched original", "text": level0},
        {"level": 1, "description": "vocabulary flattened to vague/basic words", "text": level1},
        {"level": 2, "description": "level 1 + paragraph breaks removed", "text": level2},
        {"level": 3, "description": "level 2 + 8 grammar errors injected", "text": level3},
        {"level": 4, "description": "level 3 + one body paragraph made off-topic", "text": level4},
    ]
