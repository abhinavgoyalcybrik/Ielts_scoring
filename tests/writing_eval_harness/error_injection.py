# Error injection for the recall + category-accuracy checks
# (writing_eval_harness Step 0).
#
# Each injection is a hand-specified, exact string replacement against one
# clean_corpus.py entry - not automated NLP-based injection. This is a
# deliberate choice: an automated injector can silently produce output that
# isn't actually the error type it claims to be (e.g. a naive tense-flipper
# hitting a verb where either tense is grammatical), which would corrupt the
# ground truth this whole check depends on. Hand-specifying means every
# record's "this is a genuine, unambiguous error of exactly this type" claim
# is something I can personally stand behind.
#
# The injection record IS the ground truth - it is never inferred from the
# evaluator's own output.

from clean_corpus import CLEAN_CORPUS_BY_ID

# category names must exactly match the CATEGORY REFERENCE lists in
# prompts/writing_task1_common.txt / writing_task2_prompt.txt.
INJECTIONS = [
    {
        "id": "inject_tense_slip_01",
        "source_id": "clean_t1_06",
        "error_type": "tense_slip",
        "find": "which was delivered to my home four days later",
        "replace": "which is delivered to my home four days later",
        "expected_category": "Tense Errors",
        "expected_category_alts": [],
    },
    {
        "id": "inject_wrong_preposition_01",
        "source_id": "clean_t2_01",
        "error_type": "wrong_preposition",
        "find": "feeling anxious in situations that require them",
        "replace": "feeling anxious on situations that require them",
        "expected_category": "Preposition Errors",
        "expected_category_alts": [],
    },
    {
        "id": "inject_missing_article_01",
        "source_id": "clean_t1_04",
        "error_type": "missing_article",
        "find": "coal was by far the dominant source",
        "replace": "coal was by far dominant source",
        "expected_category": "Article Errors",
        "expected_category_alts": [],
    },
    {
        "id": "inject_subject_verb_disagreement_01",
        "source_id": "clean_t2_06",
        "error_type": "subject_verb_disagreement",
        "find": "Confusing date labelling compounds the problem",
        "replace": "Confusing date labelling compound the problem",
        "expected_category": "Subject-Verb Agreement Errors",
        "expected_category_alts": [],
    },
    {
        "id": "inject_wrong_word_form_01",
        "source_id": "clean_t2_09",
        "error_type": "wrong_word_form",
        "find": "found that productivity per hour worked actually increased",
        "replace": "found that productive per hour worked actually increased",
        "expected_category": "Word Formation Errors",
        "expected_category_alts": [],
    },
    {
        "id": "inject_run_on_01",
        "source_id": "clean_t1_03",
        "error_type": "run_on",
        "find": (
            "surface residue. Once cleaned, the oranges pass through a "
            "sorting stage"
        ),
        "replace": (
            "surface residue, once cleaned, the oranges pass through a "
            "sorting stage"
        ),
        "expected_category": "Run-on Sentences",
        "expected_category_alts": ["Sentence Boundary Errors", "Comma Splice"],
    },
    {
        "id": "inject_unnatural_collocation_01",
        "source_id": "clean_t2_04",
        "error_type": "unnatural_collocation",
        "find": "mounting new exhibitions all require significant funding",
        "replace": "mounting new exhibitions all require big funding",
        "expected_category": "Collocation Errors",
        "expected_category_alts": ["Unnatural Collocation"],
    },
    {
        "id": "inject_register_mismatch_01",
        "source_id": "clean_t2_07",
        "error_type": "register_mismatch",
        "find": "There is genuine evidence supporting music education.",
        "replace": "There's like genuine evidence supporting music education.",
        "expected_category": "Register/Style Appropriacy Errors",
        "expected_category_alts": ["Register Mismatch"],
    },
    # A second instance of a few high-value types, in different source
    # texts, so recall isn't measured off a single example per type.
    {
        "id": "inject_wrong_preposition_02",
        "source_id": "clean_t2_08",
        "error_type": "wrong_preposition",
        "find": "responsibility for managing a child's exposure to advertising",
        "replace": "responsibility of managing a child's exposure to advertising",
        "expected_category": "Preposition Errors",
        "expected_category_alts": [],
    },
    {
        "id": "inject_wrong_word_form_02",
        "source_id": "clean_t2_10",
        "error_type": "wrong_word_form",
        "find": "many technologies developed for space missions",
        "replace": "many technology developed for space missions",
        "expected_category": "Word Formation Errors",
        "expected_category_alts": [],
    },
    {
        "id": "inject_subject_verb_disagreement_02",
        "source_id": "clean_t1_05",
        "error_type": "subject_verb_disagreement",
        "find": "The table presents visitor numbers",
        "replace": "The table present visitor numbers",
        "expected_category": "Subject-Verb Agreement Errors",
        "expected_category_alts": [],
    },
]


def build_damaged_corpus():
    """Returns a list of {id, source_id, task_type, question, text,
    injections: [record, ...]} - one damaged text per source corpus entry
    that has at least one injection, with ALL of that entry's injections
    applied together (so a single evaluator run per text also exercises
    multi-error recall within one essay, not just isolated single-error
    cases). Each injection record carries enough to locate the damaged
    span in the returned text via a fresh .find() at check time - offsets
    are not pre-computed and frozen, since applying multiple injections to
    the same text shifts later offsets; the check re-locates each span
    against the actual returned text instead.
    """
    by_source = {}
    for spec in INJECTIONS:
        by_source.setdefault(spec["source_id"], []).append(spec)

    damaged = []
    for source_id, specs in by_source.items():
        entry = CLEAN_CORPUS_BY_ID[source_id]
        text = entry["text"]
        applied = []
        for spec in specs:
            if spec["find"] not in text:
                raise AssertionError(
                    f"{spec['id']}: find-string not present in {source_id} - "
                    f"clean_corpus.py must have changed out from under this "
                    f"injection. Fix the injection spec, don't skip it."
                )
            text = text.replace(spec["find"], spec["replace"], 1)
            applied.append(spec)
        damaged.append({
            "id": f"damaged_{source_id}",
            "source_id": source_id,
            "task_type": entry["task_type"],
            "question": entry["question"],
            "text": text,
            "injections": applied,
        })
    return damaged
