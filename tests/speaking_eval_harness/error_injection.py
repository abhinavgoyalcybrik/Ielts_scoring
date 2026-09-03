# Error injection (Speaking eval harness) - hand-verified grammar/
# vocabulary errors injected into clean_corpus.py's transcripts at known
# locations, one error type per location, mirroring
# tests/writing_eval_harness/error_injection.py's discipline: every
# find-string is asserted present in the source transcript before the
# damaged corpus is built, so a future edit to clean_corpus.py can never
# silently break an injection without the test suite catching it.
#
# Deliberately does NOT inject punctuation/capitalization/spelling errors,
# filler words, self-corrections, or contractions - those are ASR
# artifacts/normal speech features that must NEVER be flagged (see
# asr_artifact_corpus.py), so injecting them here would test the wrong
# thing. Every injection below is a genuine grammar/vocabulary error a
# listener would actually notice.

from clean_corpus import CLEAN_PART1, CLEAN_PART3, CLEAN_PART2

INJECTIONS = [
    {
        "id": "inject_subject_verb_agreement",
        "source_id": "p1_hometown",
        "find": "It's known for its old fishing harbour",
        "replace": "It's know for its old fishing harbour",
        "expected_type": "grammar",
        # Short, distinctive fragment of the injected error - used to check
        # a returned mistake's "original" field actually points at the
        # injected span, not some unrelated part of the transcript.
        "detection_signal": "know for",
    },
    {
        "id": "inject_wrong_preposition",
        "source_id": "p1_work_study",
        "find": "I'm currently working as a junior accountant at a small firm",
        "replace": "I'm currently working as a junior accountant on a small firm",
        "expected_type": "grammar",
        "detection_signal": "on a small firm",
    },
    {
        "id": "inject_verb_tense",
        "source_id": "p1_hobbies",
        "find": "There's a river path near my house that I use most weekends",
        "replace": "There's a river path near my house that I using most weekends",
        "expected_type": "grammar",
        "detection_signal": "using most weekends",
    },
    {
        "id": "inject_wrong_word_choice",
        "source_id": "p3_travel_changes",
        "find": "has brought a lot of investment into smaller towns",
        "replace": "has bring a lot of investment into smaller towns",
        "expected_type": "grammar",
        "detection_signal": "has bring",
    },
    {
        "id": "inject_missing_article",
        "source_id": "p3_technology_travel",
        "find": "used to require a travel agent",
        "replace": "used to require travel agent",
        "expected_type": "grammar",
        "detection_signal": "require travel agent",
    },
    {
        "id": "inject_vocabulary_misuse",
        "source_id": "p3_sustainable_tourism",
        "find": "invest the revenue from tourism directly back into conservation",
        "replace": "invest the revenue from tourism directly back into conversation",
        "expected_type": "vocabulary",
        "detection_signal": "back into conversation",
    },
]


def _source_by_id():
    lookup = {item["id"]: item for item in CLEAN_PART1 + CLEAN_PART3 + [CLEAN_PART2]}
    return lookup


def build_damaged_corpus() -> list:
    """Returns a list of {id, part, question, answer, expected_type,
    injected_find, injected_replace} - one damaged transcript per
    injection, each a full item from the clean corpus with exactly one
    error substituted in. Raises AssertionError immediately if any
    injection's `find` string is no longer present in its source item -
    the same fail-loud-not-silently discipline as the Writing harness's
    build_damaged_corpus()."""
    sources = _source_by_id()
    damaged = []
    for spec in INJECTIONS:
        source = sources[spec["source_id"]]
        assert spec["find"] in source["answer"], (
            f"{spec['id']}: find-string not present in {spec['source_id']} - "
            f"clean_corpus.py must have changed out from under this injection"
        )
        damaged_answer = source["answer"].replace(spec["find"], spec["replace"], 1)
        damaged.append({
            "id": spec["id"],
            "part": source["part"],
            "question": source["question"],
            "answer": damaged_answer,
            "expected_type": spec["expected_type"],
            "injected_find": spec["replace"],
            "detection_signal": spec["detection_signal"],
        })
    return damaged


if __name__ == "__main__":
    corpus = build_damaged_corpus()
    print(f"Built {len(corpus)} damaged transcripts, all find-strings verified present.")
