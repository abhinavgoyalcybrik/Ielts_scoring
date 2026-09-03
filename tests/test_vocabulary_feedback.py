import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.vocabulary_feedback import detect_essay_topic, generate_topic_vocabulary


# ---------------------------------------------------------------------------
# detect_essay_topic: drives which topic-specific vocabulary bucket a
# Writing Task 2 essay gets. Two layered bugs found from the same real
# case (a health-policy essay whose vocabulary suggestions came back about
# technology): a bare substring match ("ai" matching inside "campaigns"),
# and "first topic in dict order with any match" instead of "topic with
# the most actual signal in the text".
# ---------------------------------------------------------------------------

def test_detect_essay_topic_does_not_false_positive_on_ai_substring():
    # "campaigns" contains "ai" as a bare substring - must not match the
    # "technology" topic's "ai" keyword (meant for the acronym AI).
    text = "the government ran several public health campaigns last year"
    assert detect_essay_topic(text) != "technology"


def test_detect_essay_topic_rejects_other_ai_substring_words():
    for text in (
        "we must maintain high standards",
        "let me explain this point",
        "it is certain to happen",
    ):
        assert detect_essay_topic(text) != "technology"


def test_detect_essay_topic_picks_dominant_topic_not_first_dict_match():
    # Regression test for the real case: an essay overwhelmingly about
    # health (multiple mentions of "health") that mentions "education"
    # only once in passing was still classified as "education", since
    # that topic happens to be checked earlier in the dict and the old
    # code stopped at the first match instead of the best one.
    question = (
        "Some people believe that governments should be responsible for "
        "people's health, while others believe individuals should take "
        "responsibility for their own health."
    )
    essay = (
        "There are two different opinions about who is responsible for "
        "people's health. Some believe the government should take charge, "
        "providing free healthcare and public health campaigns so that no "
        "person is left behind for lack of money or education."
    )
    combined = (question + " " + essay).lower()
    assert detect_essay_topic(combined) == "health"


def test_detect_essay_topic_still_detects_genuine_technology_topic():
    text = "artificial intelligence and automation are transforming modern workplaces through digital innovation"
    assert detect_essay_topic(text) == "technology"


def test_detect_essay_topic_uses_word_boundaries_not_bare_substrings():
    # "work" must match as its own word, not as a substring of an
    # unrelated longer word.
    assert detect_essay_topic("catch the train to work") == "work"
    assert detect_essay_topic("the workshop was fully booked") != "work"


def test_detect_essay_topic_falls_back_to_general_when_no_keywords_match():
    assert detect_essay_topic("") == "general"
    assert detect_essay_topic("xyz abc qrs tuv") == "general"


# ---------------------------------------------------------------------------
# generate_topic_vocabulary(): Task 1's branch computed `topic` via
# detect_essay_topic() but never actually used it - it re-detected the
# topic itself with only 4 narrow keyword buckets (transport/travel,
# population, education/students, sales/company), and gave every single
# word the identical placeholder "context-based usage" hint. Regression
# test for the real observed case: a Task 1 chart about fast food
# expenditure by income group matched none of those 4 buckets and fell
# back to a generic list with zero real per-word guidance.
# ---------------------------------------------------------------------------

def test_generate_topic_vocabulary_task1_gives_real_hints_not_identical_placeholder():
    question = "The bar chart shows average weekly expenditure on fast food by income group."
    essay = "Low-income households spend on fish and chips, hamburgers, pizza, and delivery."

    vocab = generate_topic_vocabulary(question, essay, "task_1")

    assert len(vocab) == 10
    hints = {v["usage_hint"] for v in vocab}
    assert len(hints) > 1
    assert "context-based usage" not in hints


def test_generate_topic_vocabulary_task1_reuses_detect_essay_topic_categories():
    # A topic detect_essay_topic() genuinely recognises (health) that the
    # OLD Task 1 4-bucket re-match did NOT cover at all.
    question = "The graph shows obesity and exercise rates across age groups."

    vocab = generate_topic_vocabulary(question, "", "task_1")

    words = {v["word"] for v in vocab}
    assert words & {"prevalence", "life expectancy", "obesity rate", "dietary", "expenditure"}
