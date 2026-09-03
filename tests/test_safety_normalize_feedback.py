from utils.safety import normalize_feedback


def test_normalize_feedback_preserves_abbreviation_periods():
    text = (
        "The candidate should focus on improving lexical accuracy, especially "
        "word choice (e.g. 'improvise' vs. 'improve') and address minor "
        "grammatical inconsistencies for a higher band score."
    )
    result = normalize_feedback(text)
    assert result == text
    assert "(e. g." not in result
    assert not result.rstrip(".").endswith("vs")


def test_normalize_feedback_caps_at_three_sentences():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    result = normalize_feedback(text)
    assert result == "Sentence one. Sentence two. Sentence three."


def test_normalize_feedback_dedupes_repeated_sentences():
    text = "Improve your grammar. Improve your grammar. Add more examples."
    result = normalize_feedback(text)
    assert result == "Improve your grammar. Add more examples."


def test_normalize_feedback_adds_missing_terminal_period():
    result = normalize_feedback("Only one sentence with no ending punctuation")
    assert result == "Only one sentence with no ending punctuation."


def test_normalize_feedback_empty_input_returns_empty_string():
    assert normalize_feedback("") == ""
    assert normalize_feedback(None) == ""
