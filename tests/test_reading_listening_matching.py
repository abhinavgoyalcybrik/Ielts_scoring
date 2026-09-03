import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluators.reading import evaluate_reading
from evaluators.listening import evaluate_listening


# ---------------------------------------------------------------------------
# Answer-key matching in Reading and Listening used a plain `==` comparison
# with no normalisation at all - case-sensitive AND whitespace-sensitive.
# A fully correct answer typed as "true" instead of "TRUE", or with a
# trailing space, was marked wrong. Real IELTS Reading/Listening marking
# does not penalise capitalisation (spelling must still be exact).
# ---------------------------------------------------------------------------

def test_reading_answers_match_case_insensitively():
    result = evaluate_reading({
        "questions": [
            {"question_id": "q1", "answer_key": "TRUE", "type": "TRUE_FALSE_NOT_GIVEN"},
            {"question_id": "q2", "answer_key": "Paris", "type": "FILL_IN_THE_BLANKS"},
            {"question_id": "q3", "answer_key": "B", "type": "MCQ"},
        ],
        "user_answers": {"q1": "true", "q2": "  paris  ", "q3": "b"},
    })
    assert result["accuracy"] == "3/3"


def test_reading_still_marks_genuinely_wrong_spelling_as_incorrect():
    result = evaluate_reading({
        "questions": [{"question_id": "q1", "answer_key": "London", "type": "FILL_IN_THE_BLANKS"}],
        "user_answers": {"q1": "Berlin"},
    })
    assert result["accuracy"] == "0/1"


def test_reading_blank_answer_never_counts_as_correct():
    result = evaluate_reading({
        "questions": [{"question_id": "q1", "answer_key": "TRUE", "type": "TRUE_FALSE_NOT_GIVEN"}],
        "user_answers": {"q1": ""},
    })
    assert result["accuracy"] == "0/1"


def test_listening_answers_match_case_and_whitespace_insensitively():
    result = evaluate_listening({
        "answer_key": {"q1": "Museum", "q2": "TUESDAY", "q3": "42"},
        "user_answers": {"q1": "museum", "q2": " tuesday ", "q3": "43"},
    })
    assert result["accuracy"] == "2/3"


def test_listening_blank_answer_never_counts_as_correct():
    result = evaluate_listening({
        "answer_key": {"q1": "Museum"},
        "user_answers": {"q1": ""},
    })
    assert result["accuracy"] == "0/1"


# ---------------------------------------------------------------------------
# Multiple acceptable answers per question. `answer_key` (Reading) / each
# value in `answer_key` (Listening) may be a list of several answers the
# mark scheme accepts (e.g. British vs American spelling, "cannot" vs
# "can not" vs "can't") - the user's answer is correct if it matches ANY
# one of them, still case/whitespace-insensitively. Before this, passing a
# list as answer_key would never match anything (str() of a list vs plain
# text), silently marking every such question wrong regardless of input.
# ---------------------------------------------------------------------------

def test_reading_matches_any_of_several_accepted_answers():
    result = evaluate_reading({
        "questions": [
            {"question_id": "q1", "answer_key": ["colour", "color"], "type": "FILL_IN_THE_BLANKS"},
            {"question_id": "q2", "answer_key": ["cannot", "can not", "can't"], "type": "FILL_IN_THE_BLANKS"},
            {"question_id": "q3", "answer_key": ["A", "B"], "type": "MCQ"},
        ],
        "user_answers": {"q1": "COLOR", "q2": "can't", "q3": "b"},
    })
    assert result["accuracy"] == "3/3"


def test_reading_marks_wrong_when_answer_matches_none_of_the_alternatives():
    result = evaluate_reading({
        "questions": [{"question_id": "q1", "answer_key": ["red", "blue", "green", "yellow"], "type": "FILL_IN_THE_BLANKS"}],
        "user_answers": {"q1": "purple"},
    })
    assert result["accuracy"] == "0/1"


def test_reading_single_scalar_answer_key_still_works_unchanged():
    result = evaluate_reading({
        "questions": [{"question_id": "q1", "answer_key": "TRUE", "type": "TRUE_FALSE_NOT_GIVEN"}],
        "user_answers": {"q1": "true"},
    })
    assert result["accuracy"] == "1/1"


def test_listening_matches_any_of_several_accepted_answers():
    result = evaluate_listening({
        "answer_key": {"q1": ["Museum", "Gallery"], "q2": ["42", "forty-two"]},
        "user_answers": {"q1": "gallery", "q2": "43"},
    })
    assert result["accuracy"] == "1/2"
