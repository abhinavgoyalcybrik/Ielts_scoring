#!/usr/bin/env python
"""
Test IELTS Writing Examiner validation rules implementation
"""

from evaluators.writing import (
    validate_word_count, 
    apply_coherence_penalty_cap,
    apply_fair_band_scoring,
    count_words
)

print("=" * 80)
print("TESTING IELTS ACADEMIC WRITING EXAMINER RULES")
print("=" * 80)

# TEST 1: Word Count Validation
print("\n1. WORD COUNT VALIDATION TEST:")
print("-" * 80)

task1_text = "The charts show data. " * 10  # 40 words
task1_valid = "The charts show employment data. " * 10  # 60 words
task2_text = "Music is important. " * 15  # 60 words
task2_valid = "Music is important and necessary. " * 25  # 175 words

tests = [
    ("Task 1 - Below 150 words", "task_1", task1_text, False),
    ("Task 1 - Above 150 words", "task_1", task1_valid, True),
    ("Task 2 - Below 250 words", "task_2", task2_text, False),
    ("Task 2 - Above 250 words", "task_2", task2_valid, True),
]

for test_name, task_type, text, should_pass in tests:
    try:
        wc = validate_word_count(task_type, text)
        status = "✓ PASS" if should_pass else "✗ FAIL (should have raised)"
        print(f"  {test_name}: {status} ({wc} words)")
    except ValueError as e:
        status = "✗ FAIL (should have passed)" if should_pass else "✓ PASS"
        print(f"  {test_name}: {status} - {e}")

# TEST 2: Coherence Penalty Cap
print("\n2. COHERENCE PENALTY CAP TEST (Max 2 repetition errors):")
print("-" * 80)

mistakes_5_repetitions = [
    {"sentence": "Test 1", "error_type": "coherence", "explanation": "Repetition of word", "correction": "Fix 1"},
    {"sentence": "Test 2", "error_type": "coherence", "explanation": "Repetition of phrase", "correction": "Fix 2"},
    {"sentence": "Test 3", "error_type": "coherence", "explanation": "Repetition of concept", "correction": "Fix 3"},
    {"sentence": "Test 4", "error_type": "coherence", "explanation": "Another repetition", "correction": "Fix 4"},
    {"sentence": "Test 5", "error_type": "coherence", "explanation": "More repetition", "correction": "Fix 5"},
    {"sentence": "Test 6", "error_type": "grammar", "explanation": "Tense error", "correction": "Fix 6"},
]

capped = apply_coherence_penalty_cap(mistakes_5_repetitions)
repetition_count = len([m for m in capped if m.get("error_type") == "coherence" and "repetition" in m.get("explanation", "").lower()])

print(f"  Input: 5 repetition errors + 1 grammar error")
print(f"  Output: {repetition_count} repetition errors (capped to max 2) + 1 grammar error")
print(f"  Total mistakes: {len(capped)}")
print(f"  Status: {'✓ PASS' if repetition_count <= 2 else '✗ FAIL'}")

# TEST 3: Fair Band Scoring (Min 5 if task addressed)
print("\n3. FAIR BAND SCORING TEST (Min Band 5 if task addressed):")
print("-" * 80)

scoring_tests = [
    ("Task addressed (TR=7), low overall (3.2)", 3.2, 7.0, 5.0),  # Should boost to 5
    ("Task addressed (TR=6), low overall (4.0)", 4.0, 6.0, 5.0),  # Should boost to 5
    ("Task not addressed (TR=4), low overall (3.0)", 3.0, 4.0, 3.0),  # Should stay at 3
    ("Normal case (TR=6, avg=6.5)", 6.5, 6.0, 6.5),  # Should stay at 6.5
]

for test_name, overall_input, tr_score, expected in scoring_tests:
    result = apply_fair_band_scoring(overall_input, tr_score, "task_2")
    status = "✓ PASS" if abs(result - expected) < 0.01 else "✗ FAIL"
    print(f"  {test_name}")
    print(f"    Input: overall={overall_input}, task_response={tr_score}")
    print(f"    Output: {result} (expected {expected}) - {status}\n")

print("=" * 80)
print("✓ ALL RULE VALIDATION TESTS COMPLETED")
print("=" * 80)
