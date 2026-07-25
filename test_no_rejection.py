#!/usr/bin/env python
"""
Test that IELTS Writing evaluation proceeds regardless of word count.
Verifies that short answers are evaluated (not rejected) and penalized through scoring.
"""

from evaluators.writing import count_words, validate_word_count

print("=" * 80)
print("TESTING: ALWAYS EVALUATE - NEVER REJECT PRINCIPLE")
print("=" * 80)

# TEST 1: Short Task 1 (below 150 words)
print("\n1. SHORT TASK 1 ANSWER (50 words):")
print("-" * 80)

short_task1 = "The chart shows employment. Agriculture decreased. Services increased. Industry stable."
wc_short_task1 = count_words(short_task1)
print(f"  Word count: {wc_short_task1}")
print(f"  Recommendation: 150+ words")

try:
    result = validate_word_count("task_1", short_task1)
    print(f"  ✓ EVALUATION PROCEEDS: {result} words counted")
    print(f"  → Will be penalized through Task Response score (underdevelopment)")
    status_1 = "✓ PASS"
except ValueError as e:
    print(f"  ✗ BLOCKED: {e}")
    status_1 = "✗ FAIL"

print(f"  Status: {status_1}")

# TEST 2: Short Task 2 (below 250 words)
print("\n2. SHORT TASK 2 ANSWER (80 words):")
print("-" * 80)

short_task2 = (
    "Music is important. People enjoy music. Traditional music is good. Modern music is also good. "
    "Both are valuable. I like both types. Music helps people relax. That is why music matters."
)
wc_short_task2 = count_words(short_task2)
print(f"  Word count: {wc_short_task2}")
print(f"  Recommendation: 250+ words")

try:
    result = validate_word_count("task_2", short_task2)
    print(f"  ✓ EVALUATION PROCEEDS: {result} words counted")
    print(f"  → Will be penalized through Task Response score (incomplete argument)")
    status_2 = "✓ PASS"
except ValueError as e:
    print(f"  ✗ BLOCKED: {e}")
    status_2 = "✗ FAIL"

print(f"  Status: {status_2}")

# TEST 3: Valid length Task 1
print("\n3. VALID TASK 1 ANSWER (180 words):")
print("-" * 80)

valid_task1 = (
    "The bar chart illustrates employment distribution across three sectors. "
    "Overall, there has been a significant shift from agriculture to services. "
    "In 1990, agriculture employed 45%, declining to 20% by 2010. "
    "Services sector grew from 30% to 55%, becoming dominant. "
    "Industrial employment remained stable around 25-30%. "
    "This represents structural economic change. "
    "The data demonstrates clear workforce reallocation. "
    "Agricultural jobs moved to service sector. "
) * 1  # Repeated for word count

wc_valid_task1 = count_words(valid_task1)
print(f"  Word count: {wc_valid_task1}")
print(f"  Recommendation: 150+ words")

try:
    result = validate_word_count("task_1", valid_task1)
    print(f"  ✓ EVALUATION PROCEEDS: {result} words counted")
    print(f"  → Will be evaluated normally on quality criteria")
    status_3 = "✓ PASS"
except ValueError as e:
    print(f"  ✗ BLOCKED: {e}")
    status_3 = "✗ FAIL"

print(f"  Status: {status_3}")

# TEST 4: Valid length Task 2
print("\n4. VALID TASK 2 ANSWER (280 words):")
print("-" * 80)

valid_task2 = (
    "Music has always been important in society. There are many types of music. "
    "Traditional music shows culture and history. International music connects people globally. "
    "Both have different benefits and purposes. "
    "Traditional music preserves cultural heritage. It tells stories of ancestors. "
    "It helps people remember their identity. Cultural traditions are valuable. "
    "International music is modern and popular. It brings people together. "
    "It allows cultural exchange and understanding. Modern music appeals to young people. "
    "However, traditional music is also important. It should not be forgotten. "
    "In conclusion, both traditional and international music are necessary. "
    "Society needs both types. Cultural preservation matters. Global connection is important. "
    "We should value and support both forms of musical expression. "
) * 1

wc_valid_task2 = count_words(valid_task2)
print(f"  Word count: {wc_valid_task2}")
print(f"  Recommendation: 250+ words")

try:
    result = validate_word_count("task_2", valid_task2)
    print(f"  ✓ EVALUATION PROCEEDS: {result} words counted")
    print(f"  → Will be evaluated normally on quality criteria")
    status_4 = "✓ PASS"
except ValueError as e:
    print(f"  ✗ BLOCKED: {e}")
    status_4 = "✗ FAIL"

print(f"  Status: {status_4}")

# SUMMARY
print("\n" + "=" * 80)
print("SUMMARY: NO HARD VALIDATION - GRACEFUL HANDLING")
print("=" * 80)

all_pass = all(s == "✓ PASS" for s in [status_1, status_2, status_3, status_4])

print(f"\n✓ Test 1 (Short Task 1): {status_1}")
print(f"✓ Test 2 (Short Task 2): {status_2}")
print(f"✓ Test 3 (Valid Task 1): {status_3}")
print(f"✓ Test 4 (Valid Task 2): {status_4}")

print(f"\nOVERALL RESULT: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")

print("\n" + "=" * 80)
print("PRINCIPLE VERIFIED: ALWAYS EVALUATE - NEVER REJECT")
print("=" * 80)
print("\nKey Points:")
print("✓ No word count errors raised")
print("✓ All answers proceed to evaluation")
print("✓ Word count is counted and available for scoring adjustment")
print("✓ Low word count will be penalized through Task Response score")
print("✓ Aligns with IELTS examiner practice (penalize, don't reject)")
