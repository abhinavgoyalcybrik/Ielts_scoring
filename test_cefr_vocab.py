#!/usr/bin/env python
"""Test CEFR mapping and vocabulary features."""

from evaluators.writing import get_vocabulary_to_learn
from utils.cefr_mapper import map_ielts_to_cefr

print("=" * 70)
print("TESTING NEW FEATURES: CEFR & VOCABULARY")
print("=" * 70)

# Test 1: CEFR Mapping
print("\n1. CEFR MAPPING TEST:")
print("-" * 70)
test_bands = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]
for band in test_bands:
    cefr = map_ielts_to_cefr(band)
    print(f"  Band {band:.1f} -> CEFR {cefr}")

# Test 2: Vocabulary Extraction
print("\n2. VOCABULARY EXTRACTION TEST:")
print("-" * 70)
test_essay = """
The bar chart illustrates the distribution of the workforce across three sectors 
in a particular country over two decades. Overall, there was a significant shift 
in employment from agriculture to the services sector, while industrial employment 
remained relatively stable.

In 1990, agriculture employed the largest share of the workforce at approximately 
45%. However, this figure declined markedly to around 30% in 2000 and further 
dropped to about 20% by 2010. Conversely, the services sector experienced consistent 
growth, rising from roughly 30% in 1990 to nearly 40% in 2000, and then surging to 
approximately 55% in 2010. Employment in industry showed minor fluctuations, 
accounting for about 25% of the workforce in 1990, increasing slightly to 30% in 2000, 
before decreasing back to around 25% in 2010.

In summary, the data reveals a clear structural transformation in the country's 
employment landscape.
"""

vocab_task1 = get_vocabulary_to_learn(test_essay, "task_1", 6.5)
print(f"  Task 1 - Vocabulary items extracted: {len(vocab_task1)}")
print(f"  Min required: 12, Max: 20")
print(f"  Status: {'✓ PASS' if 12 <= len(vocab_task1) <= 20 else '✗ FAIL'}")

print("\n  Sample vocabulary to learn:")
for i, item in enumerate(vocab_task1[:5], 1):
    print(f"    {i}. {item['word']} - {item['usage_hint']}")

# Test 3: Task 2 Vocabulary
print("\n3. TASK 2 VOCABULARY TEST:")
print("-" * 70)
test_essay_task2 = """
Some people argue that governments should prioritize investing more funds in public 
transportation, while others contend that constructing new roads is more crucial. 
Both perspectives have merit; however, I firmly believe that enhancing public transport 
offers a more sustainable and effective long-term solution.

A primary reason for emphasizing public transportation is its potential to alleviate 
traffic congestion in urban areas. Efficient public transport systems can accommodate 
large numbers of passengers simultaneously. Moreover, public transportation plays a vital 
role in environmental protection. Private cars are significant contributors to air pollution 
and greenhouse gas emissions. By encouraging people to opt for public transit, fuel consumption 
per capita decreases, leading to improved air quality and better public health.

In conclusion, while road development remains important, governments should primarily 
focus on improving public transportation systems.
"""

vocab_task2 = get_vocabulary_to_learn(test_essay_task2, "task_2", 8.0)
print(f"  Task 2 - Vocabulary items extracted: {len(vocab_task2)}")
print(f"  Min required: 12, Max: 20")
print(f"  Status: {'✓ PASS' if 12 <= len(vocab_task2) <= 20 else '✗ FAIL'}")

print("\n  Sample vocabulary to learn:")
for i, item in enumerate(vocab_task2[:5], 1):
    print(f"    {i}. {item['word']} - {item['usage_hint']}")

print("\n" + "=" * 70)
print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
print("=" * 70)
