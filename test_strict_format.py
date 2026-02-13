"""
Test Script: IELTS Strict A-F Format Output
============================================

This script demonstrates how to use the format_writing_strict function
to output writing evaluations in the mandatory A-F format.

Usage:
    python test_strict_format.py
"""

from evaluator import evaluate_attempt, format_writing_strict
import json

# =========================
# TASK 1 INPUT
# =========================
task_1_text = (
    "The charts show why adults decide to study and how the cost of adult education should be shared. "
    "According to the bar chart, the most common reason for studying is interest in the subject at 40%. "
    "Gaining qualifications is also an important reason at 38%. "
    "Some people study because it helps their current job, while fewer people study to meet new people. "
    "The pie chart shows that individuals pay the highest percentage of the cost at 40%. "
    "Employers pay 35%, and taxpayers pay the smallest part at 25%."
)

task_1_data = {
    "test_type": "writing",
    "metadata": {
        "task_type": "task_1",
        "question": (
            "The charts below show the results of a survey of adult education. "
            "The first chart shows the reasons why adults decide to study. "
            "The pie chart shows how people think the costs of adult education should be shared."
        )
    },
    "user_answers": {
        "text": task_1_text
    }
}

# =========================
# TASK 2 INPUT
# =========================
task_2_text = (
    "There are many types of music in the world today. People need music for entertainment "
    "and relaxation. Music can make people feel happy or calm.\n\n"
    "Some people think traditional music is important because it shows the culture of a country "
    "and helps people remember history. However, international music is popular because it is "
    "modern and easy to listen to.\n\n"
    "In my opinion, both traditional and international music are important. Traditional music "
    "keeps culture alive, while international music connects people around the world."
)

task_2_data = {
    "test_type": "writing",
    "metadata": {
        "task_type": "task_2",
        "question": (
            "There are many different types of music in the world today. "
            "Why do we need music? "
            "Is traditional music more important than international music?"
        )
    },
    "user_answers": {
        "text": task_2_text
    }
}

# =========================
# STEP 1: EVALUATE BOTH TASKS
# =========================
print("=" * 80)
print("STEP 1: EVALUATING TASKS")
print("=" * 80)

print("\nEvaluating Task 1...")
task_1_result = evaluate_attempt(task_1_data)
print(f"Task 1 Band: {task_1_result.get('overall_band', 'N/A')}")

print("\nEvaluating Task 2...")
task_2_result = evaluate_attempt(task_2_data)
print(f"Task 2 Band: {task_2_result.get('overall_band', 'N/A')}")

# =========================
# STEP 2: FORMAT TO STRICT A-F FORMAT
# =========================
print("\n" + "=" * 80)
print("STEP 2: FORMATTING OUTPUT TO STRICT A-F FORMAT")
print("=" * 80)

formatted_output = format_writing_strict(task_1_result, task_2_result)

# =========================
# STEP 3: VALIDATE STRUCTURE
# =========================
print("\n" + "=" * 80)
print("STEP 3: VALIDATION")
print("=" * 80)

required_sections = [
    "A_OVERALL_RESULT",
    "B_CRITERIA_BREAKDOWN", 
    "C_ERRORS_FOUND",
    "D_REFINED_ANSWER",
    "E_USEFUL_VOCABULARY",
    "F_HOW_WORDS_IMPROVE"
]

print("\nChecking mandatory sections:")
for section in required_sections:
    present = section in formatted_output
    status = "✓" if present else "✗"
    print(f"  {status} {section}")

# Validate section E vocabulary count
vocab_count = len(formatted_output.get("E_USEFUL_VOCABULARY", {}).get("vocabulary_list", []))
vocab_valid = 12 <= vocab_count <= 20
print(f"\nVocabulary count: {vocab_count} (valid: {vocab_valid}) [12-20 required]")

# Validate section F bullet count
bullet_count = len(formatted_output.get("F_HOW_WORDS_IMPROVE", {}).get("improvements", []))
bullet_valid = bullet_count == 3
print(f"Improvement bullets: {bullet_count} (valid: {bullet_valid}) [3 required]")

# Check section titles
section_e_title = formatted_output.get("E_USEFUL_VOCABULARY", {}).get("section", "")
section_f_title = formatted_output.get("F_HOW_WORDS_IMPROVE", {}).get("section", "")

print(f"\nSection E title: '{section_e_title}'")
print(f"  Expected: 'Useful Vocabulary'")
print(f"  Match: {section_e_title == 'Useful Vocabulary'}")

print(f"\nSection F title: '{section_f_title}'")
print(f"  Expected: 'How These Words Improve the Answer'")
print(f"  Match: {section_f_title == 'How These Words Improve the Answer'}")

# =========================
# STEP 4: DISPLAY RESULTS
# =========================
print("\n" + "=" * 80)
print("STEP 4: COMPLETE OUTPUT (JSON)")
print("=" * 80 + "\n")

print(json.dumps(formatted_output, indent=2, ensure_ascii=False))

# =========================
# STEP 5: DISPLAY SUMMARIES
# =========================
print("\n" + "=" * 80)
print("STEP 5: SUMMARY")
print("=" * 80)

overall = formatted_output.get("A_OVERALL_RESULT", {})
print(f"\nOVERALL BANDS:")
print(f"  Task 1: {overall.get('task_1_band', 'N/A')}")
print(f"  Task 2: {overall.get('task_2_band', 'N/A')}")
print(f"  Writing Module: {overall.get('overall_writing_band', 'N/A')}")

print(f"\nKEY VOCABULARY SAMPLES:")
vocab_list = formatted_output.get("E_USEFUL_VOCABULARY", {}).get("vocabulary_list", [])
for i, word in enumerate(vocab_list[:3], 1):
    print(f"  {i}. {word.get('word', 'N/A')} - {word.get('usage_hint', 'N/A')}")

print("\nIMPROVEMENT FOCUS AREAS:")
improvements = formatted_output.get("F_HOW_WORDS_IMPROVE", {}).get("improvements", [])
for imp in improvements:
    focus = imp.get('focus', 'N/A')
    desc = imp.get('description', '')[:60] + "..."
    print(f"  • {focus}: {desc}")

print("\n" + "=" * 80)
print("✓ STRICT FORMAT OUTPUT COMPLETE")
print("=" * 80)
