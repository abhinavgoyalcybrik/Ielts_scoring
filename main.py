import json
from evaluator import evaluate_attempt
from utils.band import round_band

# =========================
# TASK 1 INPUT (VALID LENGTH)
# =========================
task1_text = (
    "The charts illustrate the reasons why adults choose to study and how the cost of adult education "
    "should be shared. Overall, personal interest and gaining qualifications are the most common motivations, "
    "while meeting new people is the least common reason.\n\n"
    "According to the bar chart, 40% of adults study because they are interested in the subject, followed closely "
    "by those who want to gain qualifications at 38%. A smaller proportion study to support their current job, "
    "while only a minority study to meet new people.\n\n"
    "The pie chart shows that individuals believe they should pay the largest share of the cost at 40%. "
    "Employers are expected to contribute 35%, while taxpayers are responsible for the remaining 25%."
)

task1_data = {
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
        "text": task1_text
    }
}

# =========================
# TASK 2 INPUT (VALID LENGTH)
# =========================
task2_text = (
    "Music plays an important role in people's lives for various reasons. It provides entertainment, "
    "helps individuals relax, and can influence emotions such as happiness or calmness.\n\n"
    "Traditional music is often valued because it represents the culture and history of a country. "
    "It allows people to understand their roots and preserves cultural identity for future generations. "
    "On the other hand, international music has become increasingly popular due to globalisation and "
    "modern technology, making it accessible to people around the world.\n\n"
    "In my opinion, both traditional and international music are equally important. Traditional music "
    "maintains cultural heritage, while international music encourages cultural exchange and global "
    "connection. Therefore, a balance between the two forms of music is beneficial for society."
)

task2_data = {
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
        "text": task2_text
    }
}

# =========================
# INTERNAL EVALUATION
# =========================
task1_result = evaluate_attempt(task1_data)
task2_result = evaluate_attempt(task2_data)

task1_band = task1_result.get("overall_band", 5.0)
task2_band = task2_result.get("overall_band", 5.0)

# =========================
# OVERALL WRITING BAND
# =========================
overall_writing_band = round_band(
    (task1_band * 1 / 3) +
    (task2_band * 2 / 3)
)

# =========================
# FINAL OUTPUT
# =========================
final_output = {
    "module": "writing",
    "task_1": task1_result,
    "task_2": task2_result,
    "overall_writing_band": overall_writing_band
}

print(json.dumps(final_output, indent=2, ensure_ascii=False))
