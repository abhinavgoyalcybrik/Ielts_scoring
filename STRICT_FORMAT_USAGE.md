# IELTS Writing Evaluator - Strict A-F Format Guide

## Overview
The evaluator now supports the **strict A-F output format** as specified. This format is mandatory for production outputs.

## Sections Enforced

### A. OVERALL RESULT
Contains:
- `task_1_band`: Task 1 overall band (0-9, 0.5 increments)
- `task_2_band`: Task 2 overall band (0-9, 0.5 increments)
- `overall_writing_band`: Combined band (Task 1 = 1/3, Task 2 = 2/3)

### B. CRITERIA BREAKDOWN (Mandatory)
For **EACH TASK**, includes ALL 4 criteria:
1. **Task Response** (or Task Achievement for Task 1)
   - `band`: 0-9 numeric score
   - `reason`: 1-line explanation of score
2. **Coherence & Cohesion**
   - `band`: 0-9 numeric score
   - `reason`: 1-line explanation of score
3. **Lexical Resource**
   - `band`: 0-9 numeric score
   - `reason`: 1-line explanation of score
4. **Grammar Accuracy**
   - `band`: 0-9 numeric score
   - `reason`: 1-line explanation of score

All reasons are contextual and adaptive to the score level.

### C. ERRORS FOUND
- Minimum 1 error per task
- Maximum 10 errors total per task
- Each error includes:
  - `sentence`: The text containing the error
  - `error_type`: Type of error (grammar, coherence, vocabulary, punctuation, etc.)
  - `problem`: 1-line explanation of why it's an error
  - `correction`: Suggested correction

### D. REFINED ANSWER
- Task 1: 170-190 words (IELTS official requirement minimum is 150)
- Task 2: 260-300 words (IELTS official requirement minimum is 250)
- Maintains original meaning and arguments
- Fixes only critical errors and typing noise

### E. USEFUL VOCABULARY
**Section title MUST be exactly:** `Useful Vocabulary`

Contains 12-20 words:
- Each word with a 2-4 word usage hint
- Topic-relevant for IELTS Writing
- No explanations, only direct hints

### F. HOW THESE WORDS IMPROVE THE ANSWER
**Section title MUST be exactly:** `How These Words Improve the Answer`

Contains exactly 3 bullets covering:
1. **Lexical Resource**: How vocabulary diversity improves scoring
2. **Task Response**: How word choice clarifies task understanding
3. **Coherence**: How vocabulary prevents repetition and improves flow

---

## Usage Instructions

### Method 1: For Combined Task Evaluation (Recommended)

```python
from evaluator import format_writing_strict, evaluate_attempt

# Evaluate Task 1
task_1_data = {
    "test_type": "writing",
    "metadata": {
        "task_type": "task_1",
        "question": "The charts below show..."
    },
    "user_answers": {
        "text": "The charts show why adults decide to study..."
    }
}
task_1_result = evaluate_attempt(task_1_data)

# Evaluate Task 2
task_2_data = {
    "test_type": "writing",
    "metadata": {
        "task_type": "task_2",
        "question": "Some people think..."
    },
    "user_answers": {
        "text": "In my opinion, both types of music are important..."
    }
}
task_2_result = evaluate_attempt(task_2_data)

# Format to strict A-F format
formatted_output = format_writing_strict(task_1_result, task_2_result)

# Output structure
print(formatted_output)
# Keys: A_OVERALL_RESULT, B_CRITERIA_BREAKDOWN, C_ERRORS_FOUND, 
#       D_REFINED_ANSWER, E_USEFUL_VOCABULARY, F_HOW_WORDS_IMPROVE
```

---

## Output Structure (JSON)

```json
{
  "status": "valid",
  "strict_format": true,
  "A_OVERALL_RESULT": {
    "section": "A. OVERALL RESULT",
    "task_1_band": 6.5,
    "task_2_band": 7.0,
    "overall_writing_band": 6.8
  },
  "B_CRITERIA_BREAKDOWN": {
    "section": "B. CRITERIA BREAKDOWN",
    "task_1": {
      "Task Response": {
        "band": 6.5,
        "reason": "Addresses task with some clarity but lacks fully developed ideas"
      },
      "Coherence & Cohesion": {
        "band": 7.0,
        "reason": "Generally logical organisation with mostly clear progression"
      },
      "Lexical Resource": {
        "band": 6.5,
        "reason": "Adequate vocabulary but with some repetition"
      },
      "Grammar Accuracy": {
        "band": 6.0,
        "reason": "Accurate in simple structures with inconsistent complex use"
      }
    },
    "task_2": {
      "Task Response": {
        "band": 7.0,
        "reason": "Addresses task with clear position and adequate examples"
      },
      "Coherence & Cohesion": {
        "band": 7.0,
        "reason": "Generally logical organisation with mostly clear progression"
      },
      "Lexical Resource": {
        "band": 7.0,
        "reason": "Good range of vocabulary with appropriate word choice"
      },
      "Grammar Accuracy": {
        "band": 7.0,
        "reason": "Mostly accurate with good range of complex structures"
      }
    }
  },
  "C_ERRORS_FOUND": {
    "section": "C. ERRORS FOUND",
    "task_1": [
      {
        "index": 1,
        "sentence": "The charts show why adults decide to studying.",
        "error_type": "grammar",
        "problem": "Incorrect verb form after 'to'",
        "correction": "Change 'studying' to 'study'"
      }
    ],
    "task_2": []
  },
  "D_REFINED_ANSWER": {
    "section": "D. REFINED ANSWER",
    "task_1": "The charts demonstrate why adults pursue further education and how costs should be distributed. The bar chart reveals that interest in the subject is the primary motivation at 40%, followed closely by qualification acquisition at 38%. Employment advancement and social engagement represent smaller percentages. The pie chart indicates that individuals bear the largest cost burden at 40%, with employers contributing 35% and taxpayers providing 25%.",
    "task_2": "In my view, both traditional and contemporary music serve important roles in society. Traditional music preserves cultural heritage and historical awareness, maintaining connections to our past. International music, however, facilitates global communication and cultural exchange, reflecting modern society. Rather than competing, these genres complement each other. Traditional music grounds us culturally while international music broadens our perspective. Therefore, both deserve equal support and recognition in educational and public spheres."
  },
  "E_USEFUL_VOCABULARY": {
    "section": "Useful Vocabulary",
    "vocabulary_list": [
      {
        "word": "demonstrate",
        "usage_hint": "Show clearly with evidence"
      },
      {
        "word": "depict",
        "usage_hint": "Illustrate data or trends"
      },
      {
        "word": "proportion",
        "usage_hint": "Share or percentage"
      }
    ]
  },
  "F_HOW_WORDS_IMPROVE": {
    "section": "How These Words Improve the Answer",
    "improvements": [
      {
        "bullet": 1,
        "focus": "Lexical Resource",
        "description": "These words expand your vocabulary range from basic to advanced level, allowing you to express ideas with precision and variety across different topics."
      },
      {
        "bullet": 2,
        "focus": "Task Response",
        "description": "Topic-specific vocabulary enables you to address task requirements more directly and demonstrate clear understanding of the question."
      },
      {
        "bullet": 3,
        "focus": "Coherence",
        "description": "Using varied and accurate vocabulary reduces repetition and improves overall readability, making your arguments flow more naturally."
      }
    ]
  }
}
```

---

## Validation Rules

Before returning output, the evaluator validates:
- ✅ All sections A-F are present
- ✅ Section B includes criteria with reasons
- ✅ Section E has 12-20 words
- ✅ Section E has exact title "Useful Vocabulary"
- ✅ Section F has exact title "How These Words Improve the Answer"
- ✅ Section F has exactly 3 bullets
- ✅ All errors have required fields

**If ANY validation fails → Output is marked INVALID**

---

## Version History
- **v1.0** (Current): Full A-F strict format implementation with mandatory validations
