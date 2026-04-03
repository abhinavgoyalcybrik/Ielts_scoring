from collections import defaultdict

# attempt_id -> { "parts": {1: {...}, 2: {...}, 3: {...}} }
SPEAKING_ATTEMPTS = defaultdict(lambda: {"parts": {}})
