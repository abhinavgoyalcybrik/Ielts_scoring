"""
Manual test helper - NOT part of the API or evaluator.

Usage:
    venv/Scripts/python.exe test_task1_with_image.py "C:\\path\\to\\chart.png"

Uploads the given image to /writing/task1-image-to-data-url, then calls
/writing/evaluate with the returned data URL as task_1.image_url, so you
never have to manually copy/paste the base64 string yourself. Edit the
question/answer/task_2 text below to match whatever you're testing.
"""
import sys
import json
import mimetypes
import uuid
import urllib.request

BASE_URL = "http://127.0.0.1:8000"

QUESTION_TASK1 = (
    "The charts below show the average percentages in typical meals of "
    "three types of nutrients, all of which may be unhealthy if eaten too "
    "much. Summarise the information by selecting and reporting the main "
    "features, and make comparisons where relevant."
)

ANSWER_TASK1 = (
    "The diagrams illustrate the average proportions of three types of nutrients in "
    "typical meals, which can be unhealthy if consumed too much. The three types "
    "include sodium, saturated fats and added sugar. The data is taken from the "
    "United States of America. The first chart shows the average percentages of "
    "sodium. Dinner contains the most sodium (43%). Breakfast and snacks include an "
    "equal proportion of sodium consumed, with each of them adding up 14% of "
    "sodium. Through eating lunch, 29% sodium is consumed. The second chart shows "
    "the percentages of saturated fat in meals. By eating dinner, 37% saturated "
    "fat is consumed. Lunch contributes to a consumption of 26% saturated fat, "
    "followed by snacks with 21% and breakfast with 16%. The last chart illustrates "
    "the proportions of added sugar. Snacks contain the highest amount of added "
    "sugar (42%). Dinner includes 23%. A typical dinner includes 23% added sugar, "
    "while lunch contains 19% and breakfast includes 16%. All in all, the diagrams "
    "show that every typical meal consumed in the USA contains a percentage of at "
    "least 14% of nutrients that can be unhealthy if eaten too much."
)

QUESTION_TASK2 = "Some people believe technology makes life easier. Discuss both views."
ANSWER_TASK2 = (
    "Technology has changed the way people live and work. Some argue it "
    "simplifies daily tasks, while others believe it creates new problems. In my "
    "opinion, the benefits outweigh the drawbacks because technology saves time "
    "and increases access to information. However, overreliance on technology "
    "can reduce face to face interaction. In conclusion, a balanced approach to "
    "technology use is the most sensible path forward for modern society."
)


def upload_image(path):
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        raw = f.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="chart.png"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + raw + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/writing/task1-image-to-data-url",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))["image_url"]


def evaluate(image_url):
    payload = {
        "task_1": {"question": QUESTION_TASK1, "answer": ANSWER_TASK1, "image_url": image_url},
        "task_2": {"question": QUESTION_TASK2, "answer": ANSWER_TASK2},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/writing/evaluate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_task1_with_image.py <path-to-chart-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Uploading {image_path} ...")
    data_url = upload_image(image_path)
    print(f"Got data URL ({len(data_url)} chars). Evaluating...")

    result = evaluate(data_url)
    t1 = result["tasks"]["task_1"]
    print(json.dumps({
        "ai_evaluation_failed": t1.get("ai_evaluation_failed"),
        "image_verification_used": t1.get("image_verification_used"),
        "image_data_accuracy": t1.get("image_data_accuracy"),
        "overall_band": t1.get("overall_band"),
        "criteria_scores": t1.get("criteria_scores"),
        "mistakes_count": len(t1.get("mistakes", [])),
    }, indent=2))
