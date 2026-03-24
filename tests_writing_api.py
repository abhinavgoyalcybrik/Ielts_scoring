from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from evaluators.api import writing as writing_api


class WritingApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(writing_api.router)
        cls.client = TestClient(app)

    def test_rejects_missing_task_2(self):
        response = self.client.post("/writing/evaluate", json={"task_1": {"question": "Q1", "answer": "A1"}})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json().get("detail"), "task_2 is required")

    def test_rejects_empty_task_2_answer(self):
        response = self.client.post(
            "/writing/evaluate",
            json={"task_2": {"question": "Q2", "answer": "   "}},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json().get("detail"), "task_2.answer is required")

    @patch("evaluators.api.writing.evaluate_writing")
    def test_accepts_task_2_response_alias(self, mock_evaluate_writing):
        mock_evaluate_writing.return_value = {"overall_band": 6.5, "criteria_scores": {}}

        response = self.client.post(
            "/writing/evaluate",
            json={"task_2": {"question": "Q2", "response": "Task 2 answer text"}},
        )

        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.json().get("overall_writing_band"), 6.5)
        called_payload = mock_evaluate_writing.call_args[0][0]
        self.assertEqual(called_payload["user_answers"]["text"], "Task 2 answer text")

    @patch("evaluators.api.writing.evaluate_writing")
    def test_maps_essay_missing_value_error_to_422(self, mock_evaluate_writing):
        mock_evaluate_writing.side_effect = ValueError("Essay text missing")

        response = self.client.post(
            "/writing/evaluate",
            json={"task_2": {"question": "Q2", "answer": "valid text"}},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json().get("detail"), "task_2.answer is required")


if __name__ == "__main__":
    unittest.main()
