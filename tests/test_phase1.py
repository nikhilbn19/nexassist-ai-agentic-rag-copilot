import io
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.tools import ToolException
from fastapi.testclient import TestClient

from app.main import app
from app.services.ingestion import load_file
from app.rag.workflow import search_web, grade_kb, grade_web

class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_requirements_file(self):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("langchain-google-genai", content)
        self.assertIn("langchain-community", content)
        self.assertNotIn("langchain-openai", content)
        self.assertNotIn("openai==", content)

    def test_load_empty_file(self):
        tmp_file = Path("tests/empty_test.txt")
        tmp_file.write_text("", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_file(tmp_file)
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    def test_load_corrupted_pdf(self):
        tmp_file = Path("tests/corrupted_test.pdf")
        tmp_file.write_bytes(b"not a real pdf content")
        try:
            with self.assertRaises(ValueError) as ctx:
                load_file(tmp_file)
            self.assertIn("Failed to parse .pdf", str(ctx.exception))
        finally:
            if tmp_file.exists():
                tmp_file.unlink()

    def test_search_web_handles_tool_exception(self):
        state = {
            "question": "xyz",
            "current_query": "xyz",
            "trace": [],
            "citations": []
        }
        with patch("app.rag.workflow.web_search_tool") as mock_tool:
            mock_instance = MagicMock()
            mock_instance.invoke.side_effect = ToolException("No search results found")
            mock_tool.return_value = mock_instance

            res = search_web(state)
            self.assertEqual(res["web_results"], "")
            self.assertEqual(res["citations"], [])
            self.assertTrue(any("no results found" in t for t in res["trace"]))

    def test_search_web_handles_api_error_dict(self):
        state = {
            "question": "xyz",
            "current_query": "xyz",
            "trace": [],
            "citations": []
        }
        with patch("app.rag.workflow.web_search_tool") as mock_tool:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = {"error": "401 Unauthorized"}
            mock_tool.return_value = mock_instance

            res = search_web(state)
            self.assertEqual(res["web_results"], "")
            self.assertEqual(res["citations"], [])
            self.assertTrue(any("unavailable" in t for t in res["trace"]))

    def test_grade_kb_short_circuits_on_empty_docs(self):
        state = {
            "question": "test question",
            "kb_docs": [],
            "trace": []
        }
        with patch("app.rag.workflow.llm") as mock_llm:
            res = grade_kb(state)
            mock_llm.assert_not_called()
            self.assertEqual(res["kb_grade"], "weak")

    def test_grade_web_short_circuits_on_empty_web_results(self):
        state = {
            "question": "test question",
            "web_results": "",
            "trace": []
        }
        with patch("app.rag.workflow.llm") as mock_llm:
            res = grade_web(state)
            mock_llm.assert_not_called()
            self.assertEqual(res["web_grade"], "weak")

    def test_ingest_corrupted_file_returns_400(self):
        fake_pdf = io.BytesIO(b"garbage not pdf")
        response = self.client.post(
            "/api/ingest",
            headers={"x-admin-key": "change-me"},
            files={"file": ("corrupted.pdf", fake_pdf, "application/pdf")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Failed to parse", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
