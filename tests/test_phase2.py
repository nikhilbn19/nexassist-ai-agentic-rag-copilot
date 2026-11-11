import io
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.rag.workflow import generate_from_kb
from app.core.rate_limit import InMemoryRateLimiter

class Phase2Tests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_citation_path_disclosure_fixed(self):
        # Test that Windows backslash path is cleanly converted to file basename
        mock_doc = MagicMock()
        mock_doc.metadata = {"source": "D:\\company\\internal\\secret_path\\it_handbook.pdf"}
        mock_doc.page_content = "VPN instructions content."

        state = {
            "question": "How to connect VPN?",
            "kb_docs": [mock_doc],
            "trace": []
        }

        with patch("app.rag.workflow.llm") as mock_llm:
            mock_resp = MagicMock()
            mock_resp.content = "Use the VPN instructions."
            mock_llm.return_value.invoke.return_value = mock_resp

            result = generate_from_kb(state)
            citations = result.get("citations", [])
            self.assertEqual(len(citations), 1)
            # The title must be ONLY the basename, not the Windows filesystem path
            self.assertEqual(citations[0]["title"], "it_handbook.pdf")
            self.assertNotIn("D:\\", citations[0]["title"])
            self.assertNotIn("secret_path", citations[0]["title"])

    def test_admin_auth_invalid_key_rejected(self):
        fake_txt = io.BytesIO(b"Valid content")
        response = self.client.post(
            "/api/ingest",
            headers={"x-admin-key": "wrong-key"},
            files={"file": ("test.txt", fake_txt, "text/plain")}
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid admin key")

    def test_admin_auth_empty_key_rejected(self):
        fake_txt = io.BytesIO(b"Valid content")
        response = self.client.post(
            "/api/ingest",
            headers={"x-admin-key": ""},
            files={"file": ("test.txt", fake_txt, "text/plain")}
        )
        self.assertEqual(response.status_code, 401)

    def test_oversized_upload_rejected(self):
        # Create a stream that exceeds MAX_UPLOAD_SIZE (15MB + 1KB)
        oversized_data = b"0" * (15 * 1024 * 1024 + 1024)
        response = self.client.post(
            "/api/ingest",
            headers={"x-admin-key": "change-me"},
            files={"file": ("large.txt", io.BytesIO(oversized_data), "text/plain")}
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("File exceeds maximum upload size", response.json()["detail"])

    def test_cors_middleware_configured(self):
        response = self.client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")

    def test_rate_limiter_triggers_429(self):
        limiter = InMemoryRateLimiter(requests_per_minute=2)
        mock_req = MagicMock()
        mock_req.client.host = "192.168.1.50"

        # 2 requests succeed
        limiter.check(mock_req)
        limiter.check(mock_req)

        # 3rd request raises 429
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            limiter.check(mock_req)
        self.assertEqual(ctx.exception.status_code, 429)

if __name__ == "__main__":
    unittest.main()
