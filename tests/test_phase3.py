import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.rag.workflow import insufficient, extract_text, grade_kb
from app.rag.vectorstore import add_documents

class Phase3Tests(unittest.TestCase):

    def test_insufficient_clears_stale_citations(self):
        
        stale_citations = [
            {"title": "Irrelevant Web Link", "url": "https://example.com", "type": "web"}
        ]
        state = {
            "question": "Impossible IT question",
            "citations": stale_citations,
            "trace": []
        }
        res = insufficient(state)
        self.assertEqual(res["citations"], [])
        self.assertEqual(res["source_used"], "insufficient_evidence")
        self.assertIn("couldn't find enough reliable evidence", res["answer"])

    def test_extract_text_handles_str_and_list_types(self):
        
        self.assertEqual(extract_text("Normal response"), "Normal response")

        
        self.assertEqual(extract_text(["Part 1 ", "Part 2"]), "Part 1 Part 2")

        
        blocks = [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "World"}
        ]
        self.assertEqual(extract_text(blocks), "Hello World")

        
        self.assertEqual(extract_text(12345), "12345")

    def test_deterministic_document_ids_in_vectorstore(self):
        chunks = [
            Document(page_content="Policy 1: VPN must be used.", metadata={"source": "handbook.md"}),
            Document(page_content="Policy 2: Passwords must be 14 chars.", metadata={"source": "handbook.md"})
        ]

        with patch("app.rag.vectorstore.get_vectorstore") as mock_vs:
            mock_store = MagicMock()
            mock_vs.return_value = mock_store

            add_documents(chunks)

            
            self.assertTrue(mock_store.add_documents.called)
            call_kwargs = mock_store.add_documents.call_args
            ids = call_kwargs[1].get("ids") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs[1].get("ids")
            self.assertIsNotNone(ids)
            self.assertEqual(len(ids), 2)
            self.assertTrue(ids[0].startswith("handbook.md_0_"))
            self.assertTrue(ids[1].startswith("handbook.md_1_"))

            
            mock_store.reset_mock()
            add_documents(chunks)
            call_kwargs2 = mock_store.add_documents.call_args
            ids2 = call_kwargs2[1].get("ids")
            self.assertEqual(ids, ids2)

    def test_grade_kb_short_circuits_without_llm(self):
        state = {
            "question": "VPN question",
            "kb_docs": [],
            "trace": []
        }
        with patch("app.rag.workflow.llm") as mock_llm:
            res = grade_kb(state)
            mock_llm.assert_not_called()
            self.assertEqual(res["kb_grade"], "weak")

if __name__ == "__main__":
    unittest.main()

