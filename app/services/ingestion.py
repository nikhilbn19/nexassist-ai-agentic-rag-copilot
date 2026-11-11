import logging
from pathlib import Path
from typing import Iterable
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

def load_file(path: Path) -> list[Document]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")

    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(str(path)).load()
            if not docs or not any(d.page_content.strip() for d in docs):
                raise ValueError("PDF contains no extractable text.")
            return docs
        if suffix in {".txt", ".md"}:
            docs = TextLoader(str(path), encoding="utf-8").load()
            if not docs or not any(d.page_content.strip() for d in docs):
                raise ValueError(f"{suffix} file contains no text.")
            return docs
        if suffix == ".docx":
            doc = DocxDocument(str(path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if not text.strip():
                raise ValueError("DOCX file contains no extractable text.")
            return [Document(page_content=text, metadata={"source": str(path)})]
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Failed to parse file %s: %s", path.name, exc)
        raise ValueError(f"Failed to parse {suffix} document: {exc}") from exc

    raise ValueError(f"Unsupported file type: {suffix}")



def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, add_start_index=True)
    return splitter.split_documents(list(docs))