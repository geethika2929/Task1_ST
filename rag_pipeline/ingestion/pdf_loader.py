from pathlib import Path
from typing import List
from pydantic import BaseModel
import re

from PyPDF2 import PdfReader
from rag_pipeline.models import Document


class PDFIngestor(BaseModel):
    """
    Scans a directory for PDFs and returns extracted text for each file.
    You can extend this later for .txt, .md, etc.
    """

    def load_all_pdfs(self, folder: Path) -> List[Document]:
        docs: List[Document] = []

        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() == ".pdf":
                text = self._extract_text_from_pdf(path)
                clean_text = self._clean_whitespace(text)
                docs.append(
                    Document(
                        doc_id=path.name,
                        source_path=str(path.resolve()),
                        text=clean_text,
                    )
                )

        return docs

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        reader = PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            # PdfReader returns text with weird linebreaks sometimes
            pages_text.append(page.extract_text() or "")
        return "\n".join(pages_text)

    def _clean_whitespace(self, text: str) -> str:
        # collapse multiple spaces/newlines into something readable
        text = re.sub(r"\r", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # kill huge blank gaps
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
