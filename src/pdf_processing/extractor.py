"""PDF extraction pipeline: text, layout, tables, and page images.

Images are rendered LAZILY (on demand) to keep uploads fast.
"""

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
import io
import logging

logger = logging.getLogger(__name__)

# Lower DPI for speed on Mac — still readable
DEFAULT_DPI = 120


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    has_charts: bool = False
    has_tables: bool = False
    width: float = 0.0
    height: float = 0.0


@dataclass
class DocumentContent:
    """Represents the full extracted content from a PDF document."""
    file_path: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PDFExtractor:
    """Extracts text, tables, and layout info from PDF documents.

    Page images are NOT rendered at upload time — use render_page_image()
    to get them on demand (when viewing or when vision model needs them).
    """

    def __init__(self, dpi: int = DEFAULT_DPI, extract_tables: bool = True,
                 ocr_fallback: bool = True):
        self.dpi = dpi
        self.extract_tables = extract_tables
        self.ocr_fallback = ocr_fallback

    def extract(self, pdf_path: str, max_pages: int | None = None) -> DocumentContent:
        """Extract text and tables from a PDF (fast — no image rendering)."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        metadata = doc.metadata or {}

        document = DocumentContent(
            file_path=str(pdf_path),
            total_pages=total_pages,
            metadata=metadata,
        )

        plumber_doc = pdfplumber.open(str(pdf_path)) if self.extract_tables else None

        for page_idx in range(pages_to_process):
            page_content = self._extract_page(doc, page_idx, plumber_doc)
            document.pages.append(page_content)

        doc.close()
        if plumber_doc:
            plumber_doc.close()

        logger.info(f"Extracted {len(document.pages)} pages from {pdf_path.name}")
        return document

    def render_page_image(self, pdf_path: str, page_number: int) -> Image.Image | None:
        """Render a single page as an image ON DEMAND."""
        try:
            doc = fitz.open(pdf_path)
            if page_number < 1 or page_number > len(doc):
                doc.close()
                return None
            page = doc[page_number - 1]
            zoom = self.dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)
            img_data = pixmap.tobytes("png")
            doc.close()
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            logger.warning(f"Failed to render page {page_number}: {e}")
            return None

    def _extract_page(self, doc: fitz.Document, page_idx: int,
                      plumber_doc=None) -> PageContent:
        """Extract text content from a single page (no image rendering)."""
        page = doc[page_idx]
        rect = page.rect

        text = page.get_text("text")

        # OCR fallback for scanned pages with very little text
        if self.ocr_fallback and len(text.strip()) < 50:
            ocr_text = self._ocr_page(page)
            if ocr_text:
                text = ocr_text

        # Extract tables via pdfplumber
        tables = []
        has_tables = False
        if plumber_doc and self.extract_tables:
            try:
                plumber_page = plumber_doc.pages[page_idx]
                raw_tables = plumber_page.extract_tables()
                if raw_tables:
                    tables = [
                        [[cell or "" for cell in row] for row in table]
                        for table in raw_tables
                    ]
                    has_tables = True
            except Exception as e:
                logger.warning(f"Table extraction failed on page {page_idx + 1}: {e}")

        has_charts = self._detect_charts(page)

        return PageContent(
            page_number=page_idx + 1,
            text=text,
            tables=tables,
            has_charts=has_charts,
            has_tables=has_tables,
            width=rect.width,
            height=rect.height,
        )

    def _ocr_page(self, page: fitz.Page) -> str | None:
        """Run OCR on a page image using Tesseract."""
        try:
            import pytesseract
            zoom = self.dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            return pytesseract.image_to_string(img)
        except ImportError:
            logger.warning("pytesseract not installed, skipping OCR")
            return None
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None

    def _detect_charts(self, page: fitz.Page) -> bool:
        """Heuristic: detect if a page likely contains charts/figures."""
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                if base_image and base_image.get("width", 0) > 100 and base_image.get("height", 0) > 100:
                    return True
            except Exception:
                continue
        return False
