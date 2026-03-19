"""Sentence-aware recursive chunking for document retrieval.

Implements recommendations:
- Preserves sentence boundaries (never splits mid-sentence)
- Recursive splitting: paragraphs → sentences → word-level fallback
- Tables chunked separately with headers preserved
- Metadata-aware: page number, chunk type, visual flags
"""

import re
from dataclasses import dataclass
from .extractor import PageContent, DocumentContent


@dataclass
class Chunk:
    """A retrievable segment of a document."""
    chunk_id: str
    page_number: int
    text: str
    chunk_type: str  # "text", "table", "mixed"
    metadata: dict


# Sentence boundary pattern — handles abbreviations reasonably
SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving boundaries."""
    sentences = SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


class DocumentChunker:
    """Sentence-aware recursive chunker.

    Strategy:
    1. Split page text on paragraph boundaries (double newline)
    2. If a paragraph exceeds chunk_size, split into sentences
    3. If a sentence exceeds chunk_size, split by words (fallback)
    4. Group segments into chunks of target size with overlap
    5. Tables get their own chunks (self-contained)
    """

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: DocumentContent) -> list[Chunk]:
        """Split a full document into chunks."""
        chunks = []
        for page in document.pages:
            page_chunks = self._chunk_page(page, document.file_path)
            chunks.extend(page_chunks)
        return chunks

    def _chunk_page(self, page: PageContent, file_path: str) -> list[Chunk]:
        """Split a single page into chunks."""
        chunks = []
        base_meta = {
            "file_path": file_path,
            "page_number": page.page_number,
            "has_charts": page.has_charts,
            "has_tables": page.has_tables,
        }

        # Tables get their own chunks
        for t_idx, table in enumerate(page.tables):
            table_text = self._table_to_text(table)
            if table_text.strip():
                chunks.append(Chunk(
                    chunk_id=f"p{page.page_number}_table_{t_idx}",
                    page_number=page.page_number,
                    text=table_text,
                    chunk_type="table",
                    metadata={**base_meta, "table_index": t_idx},
                ))

        # Recursive text splitting
        text = page.text.strip()
        if not text:
            return chunks

        segments = self._recursive_split(text)
        text_chunks = self._group_segments(segments, page.page_number, base_meta)
        chunks.extend(text_chunks)

        return chunks

    def _clean_financial_text(self, text: str) -> str:
        """Clean messy financial table text from PDF extraction.

        Raw PDF extraction often gives lines like:
            iPhone
            $
            201,183
            $
            200,583
        We restructure this into: iPhone $201,183 $200,583
        """
        lines = text.split('\n')
        cleaned = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                cleaned.append("")
                i += 1
                continue

            # Check if this looks like a label followed by $ signs and numbers
            # on subsequent lines (common in SEC filings)
            if (i + 1 < len(lines) and
                    lines[i + 1].strip() in ('$', '(', ')') or
                    (i + 1 < len(lines) and
                     re.match(r'^[\$\(\)]\s*$', lines[i + 1].strip()))):
                # Collect the label and all following value lines
                combined = line
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    # Keep combining if it's $, a number, (, ), or empty
                    if (next_line in ('$', '(', ')') or
                            re.match(r'^[\$\(\)]?\s*[\d,]+\s*[\)\%]?\s*$', next_line) or
                            re.match(r'^\(\d', next_line)):
                        combined += " " + next_line
                        j += 1
                    else:
                        break
                # Clean up extra spaces around $ and parentheses
                combined = re.sub(r'\$\s+', '$', combined)
                combined = re.sub(r'\(\s+', '(', combined)
                combined = re.sub(r'\s+\)', ')', combined)
                combined = re.sub(r'\s{2,}', '  ', combined)
                cleaned.append(combined)
                i = j
            else:
                cleaned.append(line)
                i += 1

        return '\n'.join(cleaned)

    def _recursive_split(self, text: str) -> list[str]:
        """Recursively split text: paragraphs → sentences → words."""
        # Level 1: Split on paragraph boundaries
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If only one block and it's long, split on single newlines
        if len(paragraphs) == 1 and len(paragraphs[0].split()) > self.chunk_size:
            paragraphs = [line.strip() for line in text.split('\n') if line.strip()]

        segments = []
        for para in paragraphs:
            word_count = len(para.split())
            if word_count <= self.chunk_size:
                segments.append(para)
            else:
                # Level 2: Split into sentences
                sentences = _split_sentences(para)
                for sent in sentences:
                    if len(sent.split()) <= self.chunk_size:
                        segments.append(sent)
                    else:
                        # Level 3: Word-level fallback
                        words = sent.split()
                        for i in range(0, len(words), self.chunk_size):
                            segments.append(" ".join(words[i:i + self.chunk_size]))

        return segments

    def _group_segments(self, segments: list[str], page_number: int,
                        base_meta: dict) -> list[Chunk]:
        """Group segments into chunks of target size with overlap."""
        chunks = []
        current_segments = []
        current_words = 0
        chunk_idx = 0

        for seg in segments:
            seg_words = len(seg.split())

            if current_words + seg_words > self.chunk_size and current_segments:
                # Flush current chunk
                chunk_text = "\n".join(current_segments)
                chunks.append(Chunk(
                    chunk_id=f"p{page_number}_text_{chunk_idx}",
                    page_number=page_number,
                    text=chunk_text,
                    chunk_type="text",
                    metadata=base_meta,
                ))
                chunk_idx += 1

                # Overlap: keep last segment(s) up to overlap word count
                overlap_segments = []
                overlap_words = 0
                for s in reversed(current_segments):
                    s_words = len(s.split())
                    if overlap_words + s_words > self.chunk_overlap:
                        break
                    overlap_segments.insert(0, s)
                    overlap_words += s_words

                current_segments = overlap_segments
                current_words = overlap_words

            current_segments.append(seg)
            current_words += seg_words

        # Flush remaining
        if current_segments:
            chunk_text = "\n".join(current_segments)
            chunks.append(Chunk(
                chunk_id=f"p{page_number}_text_{chunk_idx}",
                page_number=page_number,
                text=chunk_text,
                chunk_type="text",
                metadata=base_meta,
            ))

        return chunks

    def _table_to_text(self, table: list[list[str]]) -> str:
        """Convert a table to a readable text representation with clear labels."""
        if not table:
            return ""

        # Extract headers from first row
        headers = [cell.strip() for cell in table[0]] if table else []

        rows = []
        for i, row in enumerate(table):
            cells = [cell.strip() for cell in row]
            if i == 0:
                non_empty = [c for c in cells if c]
                if non_empty:
                    rows.append("Headers: " + " | ".join(non_empty))
            else:
                # Associate each cell with its column header for clarity
                if headers and len(cells) == len(headers):
                    labeled_cells = []
                    for header, cell in zip(headers, cells):
                        if cell:
                            if header:
                                labeled_cells.append(f"{header}: {cell}")
                            else:
                                labeled_cells.append(cell)
                    if labeled_cells:
                        rows.append(" | ".join(labeled_cells))
                else:
                    non_empty = [c for c in cells if c]
                    if non_empty:
                        rows.append(" | ".join(non_empty))
        return "\n".join(rows)
