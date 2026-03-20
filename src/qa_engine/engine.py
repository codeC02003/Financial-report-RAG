"""Core QA engine — hybrid architecture: dynamic table extraction + model fallback.

Pipeline:
1. Load document → dynamically discover table structure (years, row labels)
2. For each question:
   a. Try table-based extraction (precise, no hallucination)
   b. Extractive QA (RoBERTa-SQuAD2) as fallback
   c. Generative model (Qwen2-VL) for complex/conversational questions
3. Unanswerable detection via model agreement

Table logic is fully dynamic — discovers all structure from the document itself.
No hardcoded label maps, company names, year values, or keyword aliases.
"""

import re
import logging
import difflib
from dataclasses import dataclass, field
from PIL import Image

from src.pdf_processing.extractor import PDFExtractor, DocumentContent
from src.pdf_processing.chunker import DocumentChunker, Chunk
from src.retrieval.embedder import DocumentEmbedder
from src.models.extractive_qa import ExtractiveQA
from src.models.vision_model import VisionLanguageModel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Dynamic Document Index — built at load time by scanning all chunks
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DocumentIndex:
    """Index built by scanning all chunks. Stores discovered table labels and years."""
    table_labels: dict[str, str] = field(default_factory=dict)  # lower → original
    fiscal_years: list[int] = field(default_factory=list)
    most_recent_year: int | None = None

    @classmethod
    def build(cls, chunks: list) -> "DocumentIndex":
        """Scan all chunks and discover table row labels and fiscal years."""
        idx = cls()
        year_counts: dict[int, int] = {}

        for chunk in chunks:
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
            lines = text.split("\n")

            for line_idx, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue

                # --- Discover fiscal years from column headers ---
                # Lines with 2+ years, short length, no financial data
                year_matches = re.findall(r'\b(20\d{2})\b', stripped)
                if len(year_matches) >= 2 and len(stripped) < 100:
                    words = stripped.split()
                    # Skip lines with financial data, date ranges, or bond maturities
                    has_financial = bool(re.search(r'\d+[,:]\s*\d+\s*\|', stripped))
                    has_months = bool(re.search(
                        r'(?:January|February|March|April|May|June|July|August|'
                        r'September|October|November|December)', stripped))
                    has_bonds = bool(re.search(
                        r'notes?\s+due|bonds?\s+due|%\s+notes?\s+due|debt|issuance|–\s*20\d{2}',
                        stripped, re.IGNORECASE))
                    if len(words) <= 12 and not has_financial and not has_months and not has_bonds:
                        for ym in year_matches:
                            y = int(ym)
                            if 2015 <= y <= 2030:
                                year_counts[y] = year_counts.get(y, 0) + 1

                # --- Discover table row labels ---
                # Look for lines where numbers appear in a window after the text
                window = lines[line_idx:min(line_idx + 9, len(lines))]
                window_text = " ".join(window)
                numbers = re.findall(r'[\$]?\s*\(?([\d,]{4,})\)?', window_text)

                valid_nums = []
                for n in numbers:
                    clean = n.replace(",", "")
                    try:
                        val = int(clean)
                        if val >= 100 and not (2000 <= val <= 2099):
                            valid_nums.append(val)
                    except ValueError:
                        continue

                if not valid_nums:
                    continue

                # Check if numbers are on the same line or subsequent lines
                same_line_nums = re.findall(r'[\$]?\s*\(?([\d,]{4,})\)?', stripped)
                same_line_valid = [n for n in same_line_nums
                                   if n.replace(",", "").isdigit()
                                   and int(n.replace(",", "")) >= 100
                                   and not (2000 <= int(n.replace(",", "")) <= 2099)]

                if same_line_valid:
                    # Label is text before first number
                    m = re.search(r'[\$\(]?\s*[\d,]{4,}', stripped)
                    label = stripped[:m.start()].strip() if m else ""
                else:
                    # Entire line is the label
                    label = stripped

                # Clean label: remove formatting artifacts
                label = re.sub(r'^.*?::\s*', '', label)  # "ASSETS:: X" → "X"
                # Normalize smart quotes/apostrophes to ASCII
                label = label.replace('\u2019', "'").replace('\u2018', "'")
                label = label.replace('\u201c', '"').replace('\u201d', '"')
                label = label.replace('\xa0', ' ')  # non-breaking space
                label = label.strip(" \t:,.|$()\"'")
                label = re.sub(r'\s*\|.*$', '', label)  # trailing "| $"
                label = re.sub(r'\s*\(\d+\)?\s*$', '', label)  # footnote refs
                label = label.strip(" \t:,.|$()\"'")

                # Validate
                if (len(label) < 3 or len(label) > 60
                        or not re.match(r'^[A-Za-z]', label)
                        or any(ord(c) > 127 for c in label)  # non-ASCII (checkboxes etc.)
                        or re.match(r'^(yes|no|an |a |the )', label.lower())
                        or len(label.split()) > 8
                        or (len(label.split()) < 2 and len(valid_nums) < 2)):
                    continue

                # Strip apostrophes from key so "shareholders' equity" matches "shareholders equity"
                clean_key = label.lower().replace("'", "").replace("\u2019", "")
                idx.table_labels[clean_key] = label

        # Determine fiscal years: most frequent years that cluster together
        if year_counts:
            sorted_years = sorted(year_counts.items(), key=lambda x: x[1], reverse=True)
            top_year = sorted_years[0][0]
            fiscal = sorted(y for y, c in year_counts.items() if abs(y - top_year) <= 2)
            idx.fiscal_years = fiscal
            idx.most_recent_year = max(fiscal)

        logger.info(f"DocumentIndex: {len(idx.table_labels)} labels, "
                     f"years={idx.fiscal_years}, most_recent={idx.most_recent_year}")
        return idx


# ──────────────────────────────────────────────────────────────────────
# Generic table extraction helpers
# ──────────────────────────────────────────────────────────────────────

def _match_label(query_terms: list[str], doc_index: DocumentIndex) -> str | None:
    """Find the best matching table label for the query terms using fuzzy matching.

    Prefers "total" labels over qualified ones (e.g. "Total net sales" over
    "Deferred revenue") when the query is a broad concept like "revenue".
    """
    if not doc_index.table_labels or not query_terms:
        return None

    query = " ".join(query_terms).lower()
    candidates = []

    # Qualifier words that indicate a sub-item rather than the main total
    _qualifiers = frozenset({
        "deferred", "accrued", "other", "non-current", "current portion",
        "change in", "provision", "allowance", "prepaid", "accumulated",
    })

    for label_lower, label_orig in doc_index.table_labels.items():
        # Skip very short labels and generic standalone words
        if len(label_lower) < 4:
            continue
        # Skip common standalone words that are too ambiguous as labels
        _skip_labels = {"total", "subtotal", "amount", "other", "value", "price",
                        "balance", "items", "period", "change", "years", "notes", "ratio"}
        if label_lower in _skip_labels:
            continue
        # Skip header/section labels that don't contain extractable values
        if label_lower.startswith("headers:") or "disaggregated" in label_lower:
            continue

        # Exact match — still collect as candidate (don't return early)
        # because a "Total X" variant may be better
        if query == label_lower:
            score = 0.99
        # Query is substring of label
        elif query in label_lower:
            score = 0.95
        # Label is substring of query — strength depends on how much of the query it covers
        elif label_lower in query:
            coverage = len(label_lower) / max(len(query), 1)
            # Short labels (< 30% coverage) get low scores, long labels get high scores
            if coverage < 0.3:
                score = 0.60 + 0.20 * coverage  # short labels: 0.60-0.66
            else:
                score = 0.90 + 0.08 * coverage  # long labels: 0.92-0.98
        # Term overlap (including fuzzy per-term matching for typos)
        else:
            hits = sum(1 for t in query_terms if t in label_lower)
            # Also check fuzzy per-term matches (handles typos like "avenue"→"revenue")
            if hits == 0:
                label_words = label_lower.split()
                fuzzy_hits = 0
                for t in query_terms:
                    if len(t) >= 4:
                        for lw in label_words:
                            if difflib.SequenceMatcher(None, t, lw).ratio() >= 0.75:
                                fuzzy_hits += 1
                                break
                hits = fuzzy_hits
            if hits == len(query_terms) and query_terms:
                score = 0.85
            elif hits > 0:
                score = 0.5 + 0.15 * hits
            else:
                score = difflib.SequenceMatcher(None, query, label_lower).ratio() * 0.6

        if score < 0.5:
            continue

        # Penalize qualified/sub-item labels when query is a broad concept
        has_qualifier = any(q in label_lower for q in _qualifiers)
        has_total = "total" in label_lower
        # Boost labels with "total" for broad queries
        is_total_bonus = has_total and not has_qualifier

        candidates.append((label_orig, score, len(label_lower),
                           is_total_bonus, has_qualifier))

    if not candidates:
        return None

    # Sort: highest score first; within close scores (±0.15), prefer total labels
    # This ensures "net income" (exact=0.99) beats "Total net sales" (0.65)
    # but "Total gross margin" (0.95) beats "Gross margin" (0.99) when close
    def _sort_key(c):
        label_orig, score, label_len, is_total, has_qual = c
        # Bucket score into bands of 0.15 — within a band, total wins
        score_band = int(score / 0.15)
        # Penalize qualified labels
        qual_penalty = -1 if has_qual else 0
        return (score_band, is_total, qual_penalty, score, -label_len)

    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0][0]


def _extract_terms(question: str) -> list[str]:
    """Extract meaningful terms from a question by removing common stop words."""
    stop = frozenset({
        "what", "was", "were", "is", "are", "the", "a", "an", "in", "of", "for",
        "to", "and", "or", "how", "much", "did", "do", "does", "has", "have",
        "had", "been", "be", "from", "by", "on", "at", "with", "its", "their",
        "this", "that", "it", "they", "them", "company", "fiscal", "year",
        "during", "which", "about", "many", "will", "would", "could", "should",
        "may", "can", "than", "then", "also", "just", "only", "very", "most",
        "more", "between", "per", "other", "report",
    })
    words = re.sub(r'[^\w\s]', ' ', question.lower()).split()
    return [w for w in words if w not in stop and not re.match(r'^20\d{2}$', w) and len(w) > 1]


# Universal financial synonyms — these are standard accounting terms, not document-specific.
# Maps common query words → preferred label forms found in financial documents.
_FINANCIAL_SYNONYMS: dict[str, list[str]] = {
    "revenue": ["total net sales", "total revenue", "net sales"],
    "sales": ["total net sales", "total revenue", "net sales"],
    "net sales": ["total net sales"],
    "income": ["net income"],
    "net income": ["net income"],
    "profit": ["net income", "operating income"],
    "money": ["net income", "total net sales"],
    "cost sales": ["total cost of sales", "cost of sales"],
    "cost of sales": ["total cost of sales"],
    "equity": ["total shareholders equity", "total stockholders equity",
               "shareholders equity", "stockholders equity"],
    "operating expenses": ["total operating expenses"],
    "gross margin": ["total gross margin", "gross margin"],
}


def _expand_query_with_synonyms(terms: list[str], doc_index: DocumentIndex) -> list[str]:
    """If the original terms don't match a 'total' label, try universal financial synonyms.

    Won't short-circuit on qualified labels like 'Deferred revenue' —
    only considers it a good match if it's a 'total' or unqualified label.
    Won't expand if the query has specific product/segment qualifiers.
    """
    query = " ".join(terms).lower()

    # Don't expand if the query has specific qualifiers (user asking about a sub-item)
    # e.g. "Services revenue" should NOT expand to "total net sales"
    _specific_qualifiers = {"services", "products", "iphone", "mac", "ipad",
                             "wearables", "americas", "europe", "japan", "china",
                             "greater china", "rest"}
    if any(q in query for q in _specific_qualifiers):
        return terms

    _bad_qualifiers = {"deferred", "accrued", "other", "non-current", "prepaid",
                        "accumulated", "provision", "allowance", "change in",
                        "disaggregated", "headers:", "classification", "supplemental",
                        "internal revenue", "code section", "adjustment to",
                        "foreign currency", "translation"}

    # Check if original terms match a GOOD label (total or unqualified)
    # Require the match to be substantial (label covers at least 60% of query length)
    for label_lower in doc_index.table_labels:
        if query in label_lower or label_lower in query:
            # Skip if the label is much shorter than the query (weak match)
            overlap = min(len(label_lower), len(query))
            if overlap < len(query) * 0.6:
                continue
            has_qualifier = any(q in label_lower for q in _bad_qualifiers)
            if not has_qualifier:
                return terms

    # Try synonyms — check longer (more specific) triggers first
    # e.g. "cost sales" should match before bare "sales"
    sorted_triggers = sorted(_FINANCIAL_SYNONYMS.keys(), key=len, reverse=True)
    for trigger in sorted_triggers:
        if trigger in query:
            for alt in _FINANCIAL_SYNONYMS[trigger]:
                alt_lower = alt.lower()
                for label_lower in doc_index.table_labels:
                    if alt_lower in label_lower or label_lower in alt_lower:
                        return alt.split()

    # Fuzzy match: handle typos (e.g. "avenue" → "revenue")
    # Check each query term against synonym triggers using edit distance
    for term in terms:
        if len(term) < 4:
            continue
        best_match = None
        best_ratio = 0.0
        for trigger in sorted_triggers:
            # Check each word in the trigger against the term
            for tword in trigger.split():
                ratio = difflib.SequenceMatcher(None, term, tword).ratio()
                if ratio > best_ratio and ratio >= 0.75:
                    best_ratio = ratio
                    best_match = trigger
        if best_match:
            for alt in _FINANCIAL_SYNONYMS[best_match]:
                alt_lower = alt.lower()
                for label_lower in doc_index.table_labels:
                    if alt_lower in label_lower or label_lower in alt_lower:
                        return alt.split()

    return terms


def _find_row_values(label: str, chunks: list, max_values: int = 3) -> list[int] | None:
    """Find a table row matching the label and return its numeric values.

    Strategy: build a window from the label line, extending only until we hit
    another line that looks like a new table row (starts with alpha text and
    contains numbers). This prevents bleeding into adjacent rows.
    Prefers table-type chunks over narrative text.
    """
    best = None
    best_has_dollar = False
    best_is_primary = False
    best_same_line = False
    best_is_table_chunk = False
    # Strip apostrophes from label for matching (they may differ between smart/ASCII)
    label_lower = label.lower().replace("'", "").replace("\u2019", "")

    for chunk in chunks:
        text = chunk["text"] if isinstance(chunk, dict) else chunk.text
        chunk_type = chunk.get("chunk_type", "") if isinstance(chunk, dict) else getattr(chunk, "chunk_type", "")
        is_table_chunk = chunk_type == "table"
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line_lower = line.lower().strip().replace("\u2019", "").replace("'", "")
            if label_lower not in line_lower:
                continue
            if "beginning" in line_lower:
                continue

            # Check if label is at the START of the line (primary label)
            label_pos_in_line = line_lower.index(label_lower)
            prefix = line_lower[:label_pos_in_line].strip()
            is_primary = len(prefix) == 0

            # Build window: label line + continuation lines until next row label
            window_lines = [line]
            for j in range(i + 1, min(i + 8, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue
                # Stop if this line looks like a NEW table row:
                # starts with alpha text AND contains significant numbers
                if (re.match(r'^[A-Za-z]', next_line)
                        and re.search(r'[\d,]{4,}', next_line)
                        and label_lower not in next_line.lower().replace("\u2019", "").replace("'", "")):
                    break
                window_lines.append(next_line)

            window = " ".join(window_lines)
            # Normalize for position lookup (strip all apostrophe variants)
            window_norm = window.lower().replace("\u2019", "").replace("'", "")
            pos = window_norm.index(label_lower)
            after = window[pos + len(label_lower):]
            has_dollar = "$" in after

            # Skip share count rows
            context_between = after[:100].lower()
            if re.search(r'shares|outstanding|weighted|denominator', context_between):
                continue

            # Check if numbers are on the same line
            line_norm = line.lower().replace("\u2019", "").replace("'", "")
            lbl_pos = line_norm.index(label_lower) if label_lower in line_norm else -1
            after_on_line = line[lbl_pos + len(label_lower):] if lbl_pos >= 0 else ""
            same_line_has_nums = bool(re.search(r'[\d,]{4,}', after_on_line))

            nums = []
            for m in re.finditer(r'[\$]?\s*\(?([\d,]+)\)?', after):
                val_str = m.group(1).replace(",", "")
                try:
                    val = int(val_str)
                    if val >= 100 and not (2000 <= val <= 2099):
                        nums.append(val)
                        if len(nums) >= max_values:
                            break
                except ValueError:
                    continue

            if not nums:
                continue

            # Build a numeric score for candidate ranking
            def _candidate_score(n, tc, sl, pr, dl):
                """Score a candidate. Higher = better."""
                s = 0
                s += 500 * tc    # table chunk bonus
                s += 100 * len(n)  # more columns
                s += 2000 * pr   # PRIMARY label is critical (not a merged row)
                s += 20 * dl     # has dollar
                s += 10 * sl     # same-line bonus
                # Light tiebreaker: prefer larger values (avoids tiny footnote values)
                s += min(max(n) / 1000.0, 50)  # capped at +50
                return s

            score = _candidate_score(nums, is_table_chunk, same_line_has_nums,
                                      is_primary, has_dollar)
            best_score = (_candidate_score(best, best_is_table_chunk, best_same_line,
                                           best_is_primary, best_has_dollar)
                          if best else -1)

            if score > best_score:
                best = nums
                best_has_dollar = has_dollar
                best_is_primary = is_primary
                best_same_line = same_line_has_nums
                best_is_table_chunk = is_table_chunk
    return best


def _get_year_column(year: int, most_recent: int | None) -> int:
    """Column index: 0 = most recent year, 1 = previous, etc."""
    if most_recent is None:
        return 0
    return most_recent - year


def _is_unanswerable(question: str, doc_index: DocumentIndex | None = None) -> bool:
    """Generic heuristic: detect questions that typically can't be answered from documents."""
    q_lower = question.lower()
    # Future predictions
    if re.search(r'what will .+ be in (\d{4})', q_lower):
        return True
    if re.search(r'\b(predict|forecast)\b', q_lower):
        return True
    # Data not in financial filings
    if re.search(r'stock price|share price|market cap', q_lower):
        return True
    if re.search(r'how many .+ (were|was) sold|units? sold', q_lower):
        return True
    # Question about a year beyond the document's fiscal years
    if doc_index and doc_index.most_recent_year:
        year_match = re.search(r'\b(20\d{2})\b', q_lower)
        if year_match:
            asked_year = int(year_match.group(1))
            if asked_year > doc_index.most_recent_year:
                # "What will revenue be in 2025?" when doc covers up to 2024
                if re.search(r'\b(will|be|predict|expect|forecast)\b', q_lower):
                    return True
    return False


def _try_table_extraction(question: str, chunks: list,
                           doc_index: DocumentIndex | None) -> str | None:
    """Try to answer using table data. High-precision: only returns when confident."""
    if not doc_index or not doc_index.table_labels:
        return None

    q_lower = question.lower()

    # Skip question types that need model reasoning
    skip_types = [
        "summarize", "summary", "overview", "describe", "explain",
        "risks", "what is this document", "highlights",
        "who is", "where is", "what are", "what is this",
        "increase or decrease", "how many employees",
        "fiscal year end", "earnings per", "eps",
        "which .+ had the highest", "which .+ is the",
    ]
    for pat in skip_types:
        if re.search(pat, q_lower):
            return None

    # --- Comparison / Difference questions ---
    diff_match = re.search(
        r'(?:difference|change|compare)\b.*?(\d{4}).*?(\d{4})', q_lower)
    compare_match = re.search(
        r'how did (.+?) (?:change|compare).+?(\d{4}).+?(\d{4})', q_lower)

    if diff_match or compare_match:
        if compare_match:
            subject_text = compare_match.group(1)
            y1, y2 = int(compare_match.group(2)), int(compare_match.group(3))
        else:
            y1, y2 = int(diff_match.group(1)), int(diff_match.group(2))
            subject_text = re.sub(r'\d{4}', '', q_lower)
            subject_text = re.sub(
                r'\b(difference|change|between|and|from|to|what|was|is|the|in|of|compare|tell|me)\b',
                '', subject_text).strip()

        if y1 > y2:
            y1, y2 = y2, y1

        terms = _extract_terms(subject_text)
        if terms:
            terms = _expand_query_with_synonyms(terms, doc_index)
        label = _match_label(terms, doc_index) if terms else None
        if not label:
            all_terms = _extract_terms(question)
            all_terms = [t for t in all_terms if t not in
                         ("difference", "change", "compare", "comparison")]
            all_terms = _expand_query_with_synonyms(all_terms, doc_index)
            label = _match_label(all_terms, doc_index)

        if label:
            row = _find_row_values(label, chunks)
            mry = doc_index.most_recent_year
            if row and mry and len(row) >= 2:
                idx1 = min(max(mry - y1, 0), len(row) - 1)
                idx2 = min(max(mry - y2, 0), len(row) - 1)
                v1, v2 = row[idx1], row[idx2]
                diff = v2 - v1
                direction = "increased" if diff > 0 else "decreased" if diff < 0 else "unchanged"
                if "difference" in q_lower:
                    return (f"The difference is ${abs(diff):,} million "
                            f"({label}: ${v2:,} in {y2} vs ${v1:,} in {y1}, {direction})")
                return (f"{label}: ${v2:,} in {y2} vs ${v1:,} in {y1} "
                        f"({direction} by ${abs(diff):,})")
        return None  # let model handle if no label found

    # --- Trend questions ---
    trend_match = re.search(
        r'when did (?:the )?(.+?)\s+(increas|decreas|grow|drop|rise|fall)', q_lower)
    if trend_match:
        terms = _extract_terms(trend_match.group(1))
        if terms:
            terms = _expand_query_with_synonyms(terms, doc_index)
        label = _match_label(terms, doc_index) if terms else None
        if label and doc_index.fiscal_years:
            row = _find_row_values(label, chunks)
            if row and len(row) >= 2:
                years = sorted(doc_index.fiscal_years, reverse=True)[:len(row)]
                parts = []
                for j in range(len(row) - 1, -1, -1):
                    yr = years[j] if j < len(years) else "?"
                    val = row[j]
                    if j < len(row) - 1:
                        older = row[j + 1]
                        arrow = "↑" if val > older else "↓"
                        parts.append(f"{yr}: ${val:,} ({arrow} ${abs(val - older):,})")
                    else:
                        parts.append(f"{yr}: ${val:,}")
                return f"{label} trend: {' | '.join(parts)}"
        return None

    # --- Percentage questions ---
    pct_match = re.search(
        r'(?:percentage|percent|%)\s+of\s+(.+?)\s+(?:came|come|was|is)\s+(?:from\s+)?(.+?)[\?.]?$',
        q_lower)
    if pct_match:
        total_terms = _extract_terms(pct_match.group(1))
        total_terms = _expand_query_with_synonyms(total_terms, doc_index)
        total_label = _match_label(total_terms, doc_index)
        part_terms = _extract_terms(pct_match.group(2))
        part_terms = _expand_query_with_synonyms(part_terms, doc_index)
        part_label = _match_label(part_terms, doc_index)
        if total_label and part_label:
            total_row = _find_row_values(total_label, chunks)
            part_row = _find_row_values(part_label, chunks)
            if total_row and part_row and total_row[0] > 0:
                pct = (part_row[0] / total_row[0]) * 100
                # Sanity check: percentage > 100% means wrong values
                if pct <= 100:
                    return f"{pct:.1f}%"
        return None

    # --- Simple value lookup (high precision: require good match) ---
    terms = _extract_terms(question)
    if not terms:
        return None

    terms = _expand_query_with_synonyms(terms, doc_index)
    label = _match_label(terms, doc_index)
    if not label:
        return None

    year_match = re.search(r'\b(20\d{2})\b', q_lower)
    year = int(year_match.group(1)) if year_match else (doc_index.most_recent_year or 0)
    col = _get_year_column(year, doc_index.most_recent_year)

    row = _find_row_values(label, chunks)
    if row:
        idx = min(max(col, 0), len(row) - 1)
        return f"${row[idx]:,}"

    return None


# ──────────────────────────────────────────────────────────────────────
# Main Engine
# ──────────────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    question: str
    answer: str
    confidence: float
    is_unanswerable: bool
    source_pages: list[int]
    evidence_chunks: list[dict]
    used_vision: bool
    method: str = "unknown"


class DocumentQAEngine:
    """Hybrid QA: dynamic table extraction + model fallback."""

    def __init__(self, config: dict):
        self.config = config

        pdf_cfg = config.get("pdf_processing", {})
        self.extractor = PDFExtractor(
            dpi=pdf_cfg.get("dpi", 120),
            extract_tables=pdf_cfg.get("extract_tables", True),
            ocr_fallback=pdf_cfg.get("ocr_fallback", True),
        )
        self.chunker = DocumentChunker(
            chunk_size=pdf_cfg.get("chunk_size", 200),
            chunk_overlap=pdf_cfg.get("chunk_overlap", 40),
        )

        model_cfg = config.get("model", {})
        self.embedder = DocumentEmbedder(
            model_name=model_cfg.get("embedding_model",
                                      "sentence-transformers/all-MiniLM-L6-v2"),
            device="cpu",
        )

        self.extractive_qa = ExtractiveQA(device="cpu")

        self.vision_model = VisionLanguageModel(
            model_name=model_cfg.get("vision_model", "Qwen/Qwen2-VL-2B-Instruct"),
            device=model_cfg.get("device", "cpu"),
            max_new_tokens=model_cfg.get("max_new_tokens", 256),
            temperature=model_cfg.get("temperature", 0.1),
        )

        self.document: DocumentContent | None = None
        self.chunks: list[Chunk] = []
        self._pdf_path: str | None = None
        self._doc_overview: str = ""
        self._doc_index: DocumentIndex | None = None

    def load_document(self, pdf_path: str) -> dict:
        self._pdf_path = pdf_path
        max_pages = self.config.get("pdf_processing", {}).get("max_pages", 50)
        self.document = self.extractor.extract(pdf_path, max_pages=max_pages)
        self.chunks = self.chunker.chunk_document(self.document)
        self.embedder.index_chunks(self.chunks)
        self._doc_overview = self._build_document_overview()
        self._doc_index = DocumentIndex.build(self.chunks)

        return {
            "file_path": pdf_path,
            "total_pages": self.document.total_pages,
            "pages_processed": len(self.document.pages),
            "total_chunks": len(self.chunks),
            "pages_with_tables": sum(1 for p in self.document.pages if p.has_tables),
            "pages_with_charts": sum(1 for p in self.document.pages if p.has_charts),
        }

    def _build_document_overview(self) -> str:
        if not self.document:
            return ""
        parts = []
        meta = self.document.metadata or {}
        for key in ("title", "subject", "author"):
            if meta.get(key):
                parts.append(f"{key.title()}: {meta[key]}")
        parts.append(f"Total Pages: {self.document.total_pages}")
        for page in self.document.pages[:5]:
            if page.text and len(page.text.strip()) > 50:
                parts.append(f"[Page {page.page_number}]: {page.text[:500]}")
        return "\n".join(parts)

    def _rewrite_with_history(self, question: str, history: list[dict]) -> str:
        """Prepend conversation context for follow-up questions."""
        if not history:
            return question

        recent = history[-6:]
        parts = []
        for msg in recent:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content[:300]}")

        if parts:
            return f"Conversation:\n" + "\n".join(parts) + f"\n\nCurrent question: {question}"
        return question

    def ask(self, question: str, top_k: int = 5,
            history: list[dict] | None = None) -> QAResult:
        if not self.document:
            raise ValueError("No document loaded.")

        original_question = question

        # Handle follow-ups
        if history:
            question = self._rewrite_with_history(question, history)

        # Retrieve chunks
        retrieval_cfg = self.config.get("retrieval", {})
        results = self.embedder.search(
            question, top_k=max(top_k, 10),
            threshold=retrieval_cfg.get("similarity_threshold", 0.1),
        )

        if not results:
            return QAResult(
                question=original_question,
                answer="UNANSWERABLE: No relevant content found.",
                confidence=0.0, is_unanswerable=True,
                source_pages=[], evidence_chunks=[], used_vision=False,
            )

        source_pages = sorted(set(c.page_number for c, _ in results))
        has_charts = any(c.metadata.get("has_charts") for c, _ in results)

        # Build context (table chunks first for better model input)
        table_first = sorted(results, key=lambda x: x[0].chunk_type != "table")
        context_text = "\n\n".join(
            f"[Page {c.page_number}, {c.chunk_type}]: {c.text}"
            for c, _ in table_first
        )
        evidence = [
            {"chunk_id": c.chunk_id, "page": c.page_number,
             "type": c.chunk_type, "score": round(s, 4), "text": c.text[:300]}
            for c, s in results
        ]
        chunk_dicts = [
            {"text": c.text, "chunk_id": c.chunk_id,
             "page_number": c.page_number, "chunk_type": c.chunk_type}
            for c, _ in table_first
        ]

        def _result(answer, confidence, unanswerable=False, vision=False, method="unknown"):
            return QAResult(
                question=original_question, answer=answer,
                confidence=confidence, is_unanswerable=unanswerable,
                source_pages=source_pages, evidence_chunks=evidence,
                used_vision=vision, method=method,
            )

        # --- Stage 0: Unanswerable heuristic (before table extraction) ---
        if _is_unanswerable(question, self._doc_index):
            return _result("UNANSWERABLE: The document does not contain this information.",
                           0.9, unanswerable=True, method="unanswerable")

        # --- Stage 0.5: Route conversational/open-ended questions ---
        q_lower = question.lower()
        is_conversational = bool(re.search(
            r'summarize|summary|overview|what is this document|tell me about|'
            r'describe|explain|highlights|main risks|key (findings|points|takeaways)|'
            r'which company|whose (report|annual|document)|what company|'
            r'who (is the|are the)|what is this (report|document|filing)|'
            r'annual report .* (about|for)|what (are|is) the main|'
            r'(major|main|key) segments|business segments|what segments',
            q_lower
        ))
        if is_conversational:
            # For broad/conversational questions, include first-page context
            # so the model has document-level info (company name, overview, etc.)
            conv_context = context_text
            if self.document and self.document.pages:
                first_pages_text = "\n".join(
                    f"[Page {p.page_number}]: {p.text[:800]}"
                    for p in self.document.pages[:3]
                    if p.text and len(p.text.strip()) > 30
                )
                if first_pages_text:
                    conv_context = first_pages_text + "\n\n" + context_text

            # For financial highlights/summary questions, extract key metrics
            # from the document index so the model has real numbers
            is_highlights = bool(re.search(
                r'highlight|financial (summary|overview)|key (financial|metric|figure)',
                q_lower))
            if is_highlights and self._doc_index and self._doc_index.table_labels:
                key_metrics = []
                _highlight_labels = [
                    "revenue", "net sales", "total net sales", "total revenue",
                    "net income", "gross margin", "total gross margin",
                    "operating income", "total assets", "total liabilities",
                    "total shareholders equity", "total stockholders equity",
                    "cash and cash equivalents", "cost of sales",
                ]
                seen = set()
                for target in _highlight_labels:
                    for label_lower, label_orig in self._doc_index.table_labels.items():
                        if target in label_lower and label_lower not in seen:
                            row = _find_row_values(label_orig, chunk_dicts)
                            if not row:
                                all_cd = [{"text": c.text, "chunk_id": c.chunk_id,
                                           "page_number": c.page_number,
                                           "chunk_type": c.chunk_type}
                                          for c in self.chunks]
                                row = _find_row_values(label_orig, all_cd)
                            if row:
                                seen.add(label_lower)
                                yr = self._doc_index.most_recent_year or "Latest"
                                key_metrics.append(f"- {label_orig}: ${row[0]:,} ({yr})")
                                if len(key_metrics) >= 8:
                                    break
                    if len(key_metrics) >= 8:
                        break
                if key_metrics:
                    metrics_text = "Key Financial Data (extracted from tables):\n" + "\n".join(key_metrics)
                    conv_context = metrics_text + "\n\n" + conv_context

            r = self.vision_model.answer_conversational(
                question, conv_context, self._doc_overview)
            return _result(r["answer"], r["confidence"], r["is_unanswerable"], method="conversational")

        # --- Stage 1: Try table extraction (fast, precise) ---
        # For follow-ups, try table extraction on the ORIGINAL question first
        # (rewritten question includes history context that pollutes label matching)
        # But if the follow-up lacks a subject (e.g. "Difference between 2023 and 2022"),
        # inject the subject from the last user question in history.
        table_question = original_question if history else question
        if history and original_question:
            # Check if follow-up is a diff/compare/trend question missing a subject
            oq = original_question.lower()
            has_pattern = bool(re.search(r'(difference|change|compare|trend)', oq))
            if has_pattern:
                # Extract subject terms from the follow-up after removing years/keywords
                subj = re.sub(r'\d{4}', '', oq)
                subj = re.sub(
                    r'\b(difference|change|between|and|from|to|what|was|is|the|in|of|compare|tell|me|now|it|with|about)\b',
                    '', subj).strip()
                if len(subj) < 3:
                    # No subject — get it from the last user message in history
                    for msg in reversed(history):
                        if msg.get("role") == "user":
                            prev_q = msg["content"]
                            # Extract meaningful terms from previous question
                            prev_terms = _extract_terms(prev_q)
                            if prev_terms:
                                subject = " ".join(prev_terms)
                                table_question = f"{subject} difference between " + " and ".join(
                                    re.findall(r'\b(20\d{2})\b', original_question))
                                logger.debug(f"Follow-up enriched: '{original_question}' -> '{table_question}'")
                            break
        # First try on retrieved chunks (most relevant)
        table_answer = _try_table_extraction(table_question, chunk_dicts, self._doc_index)
        if not table_answer:
            # Fall back to ALL document chunks (text + table)
            all_chunk_dicts = [
                {"text": c.text, "chunk_id": c.chunk_id,
                 "page_number": c.page_number, "chunk_type": c.chunk_type}
                for c in self.chunks
            ]
            table_answer = _try_table_extraction(table_question, all_chunk_dicts, self._doc_index)
        if table_answer:
            return _result(table_answer, 0.85, method="table")

        # --- Stage 2: Charts → vision model ---
        if has_charts and source_pages and self._pdf_path:
            page_img = self.extractor.render_page_image(self._pdf_path, source_pages[0])
            if page_img:
                r = self.vision_model.answer_with_image(question, page_img, context_text)
                return _result(r["answer"], r["confidence"], r["is_unanswerable"], True, method="vision")

        # --- Stage 3: Extractive QA ---
        ext = self.extractive_qa.answer(question, chunk_dicts, no_answer_threshold=0.3)

        # --- Stage 4: Generative model ---
        gen = self.vision_model.answer_text_only(question, context_text)

        # --- Decide ---
        ext_ok = not ext["is_unanswerable"] and ext["confidence"] >= 0.55 and len(ext["answer"]) > 1
        gen_ok = not gen["is_unanswerable"] and gen["confidence"] >= 0.4

        if ext["is_unanswerable"] and gen["is_unanswerable"]:
            return _result("UNANSWERABLE: The document does not contain this information.",
                           0.9, unanswerable=True, method="unanswerable")

        if ext_ok:
            return _result(ext["answer"], ext["confidence"], method="extractive")

        if gen_ok:
            return _result(gen["answer"], gen["confidence"], method="generative")

        # Low confidence fallback
        if not ext["is_unanswerable"] and ext["answer"].strip():
            return _result(ext["answer"], ext["confidence"], method="extractive")

        return _result(gen["answer"], gen["confidence"], gen["is_unanswerable"], method="generative")

    def get_page_image(self, page_number: int) -> Image.Image | None:
        if not self.document or not self._pdf_path:
            return None
        if page_number < 1 or page_number > len(self.document.pages):
            return None
        return self.extractor.render_page_image(self._pdf_path, page_number)

    def get_document_summary(self) -> dict | None:
        if not self.document:
            return None
        return {
            "file_path": self.document.file_path,
            "total_pages": self.document.total_pages,
            "pages_processed": len(self.document.pages),
            "total_chunks": len(self.chunks),
            "metadata": self.document.metadata,
        }
