"""Hybrid retrieval: Whoosh (BM25 + fuzzy + phrases + abbreviations) + FAISS semantic + Cross-Encoder reranking.

Pipeline:
1. Whoosh BM25 with custom analyzer (stemming, synonyms, abbreviation expansion, fuzzy matching)
2. Dense retrieval via FAISS (semantic similarity)
3. Reciprocal Rank Fusion (RRF) to merge results
4. Cross-encoder reranker as second stage for precision
"""

import os
import logging
import re
import shutil
import tempfile
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from src.pdf_processing.chunker import Chunk

# Whoosh imports
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import MultifieldParser, OrGroup, FuzzyTermPlugin, PhrasePlugin
from whoosh.analysis import (
    StandardAnalyzer, StemmingAnalyzer, LowercaseFilter,
    RegexTokenizer, IntraWordFilter, StemFilter, StopFilter
)
from whoosh import scoring

logger = logging.getLogger(__name__)

# RRF constant (standard value from literature)
RRF_K = 60

# ──────────────────────────────────────────────────────────────────────
# Financial abbreviation → full form mappings
# ──────────────────────────────────────────────────────────────────────

ABBREVIATIONS: dict[str, list[str]] = {
    # Income Statement
    "cogs": ["cost of goods sold", "cost of sales"],
    "sga": ["selling general and administrative"],
    "sg&a": ["selling general and administrative"],
    "r&d": ["research and development"],
    "r and d": ["research and development"],
    "eps": ["earnings per share"],
    "ebitda": ["earnings before interest taxes depreciation and amortization"],
    "ebit": ["earnings before interest and taxes", "operating income"],
    "opex": ["operating expenses"],
    "capex": ["capital expenditures"],
    "d&a": ["depreciation and amortization"],

    # Balance Sheet
    "pp&e": ["property plant and equipment"],
    "ppe": ["property plant and equipment"],
    "a/r": ["accounts receivable"],
    "a/p": ["accounts payable"],
    "ar": ["accounts receivable"],
    "ap": ["accounts payable"],
    "roi": ["return on investment"],
    "roe": ["return on equity"],
    "roa": ["return on assets"],
    "bv": ["book value"],
    "nav": ["net asset value"],
    "nwc": ["net working capital"],

    # Cash Flow
    "cfo": ["cash from operations", "operating cash flow"],
    "cfi": ["cash from investing", "investing cash flow"],
    "cff": ["cash from financing", "financing cash flow"],
    "fcf": ["free cash flow"],
    "ocf": ["operating cash flow"],

    # Ratios & Metrics
    "p/e": ["price to earnings"],
    "p/b": ["price to book"],
    "d/e": ["debt to equity"],
    "gm": ["gross margin"],
    "npm": ["net profit margin"],
    "opm": ["operating profit margin"],
    "yoy": ["year over year"],
    "qoq": ["quarter over quarter"],
    "ttm": ["trailing twelve months"],
    "ltm": ["last twelve months"],

    # Document / Filing
    "10-k": ["annual report"],
    "10-q": ["quarterly report"],
    "8-k": ["current report"],
    "md&a": ["management discussion and analysis"],
    "mda": ["management discussion and analysis"],
    "gaap": ["generally accepted accounting principles"],
    "ifrs": ["international financial reporting standards"],
    "sec": ["securities and exchange commission"],
    "fy": ["fiscal year"],
}


def _expand_abbreviations(query: str) -> str:
    """Expand known abbreviations in the query to their full forms.

    Returns the original query with abbreviation expansions appended,
    so both the abbreviation and the full form are searchable.
    """
    q_lower = query.lower().strip()
    expansions = []

    for abbr, full_forms in ABBREVIATIONS.items():
        # Match abbreviation as a whole word
        pattern = r'\b' + re.escape(abbr).replace(r'\&', r'[&]?') + r'\b'
        if re.search(pattern, q_lower, re.IGNORECASE):
            for full_form in full_forms:
                if full_form.lower() not in q_lower:
                    expansions.append(full_form)

    if expansions:
        return query + " " + " ".join(expansions)
    return query


# ──────────────────────────────────────────────────────────────────────
# Query normalizer — Whoosh-powered tokenize + lowercase + stem
# Used by all pipeline stages for consistent term matching
# ──────────────────────────────────────────────────────────────────────

# Shared analyzer instance: tokenize → lowercase → stop words → Porter stem
_query_analyzer = StemmingAnalyzer()


def normalize_query(text: str) -> str:
    """Normalize text using Whoosh's stemming analyzer.

    Tokenizes, lowercases, removes stop words, and stems each token.
    Returns a space-joined string of normalized tokens.
    E.g., "What were Total Revenues?" → "total revenu"
          "Operating Expenses for 2023" → "oper expens 2023"
    """
    tokens = [t.text for t in _query_analyzer(text)]
    return " ".join(tokens)


def normalize_terms(text: str) -> list[str]:
    """Like normalize_query but returns individual stemmed tokens as a list.

    E.g., "Total Net Sales" → ["total", "net", "sale"]
          "Earnings Per Share" → ["earn", "per", "share"]
    """
    return [t.text for t in _query_analyzer(text)]


def expand_and_normalize(query: str) -> str:
    """Full query preprocessing: abbreviation expansion + Whoosh normalization.

    E.g., "What was COGS?" → "what was cogs cost good sold cost sale"
    """
    expanded = _expand_abbreviations(query)
    return normalize_query(expanded)


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def _rrf_merge(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion — merge multiple ranked lists by rank, not score.

    RRF(d) = sum over lists: 1 / (k + rank(d))
    This is robust to score scale differences between retrievers.
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


class DocumentEmbedder:
    """Hybrid retrieval with Whoosh BM25 + FAISS semantic + cross-encoder reranking.

    Pipeline:
    1. Whoosh BM25 (keyword matching, stemming, fuzzy, phrase queries, abbreviation expansion)
    2. Dense retrieval via FAISS (semantic similarity)
    3. Reciprocal Rank Fusion to merge results
    4. Cross-encoder reranking on top candidates for precision
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        self.device = device
        self.chunks: list[Chunk] = []
        self._chunk_map: dict[str, Chunk] = {}

        # --- Whoosh index directory ---
        self._whoosh_dir = tempfile.mkdtemp(prefix="docqa_whoosh_")
        self._whoosh_ix = None

        # --- Whoosh schema ---
        # StemmingAnalyzer: tokenize → lowercase → stem (Porter)
        analyzer = StemmingAnalyzer() | IntraWordFilter(mergewords=True, mergenums=True)
        self._schema = Schema(
            chunk_id=ID(stored=True, unique=True),
            content=TEXT(analyzer=StemmingAnalyzer(), stored=False),
            chunk_type=STORED,
            page_number=STORED,
        )

        # --- FAISS for dense retrieval ---
        logger.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embed_model = AutoModel.from_pretrained(model_name).to(device)
        self.embed_model.eval()
        self.dimension = self.embed_model.config.hidden_size
        self.faiss_index: faiss.IndexFlatIP | None = None

        # --- Cross-encoder reranker ---
        logger.info("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", device=device
        )
        logger.info("Reranker loaded.")

    def index_chunks(self, chunks: list[Chunk]):
        """Build both Whoosh and FAISS indexes."""
        self.chunks = chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}
        texts = [c.text for c in chunks]

        # --- Whoosh BM25 Index ---
        # Recreate index dir
        if os.path.exists(self._whoosh_dir):
            shutil.rmtree(self._whoosh_dir)
        os.makedirs(self._whoosh_dir, exist_ok=True)

        self._whoosh_ix = create_in(self._whoosh_dir, self._schema)
        writer = self._whoosh_ix.writer()
        for chunk in chunks:
            writer.add_document(
                chunk_id=chunk.chunk_id,
                content=chunk.text,
                chunk_type=chunk.chunk_type,
                page_number=chunk.page_number,
            )
        writer.commit()
        logger.info(f"Whoosh index built with {len(chunks)} documents.")

        # --- FAISS Semantic Index ---
        logger.info(f"Embedding {len(texts)} chunks for semantic index...")
        embeddings = self._encode(texts, batch_size=64)
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors.")

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts into normalized embeddings in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.embed_model(**encoded)

            embeddings = _mean_pooling(output, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def _bm25_search(self, query: str, top_k: int) -> list[str]:
        """BM25 search via Whoosh with fuzzy matching and abbreviation expansion."""
        if not self._whoosh_ix:
            return []

        # Expand abbreviations in the query
        expanded_query = _expand_abbreviations(query)

        # Clean query for Whoosh parser
        clean_query = re.sub(r'[^\w\s&/\-]', ' ', expanded_query).strip()
        if not clean_query:
            return []

        try:
            with self._whoosh_ix.searcher(weighting=scoring.BM25F()) as searcher:
                # Parse with OR group (any term can match) + fuzzy plugin
                parser = MultifieldParser(
                    ["content"], schema=self._schema, group=OrGroup
                )
                parser.add_plugin(FuzzyTermPlugin())

                # Try exact query first
                parsed = parser.parse(clean_query)
                results = searcher.search(parsed, limit=top_k)
                chunk_ids = [hit["chunk_id"] for hit in results]

                # If few results, also try fuzzy variants
                if len(chunk_ids) < top_k // 2:
                    # Add fuzzy suffix (~1 edit distance) to each term
                    fuzzy_terms = []
                    for word in clean_query.split():
                        if len(word) >= 4:
                            fuzzy_terms.append(f"{word}~1")
                        else:
                            fuzzy_terms.append(word)
                    fuzzy_query = " ".join(fuzzy_terms)
                    try:
                        fuzzy_parsed = parser.parse(fuzzy_query)
                        fuzzy_results = searcher.search(fuzzy_parsed, limit=top_k)
                        for hit in fuzzy_results:
                            cid = hit["chunk_id"]
                            if cid not in chunk_ids:
                                chunk_ids.append(cid)
                    except Exception:
                        pass  # fuzzy parse can fail on edge cases

                return chunk_ids[:top_k]
        except Exception as e:
            logger.warning(f"Whoosh search failed: {e}")
            return []

    def _semantic_search(self, query: str, top_k: int) -> list[str]:
        """Dense retrieval via FAISS. Returns ranked list of chunk_ids."""
        if self.faiss_index is None or not self.chunks:
            return []

        # Also expand abbreviations for semantic search
        expanded_query = _expand_abbreviations(query)
        query_embedding = self._encode([expanded_query])
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(self.chunks[idx].chunk_id)
        return results

    def _rerank(self, query: str, chunk_ids: list[str], top_k: int) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder for precision."""
        if not chunk_ids:
            return []

        pairs = [(query, self._chunk_map[cid].text[:512]) for cid in chunk_ids]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _classify_query(query: str) -> str:
        """Classify query intent for metadata-aware chunk boosting.

        Returns: 'numeric', 'narrative', or 'mixed'
        """
        q = query.lower()
        numeric_signals = [
            r'what (was|were|is|are) (the )?(total|net|gross)',
            r'how much', r'how many', r'revenue', r'income', r'cost',
            r'assets', r'liabilities', r'equity', r'margin', r'debt',
            r'earnings', r'sales', r'expenses', r'profit', r'cash',
            r'compare.*\d{4}', r'difference.*\d{4}', r'trend',
            r'percentage', r'percent', r'ratio', r'\d{4}.*vs',
            r'capex', r'ebitda', r'cogs', r'eps', r'roe', r'roi',
        ]
        narrative_signals = [
            r'summarize', r'summary', r'explain', r'describe', r'overview',
            r'risk', r'who is', r'who are', r'what type', r'what kind',
            r'tell me about', r'business model', r'how does .+ (generate|earn)',
            r'what caused', r'what drove', r'why did', r'main .+ factors',
            r'segments', r'strategy', r'outlook',
        ]
        num_score = sum(1 for p in numeric_signals if re.search(p, q))
        nar_score = sum(1 for p in narrative_signals if re.search(p, q))
        if num_score > nar_score:
            return "numeric"
        if nar_score > num_score:
            return "narrative"
        return "mixed"

    def search(self, query: str, top_k: int = 8,
               threshold: float = 0.0) -> list[tuple[Chunk, float]]:
        """Hybrid search: Whoosh BM25 + FAISS semantic via RRF, then cross-encoder reranking.

        Includes metadata-aware chunk boosting: table chunks are boosted for
        numeric queries, text chunks are boosted for narrative queries.
        """
        if not self.chunks:
            return []

        # Classify query for metadata boosting
        query_type = self._classify_query(query)

        # Stage 1: Get candidates from both retrievers
        candidate_k = max(top_k * 5, 30)
        all_bm25 = self._bm25_search(query, top_k=candidate_k)
        all_semantic = self._semantic_search(query, top_k=candidate_k)

        # Deduplicate while preserving order
        bm25_ranked = list(dict.fromkeys(all_bm25))
        semantic_ranked = list(dict.fromkeys(all_semantic))

        # Stage 2: Reciprocal Rank Fusion
        rrf_scores = _rrf_merge([bm25_ranked, semantic_ranked], k=RRF_K)

        if not rrf_scores:
            return []

        # Stage 2.5: Metadata-aware boosting
        # Boost table chunks for numeric queries, text chunks for narrative queries
        if query_type != "mixed":
            preferred_type = "table" if query_type == "numeric" else "text"
            for cid in rrf_scores:
                chunk = self._chunk_map.get(cid)
                if chunk and chunk.chunk_type == preferred_type:
                    rrf_scores[cid] *= 1.3  # 30% boost for preferred type

        # Sort by RRF score, take more candidates for reranking
        rrf_ranked = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        rerank_candidates = rrf_ranked[:max(top_k * 3, 20)]

        # Stage 3: Cross-encoder reranking
        reranked = self._rerank(query, rerank_candidates, top_k=top_k)

        # Build results
        results = []
        for cid, score in reranked:
            if cid in self._chunk_map:
                results.append((self._chunk_map[cid], float(score)))

        return results

    def reset(self):
        """Clear all indexes."""
        self.faiss_index = None
        self.chunks = []
        self._chunk_map = {}
        self._whoosh_ix = None
        if os.path.exists(self._whoosh_dir):
            try:
                shutil.rmtree(self._whoosh_dir)
            except OSError:
                pass
