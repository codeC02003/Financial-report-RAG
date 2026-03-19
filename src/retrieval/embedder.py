"""Hybrid retrieval: SQLite FTS5 (BM25) + FAISS semantic + Cross-Encoder reranking.

Implements recommendations from best-practice RAG stack analysis:
- SQLite FTS5 replaces Whoosh (actively maintained, built-in BM25)
- Reciprocal Rank Fusion (RRF) replaces weighted score mixing
- Cross-encoder reranker as second stage for precision
- BGE-M3 / modern embedding model support
"""

import os
import sqlite3
import logging
import re
import tempfile
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from src.pdf_processing.chunker import Chunk

logger = logging.getLogger(__name__)

# RRF constant (standard value from literature)
RRF_K = 60


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
    """Hybrid retrieval with SQLite FTS5 BM25 + FAISS semantic + cross-encoder reranking.

    Pipeline:
    1. BM25 via SQLite FTS5 (keyword matching, stemming-like via tokenizer)
    2. Dense retrieval via FAISS (semantic similarity)
    3. Reciprocal Rank Fusion to merge results
    4. Cross-encoder reranking on top candidates for precision
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        self.device = device
        self.chunks: list[Chunk] = []
        self._chunk_map: dict[str, Chunk] = {}

        # --- SQLite FTS5 for BM25 ---
        self._db_path = os.path.join(tempfile.mkdtemp(prefix="docqa_fts_"), "fts.db")
        self._conn: sqlite3.Connection | None = None

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

    def _init_fts_db(self):
        """Create SQLite FTS5 virtual table."""
        if self._conn:
            self._conn.close()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS chunks_fts")
        cur.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id,
                content,
                chunk_type,
                page_number,
                tokenize='porter unicode61'
            )
        """)
        self._conn.commit()

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

    def index_chunks(self, chunks: list[Chunk]):
        """Build both FTS5 and FAISS indexes."""
        self.chunks = chunks
        self._chunk_map = {c.chunk_id: c for c in chunks}
        texts = [c.text for c in chunks]

        # --- SQLite FTS5 BM25 Index ---
        self._init_fts_db()
        cur = self._conn.cursor()
        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunks_fts (chunk_id, content, chunk_type, page_number) VALUES (?, ?, ?, ?)",
                (chunk.chunk_id, chunk.text, chunk.chunk_type, str(chunk.page_number)),
            )
        self._conn.commit()
        logger.info(f"SQLite FTS5 index built with {len(chunks)} documents.")

        # --- FAISS Semantic Index ---
        logger.info(f"Embedding {len(texts)} chunks for semantic index...")
        embeddings = self._encode(texts, batch_size=64)
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors.")

    def _bm25_search(self, query: str, top_k: int) -> list[str]:
        """BM25 search via SQLite FTS5. Returns ranked list of chunk_ids."""
        if not self._conn:
            return []

        # Clean query for FTS5 (remove special chars)
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not clean_query:
            return []

        cur = self._conn.cursor()
        try:
            cur.execute(
                """SELECT chunk_id, bm25(chunks_fts) as score
                   FROM chunks_fts
                   WHERE chunks_fts MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (clean_query, top_k),
            )
            return [row[0] for row in cur.fetchall()]
        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 search failed: {e}")
            return []

    def _semantic_search(self, query: str, top_k: int) -> list[str]:
        """Dense retrieval via FAISS. Returns ranked list of chunk_ids."""
        if self.faiss_index is None or not self.chunks:
            return []

        query_embedding = self._encode([query])
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

    def _augment_query(self, query: str) -> list[str]:
        """Generate query variants to improve recall.

        No hardcoded expansions — just uses the original query.
        The hybrid BM25 + semantic retrieval handles synonyms and related terms.
        """
        return [query]

    def search(self, query: str, top_k: int = 8,
               threshold: float = 0.0) -> list[tuple[Chunk, float]]:
        """Hybrid search: BM25 + semantic via RRF, then cross-encoder reranking."""
        if not self.chunks:
            return []

        # Stage 1: Get candidates from both retrievers
        # Over-fetch significantly — let the cross-encoder reranker pick the best
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

        # Sort by RRF score, take more candidates for reranking (better precision)
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
        if self._conn:
            self._conn.close()
            self._conn = None
