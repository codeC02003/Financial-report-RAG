---
title: FinRAG API
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# FinRAG — Multimodal Document QA

An intelligent document question-answering system that combines **vision-language models**, **hybrid retrieval (BM25 + semantic search)**, and **dynamic table extraction** to answer questions about uploaded PDF documents with high accuracy.

Upload any annual report, financial filing, or document PDF and ask natural language questions — the system extracts precise values from tables, answers factual questions, handles follow-up conversations, and detects unanswerable queries.

## Key Features

- **Dynamic Table Extraction** — Automatically discovers table structure (row labels, fiscal years, column headers) from any document at upload time. No hardcoded mappings or company-specific logic.
- **Hybrid Retrieval Pipeline** — BM25 keyword search (SQLite FTS5) + dense semantic search (FAISS) merged via Reciprocal Rank Fusion, then reranked with a cross-encoder for precision.
- **Multi-Model Architecture** — Extractive QA (RoBERTa-SQuAD2) for factual questions, generative VLM (Qwen2-VL-2B) for complex/conversational queries, with automatic routing.
- **Conversational Memory** — Follow-up questions with context-aware history rewriting. Ask "What was the revenue?" then "How does it compare to last year?" and get coherent answers.
- **Unanswerable Detection** — Identifies questions about future predictions, stock prices, unit sales, and out-of-scope data rather than hallucinating answers.
- **Document Page Viewer** — View rendered PDF pages alongside answers with source page references.
- **React Frontend** — Modern chat-style interface with drag-and-drop PDF upload and real-time Q&A.

## Architecture

```
PDF Upload → Text/Table Extraction (PyMuPDF + pdfplumber)
          → Chunking (text + table-aware)
          → Dual Indexing: SQLite FTS5 (BM25) + FAISS (semantic)
          → DocumentIndex (discovers labels, years, structure)

Question → Unanswerable Check
        → Conversational Routing (summaries, overviews, risks)
        → Table Extraction (dynamic label matching, row extraction)
        → Extractive QA (RoBERTa-SQuAD2)
        → Generative VLM (Qwen2-VL-2B with page images)
        → Confidence-ranked answer selection
```

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Vision-Language | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Multimodal document understanding, conversational QA |
| Extractive QA | [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) | Precise span extraction from text |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Semantic search embeddings |
| Reranker | [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | Cross-encoder reranking for retrieval precision |

## Tech Stack

**Backend:** Python, FastAPI, PyTorch, Transformers, FAISS, SQLite FTS5, PyMuPDF, pdfplumber

**Frontend:** React, Vite, React Icons

**Retrieval:** Hybrid BM25 + dense retrieval, Reciprocal Rank Fusion, cross-encoder reranking

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- ~4 GB disk space for model weights (downloaded automatically on first run)

### Installation

```bash
# Clone the repository
git clone https://github.com/codeC02003/Financial-report-RAG.git
cd Financial-report-RAG

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Running the Application

**Start the backend:**

```bash
python run_backend.py
```

The API server starts at `http://localhost:8000`. Models are downloaded automatically on first launch.

**Start the frontend (in a separate terminal):**

```bash
cd frontend
npm run dev
```

The UI opens at `http://localhost:5173`.

### Usage

1. Open the frontend in your browser
2. Upload a PDF document (annual reports, 10-K filings, financial documents, etc.)
3. Ask questions in natural language:
   - *"What was the total revenue?"*
   - *"How did net income change from 2023 to 2024?"*
   - *"What percentage of revenue came from services?"*
   - *"Summarize the main risks."*
   - *"What will revenue be in 2030?"* → Correctly identified as unanswerable

## Evaluation Results

Tested on a 50-question benchmark covering financial data extraction, comparisons, trends, conversational QA, follow-up questions, and unanswerable detection.

| Document | Questions | Accuracy |
|----------|-----------|----------|
| Apple 10-K (2024) | 50 | **94%** (47/50) |
| Aaron's Holdings 10-K (2023) | 14 | **100%** (14/14) |
| Materion Corp 10-K (2024) | 14 | **100%** (14/14) |

**Question categories tested:**
- Direct value lookup (revenue, net income, cost of sales)
- Year-over-year comparisons and differences
- Percentage calculations
- Trend analysis across multiple years
- Conversational questions (summaries, overviews, risk analysis)
- Follow-up questions with conversational context
- Unanswerable detection (future predictions, stock prices, unit sales)
- Alternate phrasings and informal queries

## Project Structure

```
Financial-report-RAG/
├── src/
│   ├── api/server.py              # FastAPI REST endpoints
│   ├── qa_engine/engine.py        # Core QA pipeline & table extraction
│   ├── models/
│   │   ├── vision_model.py        # Qwen2-VL wrapper
│   │   └── extractive_qa.py       # RoBERTa-SQuAD2 wrapper
│   ├── retrieval/embedder.py      # Hybrid BM25 + FAISS + reranker
│   └── pdf_processing/
│       ├── extractor.py           # PDF text & image extraction
│       └── chunker.py             # Document chunking
├── frontend/src/                  # React chat interface
├── configs/config.yaml            # Model & pipeline configuration
├── run_backend.py                 # Backend entry point
├── run_tests.py                   # Apple 10-K benchmark (50 questions)
├── test_generic.py                # Generic benchmark for any document
└── requirements.txt
```

## API Reference

### `POST /api/upload`

Upload a PDF document for processing.

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

### `POST /api/ask`

Ask a question about the uploaded document.

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue?", "top_k": 5}'
```

**With conversation history (follow-ups):**

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does it compare to last year?",
    "top_k": 5,
    "history": [
      {"role": "user", "content": "What was the total revenue?"},
      {"role": "assistant", "content": "$391,035"}
    ]
  }'
```

### `GET /api/page-image/{page_number}`

Get a rendered page image for document viewing.

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
pdf_processing:
  dpi: 120
  max_pages: 150
  chunk_size: 200

model:
  vision_model: "Qwen/Qwen2-VL-2B-Instruct"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"

retrieval:
  top_k: 8
  similarity_threshold: 0.1
```

## How It Works

1. **PDF Processing** — Extracts text and tables using PyMuPDF and pdfplumber. Tables are detected and chunked separately with headers preserved for context.

2. **Dynamic Indexing** — At upload time, all chunks are scanned to discover table row labels and fiscal year columns. This builds a `DocumentIndex` that enables precise table lookups without any hardcoded knowledge.

3. **Hybrid Retrieval** — Questions trigger parallel BM25 (keyword) and FAISS (semantic) searches. Results are merged via Reciprocal Rank Fusion and reranked with a cross-encoder.

4. **Multi-Stage QA** — The engine tries strategies in order of precision: (1) dynamic table extraction for numeric lookups, (2) extractive QA for factual spans, (3) generative VLM for complex/conversational answers. The highest-confidence result is returned.

5. **Smart Follow-ups** — Follow-up questions like "What about services?" or "Compare it with 2023" are enriched with context from conversation history before being routed through the pipeline.

## Author

**Chinmay Mhatre**

- [Portfolio](https://my-portfolio-mu-ten-24.vercel.app/)
- [GitHub](https://github.com/codeC02003)
- [Email](mailto:chinmaymhatre@arizona.edu)

## License

MIT
