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

Upload any annual report, financial filing, or document PDF and ask natural language questions — the system extracts precise values from tables, elaborates answers using LLMs, handles typos, supports follow-up conversations, and detects unanswerable queries.

## Key Features

- **Dynamic Table Extraction** — Automatically discovers table structure (row labels, fiscal years, column headers) from any document at upload time. No hardcoded mappings or company-specific logic.
- **LLM-Elaborated Answers** — Simple table lookups return instant values; complex answers (trends, comparisons, qualitative) are elaborated by the VLM into rich, document-quality responses with bold formatting and structured bullet points.
- **Hybrid Retrieval Pipeline** — BM25 keyword search (SQLite FTS5) + dense semantic search (FAISS) merged via Reciprocal Rank Fusion, then reranked with a cross-encoder for precision.
- **Adjacent Chunk Expansion** — After retrieving top-k chunks, automatically includes all chunks from the same pages for richer context.
- **Multi-Model Architecture** — Extractive QA (RoBERTa-SQuAD2) for factual questions, generative VLM (Qwen2-VL-2B) for complex/conversational queries, with automatic routing.
- **Conversational Memory** — Follow-up questions with context-aware history rewriting. Ask "What was the revenue?" then "How does it compare to last year?" and get coherent answers.
- **Smart Follow-Up Suggestions** — Rule-based generation of 3 contextual follow-up questions after each answer (zero latency, no extra LLM call).
- **Typo Tolerance** — Fuzzy synonym matching with context-aware overlap scoring handles misspelled queries (e.g., "reveneu" → "revenue", "net incme" → "net income").
- **Unanswerable Detection** — Identifies questions about future predictions, stock prices, and out-of-scope data rather than hallucinating answers.
- **Document Overview** — Built at load time with key financial metrics, passed to all LLM calls for grounded answers.
- **Markdown Answer Rendering** — Frontend renders **bold**, bullet lists, and line breaks from VLM output for readable responses.
- **Document Page Viewer** — View rendered PDF pages alongside answers with source page navigation.
- **React Frontend** — Modern chat-style interface with drag-and-drop PDF upload, animated loading stages, and follow-up chips.

## Architecture

```
PDF Upload → Text/Table Extraction (PyMuPDF + pdfplumber)
          → Chunking (text + table-aware)
          → Dual Indexing: SQLite FTS5 (BM25) + FAISS (semantic)
          → DocumentIndex (discovers labels, years, structure)
          → Document Overview (key metrics for LLM context)

Question → Typo Correction (fuzzy synonym expansion)
        → Unanswerable Check
        → Conversational Routing (summaries, overviews, risks)
        → Table Extraction (dynamic label matching, full-document scan)
           ├─ Simple value ($X,XXX) → instant return (no LLM)
           └─ Complex answer → VLM elaboration
        → Extractive QA (RoBERTa-SQuAD2) → VLM elaboration
        → Generative VLM (Qwen2-VL-2B with page images)
        → Confidence-ranked answer selection
        → Follow-up suggestion generation
```

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Vision-Language | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | Multimodal understanding, answer elaboration, conversational QA |
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
   - *"What was the total revenue?"* → Instant table lookup
   - *"How did net income change from 2023 to 2024?"* → Comparison with elaboration
   - *"How did revenue change over the years?"* → Multi-year trend analysis
   - *"What percentage of revenue came from services?"* → Calculated percentage
   - *"Summarize the main risks."* → VLM-powered qualitative answer
   - *"What was total reveneu?"* → Typo handled correctly
   - *"What will revenue be in 2030?"* → Correctly identified as unanswerable
4. Click follow-up suggestion chips to continue exploring

## Evaluation Results

Tested across multiple documents covering financial data extraction, comparisons, trends, conversational QA, follow-up questions, typo handling, and unanswerable detection.

| Document | Questions | Accuracy |
|----------|-----------|----------|
| Tandy Leather 10-K (2024) | 27 | **100%** (27/27) |
| Apple 10-K (2024) | 50 | **94%** (47/50) |
| Aaron's Holdings 10-K (2023) | 14 | **100%** (14/14) |
| Materion Corp 10-K (2024) | 14 | **100%** (14/14) |

**Question categories tested:**
- Direct value lookup (revenue, net income, total assets)
- Year-over-year comparisons and differences
- Percentage calculations
- Trend analysis across multiple years
- Conversational questions (summaries, overviews, risk analysis)
- Follow-up questions with conversational context
- Typo handling (misspelled financial terms)
- Unanswerable detection (future predictions, stock prices)
- Alternate phrasings and informal queries

## Project Structure

```
Financial-report-RAG/
├── src/
│   ├── api/server.py              # FastAPI REST endpoints
│   ├── qa_engine/engine.py        # Core QA pipeline, table extraction, follow-ups
│   ├── models/
│   │   ├── vision_model.py        # Qwen2-VL wrapper (answer + elaborate)
│   │   └── extractive_qa.py       # RoBERTa-SQuAD2 wrapper
│   ├── retrieval/embedder.py      # Hybrid BM25 + FAISS + reranker
│   └── pdf_processing/
│       ├── extractor.py           # PDF text & image extraction
│       └── chunker.py             # Document chunking
├── frontend/src/                  # React chat interface
├── configs/config.yaml            # Model & pipeline configuration
├── run_backend.py                 # Backend entry point
├── run_tests.py                   # Multi-document benchmark
├── test_generic.py                # Generic benchmark for any document
├── test_comprehensive.py          # Comprehensive QA test suite (27 questions)
├── Dockerfile                     # Docker config for HF Spaces deployment
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

**Response includes:**
- `answer` — The answer text (with markdown formatting)
- `confidence` — Confidence score (0-1)
- `method` — Which pipeline stage answered (table, extractive, vision, conversational, unanswerable)
- `source_pages` — Page numbers where evidence was found
- `follow_ups` — 3 suggested follow-up questions
- `is_unanswerable` — Whether the question is out of scope

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

### `GET /api/page/{page_number}`

Get a rendered page image for document viewing.

### `GET /api/document`

Get document summary and metadata.

### `GET /api/health`

Health check endpoint.

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

2. **Dynamic Indexing** — At upload time, all chunks are scanned to discover table row labels and fiscal year columns. A `DocumentIndex` enables precise table lookups, and a document overview with key metrics is generated for LLM context.

3. **Hybrid Retrieval** — Questions trigger parallel BM25 (keyword) and FAISS (semantic) searches. Results are merged via Reciprocal Rank Fusion and reranked with a cross-encoder. Adjacent chunks from the same pages are automatically included for richer context.

4. **Multi-Stage QA** — The engine tries strategies in order of precision:
   - **Table extraction** scans all document chunks (not just retrieved ones) for precise numeric lookups. Simple values ($X,XXX) are returned instantly; complex answers are elaborated by the VLM.
   - **Extractive QA** (RoBERTa-SQuAD2) extracts factual spans, then the VLM elaborates them.
   - **Generative VLM** (Qwen2-VL-2B) handles conversational and complex queries with page images and document overview context.

5. **Smart Follow-ups** — Follow-up questions like "What about services?" or "Compare it with 2023" are enriched with context from conversation history. After each answer, 3 contextual follow-up suggestions are generated using rule-based logic (zero latency).

6. **Typo Handling** — Fuzzy synonym matching expands misspelled terms to known table labels using sequence matching with context-aware overlap scoring, so "net incme" correctly maps to "Net income" rather than "Operating income".

## Deployment

**Docker (Hugging Face Spaces):**

```bash
docker build -t finrag .
docker run -p 7860:7860 finrag
```

**Frontend (Vercel):**

The React frontend can be deployed as a static site on Vercel. Update the API base URL in the frontend config to point to your backend.

## Author

**Chinmay Mhatre**

- [Portfolio](https://my-portfolio-mu-ten-24.vercel.app/)
- [GitHub](https://github.com/codeC02003)
- [Email](mailto:chinmaymhatre@arizona.edu)

## License

MIT
