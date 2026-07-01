# Multimodal RAG System

**Phase 1** — Document Ingestion & Parsing  
**Phase 2** — Semantic Chunking, Embeddings & Text Search  
**Phase 3** — Multimodal Indexing & Hybrid Visual Retrieval

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 3 — MULTIMODAL PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1 Output (images + tables + JSON)                            │
│       │                                                             │
│       ▼                                                             │
│  POST /api/v1/multimodal/index                                      │
│       │                                                             │
│       ▼                                                             │
│  ImageProcessor ──────── OpenCV heuristics ─► ImageClassification  │
│   ├── FIGURE                                                        │
│   ├── DIAGRAM                                                       │
│   ├── FLOWCHART                                                     │
│   ├── ARCHITECTURE                                                  │
│   ├── CHART ────────────► ChartProcessor (ChartOCR + OCR)          │
│   ├── TABLE ────────────► TableProcessor (Camelot + Docling + OCR)  │
│   ├── EQUATION ─────────► EquationProcessor (pix2tex + OCR)        │
│   └── SCREENSHOT                                                    │
│       │                                                             │
│       ▼                                                             │
│  FigureCaptioner (Qwen2-VL-7B-Instruct)                             │
│   └── "This figure illustrates a 4-layer CNN with skip connections" │
│       │                                                             │
│       ├──► ColPali (visual embedding, 128-dim)                      │
│       └──► BGE-M3  (text embedding of description, 1024-dim)        │
│                │                                                    │
│                ▼                                                    │
│  Qdrant: research_multimodal                                        │
│   Point { vector: {visual: [...], text: [...]}, payload: {...} }    │
│                                                                     │
│  POST /api/v1/multimodal/search                                     │
│       │                                                             │
│       ├── Text search:   BGE-M3(query) → Qdrant text vector space  │
│       ├── Visual search: ColPali(query) → Qdrant visual space       │
│       └── Hybrid: weighted score fusion + dedup + rank              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## GPU Requirements

| Component | VRAM Required | Notes |
|-----------|--------------|-------|
| Qwen2-VL-7B | ~16 GB | bfloat16 on A100/RTX 3090+ |
| Qwen2-VL-7B | ~14 GB | 4-bit quantised (AWQ) |
| ColPali | ~8 GB | PaliGemma-3B backbone |
| BGE-M3 | ~4 GB | 1024-dim, fits on most GPUs |
| **Total (ideal)** | **24 GB** | A100 40GB recommended |
| **CPU fallback** | RAM only | ~8 GB RAM, much slower |

For CPU-only systems, disable captioning during indexing:
```json
{"document_id": "abc123", "run_captioning": false}
```

---

## Installing Phase 3 Dependencies

### 1. Install PyTorch (choose your CUDA version)

```bash
# CUDA 12.1 (recommended for A100, RTX 40xx)
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.4.1
```

### 2. Install Transformers for Qwen2-VL

```bash
pip install transformers>=4.45.0 accelerate>=0.26.0 qwen-vl-utils
```

### 3. Install ColPali

```bash
pip install colpali-engine>=0.3.0
```

ColPali downloads the `vidore/colpali-v1.2` checkpoint (~6GB) on first use.

### 4. Install Camelot for table extraction

```bash
# Install Ghostscript first (required by Camelot)
# Ubuntu/Debian:
sudo apt-get install -y ghostscript
# macOS:
brew install ghostscript

pip install "camelot-py[cv]"
```

### 5. Install ChartOCR

ChartOCR must be installed from source:

```bash
git clone https://github.com/soap117/DeepRule.git
cd DeepRule
pip install -e .
cd ..
```

If ChartOCR is unavailable, the system automatically falls back to
OpenCV heuristics + PaddleOCR for chart type detection.

### 6. (Optional) Install LaTeX-OCR for equations

```bash
pip install pix2tex
```

---

## Running Qdrant (Phase 3 Collections)

Phase 3 adds a second Qdrant collection: `research_multimodal`

```bash
# Start Qdrant (same instance as Phase 2)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Verify collections after startup
curl http://localhost:6333/collections
```

The `research_multimodal` collection is created automatically on startup
with dual named vectors:
- `"text"`: 1024-dim (BGE-M3)
- `"visual"`: 128-dim (ColPali)

---

## Complete 3-Phase Workflow

### Phase 1 — Upload and parse

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "document=@attention_is_all_you_need.pdf"
# Returns: {"document_id": "abc123def456", ...}
```

### Phase 2 — Index text chunks

```bash
curl -X POST "http://localhost:8000/api/v1/index" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "abc123def456"}'
```

### Phase 3 — Index multimodal elements

```bash
curl -X POST "http://localhost:8000/api/v1/multimodal/index" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "abc123def456",
    "force_reindex": false,
    "run_captioning": true,
    "run_chart_ocr": true,
    "run_table_extraction": true,
    "run_equation_detection": true
  }'
```

**Response:**
```json
{
  "document_id": "abc123def456",
  "figures_processed": 12,
  "charts_processed": 3,
  "tables_processed": 5,
  "equations_detected": 8,
  "vectors_stored": 28,
  "collection_name": "research_multimodal",
  "duration_seconds": 187.4,
  "message": "Indexed 28 multimodal vectors (12 figures, 3 charts, 5 tables, 8 equations) in 187.4s."
}
```

---

## Multimodal Search Examples

### Find architecture diagrams

```bash
curl -X POST "http://localhost:8000/api/v1/multimodal/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find the CNN architecture diagram",
    "top_k": 5,
    "search_mode": "hybrid"
  }'
```

### Find all confusion matrices

```bash
curl -X POST "http://localhost:8000/api/v1/multimodal/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show all confusion matrices",
    "top_k": 5,
    "chunk_types": ["figure", "chart"]
  }'
```

### Search only tables comparing accuracy

```bash
curl -X POST "http://localhost:8000/api/v1/multimodal/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "tables comparing model accuracy metrics",
    "top_k": 3,
    "chunk_types": ["table"],
    "search_mode": "text"
  }'
```

### Find loss curve charts

```bash
curl -X POST "http://localhost:8000/api/v1/multimodal/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "training loss curve over epochs",
    "top_k": 5,
    "chunk_types": ["chart"]
  }'
```

### Find cross-entropy equations

```bash
curl -X POST "http://localhost:8000/api/v1/multimodal/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cross entropy loss equation",
    "top_k": 3,
    "chunk_types": ["equation"]
  }'
```

### Search within a specific document only

```bash
curl -X POST "http://localhost:8000/api/v1/multimodal/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "attention mechanism",
    "top_k": 5,
    "document_id": "abc123def456"
  }'
```

**Sample Response:**
```json
{
  "query": "CNN architecture diagram",
  "total_results": 3,
  "results": [
    {
      "score": 0.943,
      "chunk_type": "architecture",
      "page": 4,
      "caption": "Architecture on page 4",
      "description": "This figure illustrates a convolutional neural network with four convolutional layers, two max-pooling layers, and three fully connected layers. Skip connections are visible between the second and fourth convolutional layers.",
      "image_path": "output/images/abc123def456/img_003_p4.png",
      "document_id": "abc123def456",
      "source_file": "attention_is_all_you_need.pdf",
      "chunk_id": "mm_a1b2c3d4e5f6"
    }
  ],
  "latency_ms": 45.2,
  "search_mode": "hybrid"
}
```

---

## Search Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `hybrid` | Weighted sum of text + visual scores (default) | General search |
| `text` | BGE-M3 description embedding only | Table/equation search |
| `visual` | ColPali visual embedding only | Finding similar-looking diagrams |

Adjust fusion weights for custom retrieval:
```json
{
  "query": "...",
  "search_mode": "hybrid",
  "text_weight": 0.7,
  "visual_weight": 0.3
}
```

---

## Running Tests

```bash
# Phase 3 tests only
pytest tests/test_image_processor.py -v
pytest tests/test_figure_captioner.py -v
pytest tests/test_chart_processor.py -v
pytest tests/test_table_processor.py -v
pytest tests/test_image_embedder.py -v
pytest tests/test_multimodal_indexer.py -v
pytest tests/test_multimodal_search.py -v

# All tests (Phase 1 + 2 + 3)
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Phase 4 Roadmap

| Feature | Extension Point |
|---------|----------------|
| Cross-encoder reranking | `rerank()` step in `search_service.py` + `multimodal_service.py` |
| Hybrid BM25 + dense | Add sparse vectors to `multimodal_indexer.py` |
| Citation-aware RAG | Use `chunk.citation_context` in LLM prompt builder |
| Conversational memory | New `services/memory_service.py` |
| LLM answer generation | New `services/generation_service.py` |
| CLIP cross-modal search | Add `encode_image()` to `image_embedder.py` |
| Late interaction (MaxSim) | Phase 4 ColPali patch-level scoring |
| Equation solving hints | Connect equation chunks to Wolfram/symbolic solver |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/upload` | Phase 1: Upload document |
| `POST` | `/api/v1/index` | Phase 2: Index text chunks |
| `POST` | `/api/v1/search` | Phase 2: Text semantic search |
| `POST` | `/api/v1/multimodal/index` | Phase 3: Multimodal indexing |
| `POST` | `/api/v1/multimodal/search` | Phase 3: Hybrid search |
| `GET` | `/api/v1/multimodal/health` | Phase 3: Collection health |
| `GET` | `/health` | System health check |
| `GET` | `/docs` | Swagger UI |