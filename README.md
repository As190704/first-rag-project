# first-rag-project
A production-grade document ingestion pipeline that parses research papers (PDF, scanned PDF, images, DOCX, PPTX) and extracts structured content ready for embedding and retrieval in later phases.  ---


# Multimodal RAG System — Phase 1: Document Ingestion Pipeline

A production-grade document ingestion pipeline that parses research papers
(PDF, scanned PDF, images, DOCX, PPTX) and extracts structured content
ready for embedding and retrieval in later phases.

---

## Architecture Overview

```
Upload Request
     │
     ▼
POST /api/v1/upload
     │
     ▼
DocumentRouter  ──── detect file type
     │
     ├── PDF ──────► PDFParser
     │                   ├── Docling  (structured extraction)
     │                   ├── PyMuPDF  (page rendering + image extraction)
     │                   └── PaddleOCR (scanned page fallback)
     │
     ├── Image ────► OCREngine (PaddleOCR)
     │
     ├── DOCX ─────► Placeholder (Phase 2)
     │
     └── PPTX ─────► Placeholder (Phase 2)
          │
          ▼
     ParsedDocument (Pydantic model)
          │
          ▼
     ExportService ──► output/json/<id>.json
          │
          ▼
     UploadResponse (API JSON)
```

---

## Requirements

- Python 3.12
- pip

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd backend
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PaddlePaddle:**  
> If you encounter issues installing `paddlepaddle`, install it separately first:
> ```bash
> pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
> pip install paddleocr
> ```

### 4. Install system dependencies (Linux/macOS)

```bash
# Ubuntu / Debian
sudo apt-get install -y libmagic1 libgl1-mesa-glx libglib2.0-0

# macOS
brew install libmagic
```

### 5. Start the server

```bash
python app.py
# OR with uvicorn directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at: `http://localhost:8000`

---

## API Reference

### POST `/api/v1/upload`

Upload a research document for parsing.

**Request:**
```
Content-Type: multipart/form-data
Field: document (file)
```

**Supported formats:**
| Format | Extension | Parser |
|--------|-----------|--------|
| PDF | `.pdf` | Docling + PyMuPDF + PaddleOCR |
| Scanned PDF | `.pdf` | PaddleOCR (auto-detected) |
| Image | `.png .jpg .jpeg` | PaddleOCR |
| DOCX | `.docx` | Placeholder (Phase 2) |
| PPTX | `.pptx` | Placeholder (Phase 2) |

**Response:**
```json
{
  "document_id": "abc123def456",
  "filename": "attention_is_all_you_need.pdf",
  "pages": 15,
  "sections": [
    {
      "page": 1,
      "heading": "Abstract",
      "text": "The dominant sequence transduction models...",
      "level": 1
    }
  ],
  "images": [
    {
      "page": 3,
      "image_number": 1,
      "image_path": "output/images/abc123def456/img_001_p3.png",
      "width": 640,
      "height": 480,
      "bounding_box": {"x0": 72.0, "y0": 150.0, "x1": 540.0, "y1": 420.0}
    }
  ],
  "tables": [],
  "metadata": {
    "authors": [],
    "year": "2017",
    "pages": 15,
    "doi": "",
    "keywords": []
  },
  "ocr_applied": false,
  "output_json_path": "output/json/abc123def456.json",
  "message": "Document parsed successfully."
}
```

### GET `/health`

Health check endpoint.

```json
{
  "status": "healthy",
  "phase": 1,
  "service": "document-ingestion",
  "version": "1.0.0"
}
```

### Interactive Docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Testing with cURL

```bash
# Upload a PDF
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "accept: application/json" \
  -F "document=@/path/to/paper.pdf"

# Upload an image
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "accept: application/json" \
  -F "document=@/path/to/figure.png"
```

---

## Testing with Python

```python
import httpx

with open("paper.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/upload",
        files={"document": ("paper.pdf", f, "application/pdf")},
    )

print(response.json())
```

---

## Output Files

```
output/
├── json/
│   └── abc123def456.json          # Full structured document
└── images/
    └── abc123def456/
        ├── img_001_p3.png         # Image from page 3
        └── img_002_p7.jpg         # Image from page 7

logs/
└── pipeline.log                   # Full debug log
```

---

## Error Codes

| Code | Condition |
|------|-----------|
| 400  | Empty file, file too large, missing filename |
| 415  | Unsupported file format |
| 422  | Encrypted/corrupted PDF |
| 500  | Internal parsing or export failure |

---

## Phase 2 Roadmap

- [ ] DOCX parsing with `python-docx`
- [ ] PPTX parsing with `python-pptx`
- [ ] Text chunking with configurable overlap
- [ ] Embedding generation (OpenAI / local models)
- [ ] Vector database ingestion (Qdrant / Chroma)
- [ ] Multimodal image embeddings (CLIP)
- [ ] LLM query integration

---

## Project Structure

```
backend/
├── app.py                    # FastAPI app factory and entry point
├── api/
│   └── upload.py             # POST /upload endpoint
├── parser/
│   ├── pdf_parser.py         # PDF parsing (Docling + MuPDF + OCR)
│   ├── image_extractor.py    # Embedded image extraction
│   ├── ocr.py                # PaddleOCR wrapper
│   └── document_router.py    # File type detection and dispatch
├── models/
│   └── document.py           # Pydantic data models
├── services/
│   └── export_service.py     # JSON serialisation and persistence
├── utils/
│   ├── logger.py             # Logging configuration
│   └── file_utils.py         # File helpers and MIME detection
├── uploads/                  # Temporary upload storage
├── output/
│   ├── json/                 # Parsed document JSON files
│   └── images/               # Extracted document images
└── logs/                     # Application logs
```


┌─────────────────────────────────────────────────────────────────┐
│                    DESIGN DECISION NOTES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SINGLETON OCR ENGINE                                        │
│     PaddleOCR takes ~5s to initialise. The singleton pattern   │
│     ensures it loads once and is reused across all requests.    │
│                                                                 │
│  2. DOCLING + MUPDF DUAL STRATEGY                               │
│     Docling gives semantic structure (headings, tables).        │
│     PyMuPDF gives reliable page-level text for OCR detection    │
│     and image extraction with bounding boxes.                   │
│                                                                 │
│  3. SCANNED PAGE DETECTION                                      │
│     A page with < 50 chars of native text is assumed scanned.  │
│     This threshold is configurable via SCANNED_TEXT_THRESHOLD.  │
│                                                                 │
│  4. PYDANTIC MODELS AS CONTRACT                                 │
│     ParsedDocument is the single source of truth shared by      │
│     parser, exporter, and API response. No ad-hoc dicts.       │
│                                                                 │
│  5. ASYNC FILE I/O                                              │
│     aiofiles ensures the upload endpoint never blocks the       │
│     event loop while streaming large files to disk.             │
│                                                                 │
│  6. SECTION MERGING STRATEGY                                    │
│     OCR sections replace Docling sections for the same page.   │
│     This prevents duplicate content on partially-scanned docs.  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘