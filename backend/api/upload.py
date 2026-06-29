"""
Upload API router — handles POST /upload multipart file submissions.

Coordinates the full ingestion pipeline:
  1. Receive and validate the uploaded file
  2. Persist it to uploads/
  3. Route to the correct parser via DocumentRouter
  4. Export the parsed result as JSON
  5. Return a structured UploadResponse
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse

from models.document import UploadResponse
from parser.document_router import DocumentRouter
from services.export_service import ExportService
from utils.file_utils import generate_document_id, detect_file_type, ensure_directories
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ── Directory constants ───────────────────────────────────────────────────────

UPLOADS_DIR = Path("uploads")
OUTPUT_IMAGES_DIR = Path("output/images")
OUTPUT_JSON_DIR = Path("output/json")

# Maximum upload size: 100 MB
MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and parse a research document",
    description=(
        "Accepts a PDF, image, DOCX, or PPTX file via multipart/form-data. "
        "Extracts structured content including sections, images, and tables. "
        "Returns a structured JSON response and persists output to disk."
    ),
)
async def upload_document(
    document: UploadFile = File(..., description="The document file to parse"),
) -> UploadResponse:
    """
    Main upload endpoint for the document ingestion pipeline.

    Args:
        document: The uploaded file from multipart form data.

    Returns:
        UploadResponse with extracted document structure.

    Raises:
        HTTPException 400: Invalid or missing file.
        HTTPException 415: Unsupported file type.
        HTTPException 422: Encrypted or corrupted document.
        HTTPException 500: Internal parsing failure.
    """
    # ── 1. Basic validation ───────────────────────────────────────────────────
    if not document.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided in the upload.",
        )

    filename = document.filename
    logger.info("Upload received: filename=%s content_type=%s", filename, document.content_type)

    # ── 2. Generate IDs and paths ─────────────────────────────────────────────
    document_id = generate_document_id(filename)
    upload_path = UPLOADS_DIR / f"{document_id}_{filename}"
    output_image_dir = OUTPUT_IMAGES_DIR / document_id

    ensure_directories(UPLOADS_DIR, output_image_dir, OUTPUT_JSON_DIR)

    # ── 3. Stream file to disk ────────────────────────────────────────────────
    await _save_upload(document, upload_path)

    # ── 4. Validate file type ─────────────────────────────────────────────────
    try:
        file_type = detect_file_type(upload_path)
        logger.info("Detected file type: %s for document_id=%s", file_type, document_id)
    except ValueError as exc:
        logger.warning("Unsupported file type for %s: %s", filename, exc)
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )

    # ── 5. Parse document ─────────────────────────────────────────────────────
    try:
        parser_router = DocumentRouter(
            file_path=upload_path,
            document_id=document_id,
            output_image_dir=output_image_dir,
        )
        parsed = parser_router.route()
        logger.info(
            "Parsing complete for document_id=%s: sections=%d images=%d tables=%d",
            document_id,
            len(parsed.sections),
            len(parsed.images),
            len(parsed.tables),
        )
    except ValueError as exc:
        # Encrypted, password-protected, or structurally invalid document
        logger.error("Document validation error for %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.error("Parse failure for %s: %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document parsing failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error parsing %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during parsing: {exc}",
        )

    # ── 6. Export JSON ────────────────────────────────────────────────────────
    try:
        exporter = ExportService(OUTPUT_JSON_DIR)
        json_path = exporter.save(parsed)
        logger.info("JSON export saved: %s", json_path)
    except Exception as exc:
        logger.error("JSON export failed for %s: %s", document_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export parsed document: {exc}",
        )

    # ── 7. Build and return response ──────────────────────────────────────────
    response = UploadResponse(
        document_id=document_id,
        filename=filename,
        pages=parsed.metadata.pages,
        sections=parsed.sections,
        images=parsed.images,
        tables=parsed.tables,
        metadata=parsed.metadata,
        ocr_applied=parsed.ocr_applied,
        output_json_path=str(json_path),
        message=(
            "Document parsed successfully."
            if not parsed.parse_errors
            else f"Parsed with {len(parsed.parse_errors)} warning(s)."
        ),
    )

    logger.info("Upload pipeline complete for document_id=%s", document_id)
    return response


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _save_upload(upload_file: UploadFile, destination: Path) -> None:
    """
    Stream an UploadFile to disk asynchronously with size enforcement.

    Args:
        upload_file:  FastAPI UploadFile object.
        destination:  Target path to write the file.

    Raises:
        HTTPException 400: If the file exceeds MAX_FILE_SIZE_BYTES.
        HTTPException 400: If the file is empty.
    """
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KB chunks

    try:
        async with aiofiles.open(destination, "wb") as out_file:
            while chunk := await upload_file.read(chunk_size):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    await out_file.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"File size exceeds the maximum allowed limit of "
                            f"{MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
                        ),
                    )
                await out_file.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.error("Failed to save upload to %s: %s", destination, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        )

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    logger.info(
        "Upload saved: %s (%.2f KB)", destination.name, total_bytes / 1024
    )