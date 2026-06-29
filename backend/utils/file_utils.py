"""
File system utility helpers for the ingestion pipeline.

Handles directory creation, MIME type detection, unique ID generation,
and safe file operations used across parser modules.
"""

import hashlib
import mimetypes
import uuid
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Supported MIME types ──────────────────────────────────────────────────────
SUPPORTED_MIME_TYPES: dict[str, str] = {
    "application/pdf": "pdf",
    "image/png": "image",
    "image/jpeg": "image",
    "image/jpg": "image",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
}

# Extension-based fallback map
EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".docx": "docx",
    ".pptx": "pptx",
}


def generate_document_id(filename: str) -> str:
    """
    Generate a short, reproducible document ID derived from filename + UUID.

    Args:
        filename: Original uploaded filename.

    Returns:
        An 8-character hex string prefixed with a UUID fragment.
    """
    unique = uuid.uuid4().hex[:8]
    name_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    doc_id = f"{name_hash}{unique}"
    logger.debug("Generated document_id=%s for filename=%s", doc_id, filename)
    return doc_id


def detect_file_type(file_path: Path) -> str:
    """
    Detect the semantic file type using MIME type and extension fallback.

    Args:
        file_path: Path to the file on disk.

    Returns:
        One of: 'pdf', 'image', 'docx', 'pptx'.

    Raises:
        ValueError: If the file type is unsupported.
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    logger.debug("Detected MIME type=%s for %s", mime_type, file_path.name)

    if mime_type and mime_type in SUPPORTED_MIME_TYPES:
        return SUPPORTED_MIME_TYPES[mime_type]

    # Fallback to extension
    suffix = file_path.suffix.lower()
    if suffix in EXTENSION_MAP:
        logger.warning(
            "MIME detection failed for %s; falling back to extension=%s",
            file_path.name,
            suffix,
        )
        return EXTENSION_MAP[suffix]

    raise ValueError(
        f"Unsupported file type: MIME='{mime_type}', extension='{suffix}'. "
        f"Supported formats: PDF, PNG, JPG, JPEG, DOCX, PPTX."
    )


def ensure_directories(*paths: Path) -> None:
    """
    Create one or more directories (including parents) if they do not exist.

    Args:
        *paths: One or more Path objects to create.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", path)


def safe_delete(file_path: Path) -> None:
    """
    Delete a file if it exists, silently ignoring missing files.

    Args:
        file_path: Path to the file to remove.
    """
    try:
        file_path.unlink(missing_ok=True)
        logger.debug("Deleted temporary file: %s", file_path)
    except OSError as exc:
        logger.warning("Could not delete file %s: %s", file_path, exc)