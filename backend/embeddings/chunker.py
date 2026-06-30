"""
Structure-aware semantic chunker for parsed research documents.

Strategy
--------
1. Walk the parsed document's sections in order.
2. Classify each section by its heading (abstract, conclusion, etc.).
3. If the section text fits within MAX_TOKENS, emit it as a single chunk.
4. If it exceeds MAX_TOKENS, apply a sliding window with token overlap
   so that each sub-chunk still carries meaningful context.
5. Tables and figure captions are always emitted as atomic chunks
   (never split) regardless of size.

This module deliberately does NOT use LangChain splitters so that
the chunking logic is fully transparent, testable, and customisable
for Phase 3 multimodal extensions.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from schemas.chunk import Chunk, ChunkType
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Tuneable constants ────────────────────────────────────────────────────────

# Rough token estimate: 1 token ≈ 4 characters (works well for English)
CHARS_PER_TOKEN: int = 4

TARGET_MIN_TOKENS: int = 400
TARGET_MAX_TOKENS: int = 700
OVERLAP_TOKENS: int = 80
HARD_MAX_TOKENS: int = 1200  # Paragraphs above this are always split

TARGET_MIN_CHARS: int = TARGET_MIN_TOKENS * CHARS_PER_TOKEN   # 1600
TARGET_MAX_CHARS: int = TARGET_MAX_TOKENS * CHARS_PER_TOKEN   # 2800
OVERLAP_CHARS: int = OVERLAP_TOKENS * CHARS_PER_TOKEN         # 320
HARD_MAX_CHARS: int = HARD_MAX_TOKENS * CHARS_PER_TOKEN       # 4800


# ── Heading → ChunkType classification ───────────────────────────────────────

HEADING_PATTERNS: list[tuple[re.Pattern, ChunkType]] = [
    (re.compile(r"\babstract\b", re.I), ChunkType.ABSTRACT),
    (re.compile(r"\bintroduction\b", re.I), ChunkType.INTRODUCTION),
    (re.compile(r"\bconclusion", re.I), ChunkType.CONCLUSION),
    (re.compile(r"\breference", re.I), ChunkType.REFERENCES),
    (re.compile(r"\bfigure\b|\bfig\.\b|\bcaption\b", re.I), ChunkType.FIGURE_CAPTION),
    (re.compile(r"\btable\b", re.I), ChunkType.TABLE),
    (re.compile(r"\bmethod", re.I), ChunkType.SUBSECTION),
    (re.compile(r"\bexperiment", re.I), ChunkType.SUBSECTION),
    (re.compile(r"\brelated\s+work", re.I), ChunkType.SUBSECTION),
    (re.compile(r"\bdiscussion", re.I), ChunkType.SUBSECTION),
    (re.compile(r"\bappendix", re.I), ChunkType.SUBSECTION),
]


def classify_heading(heading: str) -> ChunkType:
    """
    Map a section heading string to a ChunkType enum value.

    Args:
        heading: Raw heading text from the parsed document.

    Returns:
        The most specific matching ChunkType, or PARAGRAPH as default.
    """
    if not heading.strip():
        return ChunkType.PARAGRAPH

    for pattern, chunk_type in HEADING_PATTERNS:
        if pattern.search(heading):
            return chunk_type

    return ChunkType.HEADING


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.

    Uses the 1 token ≈ 4 chars heuristic which is accurate enough
    for chunking decisions without loading a full tokeniser.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


# ── Main chunker class ────────────────────────────────────────────────────────


class SemanticChunker:
    """
    Converts a parsed document JSON dict into a list of Chunk objects.

    The chunker respects document structure (sections, tables, titles)
    and applies sliding-window splitting only when necessary.

    Usage::

        chunker = SemanticChunker()
        chunks = chunker.chunk_document(parsed_doc_dict)
    """

    def __init__(
        self,
        target_max_chars: int = TARGET_MAX_CHARS,
        overlap_chars: int = OVERLAP_CHARS,
        hard_max_chars: int = HARD_MAX_CHARS,
    ) -> None:
        """
        Args:
            target_max_chars: Soft upper limit for chunk size in characters.
            overlap_chars:    Character overlap between consecutive sub-chunks.
            hard_max_chars:   Force-split any text exceeding this size.
        """
        self.target_max_chars = target_max_chars
        self.overlap_chars = overlap_chars
        self.hard_max_chars = hard_max_chars
        self._chunk_counter: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_document(self, doc: dict) -> list[Chunk]:
        """
        Convert a Phase 1 parsed document dict into semantic chunks.

        Processes in order:
          1. Title chunk
          2. Section chunks (with overflow splitting)
          3. Table chunks
          4. Figure caption chunks (from images metadata)

        Args:
            doc: Parsed document dict from output/json/<id>.json.

        Returns:
            Ordered list of Chunk objects ready for embedding.

        Raises:
            ValueError: If the document dict is missing required fields.
        """
        self._validate_document(doc)
        self._chunk_counter = 0

        document_id: str = doc["document_id"]
        source_file: str = doc.get("filename", "")
        title: str = doc.get("title", "")

        logger.info(
            "[Chunker] Starting chunking for document_id=%s title='%s'",
            document_id,
            title[:60],
        )

        chunks: list[Chunk] = []

        # ── 1. Title chunk ────────────────────────────────────────────────────
        if title.strip():
            chunks.append(
                self._make_chunk(
                    text=title,
                    document_id=document_id,
                    page=1,
                    heading="Title",
                    chunk_type=ChunkType.TITLE,
                    source_file=source_file,
                )
            )

        # ── 2. Section chunks ─────────────────────────────────────────────────
        for section in doc.get("sections", []):
            section_chunks = self._process_section(
                section=section,
                document_id=document_id,
                source_file=source_file,
            )
            chunks.extend(section_chunks)

        # ── 3. Table chunks ───────────────────────────────────────────────────
        for table in doc.get("tables", []):
            table_chunk = self._process_table(
                table=table,
                document_id=document_id,
                source_file=source_file,
            )
            if table_chunk:
                chunks.append(table_chunk)

        # ── 4. Figure caption chunks ──────────────────────────────────────────
        for image in doc.get("images", []):
            caption_chunk = self._process_image_caption(
                image=image,
                document_id=document_id,
                source_file=source_file,
            )
            if caption_chunk:
                chunks.append(caption_chunk)

        # Assign final sequential indices
        for idx, chunk in enumerate(chunks):
            chunk.chunk_index = idx

        logger.info(
            "[Chunker] Complete: %d chunks created for document_id=%s",
            len(chunks),
            document_id,
        )
        return chunks

    # ── Section processing ────────────────────────────────────────────────────

    def _process_section(
        self,
        section: dict,
        document_id: str,
        source_file: str,
    ) -> list[Chunk]:
        """
        Process a single section dict into one or more chunks.

        A short section → single chunk.
        A long section → emits a heading chunk + overlapping sub-chunks.

        Args:
            section:     Section dict with keys: page, heading, text.
            document_id: Parent document ID.
            source_file: Original filename.

        Returns:
            List of Chunk objects for this section.
        """
        heading: str = section.get("heading", "")
        text: str = section.get("text", "").strip()
        page: int = section.get("page", 1)
        chunk_type = classify_heading(heading)

        if not text:
            logger.debug("[Chunker] Skipping empty section: heading='%s'", heading)
            return []

        chunks: list[Chunk] = []

        # Emit a standalone heading chunk if heading is meaningful
        if heading.strip() and chunk_type not in (ChunkType.PARAGRAPH, ChunkType.UNKNOWN):
            chunks.append(
                self._make_chunk(
                    text=heading,
                    document_id=document_id,
                    page=page,
                    heading=heading,
                    chunk_type=ChunkType.HEADING,
                    source_file=source_file,
                )
            )

        # Split or emit body text
        if len(text) <= self.target_max_chars:
            chunks.append(
                self._make_chunk(
                    text=text,
                    document_id=document_id,
                    page=page,
                    heading=heading,
                    chunk_type=chunk_type,
                    source_file=source_file,
                )
            )
        else:
            # Long section — split into overlapping windows
            sub_chunks = self._split_with_overlap(
                text=text,
                document_id=document_id,
                page=page,
                heading=heading,
                chunk_type=chunk_type,
                source_file=source_file,
            )
            chunks.extend(sub_chunks)

        return chunks

    def _split_with_overlap(
        self,
        text: str,
        document_id: str,
        page: int,
        heading: str,
        chunk_type: ChunkType,
        source_file: str,
    ) -> list[Chunk]:
        """
        Apply a sentence-aware sliding window to split long text.

        The algorithm:
          1. Split text into sentences at '. ', '? ', '! ' boundaries.
          2. Accumulate sentences until target_max_chars is reached.
          3. Emit the window as a chunk.
          4. Step back overlap_chars worth of sentences for the next window.
          5. Repeat until the full text is covered.

        This produces much more coherent splits than character-level slicing
        because chunk boundaries always fall on sentence endings.

        Args:
            text:        Full section body text.
            document_id: Parent document ID.
            page:        Page number.
            heading:     Section heading.
            chunk_type:  Classified chunk type.
            source_file: Original filename.

        Returns:
            List of overlapping sub-chunks.
        """
        sentences = self._split_into_sentences(text)
        chunks: list[Chunk] = []
        window: list[str] = []
        window_len: int = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_len = len(sentence)

            # If adding this sentence keeps us within limit, accumulate
            if window_len + sentence_len <= self.target_max_chars:
                window.append(sentence)
                window_len += sentence_len
                i += 1
            else:
                # Flush window as a chunk (if non-empty)
                if window:
                    chunk_text = " ".join(window).strip()
                    if chunk_text:
                        chunks.append(
                            self._make_chunk(
                                text=chunk_text,
                                document_id=document_id,
                                page=page,
                                heading=heading,
                                chunk_type=chunk_type,
                                source_file=source_file,
                            )
                        )

                    # Step back: keep overlap_chars worth of sentences
                    overlap_window: list[str] = []
                    overlap_len: int = 0
                    for sent in reversed(window):
                        if overlap_len + len(sent) <= self.overlap_chars:
                            overlap_window.insert(0, sent)
                            overlap_len += len(sent)
                        else:
                            break

                    window = overlap_window
                    window_len = sum(len(s) for s in window)
                else:
                    # Single sentence exceeds limit — emit it anyway
                    chunks.append(
                        self._make_chunk(
                            text=sentence.strip(),
                            document_id=document_id,
                            page=page,
                            heading=heading,
                            chunk_type=chunk_type,
                            source_file=source_file,
                        )
                    )
                    window = []
                    window_len = 0
                    i += 1

        # Flush any remaining content
        if window:
            chunk_text = " ".join(window).strip()
            if chunk_text:
                chunks.append(
                    self._make_chunk(
                        text=chunk_text,
                        document_id=document_id,
                        page=page,
                        heading=heading,
                        chunk_type=chunk_type,
                        source_file=source_file,
                    )
                )

        logger.debug(
            "[Chunker] Split long section '%s' into %d sub-chunks",
            heading[:40],
            len(chunks),
        )
        return chunks

    # ── Table processing ──────────────────────────────────────────────────────

    def _process_table(
        self,
        table: dict,
        document_id: str,
        source_file: str,
    ) -> Chunk | None:
        """
        Convert a table dict into a single atomic TABLE chunk.

        Tables are never split because their row/column relationships
        would become meaningless if fragmented.

        Args:
            table:       Table dict with keys: page, headers, rows, raw_text.
            document_id: Parent document ID.
            source_file: Original filename.

        Returns:
            A single Chunk or None if the table has no usable text.
        """
        page: int = table.get("page", 1)
        table_number: int = table.get("table_number", 0)
        headers: list[str] = table.get("headers", [])
        rows: list[list[str]] = table.get("rows", [])
        raw_text: str = table.get("raw_text", "")

        # Build readable text representation
        text_parts: list[str] = []

        if headers:
            text_parts.append("Headers: " + " | ".join(str(h) for h in headers))

        for row in rows[:20]:  # Cap at 20 rows to avoid gigantic embeddings
            text_parts.append(" | ".join(str(cell) for cell in row))

        table_text = "\n".join(text_parts).strip() or raw_text.strip()

        if not table_text:
            logger.debug("[Chunker] Skipping empty table %d on page %d", table_number, page)
            return None

        return self._make_chunk(
            text=table_text,
            document_id=document_id,
            page=page,
            heading=f"Table {table_number}",
            chunk_type=ChunkType.TABLE,
            source_file=source_file,
        )

    # ── Figure caption processing ─────────────────────────────────────────────

    def _process_image_caption(
        self,
        image: dict,
        document_id: str,
        source_file: str,
    ) -> Chunk | None:
        """
        Create a FIGURE_CAPTION chunk for an extracted image.

        In Phase 1, images don't have caption text extracted separately,
        so we synthesise a minimal caption. Phase 3 can enhance this
        by running a vision model to generate richer captions.

        Args:
            image:       Image dict with keys: page, image_path, image_number.
            document_id: Parent document ID.
            source_file: Original filename.

        Returns:
            A Chunk with figure reference info, or None.
        """
        page: int = image.get("page", 1)
        image_path: str = image.get("image_path", "")
        image_number: int = image.get("image_number", 0)

        if not image_path:
            return None

        text = f"Figure {image_number} on page {page}. Source: {image_path}"

        return self._make_chunk(
            text=text,
            document_id=document_id,
            page=page,
            heading=f"Figure {image_number}",
            chunk_type=ChunkType.FIGURE_CAPTION,
            source_file=source_file,
            multimodal_ref=image_path,  # Phase 3: attach image path for multimodal retrieval
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_chunk(
        self,
        text: str,
        document_id: str,
        page: int,
        heading: str,
        chunk_type: ChunkType,
        source_file: str,
        multimodal_ref: str | None = None,
    ) -> Chunk:
        """
        Construct a Chunk with a sequential ID and token estimate.

        Args:
            text:           Chunk body text.
            document_id:    Parent document ID.
            page:           1-based page number.
            heading:        Section heading.
            chunk_type:     Semantic classification.
            source_file:    Original filename.
            multimodal_ref: Optional path to associated image (Phase 3).

        Returns:
            Populated Chunk instance.
        """
        self._chunk_counter += 1
        chunk_id = f"chunk_{self._chunk_counter:05d}_{document_id[:8]}"

        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            page=page,
            heading=heading,
            chunk_type=chunk_type,
            text=text,
            source_file=source_file,
            token_count=estimate_tokens(text),
            multimodal_ref=multimodal_ref,
        )

        logger.debug(
            "[Chunker] Created %s chunk_id=%s page=%d tokens≈%d heading='%s'",
            chunk_type.value,
            chunk_id,
            page,
            chunk.token_count,
            heading[:40],
        )
        return chunk

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Split text into sentences using punctuation boundaries.

        Uses a simple but robust regex that handles abbreviations like
        "et al." and "Fig." better than naive period splitting.

        Args:
            text: Raw paragraph text.

        Returns:
            List of sentence strings (may still be long for dense text).
        """
        # Split on sentence-ending punctuation followed by whitespace + capital
        sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        sentences = sentence_pattern.split(text)
        # Remove empty strings
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _validate_document(doc: dict) -> None:
        """
        Validate that the document dict has the minimum required fields.

        Args:
            doc: Parsed document dictionary.

        Raises:
            ValueError: If required fields are missing.
        """
        required = ("document_id",)
        missing = [k for k in required if k not in doc]
        if missing:
            raise ValueError(
                f"Document dict is missing required fields: {missing}. "
                "Ensure Phase 1 parsing completed successfully."
            )