"""
Multimodal indexing and search service.

Orchestrates the complete Phase 3 pipeline:
  1. Load Phase 1 JSON for image/table metadata
  2. Process every image (classify → describe → embed)
  3. Process every table (Camelot → Docling → OCR)
  4. Detect equations
  5. Store all multimodal chunks in Qdrant
  6. Provide hybrid multimodal search

This service is the sole dependency of the multimodal API layer.
Phase 4 can add reranking by inserting a rerank() step before return.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from embeddings.embedder import embedding_engine
from multimodal.chart_processor import ChartProcessor
from multimodal.equation_processor import EquationProcessor
from multimodal.figure_captioner import figure_captioner
from multimodal.image_embedder import colpali_embedder
from multimodal.image_processor import ImageProcessor, ProcessedImage
from multimodal.table_processor import TableProcessor
from schemas.multimodal_chunk import (
    ChartData,
    EmbeddedMultimodalChunk,
    EquationData,
    MultimodalChunk,
    MultimodalIndexRequest,
    MultimodalSearchRequest,
    MultimodalSearchResponse,
    MultimodalSearchResult,
    TableData,
    VisualChunkType,
)
from vector_db.multimodal_indexer import (
    MULTIMODAL_COLLECTION,
    MultimodalIndexer,
    delete_document_multimodal_vectors,
    ensure_multimodal_collection,
)
from vector_db.qdrant_client import get_qdrant_client
from qdrant_client.http import models as qmodels
from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_JSON_DIR = Path("output/json")


class MultimodalService:
    """
    Orchestrates multimodal document indexing and hybrid search.

    Usage::

        service = MultimodalService()

        # Index
        stats = service.index_document(request)

        # Search
        response = service.search(request)
    """

    def __init__(self) -> None:
        self.image_processor = ImageProcessor()
        self.chart_processor = ChartProcessor()
        self.table_processor = TableProcessor()
        self.equation_processor = EquationProcessor()
        self.indexer = MultimodalIndexer()

    # ── Indexing pipeline ─────────────────────────────────────────────────────

    def index_document(self, request: MultimodalIndexRequest) -> dict:
        """
        Run the full multimodal indexing pipeline for one document.

        Args:
            request: MultimodalIndexRequest with document_id and feature flags.

        Returns:
            Statistics dictionary with counts of each visual element type
            processed and the total vectors stored.
        """
        t_start = time.perf_counter()
        document_id = request.document_id

        logger.info(
            "[MultimodalService] Indexing document_id=%s", document_id
        )

        # ── Load Phase 1 JSON ─────────────────────────────────────────────────
        doc = self._load_document_json(document_id)
        pdf_path = self._resolve_pdf_path(doc)

        # ── Ensure collection exists ──────────────────────────────────────────
        ensure_multimodal_collection()

        # ── Handle re-indexing ────────────────────────────────────────────────
        if request.force_reindex and self.indexer.document_is_indexed(document_id):
            logger.info(
                "[MultimodalService] force_reindex: removing existing vectors."
            )
            delete_document_multimodal_vectors(document_id)

        # ── Process visual elements ───────────────────────────────────────────
        all_embedded: list[EmbeddedMultimodalChunk] = []
        stats = {
            "figures": 0,
            "charts": 0,
            "tables": 0,
            "equations": 0,
        }

        # 1. Images (figures, charts, diagrams, equations)
        image_records = doc.get("images", [])
        if image_records:
            processed_images = self.image_processor.process_image_list(image_records)
            embedded_images = self._process_images(
                processed_images=processed_images,
                document_id=document_id,
                source_file=doc.get("filename", ""),
                run_captioning=request.run_captioning,
                run_chart_ocr=request.run_chart_ocr,
                run_equation_detection=request.run_equation_detection,
            )
            all_embedded.extend(embedded_images)

            # Update stats
            for ec in embedded_images:
                ct = ec.chunk.chunk_type
                if ct == VisualChunkType.CHART:
                    stats["charts"] += 1
                elif ct == VisualChunkType.EQUATION:
                    stats["equations"] += 1
                else:
                    stats["figures"] += 1

        # 2. Tables from Phase 1 JSON
        if request.run_table_extraction:
            table_records = doc.get("tables", [])
            embedded_tables = self._process_phase1_tables(
                table_records=table_records,
                document_id=document_id,
                source_file=doc.get("filename", ""),
                pdf_path=pdf_path,
            )
            all_embedded.extend(embedded_tables)
            stats["tables"] += len(embedded_tables)

        # ── Store in Qdrant ───────────────────────────────────────────────────
        vectors_stored = 0
        if all_embedded:
            vectors_stored = self.indexer.index(all_embedded)
        else:
            logger.warning(
                "[MultimodalService] No multimodal content found for %s",
                document_id,
            )

        elapsed = time.perf_counter() - t_start

        return {
            "document_id": document_id,
            "figures_processed": stats["figures"],
            "charts_processed": stats["charts"],
            "tables_processed": stats["tables"],
            "equations_detected": stats["equations"],
            "vectors_stored": vectors_stored,
            "collection_name": MULTIMODAL_COLLECTION,
            "duration_seconds": round(elapsed, 3),
            "message": (
                f"Indexed {vectors_stored} multimodal vectors "
                f"({stats['figures']} figures, {stats['charts']} charts, "
                f"{stats['tables']} tables, {stats['equations']} equations) "
                f"in {elapsed:.2f}s."
            ),
        }

    # ── Search pipeline ───────────────────────────────────────────────────────

    def search(self, request: MultimodalSearchRequest) -> MultimodalSearchResponse:
        """
        Execute hybrid multimodal search.

        Generates both a text embedding (BGE-M3) and a visual query
        embedding (ColPali), searches both vector spaces, merges results
        by weighted score fusion, deduplicates, and returns top-K.

        Args:
            request: MultimodalSearchRequest with query and filters.

        Returns:
            MultimodalSearchResponse with ranked results.
        """
        t_start = time.perf_counter()

        logger.info(
            "[MultimodalService] Search query='%s' mode=%s top_k=%d",
            request.query[:80],
            request.search_mode,
            request.top_k,
        )

        query_filter = self._build_search_filter(request)
        results_map: dict[str, tuple[float, dict]] = {}

        # ── Text-side search ──────────────────────────────────────────────────
        if request.search_mode in ("text", "hybrid"):
            text_results = self._search_by_text(
                query=request.query,
                top_k=request.top_k * 2,  # Over-fetch for fusion
                query_filter=query_filter,
                weight=request.text_weight,
            )
            for chunk_id, score, payload in text_results:
                results_map[chunk_id] = (
                    results_map.get(chunk_id, (0.0, payload))[0] + score,
                    payload,
                )

        # ── Visual-side search ────────────────────────────────────────────────
        if request.search_mode in ("visual", "hybrid"):
            visual_results = self._search_by_visual(
                query=request.query,
                top_k=request.top_k * 2,
                query_filter=query_filter,
                weight=request.visual_weight,
            )
            for chunk_id, score, payload in visual_results:
                current_score = results_map.get(chunk_id, (0.0, payload))[0]
                results_map[chunk_id] = (current_score + score, payload)

        # ── Sort, deduplicate, and trim ───────────────────────────────────────
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x[0],
            reverse=True,
        )[: request.top_k]

        latency_ms = (time.perf_counter() - t_start) * 1000

        search_results = [
            self._payload_to_result(score, payload)
            for score, payload in sorted_results
        ]

        logger.info(
            "[MultimodalService] Search complete: %d results in %.2fms",
            len(search_results),
            latency_ms,
        )

        return MultimodalSearchResponse(
            query=request.query,
            total_results=len(search_results),
            results=search_results,
            latency_ms=round(latency_ms, 2),
            search_mode=request.search_mode,
        )

    # ── Private: image pipeline ───────────────────────────────────────────────

    def _process_images(
        self,
        processed_images: list[ProcessedImage],
        document_id: str,
        source_file: str,
        run_captioning: bool,
        run_chart_ocr: bool,
        run_equation_detection: bool,
    ) -> list[EmbeddedMultimodalChunk]:
        """
        Process classified images through captioning and embedding.

        For each image:
          1. Generate Qwen2-VL description (if run_captioning)
          2. Run type-specific processor (chart/equation)
          3. Generate ColPali visual embedding
          4. Generate BGE-M3 text embedding from description
          5. Assemble EmbeddedMultimodalChunk

        Args:
            processed_images: List of classified ProcessedImage objects.
            document_id:      Parent document ID.
            source_file:      Original filename.
            run_captioning:   Whether to use Qwen2-VL.
            run_chart_ocr:    Whether to run ChartOCR on chart images.
            run_equation_detection: Whether to process equations.

        Returns:
            List of EmbeddedMultimodalChunk objects.
        """
        if not processed_images:
            return []

        logger.info(
            "[MultimodalService] Processing %d images...", len(processed_images)
        )

        # ── Step 1: Batch captioning ──────────────────────────────────────────
        descriptions: list[str] = []
        if run_captioning:
            images_and_types = [
                (img.pil_image, img.classification) for img in processed_images
            ]
            descriptions = figure_captioner.describe_batch(images_and_types)
        else:
            descriptions = [
                figure_captioner._fallback_description(img.classification)
                for img in processed_images
            ]

        # ── Step 2: Type-specific processing ─────────────────────────────────
        chunks: list[MultimodalChunk] = []

        for idx, (proc_img, description) in enumerate(
            zip(processed_images, descriptions)
        ):
            chunk = self._build_image_chunk(
                proc_img=proc_img,
                description=description,
                document_id=document_id,
                source_file=source_file,
                run_chart_ocr=run_chart_ocr,
                run_equation_detection=run_equation_detection,
            )
            chunks.append(chunk)

        # ── Step 3: Visual embeddings (ColPali batch) ─────────────────────────
        pil_images = [pi.pil_image for pi in processed_images]
        visual_vectors = colpali_embedder.embed_images(pil_images)

        # ── Step 4: Text embeddings (BGE-M3 batch) ────────────────────────────
        embedding_texts = [chunk.get_embedding_text() for chunk in chunks]
        text_vectors = embedding_engine.embed_texts(embedding_texts)

        # ── Step 5: Assemble EmbeddedMultimodalChunks ─────────────────────────
        embedded: list[EmbeddedMultimodalChunk] = []
        for chunk, vis_vec, txt_vec in zip(chunks, visual_vectors, text_vectors):
            embedded.append(
                EmbeddedMultimodalChunk(
                    chunk=chunk,
                    visual_embedding=vis_vec,
                    text_embedding=txt_vec,
                )
            )

        logger.info(
            "[MultimodalService] Embedded %d image chunks.", len(embedded)
        )
        return embedded

    def _build_image_chunk(
        self,
        proc_img: ProcessedImage,
        description: str,
        document_id: str,
        source_file: str,
        run_chart_ocr: bool,
        run_equation_detection: bool,
    ) -> MultimodalChunk:
        """
        Build a MultimodalChunk for a single processed image.

        Args:
            proc_img:              Classified ProcessedImage.
            description:           Qwen2-VL description.
            document_id:           Parent document ID.
            source_file:           Original filename.
            run_chart_ocr:         Whether to run ChartOCR.
            run_equation_detection: Whether to process equations.

        Returns:
            Populated MultimodalChunk.
        """
        from schemas.multimodal_chunk import ImageClassification

        classification = proc_img.classification

        # Map ImageClassification → VisualChunkType
        type_map = {
            ImageClassification.FIGURE: VisualChunkType.FIGURE,
            ImageClassification.DIAGRAM: VisualChunkType.DIAGRAM,
            ImageClassification.FLOWCHART: VisualChunkType.FLOWCHART,
            ImageClassification.ARCHITECTURE: VisualChunkType.ARCHITECTURE,
            ImageClassification.SCREENSHOT: VisualChunkType.SCREENSHOT,
            ImageClassification.CHART: VisualChunkType.CHART,
            ImageClassification.TABLE: VisualChunkType.TABLE,
            ImageClassification.EQUATION: VisualChunkType.EQUATION,
            ImageClassification.UNKNOWN: VisualChunkType.UNKNOWN,
        }
        chunk_type = type_map.get(classification, VisualChunkType.UNKNOWN)

        chunk = MultimodalChunk(
            document_id=document_id,
            page=proc_img.page,
            chunk_type=chunk_type,
            image_number=proc_img.image_number,
            image_path=str(proc_img.path),
            caption=f"{chunk_type.value.title()} on page {proc_img.page}",
            description=description,
            bounding_box=proc_img.bounding_box,
            source_file=source_file,
        )

        # ── Chart-specific processing ─────────────────────────────────────────
        if chunk_type == VisualChunkType.CHART and run_chart_ocr:
            try:
                chunk.chart_data = self.chart_processor.process(
                    image_path=proc_img.path,
                    pil_image=proc_img.pil_image,
                    vlm_description=description,
                )
            except Exception as exc:
                logger.warning("[MultimodalService] Chart processing failed: %s", exc)

        # ── Equation-specific processing ──────────────────────────────────────
        elif chunk_type == VisualChunkType.EQUATION and run_equation_detection:
            try:
                chunk.equation_data = self.equation_processor.process(
                    image_path=proc_img.path,
                    pil_image=proc_img.pil_image,
                    vlm_description=description,
                )
            except Exception as exc:
                logger.warning("[MultimodalService] Equation processing failed: %s", exc)

        # ── Table image processing ────────────────────────────────────────────
        elif chunk_type == VisualChunkType.TABLE:
            try:
                chunk.table_data = self.table_processor.extract_from_image(
                    image_path=proc_img.path,
                    pil_image=proc_img.pil_image,
                    vlm_description=description,
                )
            except Exception as exc:
                logger.warning("[MultimodalService] Table-image processing failed: %s", exc)

        return chunk

    def _process_phase1_tables(
        self,
        table_records: list[dict],
        document_id: str,
        source_file: str,
        pdf_path: Path | None,
    ) -> list[EmbeddedMultimodalChunk]:
        """
        Convert Phase 1 table records into embedded multimodal chunks.

        Uses Camelot to re-extract tables when a PDF path is available,
        falling back to the Docling-extracted data in the JSON.

        Args:
            table_records: List of table dicts from Phase 1 JSON.
            document_id:   Parent document ID.
            source_file:   Original filename.
            pdf_path:      Path to source PDF (for Camelot re-extraction).

        Returns:
            List of EmbeddedMultimodalChunk for each table.
        """
        if not table_records:
            return []

        logger.info(
            "[MultimodalService] Processing %d Phase 1 tables...",
            len(table_records),
        )

        chunks: list[MultimodalChunk] = []

        for table_dict in table_records:
            page = table_dict.get("page", 1)
            table_number = table_dict.get("table_number", 0)

            # Try Camelot first if PDF available
            table_data: TableData | None = None

            if pdf_path and pdf_path.exists():
                camelot_results = self.table_processor.extract_from_pdf_page(
                    pdf_path, page
                )
                if camelot_results:
                    table_data = camelot_results[0]

            # Fall back to Phase 1 Docling data
            if table_data is None:
                table_data = self.table_processor.from_phase1_table(table_dict)

            chunk = MultimodalChunk(
                document_id=document_id,
                page=page,
                chunk_type=VisualChunkType.TABLE,
                image_number=table_number,
                image_path="",
                caption=f"Table {table_number} on page {page}",
                description=table_data.description,
                source_file=source_file,
                table_data=table_data,
            )
            chunks.append(chunk)

        # ── Text embeddings for tables ────────────────────────────────────────
        embedding_texts = [chunk.get_embedding_text() for chunk in chunks]
        text_vectors = embedding_engine.embed_texts(embedding_texts)

        embedded: list[EmbeddedMultimodalChunk] = []
        for chunk, txt_vec in zip(chunks, text_vectors):
            embedded.append(
                EmbeddedMultimodalChunk(
                    chunk=chunk,
                    visual_embedding=[0.0] * colpali_embedder.visual_dim,
                    text_embedding=txt_vec,
                )
            )

        logger.info(
            "[MultimodalService] Embedded %d table chunks.", len(embedded)
        )
        return embedded

    # ── Private: search helpers ───────────────────────────────────────────────

    def _search_by_text(
        self,
        query: str,
        top_k: int,
        query_filter,
        weight: float,
    ) -> list[tuple[str, float, dict]]:
        """
        Search the 'text' named vector space.

        Args:
            query:        Query string.
            top_k:        Number of results.
            query_filter: Qdrant filter object.
            weight:       Score multiplier for fusion.

        Returns:
            List of (chunk_id, weighted_score, payload) tuples.
        """
        try:
            text_vector = embedding_engine.embed_query(query)
            client = get_qdrant_client()

            hits = client.search(
                collection_name=MULTIMODAL_COLLECTION,
                query_vector=("text", text_vector),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            return [
                (
                    str(hit.payload.get("chunk_id", hit.id)),
                    float(hit.score) * weight,
                    hit.payload or {},
                )
                for hit in hits
            ]
        except Exception as exc:
            logger.warning("[MultimodalService] Text search failed: %s", exc)
            return []

    def _search_by_visual(
        self,
        query: str,
        top_k: int,
        query_filter,
        weight: float,
    ) -> list[tuple[str, float, dict]]:
        """
        Search the 'visual' named vector space using ColPali query encoder.

        Args:
            query:        Query string (ColPali encodes it for visual search).
            top_k:        Number of results.
            query_filter: Qdrant filter object.
            weight:       Score multiplier for fusion.

        Returns:
            List of (chunk_id, weighted_score, payload) tuples.
        """
        try:
            visual_vector = colpali_embedder.embed_query_text(query)
            client = get_qdrant_client()

            hits = client.search(
                collection_name=MULTIMODAL_COLLECTION,
                query_vector=("visual", visual_vector),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            return [
                (
                    str(hit.payload.get("chunk_id", hit.id)),
                    float(hit.score) * weight,
                    hit.payload or {},
                )
                for hit in hits
            ]
        except Exception as exc:
            logger.warning("[MultimodalService] Visual search failed: %s", exc)
            return []

    @staticmethod
    def _build_search_filter(
        request: MultimodalSearchRequest,
    ) -> qmodels.Filter | None:
        """
        Build a Qdrant filter from MultimodalSearchRequest parameters.

        Args:
            request: Search request with optional filter fields.

        Returns:
            Qdrant Filter or None.
        """
        conditions: list[qmodels.Condition] = []

        if request.document_id:
            conditions.append(
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=request.document_id),
                )
            )

        if request.chunk_types:
            type_values = [ct.value for ct in request.chunk_types]
            conditions.append(
                qmodels.FieldCondition(
                    key="chunk_type",
                    match=qmodels.MatchAny(any=type_values),
                )
            )

        return qmodels.Filter(must=conditions) if conditions else None

    @staticmethod
    def _payload_to_result(
        score: float,
        payload: dict,
    ) -> MultimodalSearchResult:
        """Convert a Qdrant payload dict into a MultimodalSearchResult."""
        return MultimodalSearchResult(
            score=round(score, 6),
            chunk_type=payload.get("chunk_type", "unknown"),
            page=int(payload.get("page", 1)),
            caption=payload.get("caption", ""),
            description=payload.get("description", ""),
            image_path=payload.get("image_path", ""),
            document_id=payload.get("document_id", ""),
            source_file=payload.get("source_file", ""),
            chunk_id=payload.get("chunk_id", ""),
            chart_data=payload.get("chart_json"),
            table_data=payload.get("table_json"),
            equation_data=payload.get("equation_json"),
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_document_json(document_id: str) -> dict:
        """Load Phase 1 parsed document JSON."""
        json_path = OUTPUT_JSON_DIR / f"{document_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Phase 1 JSON not found: {json_path}. "
                "Run Phase 1 parsing before multimodal indexing."
            )
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _resolve_pdf_path(doc: dict) -> Path | None:
        """
        Attempt to resolve the original PDF path from document metadata.

        Looks in the uploads directory for the original file.
        Returns None if not found (Camelot will be skipped).
        """
        filename = doc.get("filename", "")
        document_id = doc.get("document_id", "")

        uploads_dir = Path("uploads")
        # Phase 1 saves as: uploads/<document_id>_<filename>
        candidates = list(uploads_dir.glob(f"{document_id}*"))
        if candidates:
            return candidates[0]

        # Fallback: direct filename match
        direct = uploads_dir / filename
        if direct.exists():
            return direct

        return None