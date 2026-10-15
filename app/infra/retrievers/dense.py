import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MitreEmbeddingRetriever:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        batch_size: int = 64,
        max_chars: int = 2000,
        techniques_only: bool = True,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        use_fp16_on_gpu: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if techniques_only:
            documents = [d for d in documents if d.get("object_type") == "technique"]

        self.documents = documents
        self.max_chars = max_chars
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        logger.info("Loading embedding model %s", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512

        if use_fp16_on_gpu and self.model.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Embedding model using FP16 on %s", self.model.device)

        texts = [f"{passage_prefix}{self._document_text(d)}" for d in documents]
        logger.info(
            "Encoding %d docs (batch_size=%d, device=%s)",
            len(texts),
            batch_size,
            self.model.device,
        )
        self.doc_embeddings: np.ndarray = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if not query or not query.strip() or not self.documents:
            return []

        q_emb = self.model.encode(
            [f"{self.query_prefix}{query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        scores = self.doc_embeddings @ q_emb
        k = min(top_k, len(self.documents))

        if k >= len(self.documents):
            top_idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            top_idx = part[np.argsort(-scores[part])]

        out: list[dict[str, Any]] = []
        for idx in top_idx:
            i = int(idx)
            doc = self.documents[i]
            out.append(
                {
                    "doc_id": doc["doc_id"],
                    "object_type": doc["object_type"],
                    "external_id": doc["external_id"],
                    "name": doc.get("name"),
                    "score": round(float(scores[i]), 6),
                }
            )
        return out

    def _document_text(self, doc: dict[str, Any]) -> str:
        description = str(doc.get("description") or "")[: self.max_chars]
        tactics = ", ".join(doc.get("tactic_shortnames") or [])
        parts = [
            f"MITRE ATT&CK technique: {doc.get('name') or ''}",
            f"Technique ID: {doc.get('external_id') or ''}",
            f"Applicable tactics: {tactics}" if tactics else "",
            f"Behavior definition: {description}" if description else "",
        ]
        return "\n".join(p for p in parts if p).strip()
