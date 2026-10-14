import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class MitreReranker:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        max_chars: int = 2000,
        batch_size: int = 16,
        use_fp16_on_gpu: bool = True,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.documents_by_id = {d["doc_id"]: d for d in documents}
        self.max_chars = max_chars
        self.batch_size = batch_size

        logger.info("Loading reranker %s", model_name)
        self.model = CrossEncoder(model_name, device=device)
        self.model.max_length = 512

        hf_device = getattr(self.model.model, "device", None)
        hf_device_str = str(hf_device) if hf_device is not None else "unknown"
        logger.info("Reranker loaded on %s (dtype=%s)",
                    hf_device_str, getattr(self.model.model, "dtype", "unknown"))

        if use_fp16_on_gpu and hf_device_str.startswith("cuda"):
            self.model.model = self.model.model.half()
            logger.info("Reranker switched to FP16")
            self._fp16_selftest()

    def _fp16_selftest(self) -> None:
        try:
            scores = self.model.predict(
                [("dns tunneling suspicious", "DNS protocol tunneling technique")],
                show_progress_bar=False,
            )
            val = float(scores[0])
            if math.isnan(val) or math.isinf(val):
                logger.warning(
                    "FP16 self-test returned %s — falling back to FP32", val
                )
                self.model.model = self.model.model.float()
            else:
                logger.info("FP16 self-test OK: score=%.4f", val)
        except Exception:
            logger.exception("FP16 self-test failed — falling back to FP32")
            try:
                self.model.model = self.model.model.float()
            except Exception:
                logger.exception("Failed to restore FP32 — model left as-is")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not query or not candidates:
            return []

        pairs: list[tuple[str, str]] = []
        kept: list[dict[str, Any]] = []

        for c in candidates:
            doc = self.documents_by_id.get(c["doc_id"])
            if doc is None:
                continue
            pairs.append((query, self._document_text(doc)))
            kept.append(c)

        if not pairs:
            return []

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []
        for c, score in zip(kept, scores):
            s = float(score)
            if math.isnan(s) or math.isinf(s):
                logger.warning(
                    "Reranker returned %s for %s — clamped to 0.0",
                    s, c.get("external_id", "?"),
                )
                s = 0.0
            enriched = dict(c)
            enriched["rerank_score"] = round(s, 6)
            reranked.append(enriched)

        reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _document_text(self, doc: dict[str, Any]) -> str:
        description = str(doc.get("description") or "")[: self.max_chars]
        parts = [
            str(doc.get("external_id") or ""),
            str(doc.get("name") or ""),
            " ".join(doc.get("tactic_shortnames") or []),
            description,
        ]
        return " ".join(p for p in parts if p).strip()
