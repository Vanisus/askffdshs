import logging
from typing import Any

import bm25s

logger = logging.getLogger(__name__)


def _try_build_stemmer(language: str):
    try:
        import Stemmer  # PyStemmer
    except ImportError:
        logger.warning(
            "PyStemmer not installed — BM25 will work without stemming. "
            "This is fine for MITRE (mostly EN, short names)."
        )
        return None
    try:
        return Stemmer.Stemmer(language)
    except Exception as exc:
        logger.warning("Stemmer init failed (%s), continuing without", exc)
        return None


class MitreBm25Retriever:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        techniques_only: bool = True,
        stemmer_language: str = "english",
        stopwords: str | None = "en",
    ) -> None:
        if techniques_only:
            documents = [d for d in documents if d.get("object_type") == "technique"]

        self.documents = documents
        self.stopwords = stopwords
        self.stemmer = _try_build_stemmer(stemmer_language)

        corpus_texts = [self._document_text(d) for d in documents]
        logger.info(
            "Indexing %d docs with bm25s (stemmer=%s)",
            len(corpus_texts),
            type(self.stemmer).__name__ if self.stemmer else "none",
        )

        tokens = bm25s.tokenize(
            corpus_texts,
            stopwords=stopwords,
            stemmer=self.stemmer,
            show_progress=False,
        )

        self.retriever = bm25s.BM25()
        self.retriever.index(tokens, show_progress=False)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        if not query or not query.strip() or not self.documents:
            return []

        q_tokens = bm25s.tokenize(
            [query],
            stopwords=self.stopwords,
            stemmer=self.stemmer,
            show_progress=False,
        )
        k = min(top_k, len(self.documents))
        results, scores = self.retriever.retrieve(q_tokens, k=k)

        out: list[dict[str, Any]] = []
        for idx, score in zip(results[0], scores[0]):
            doc = self.documents[int(idx)]
            out.append(
                {
                    "doc_id": doc["doc_id"],
                    "object_type": doc["object_type"],
                    "external_id": doc["external_id"],
                    "name": doc.get("name"),
                    "short_name": doc.get("short_name"),
                    "score": round(float(score), 6),
                }
            )
        return out

    def _document_text(self, doc: dict[str, Any]) -> str:
        if doc.get("search_text"):
            return doc["search_text"]
        parts = [
            doc.get("external_id"),
            doc.get("name"),
            doc.get("description"),
            " ".join(doc.get("tactic_shortnames") or []),
        ]
        return " ".join(str(p) for p in parts if p)
