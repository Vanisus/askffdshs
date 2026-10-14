import logging
from typing import Any

logger = logging.getLogger(__name__)


class MitreHybridRetriever:
    def __init__(
        self,
        lexical_retriever,
        embedding_retriever,
        *,
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        self.lexical = lexical_retriever
        self.embedding = embedding_retriever
        self.rrf_k = rrf_k
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight

    def search(self, query: str, top_k: int = 25) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        pool = max(top_k * 2, 40)
        lexical_results = self.lexical.search(query, top_k=pool)
        dense_results = self.embedding.search(query, top_k=pool)

        scores: dict[str, float] = {}
        docs: dict[str, dict[str, Any]] = {}
        sources: dict[str, set[str]] = {}

        for rank, item in enumerate(lexical_results):
            did = item["doc_id"]
            scores[did] = scores.get(did, 0.0) + self.lexical_weight / (
                self.rrf_k + rank + 1
            )
            docs[did] = item
            sources.setdefault(did, set()).add("bm25")

        for rank, item in enumerate(dense_results):
            did = item["doc_id"]
            scores[did] = scores.get(did, 0.0) + self.dense_weight / (
                self.rrf_k + rank + 1
            )
            docs.setdefault(did, item)
            sources.setdefault(did, set()).add("dense")

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for did, score in ranked:
            doc = dict(docs[did])
            doc["rrf_score"] = round(score, 6)
            doc["sources"] = sorted(sources[did])
            results.append(doc)
        return results
