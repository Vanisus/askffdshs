from dataclasses import dataclass, field
from typing import Any, Protocol


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]: ...


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class AppContainer:
    settings: Any
    attack_store: Any
    mitre_documents: list[dict[str, Any]]

    bm25_retriever: Retriever
    hybrid_retriever: Retriever | None = None
    reranker: Reranker | None = None
    translator: Any | None = None

    warnings: list[str] = field(default_factory=list)

    @property
    def retriever(self) -> Retriever:
        return self.hybrid_retriever or self.bm25_retriever

    @property
    def is_ready(self) -> bool:
        return bool(self.attack_store.loaded)
    