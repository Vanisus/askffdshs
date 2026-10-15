import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Any, Protocol


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]: ...


class ResultCache:
    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.max_size = max(0, max_size)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        if not self.max_size or not self.ttl_seconds:
            return None
        now = monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            created_at, value = item
            if now - created_at > self.ttl_seconds:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return deepcopy(value)

    def put(self, key: str, value: Any) -> None:
        if not self.max_size or not self.ttl_seconds:
            return
        with self._lock:
            self._data[key] = (monotonic(), deepcopy(value))
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)


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
    inference_semaphore: asyncio.Semaphore | None = None
    result_cache: ResultCache | None = None

    @property
    def retriever(self) -> Retriever:
        return self.hybrid_retriever or self.bm25_retriever

    @property
    def is_ready(self) -> bool:
        return bool(self.attack_store.loaded)
