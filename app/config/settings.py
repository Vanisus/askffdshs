import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str | None) -> str | None:
    v = os.environ.get(name)
    return v if v else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else float(v)


def _default_attack_path() -> Path:
    return Path(os.environ.get(
        "SIEM_ATTACK_DATA_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "enterprise-attack-19.2.json"),
    ))


@dataclass(frozen=True, slots=True)
class Settings:
    debug: bool = field(default_factory=lambda: _env_bool("SIEM_DEBUG", False))
    attack_version: str = field(default_factory=lambda: _env_str("SIEM_ATTACK_VERSION", "19.2") or "19.2")
    attack_data_path: Path = field(default_factory=_default_attack_path)

    force_cpu: bool = field(default_factory=lambda: _env_bool("SIEM_FORCE_CPU", False))
    translator_device: str | None = field(default_factory=lambda: _env_str("SIEM_TRANSLATOR_DEVICE", None))
    embedding_device: str | None = field(default_factory=lambda: _env_str("SIEM_EMBEDDING_DEVICE", None))
    reranker_device: str | None = field(default_factory=lambda: _env_str("SIEM_RERANKER_DEVICE", None))

    enable_translator: bool = field(default_factory=lambda: _env_bool("SIEM_ENABLE_TRANSLATOR", True))
    translator_model: str = field(default_factory=lambda: _env_str("SIEM_TRANSLATOR_MODEL", "Helsinki-NLP/opus-mt-ru-en") or "Helsinki-NLP/opus-mt-ru-en")
    translator_max_input_tokens: int = field(default_factory=lambda: _env_int("SIEM_TRANSLATOR_MAX_INPUT_TOKENS", 512))
    translator_cache_size: int = field(default_factory=lambda: _env_int("SIEM_TRANSLATOR_CACHE_SIZE", 8192))
    translator_num_beams: int = field(default_factory=lambda: _env_int("SIEM_TRANSLATOR_NUM_BEAMS", 1))
    translator_no_repeat_ngram: int = field(default_factory=lambda: _env_int("SIEM_TRANSLATOR_NO_REPEAT_NGRAM", 0))
    translator_use_fp16: bool = field(default_factory=lambda: _env_bool("SIEM_TRANSLATOR_USE_FP16", False))

    enable_hybrid: bool = field(default_factory=lambda: _env_bool("SIEM_ENABLE_HYBRID", True))
    embedding_model: str = field(default_factory=lambda: _env_str("SIEM_EMBEDDING_MODEL", "intfloat/multilingual-e5-small") or "intfloat/multilingual-e5-small")
    embedding_batch_size: int = field(default_factory=lambda: _env_int("SIEM_EMBEDDING_BATCH_SIZE", 64))
    embedding_use_fp16: bool = field(default_factory=lambda: _env_bool("SIEM_EMBEDDING_USE_FP16", True))
    max_doc_chars: int = field(default_factory=lambda: _env_int("SIEM_MAX_DOC_CHARS", 2000))

    bm25_stemmer_language: str = field(default_factory=lambda: _env_str("SIEM_BM25_STEMMER_LANGUAGE", "english") or "english")
    bm25_stopwords: str | None = field(default_factory=lambda: _env_str("SIEM_BM25_STOPWORDS", "en"))

    enable_reranker: bool = field(default_factory=lambda: _env_bool("SIEM_ENABLE_RERANKER", True))
    reranker_model: str = field(default_factory=lambda: _env_str("SIEM_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3") or "BAAI/bge-reranker-v2-m3")
    reranker_batch_size: int = field(default_factory=lambda: _env_int("SIEM_RERANKER_BATCH_SIZE", 16))
    reranker_use_fp16: bool = field(default_factory=lambda: _env_bool("SIEM_RERANKER_USE_FP16", True))

    retrieval_top_k: int = field(default_factory=lambda: _env_int("SIEM_RETRIEVAL_TOP_K", 20))
    rerank_top_k: int = field(default_factory=lambda: _env_int("SIEM_RERANK_TOP_K", 5))
    rerank_score_threshold: float = field(default_factory=lambda: _env_float("SIEM_RERANK_SCORE_THRESHOLD", 0.35))
    rerank_relative_margin: float = field(default_factory=lambda: _env_float("SIEM_RERANK_RELATIVE_MARGIN", 0.65))
    rerank_min_gap: float = field(default_factory=lambda: _env_float("SIEM_RERANK_MIN_GAP", 0.05))
    max_auto_techniques: int = field(default_factory=lambda: _env_int("SIEM_MAX_AUTO_TECHNIQUES", 3))
    rrf_k: int = field(default_factory=lambda: _env_int("SIEM_RRF_K", 60))
    lexical_weight: float = field(default_factory=lambda: _env_float("SIEM_LEXICAL_WEIGHT", 1.0))
    dense_weight: float = field(default_factory=lambda: _env_float("SIEM_DENSE_WEIGHT", 1.0))

    inference_concurrency: int = field(default_factory=lambda: _env_int("SIEM_INFERENCE_CONCURRENCY", 1))
    request_timeout_seconds: float = field(default_factory=lambda: _env_float("SIEM_REQUEST_TIMEOUT_SECONDS", 30.0))
    batch_max_size: int = field(default_factory=lambda: _env_int("SIEM_BATCH_MAX_SIZE", 100))
    result_cache_size: int = field(default_factory=lambda: _env_int("SIEM_RESULT_CACHE_SIZE", 10000))
    result_cache_ttl_seconds: float = field(default_factory=lambda: _env_float("SIEM_RESULT_CACHE_TTL_SECONDS", 300.0))
