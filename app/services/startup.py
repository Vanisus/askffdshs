import asyncio
import logging

import torch

from app.config import Settings
from app.core import AttackStore, MitreDocumentBuilder
from app.infra import Translator
from app.infra.retrievers import MitreBm25Retriever, MitreEmbeddingRetriever, MitreHybridRetriever, MitreReranker
from app.services.container import AppContainer, ResultCache

logger = logging.getLogger(__name__)


def _resolve_device(explicit: str | None, *, force_cpu: bool = False, label: str = "") -> str:
    if force_cpu:
        return "cpu"
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_container(settings: Settings) -> AppContainer:
    warnings: list[str] = []
    attack_store = AttackStore(version=settings.attack_version, data_path=settings.attack_data_path)
    attack_store.load()
    docs = MitreDocumentBuilder(attack_store).build()

    translator = None
    if settings.enable_translator:
        try:
            translator = Translator(model_name=settings.translator_model, device=_resolve_device(settings.translator_device, force_cpu=settings.force_cpu), max_input_tokens=settings.translator_max_input_tokens, cache_size=settings.translator_cache_size, num_beams=settings.translator_num_beams, no_repeat_ngram_size=settings.translator_no_repeat_ngram, use_fp16=settings.translator_use_fp16)
        except Exception as exc:
            logger.exception("Translator init failed")
            warnings.append(f"translator_init_failed: {exc}")

    bm25 = MitreBm25Retriever(docs, stemmer_language=settings.bm25_stemmer_language, stopwords=settings.bm25_stopwords)
    hybrid = None
    if settings.enable_hybrid:
        try:
            dense = MitreEmbeddingRetriever(docs, model_name=settings.embedding_model, device=_resolve_device(settings.embedding_device, force_cpu=settings.force_cpu), batch_size=settings.embedding_batch_size, max_chars=settings.max_doc_chars, use_fp16_on_gpu=settings.embedding_use_fp16)
            hybrid = MitreHybridRetriever(bm25, dense, rrf_k=settings.rrf_k, lexical_weight=settings.lexical_weight, dense_weight=settings.dense_weight)
        except Exception as exc:
            logger.exception("Embedding init failed")
            warnings.append(f"hybrid_init_failed: {exc}")

    reranker = None
    if settings.enable_reranker:
        try:
            reranker = MitreReranker(docs, model_name=settings.reranker_model, device=_resolve_device(settings.reranker_device, force_cpu=settings.force_cpu), batch_size=settings.reranker_batch_size, use_fp16_on_gpu=settings.reranker_use_fp16)
        except Exception as exc:
            logger.exception("Reranker init failed")
            warnings.append(f"reranker_init_failed: {exc}")

    return AppContainer(settings=settings, attack_store=attack_store, mitre_documents=docs, bm25_retriever=bm25, hybrid_retriever=hybrid, reranker=reranker, translator=translator, warnings=warnings, inference_semaphore=asyncio.Semaphore(max(1, settings.inference_concurrency)), result_cache=ResultCache(settings.result_cache_size, settings.result_cache_ttl_seconds))
