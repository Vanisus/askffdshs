import logging

import torch

from app.config import Settings
from app.core import AttackStore, MitreDocumentBuilder
from app.infra import Translator
from app.infra.retrievers import (
    MitreBm25Retriever,
    MitreEmbeddingRetriever,
    MitreHybridRetriever,
    MitreReranker,
)
from app.services.container import AppContainer

logger = logging.getLogger(__name__)


def _resolve_device(
    explicit: str | None,
    *,
    force_cpu: bool = False,
    label: str = "",
) -> str:
    if force_cpu:
        logger.info("%s: forced CPU", label)
        return "cpu"

    if explicit:
        logger.info("%s: explicit device %s", label, explicit)
        return explicit

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("%s: auto-selected CUDA (%s)", label, name)
        return "cuda"

    logger.info("%s: auto-selected CPU (no CUDA)", label)
    return "cpu"


def build_container(settings: Settings) -> AppContainer:
    warnings: list[str] = []

    # 1. Attack data
    attack_store = AttackStore(
        version=settings.attack_version,
        data_path=settings.attack_data_path,
    )
    attack_store.load()

    # 2. Documents
    mitre_documents = MitreDocumentBuilder(attack_store).build()

    # 3. Translator
    translator: Translator | None = None
    if settings.enable_translator:
        try:
            device = _resolve_device(
                settings.translator_device,
                force_cpu=settings.force_cpu,
                label="translator",
            )
            translator = Translator(
                model_name=settings.translator_model,
                device=device,
                max_input_tokens=settings.translator_max_input_tokens,
                cache_size=settings.translator_cache_size,
                num_beams=settings.translator_num_beams,
                no_repeat_ngram_size=settings.translator_no_repeat_ngram,
                use_fp16=settings.translator_use_fp16,
            )
        except Exception as exc:
            logger.exception("Translator init failed")
            warnings.append(f"translator_init_failed: {exc}")

    # 4. BM25 (CPU-only, не ускоряется на GPU)
    bm25_retriever = MitreBm25Retriever(
        mitre_documents,
        stemmer_language=settings.bm25_stemmer_language,
        stopwords=settings.bm25_stopwords,
    )

    # 5. Dense + Hybrid
    hybrid_retriever: MitreHybridRetriever | None = None
    if settings.enable_hybrid:
        try:
            device = _resolve_device(
                settings.embedding_device,
                force_cpu=settings.force_cpu,
                label="embedding",
            )
            embedding_retriever = MitreEmbeddingRetriever(
                mitre_documents,
                model_name=settings.embedding_model,
                device=device,
                batch_size=settings.embedding_batch_size,
                max_chars=settings.max_doc_chars,
                use_fp16_on_gpu=settings.embedding_use_fp16,
            )
            hybrid_retriever = MitreHybridRetriever(
                bm25_retriever,
                embedding_retriever,
                rrf_k=settings.rrf_k,
                lexical_weight=settings.lexical_weight,
                dense_weight=settings.dense_weight,
            )
        except Exception as exc:
            logger.exception("Embedding/hybrid init failed")
            warnings.append(f"hybrid_init_failed: {exc}")

    # 6. Reranker
    reranker: MitreReranker | None = None
    if settings.enable_reranker:
        try:
            device = _resolve_device(
                settings.reranker_device,
                force_cpu=settings.force_cpu,
                label="reranker",
            )
            reranker = MitreReranker(
                mitre_documents,
                model_name=settings.reranker_model,
                device=device,
                batch_size=settings.reranker_batch_size,
                use_fp16_on_gpu=settings.reranker_use_fp16,
            )
        except Exception as exc:
            logger.exception("Reranker init failed")
            warnings.append(f"reranker_init_failed: {exc}")

    if warnings:
        logger.warning("Startup warnings: %s", warnings)

    return AppContainer(
        settings=settings,
        attack_store=attack_store,
        mitre_documents=mitre_documents,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        translator=translator,
        warnings=warnings,
    )
