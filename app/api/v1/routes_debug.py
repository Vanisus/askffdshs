from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies import get_container
from app.api.schemas import DebugSearchResponse, DebugSearchTextResponse
from app.core import query as incident_query
from app.services import AppContainer

router = APIRouter(prefix="/debug", tags=["debug"])


@router.post("/mitre-search", response_model=DebugSearchResponse)
async def debug_search_incident(
    payload: dict[str, Any],
    container: AppContainer = Depends(get_container),
) -> DebugSearchResponse:
    retriever = container.retriever

    query = await run_in_threadpool(
        incident_query.build_query_text, payload, container.translator
    )
    candidates = await run_in_threadpool(retriever.search, query, 20)

    reranked: list[dict[str, Any]] = []
    if container.reranker is not None and candidates:
        reranked = await run_in_threadpool(
            container.reranker.rerank, query, candidates, 5
        )

    return DebugSearchResponse(query=query, results=candidates, reranked=reranked)


@router.get("/mitre-search-text", response_model=DebugSearchTextResponse)
async def debug_search_text(
    q: str = Query(..., min_length=1, description="Free-form query"),
    container: AppContainer = Depends(get_container),
) -> DebugSearchTextResponse:
    retriever = container.retriever

    translated = ""
    if container.translator is not None:
        translated = await run_in_threadpool(container.translator.translate, q)

    final_query = f"{translated} {q}".strip() if translated else q
    candidates = await run_in_threadpool(retriever.search, final_query, 20)

    reranked: list[dict[str, Any]] = []
    if container.reranker is not None and candidates:
        reranked = await run_in_threadpool(
            container.reranker.rerank, final_query, candidates, 5
        )

    return DebugSearchTextResponse(
        original_query=q,
        translated_query=translated,
        final_query=final_query,
        results=candidates,
        reranked=reranked,
    )
