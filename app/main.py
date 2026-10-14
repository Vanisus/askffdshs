import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import api_router, register_exception_handlers
from app.config import Settings
from app.services import build_container

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    container = build_container(settings)
    app.state.container = container
    app.state.settings = settings
    try:
        yield
    finally:
        logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="SIEM Enrichment Service",
        version="0.3.0",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
