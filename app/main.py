import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import api_router, register_exception_handlers
from app.config import Settings
from app.services import build_container

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.container = build_container(settings)
    app.state.settings = settings
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(title="SIEM Enrichment Service", version="1.0.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(api_router)
    Instrumentator(excluded_handlers=["/metrics", "/health/.*"]).instrument(app).expose(app, include_in_schema=False)
    return app


app = create_app()
