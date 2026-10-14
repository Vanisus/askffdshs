from fastapi import APIRouter

from . import routes_debug, routes_enrich, routes_health

api_router = APIRouter()
api_router.include_router(routes_health.router)
api_router.include_router(routes_enrich.router)
api_router.include_router(routes_debug.router)