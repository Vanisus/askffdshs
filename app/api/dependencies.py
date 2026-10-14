from fastapi import Request

from app.services import AppContainer


def get_container(request: Request) -> AppContainer:
    container: AppContainer = request.app.state.container
    return container
