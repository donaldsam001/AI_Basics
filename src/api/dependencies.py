# src/api/dependencies.py
"""FastAPI dependency utilities for the AI CV matching service.

Provides dependency injection for the shared :class:`ModelRegistry` and
:class:`MatchingService` instances created during application startup.
"""

from fastapi import Depends, Request
from src.services.matching_service import ModelRegistry, MatchingService


def get_registry(request: Request) -> ModelRegistry:
    """Retrieve the :class:`ModelRegistry` from the FastAPI app state.

    The registry is created in the ``lifespan`` function of ``app.create_app``
    and stored as ``request.app.state.registry``.
    """
    return request.app.state.registry


def get_service(registry: ModelRegistry = Depends(get_registry)) -> MatchingService:
    """Retrieve the :class:`MatchingService` associated with the registry.

    ``MatchingService`` is lightweight and holds a reference to the registry, so
    it is safe to create a new instance per request.
    """
    return MatchingService(registry)
}
