"""Health check endpoints."""

from fastapi import APIRouter

from huxley.core.config import get_config

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check() -> dict:
    """
    Readiness check for Kubernetes.

    Checks that all required services are available.
    """
    # Add checks for required services here
    # e.g., database, redis, LLM provider
    return {
        "status": "ready",
        "checks": {
            "config": True,
        },
    }


@router.get("/health/live")
async def liveness_check() -> dict:
    """Liveness check for Kubernetes."""
    return {"status": "alive"}


@router.get("/info")
async def info() -> dict:
    """Server information endpoint."""
    config = get_config()
    return {
        "name": "Huxley",
        "version": "0.1.0",
        "environment": config.env,
        "default_provider": config.default_provider,
        "default_model": config.default_model,
    }
