"""
FastAPI application factory.

Creates the main Huxley API server with:
- OpenAI-compatible endpoints
- Agent execution endpoints
- Tool management endpoints
- Health and metrics endpoints
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from huxley.api.routes import agents, completions, health, tools
from huxley.core.config import get_config
from huxley.core.exceptions import HuxleyError
from huxley.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    # Startup
    configure_logging()
    config = get_config()
    config.ensure_directories()
    logger.info(
        "server_starting",
        env=config.env,
        debug=config.debug,
    )
    yield
    # Shutdown
    logger.info("server_stopping")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance
    """
    config = get_config()

    app = FastAPI(
        title="Huxley",
        description="Agentic, model-agnostic biological intelligence framework",
        version="0.1.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(HuxleyError)
    async def huxley_error_handler(
        request: Request, exc: HuxleyError
    ) -> JSONResponse:
        logger.error(
            "api_error",
            error_type=type(exc).__name__,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "InternalError",
                    "message": "An unexpected error occurred",
                }
            },
        )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(completions.router, prefix="/v1", tags=["Completions"])
    app.include_router(agents.router, prefix="/v1", tags=["Agents"])
    app.include_router(tools.router, prefix="/v1", tags=["Tools"])

    return app


# For running with uvicorn directly
app = create_app()
