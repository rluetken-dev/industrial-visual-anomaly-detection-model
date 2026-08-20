from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .model_registry_config import (
    load_model_registry_configuration,
)
from .model_routes import router as model_router
from .prediction_routes import router as prediction_router
from .runtime import InferenceRuntime
from .runtime_registry import InferenceRuntimeRegistry
from .settings import InferenceServiceSettings


def create_app(
    runtime: InferenceRuntime | InferenceRuntimeRegistry | None = None,
) -> FastAPI:
    """Create the internal Python inference service."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_runtime = runtime

        if active_runtime is None:
            settings = (
                InferenceServiceSettings.from_environment()
            )

            if settings.model_registry_path is not None:
                configuration = (
                    load_model_registry_configuration(
                        settings.model_registry_path
                    )
                )
                active_runtime = InferenceRuntimeRegistry.load(
                    configuration=configuration,
                    memory_chunk_size=(
                        settings.memory_chunk_size
                    ),
                )
            else:
                active_runtime = InferenceRuntime.load(
                    settings
                )

        app.state.inference_runtime = active_runtime
        yield

    application = FastAPI(
        title=(
            "Industrial Visual Anomaly Detection "
            "Inference Service"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.include_router(model_router)
    application.include_router(prediction_router)

    @application.get("/health/live", tags=["health"])
    def get_liveness() -> dict[str, str]:
        return {"status": "healthy"}

    return application


app = create_app()