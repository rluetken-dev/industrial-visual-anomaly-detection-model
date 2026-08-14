from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from .runtime import InferenceRuntime
from .settings import InferenceServiceSettings

from .prediction_routes import router as prediction_router

def create_app(
    runtime: InferenceRuntime | None = None,
) -> FastAPI:
    """Create the internal Python inference service."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_runtime = runtime

        if active_runtime is None:
            settings = InferenceServiceSettings.from_environment()
            active_runtime = InferenceRuntime.load(settings)

        app.state.inference_runtime = active_runtime
        yield

    application = FastAPI(
        title="Industrial Visual Anomaly Detection Inference Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.include_router(prediction_router)

    @application.get("/health/live", tags=["health"])
    def get_liveness() -> dict[str, str]:
        return {"status": "healthy"}

    return application


app = create_app()