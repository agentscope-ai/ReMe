"""HTTP service implementation using FastAPI."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .base_service import BaseService
from ..flow import BaseFlow
from ..schema import Response
from ..utils.common_utils import execute_stream_task


class HttpService(BaseService):
    """Expose flows via HTTP REST and SSE endpoints."""

    def __init__(self, **kwargs):
        """Initialize FastAPI app with CORS and health checks."""
        super().__init__(**kwargs)
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # 启动时的逻辑
            print("HTTP service is starting...")
            await asyncio.sleep(1)
            print("HTTP service started successfully!")
            yield
            # 结束时的逻辑
            print("HTTP service is shutting down...")
            await asyncio.sleep(1)
            print("HTTP service shutdown complete!")
        
        self.app = FastAPI(title=self.service_config.app_name, lifespan=lifespan)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.get("/health")(lambda: {"status": "healthy"})

    def _integrate_flow(self, flow: BaseFlow) -> str:
        """Register a standard flow as a POST endpoint."""
        tool_call, request_model = self._prepare_route(flow)

        async def execute_endpoint(request: request_model) -> Response:
            return await flow.call(**request.model_dump(exclude_none=True))

        self.app.post(
            path=f"/{tool_call.name}",
            response_model=Response,
            description=tool_call.description,
        )(execute_endpoint)
        return tool_call.name

    def _integrate_stream_flow(self, flow: BaseFlow) -> str:
        """Register a streaming flow as an SSE endpoint."""
        tool_call, request_model = self._prepare_route(flow)

        async def execute_stream_endpoint(request: request_model) -> StreamingResponse:
            stream_queue = asyncio.Queue()
            task = asyncio.create_task(flow.call(stream_queue=stream_queue, **request.model_dump(exclude_none=True)))

            async def generate_stream() -> AsyncGenerator[bytes, None]:
                async for chunk in execute_stream_task(
                    stream_queue=stream_queue,
                    task=task,
                    task_name=tool_call.name,
                    as_bytes=True,
                ):
                    yield chunk

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        self.app.post(f"/{tool_call.name}")(execute_stream_endpoint)
        return tool_call.name

    def integrate_flow(self, flow: BaseFlow) -> str | None:
        """Register a flow based on its streaming configuration."""
        return self._integrate_stream_flow(flow) if flow.stream else self._integrate_flow(flow)

    def run(self):
        """Start the Uvicorn server."""
        super().run()
        cfg = self.service_config.http
        uvicorn.run(
            self.app,
            host=cfg.host,
            port=cfg.port,
            timeout_keep_alive=cfg.timeout_keep_alive,
            limit_concurrency=cfg.limit_concurrency,
            **cfg.model_extra,
        )
