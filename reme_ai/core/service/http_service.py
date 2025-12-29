"""HTTP service for serving flows via FastAPI.

This service exposes registered flows as HTTP endpoints, including regular,
tool-callable, and streaming flows. It also provides a health check endpoint.
"""

import asyncio
import os
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from .base_service import BaseService
from ..context import C
from ..enumeration import ChunkEnum
from ..flow import BaseFlow, BaseToolFlow
from ..schema import FlowResponse, FlowStreamChunk
from ..utils.pydantic_utils import create_pydantic_model


@C.register_service("http")
class HttpService(BaseService):
    """FastAPI-based HTTP server that exposes registered flows as endpoints."""

    def __init__(self, **kwargs):
        """Initialize the FastAPI application and middleware."""
        super().__init__(**kwargs)
        self.app = FastAPI(title=os.getenv("FLOW_APP_NAME"))
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        def health_check():
            return {"status": "healthy"}

        self.app.get("/health")(health_check)

    def integrate_flow(self, flow: BaseFlow) -> bool:
        """Expose a non-tool, non-stream flow as a POST endpoint returning JSON."""
        request_model = create_pydantic_model(flow.name)

        async def execute_endpoint(request: request_model) -> FlowResponse:
            return await flow.async_call(**request.model_dump(exclude_none=True))

        self.app.post(f"/{flow.name}", response_model=FlowResponse)(execute_endpoint)
        return True

    def integrate_tool_flow(self, flow: BaseToolFlow) -> bool:
        """Expose a tool-callable flow as a POST endpoint with OpenAPI details."""
        request_model = create_pydantic_model(flow.name, input_schema=flow.tool_call.input_schema)

        async def execute_endpoint(request: request_model) -> FlowResponse:
            return await flow.async_call(**request.model_dump(exclude_none=True))

        # include tool description in OpenAPI, parameters are described via request_model fields
        self.app.post(
            f"/{flow.name}",
            response_model=FlowResponse,
            description=flow.tool_call.description,
        )(execute_endpoint)
        return True

    def integrate_stream_flow(self, flow: BaseFlow) -> bool:
        """Expose a streaming flow as a Server-Sent Events endpoint."""
        request_model = create_pydantic_model(flow.name)

        async def execute_stream_endpoint(request: request_model) -> StreamingResponse:
            stream_queue = asyncio.Queue()
            task = asyncio.create_task(
                flow.async_call(stream_queue=stream_queue, **request.model_dump(exclude_none=True)),
            )

            async def generate_stream() -> AsyncGenerator[bytes, None]:
                while True:
                    try:
                        stream_chunk: FlowStreamChunk = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                        if stream_chunk.done:
                            yield "data:[DONE]\n\n".encode("utf-8")
                            await task  # Ensure task completes
                            break
                        yield f"data:{stream_chunk.model_dump_json()}\n\n".encode("utf-8")

                    except asyncio.TimeoutError:
                        # Timeout: check if task has completed or failed
                        if task.done():
                            try:
                                await task  # This will raise exception if task failed
                            except Exception as e:
                                logger.exception(f"flow={flow.name} encounter error with args={e.args}")

                                # Task failed, send error chunk
                                error_chunk = FlowStreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e), done=True)
                                yield f"data:{error_chunk.model_dump_json()}\n\n".encode("utf-8")
                                yield "data:[DONE]\n\n".encode("utf-8")
                                break

                            else:
                                # Task completed successfully but no done chunk received
                                # This shouldn't happen normally, but handle gracefully
                                yield "data:[DONE]\n\n".encode("utf-8")
                                break

                        # Task still running, continue waiting for chunks
                        continue

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        self.app.post(f"/{flow.name}")(execute_stream_endpoint)
        return True

    def run(self):
        """Run the FastAPI app with the configured Uvicorn settings."""
        super().run()
        http_config = self.service_config.http
        uvicorn.run(
            self.app,
            host=http_config.host,
            port=http_config.port,
            timeout_keep_alive=http_config.timeout_keep_alive,
            limit_concurrency=http_config.limit_concurrency,
        )
