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
from ..flow import BaseFlow
from ..schema import Response, StreamChunk
from ..utils import create_pydantic_model


@C.register_service("http")
class HttpService(BaseService):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = FastAPI(title=os.getenv("APP_NAME", "ReMe"))
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
        request_model = create_pydantic_model(flow.name, input_schema=flow.tool_call.input_schema)

        async def execute_endpoint(request: request_model) -> Response:
            return await flow.call(**request.model_dump(exclude_none=True))

        self.app.post(path=f"/{flow.name}", response_model=Response,
                      description=flow.tool_call.description)(execute_endpoint)
        return True

    def integrate_stream_flow(self, flow: BaseFlow) -> bool:
        request_model = create_pydantic_model(flow.name)

        async def execute_stream_endpoint(request: request_model) -> StreamingResponse:
            stream_queue = asyncio.Queue()
            task = asyncio.create_task(flow.call(stream_queue=stream_queue, **request.model_dump(exclude_none=True)))

            async def generate_stream() -> AsyncGenerator[bytes, None]:
                while True:
                    try:
                        stream_chunk: StreamChunk = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                        if stream_chunk.done:
                            yield "data:[DONE]\n\n".encode("utf-8")
                            await task
                            break

                        yield f"data:{stream_chunk.model_dump_json()}\n\n".encode("utf-8")

                    except asyncio.TimeoutError:
                        # Timeout: check if task has completed or failed
                        if task.done():
                            try:
                                await task

                            except Exception as e:
                                logger.exception(f"flow={flow.name} encounter error with args={e.args}")

                                error_chunk = StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e), done=True)
                                yield f"data:{error_chunk.model_dump_json()}\n\n".encode("utf-8")
                                yield "data:[DONE]\n\n".encode("utf-8")
                                break

                            else:
                                yield "data:[DONE]\n\n".encode("utf-8")
                                break

                        continue

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        self.app.post(f"/{flow.name}")(execute_stream_endpoint)
        return True

    def run(self):
        super().run()
        http_config = self.service_config.http
        uvicorn.run(
            self.app,
            host=http_config.host,
            port=http_config.port,
            timeout_keep_alive=http_config.timeout_keep_alive,
            limit_concurrency=http_config.limit_concurrency,
            **http_config.model_extra
        )
