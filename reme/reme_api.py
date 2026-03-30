"""FastAPI server that exposes ReMe memory operations as REST endpoints."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .reme import ReMe


# ---------------------------------------------------------------------------
# Singleton instance (populated during lifespan startup)
# ---------------------------------------------------------------------------

_reme: ReMe | None = None


def get_reme() -> ReMe:
    if _reme is None:
        raise RuntimeError("ReMe is not initialized.")
    return _reme


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _reme
    _reme = ReMe(
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL"),
        working_dir=os.getenv("REME_WORKING_DIR", ".reme"),
        enable_logo=False,
        log_to_console=True,
    )
    await _reme.start()
    yield
    await _reme.close()
    _reme = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ReMe API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class MessageDict(BaseModel):
    role: str = "user"
    content: str = ""
    name: str | None = None
    time_created: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SummarizeMemoryRequest(BaseModel):
    messages: list[MessageDict]
    description: str = ""
    user_name: str | list[str] = ""
    task_name: str | list[str] = ""
    tool_name: str | list[str] = ""
    enable_thinking_params: bool = True
    version: str = "default"
    retrieve_top_k: int = 20
    return_dict: bool = False
    raise_exception: bool = False
    llm_config_name: str = "default"


class RetrieveMemoryRequest(BaseModel):
    query: str = ""
    description: str = ""
    messages: list[MessageDict] | None = None
    user_name: str | list[str] = ""
    task_name: str | list[str] = ""
    tool_name: str | list[str] = ""
    enable_thinking_params: bool = True
    version: str = "default"
    retrieve_top_k: int = 20
    enable_time_filter: bool = True
    return_dict: bool = False
    raise_exception: bool = False
    llm_config_name: str = "default"


class AddMemoryRequest(BaseModel):
    memory_content: str
    user_name: str = ""
    task_name: str = ""
    tool_name: str = ""
    when_to_use: str = ""
    message_time: str = ""
    ref_memory_id: str = ""
    author: str = ""
    score: float = 0.0


class UpdateMemoryRequest(BaseModel):
    user_name: str = ""
    task_name: str = ""
    tool_name: str = ""
    memory_content: str | None = None
    when_to_use: str | None = None
    message_time: str | None = None
    ref_memory_id: str | None = None
    author: str | None = None
    score: float | None = None


class ListMemoryRequest(BaseModel):
    user_name: str = ""
    task_name: str = ""
    tool_name: str = ""
    filters: dict[str, Any] | None = None
    limit: int | None = None
    sort_key: str | None = None
    reverse: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


@app.post("/summarize_memory")
async def summarize_memory(request: SummarizeMemoryRequest):
    """Summarize conversation messages into long-term memory."""
    try:
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        result = await get_reme().summarize_memory(
            messages=messages,
            description=request.description,
            user_name=request.user_name,
            task_name=request.task_name,
            tool_name=request.tool_name,
            enable_thinking_params=request.enable_thinking_params,
            version=request.version,
            retrieve_top_k=request.retrieve_top_k,
            return_dict=request.return_dict,
            raise_exception=request.raise_exception,
            llm_config_name=request.llm_config_name,
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retrieve_memory")
async def retrieve_memory(request: RetrieveMemoryRequest):
    """Retrieve relevant memories for a query or conversation."""
    try:
        messages = [m.model_dump(exclude_none=True) for m in request.messages] if request.messages else None
        result = await get_reme().retrieve_memory(
            query=request.query,
            description=request.description,
            messages=messages,
            user_name=request.user_name,
            task_name=request.task_name,
            tool_name=request.tool_name,
            enable_thinking_params=request.enable_thinking_params,
            version=request.version,
            retrieve_top_k=request.retrieve_top_k,
            enable_time_filter=request.enable_time_filter,
            return_dict=request.return_dict,
            raise_exception=request.raise_exception,
            llm_config_name=request.llm_config_name,
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/add_memory")
async def add_memory(request: AddMemoryRequest):
    """Add a memory entry directly to the vector store."""
    try:
        memory_node = await get_reme().add_memory(
            memory_content=request.memory_content,
            user_name=request.user_name,
            task_name=request.task_name,
            tool_name=request.tool_name,
            when_to_use=request.when_to_use,
            message_time=request.message_time,
            ref_memory_id=request.ref_memory_id,
            author=request.author,
            score=request.score,
        )
        return {"memory_node": memory_node.model_dump() if hasattr(memory_node, "model_dump") else memory_node}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/get_memory/{memory_id}")
async def get_memory(memory_id: str):
    """Retrieve a single memory node by its ID."""
    try:
        memory_node = await get_reme().get_memory(memory_id)
        return {"memory_node": memory_node.model_dump() if hasattr(memory_node, "model_dump") else memory_node}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.delete("/delete_memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory node by its ID."""
    try:
        await get_reme().delete_memory(memory_id)
        return {"deleted": memory_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/delete_all")
async def delete_all():
    """Delete all memory nodes in the vector store."""
    try:
        await get_reme().delete_all()
        return {"deleted": "all"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.put("/update_memory/{memory_id}")
async def update_memory(memory_id: str, request: UpdateMemoryRequest):
    """Update content and/or metadata of an existing memory node."""
    try:
        memory_node = await get_reme().update_memory(
            memory_id=memory_id,
            user_name=request.user_name,
            task_name=request.task_name,
            tool_name=request.tool_name,
            memory_content=request.memory_content,
            when_to_use=request.when_to_use,
            message_time=request.message_time,
            ref_memory_id=request.ref_memory_id,
            author=request.author,
            score=request.score,
        )
        return {"memory_node": memory_node.model_dump() if hasattr(memory_node, "model_dump") else memory_node}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/list_memory")
async def list_memory(request: ListMemoryRequest):
    """List memory nodes with optional filtering and sorting."""
    try:
        memory_nodes = await get_reme().list_memory(
            user_name=request.user_name,
            task_name=request.task_name,
            tool_name=request.tool_name,
            filters=request.filters,
            limit=request.limit,
            sort_key=request.sort_key,
            reverse=request.reverse,
        )
        nodes = [n.model_dump() if hasattr(n, "model_dump") else n for n in memory_nodes]
        return {"memory_nodes": nodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Start the ReMe API server."""
    uvicorn.run(app, host=host, port=port, **kwargs)


if __name__ == "__main__":
    serve()
