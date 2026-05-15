import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import settings
from graph.nodes import streaming_registry
from graph.workflow import workflow

log = logging.getLogger("zenno")
router = APIRouter()


async def _run_graph(state: dict) -> dict:
    return await workflow.ainvoke(state)


@router.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="'messages' is required")

    model = body.get("model", settings.default_model)
    do_stream = body.get("stream", False)
    temperature = float(body.get("temperature", 0.7))
    tools = body.get("tools") or None
    tool_choice = body.get("tool_choice", None)
    request_id = f"req-{uuid.uuid4().hex[:8]}"

    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    log.info("[%s] REQUEST | model=%s stream=%s tools=%s | %.100s",
             request_id, model, do_stream, bool(tools), last_user)

    state = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "do_stream": do_stream,
        "tools": tools,
        "tool_choice": tool_choice,
        "complexity": "",
        "quality": "",
        "iterations": 0,
        "max_iterations": settings.max_reasoning_iterations,
        "thoughts": [],
        "critique": "",
        "final_response": "",
        "tool_calls": None,
        "finish_reason": "stop",
        "request_id": request_id,
    }

    if do_stream:
        queue: asyncio.Queue = asyncio.Queue()
        streaming_registry[request_id] = queue

        async def _bg() -> None:
            try:
                final = await _run_graph(state)
                log.info("[%s] DONE (stream) | complexity=%s iterations=%d",
                         request_id,
                         final.get("complexity", "?").upper(),
                         final.get("iterations", 0))
            except Exception as exc:
                log.error("[%s] ERROR in graph: %s", request_id, exc, exc_info=True)
                await queue.put({"error": str(exc)})
                await queue.put(None)
            finally:
                streaming_registry.pop(request_id, None)

        bg_task: asyncio.Task = asyncio.create_task(_bg())

        async def _generate():
            try:
                while True:
                    chunk = await asyncio.wait_for(queue.get(), timeout=120)
                    if chunk is None:
                        yield "data: [DONE]\n\n"
                        break
                    if "error" in chunk:
                        yield f"data: {json.dumps(chunk)}\n\n"
                        break
                    yield f"data: {json.dumps(chunk)}\n\n"

                    # Check if the client disconnected
                    if await request.is_disconnected():
                        log.info("[%s] Client disconnected — cancelling background task", request_id)
                        break
            except asyncio.TimeoutError:
                yield "data: [DONE]\n\n"

        async def _stream_with_cleanup():
            try:
                async for item in _generate():
                    yield item
            finally:
                # Cancel the background task if the client disconnected early
                if not bg_task.done():
                    log.info("[%s] Cancelling graph execution (client gone)", request_id)
                    bg_task.cancel()
                    try:
                        await bg_task
                    except (asyncio.CancelledError, Exception):
                        pass
                streaming_registry.pop(request_id, None)

        return StreamingResponse(
            _stream_with_cleanup(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: run graph to completion, return assembled response
    final_state = await _run_graph(state)
    content = final_state.get("final_response", "")
    returned_tool_calls = final_state.get("tool_calls")
    finish_reason = final_state.get("finish_reason", "stop")
    complexity = final_state.get("complexity", "unknown")
    iterations = final_state.get("iterations", 0)
    log.info("[%s] DONE | complexity=%s iterations=%d finish=%s chars=%d",
             request_id, complexity.upper(), iterations, finish_reason, len(content))
    prompt_tokens = sum(len(m.get("content") or "") // 4 for m in messages)
    completion_tokens = len(content) // 4

    message: dict = {"role": "assistant"}
    if returned_tool_calls:
        message["content"] = None
        message["tool_calls"] = returned_tool_calls
    else:
        message["content"] = content

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
        headers={
            "X-Zenno-Complexity": complexity,
            "X-Zenno-Iterations": str(iterations),
        },
    )


@router.get("/models")
async def list_models():
    """Minimal OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.default_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
            }
        ],
    }
