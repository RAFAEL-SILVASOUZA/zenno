import asyncio
import logging
import time
import uuid

from openai import AsyncOpenAI

from core.config import settings
from graph.state import GraphState

log = logging.getLogger("zenno")

# Global registry: request_id -> asyncio.Queue
streaming_registry: dict[str, asyncio.Queue] = {}


def _text(content) -> str:
    """Extract plain text from a string or a multimodal content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
    )


def _sse_chunk(content: str, model: str, response_id: str, finish: bool = False) -> dict:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None,
            }
        ],
    }


async def _push(request_id: str, chunk: dict) -> None:
    q = streaming_registry.get(request_id)
    if q:
        await q.put(chunk)


async def _close(request_id: str) -> None:
    q = streaming_registry.get(request_id)
    if q:
        await q.put(None)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def classify_node(state: GraphState) -> GraphState:
    """Quick LLM call to decide if the request needs reasoning or not."""
    last_user = _text(next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    ))

    log.info("[%s] CLASSIFY | question: %.120s", state["request_id"], last_user)

    prompt = (
        "Analyze the user request below and respond with ONLY one word: "
        '"simple" or "complex".\n\n'
        "simple = greetings, factual lookups, basic translation, short creative tasks\n"
        "complex = multi-step reasoning, debugging, analysis, planning, math, coding\n\n"
        f"Request: {last_user}\n\nAnswer:"
    )

    client = _client()
    response = await client.chat.completions.create(
        model=state["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip().lower()
    complexity = "complex" if "complex" in raw else "simple"
    log.info("[%s] CLASSIFY | raw=%r  ->  route=%s", state["request_id"], raw, complexity.upper())
    return {**state, "complexity": complexity}


async def direct_response_node(state: GraphState) -> GraphState:
    """Stream Ollama response directly for simple requests and all tool-call requests."""
    has_tools = bool(state.get("tools"))
    log.info("[%s] DIRECT | stream=%s tools=%s", state["request_id"], state["do_stream"], has_tools)
    client = _client()

    kwargs: dict = {
        "model": state["model"],
        "messages": state["messages"],
        "temperature": state["temperature"],
    }
    if state.get("tools"):
        kwargs["tools"] = state["tools"]
    if state.get("tool_choice") is not None:
        kwargs["tool_choice"] = state["tool_choice"]

    if state["do_stream"]:
        stream = await client.chat.completions.create(**kwargs, stream=True)
        async for chunk in stream:
            # Use model_dump() to preserve tool_call deltas and finish_reason as-is
            await _push(state["request_id"], chunk.model_dump())
        await _close(state["request_id"])
        log.info("[%s] DIRECT | stream complete", state["request_id"])
        return {**state, "final_response": "", "tool_calls": None, "finish_reason": "stop"}
    else:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [tc.model_dump() for tc in choice.message.tool_calls]
            log.info("[%s] DIRECT | finish_reason=tool_calls  calls=%s",
                     state["request_id"], [tc["function"]["name"] for tc in tool_calls])
        else:
            log.info("[%s] DIRECT | finish_reason=stop  chars=%d",
                     state["request_id"], len(choice.message.content or ""))
        return {
            **state,
            "final_response": choice.message.content or "",
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason or "stop",
        }


async def reasoning_step_node(state: GraphState) -> GraphState:
    """One reasoning iteration. First pass uses chain-of-thought; subsequent passes
    show the model its own previous attempt plus specific critique as a real conversation,
    which is far more effective than system-prompt injection."""
    client = _client()
    thoughts = state["thoughts"]
    iteration = state["iterations"]

    log.info("[%s] REASONING | iteration=%d", state["request_id"], iteration + 1)

    if not thoughts:
        system_content = (
            "You are a precise and thorough assistant. "
            "Before answering, reason step by step inside <thinking> tags, "
            "then write your final answer after them. "
            "Be thorough and consider edge cases."
        )
        msgs = [{"role": "system", "content": system_content}, *state["messages"]]
    else:
        critique = state.get("critique", "The response can be improved.")
        log.info("[%s] REASONING | applying critique: %.200s", state["request_id"], critique)
        msgs = [
            *state["messages"],
            {"role": "assistant", "content": thoughts[-1]},
            {
                "role": "user",
                "content": (
                    "A critical reviewer evaluated your response and found these issues:\n\n"
                    f"{critique}\n\n"
                    "Please provide an improved response that addresses every point above. "
                    "Be direct and complete."
                ),
            },
        ]

    response = await client.chat.completions.create(
        model=state["model"],
        messages=msgs,
        temperature=state["temperature"],
    )

    thought = response.choices[0].message.content.strip()
    log.info("[%s] REASONING | iteration=%d complete  chars=%d  preview: %.100s",
             state["request_id"], iteration + 1, len(thought), thought.replace("\n", " "))
    return {
        **state,
        "thoughts": [*thoughts, thought],
        "iterations": iteration + 1,
        "final_response": thought,
    }


async def evaluate_node(state: GraphState) -> GraphState:
    """Evaluate the current response and return a specific, actionable critique."""
    client = _client()

    last_user = _text(next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    ))

    prompt = (
        "You are a critical reviewer. Evaluate the response below against the request.\n\n"
        "Reply in this exact format — nothing else:\n"
        "VERDICT: GOOD\n"
        "or\n"
        "VERDICT: NEEDS_WORK\n"
        "CRITIQUE: <bullet list of specific issues and what is missing or wrong>\n\n"
        f"REQUEST:\n{last_user}\n\n"
        f"RESPONSE:\n{state['final_response']}"
    )

    response = await client.chat.completions.create(
        model=state["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    quality = "good" if "VERDICT: GOOD" in raw.upper() else "needs_improvement"

    log.info("[%s] EVALUATE | verdict=%s", state["request_id"], quality.upper())

    # Extract critique section for the next reasoning step
    critique = ""
    if quality == "needs_improvement":
        for line in raw.splitlines():
            if line.upper().startswith("CRITIQUE:"):
                critique = line[len("CRITIQUE:"):].strip()
                break
        # If critique spans multiple lines
        if not critique:
            lines = raw.splitlines()
            in_critique = False
            parts = []
            for line in lines:
                if line.upper().startswith("CRITIQUE:"):
                    in_critique = True
                    parts.append(line[len("CRITIQUE:"):].strip())
                elif in_critique:
                    parts.append(line)
            critique = "\n".join(parts).strip()

        log.info("[%s] EVALUATE | critique: %s", state["request_id"], critique)

    return {**state, "quality": quality, "critique": critique}


def _strip_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> blocks the model uses for internal reasoning."""
    import re
    clean = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return clean.strip()


async def stream_final_node(state: GraphState) -> GraphState:
    """Deliver the reasoned response to the user.

    No extra LLM call — the reasoning already produced a good answer (EVALUATE=GOOD).
    We strip internal <thinking> tags and stream the result directly, avoiding
    context-size issues and the latency of a second generation.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    clean = _strip_thinking(state["final_response"])

    log.info("[%s] STREAM_FINAL | after %d iteration(s)  stream=%s  chars=%d",
             state["request_id"], state["iterations"], state["do_stream"], len(clean))

    if state["do_stream"]:
        # Send the complete response as a single SSE chunk — the user already waited
        # for the reasoning loop; there is no benefit in simulating token-by-token
        # streaming here, and chunked loops stall the asyncio event loop.
        await _push(state["request_id"], _sse_chunk(clean, state["model"], response_id))
        await _push(state["request_id"], _sse_chunk("", state["model"], response_id, finish=True))
        await _close(state["request_id"])
        log.info("[%s] STREAM_FINAL | pushed to queue — done", state["request_id"])

    return {**state, "final_response": clean}
