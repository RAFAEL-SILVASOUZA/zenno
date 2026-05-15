import asyncio
import logging
import random
import time
import uuid

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

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
        timeout=settings.api_request_timeout,
    )


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)


async def _api_call(request_id: str, func, max_retries: int | None = None) -> any:
    """Execute an async API call with exponential backoff retry on transient errors."""
    retries = max_retries if max_retries is not None else settings.max_api_retries

    for attempt in range(1, retries + 1):
        try:
            return await func()
        except _RETRYABLE_ERRORS as exc:
            if attempt >= retries:
                log.error("[%s] API call failed after %d attempt(s): %s", request_id, attempt, exc)
                raise
            delay = min(settings.retry_base_delay * (2 ** (attempt - 1)), settings.retry_max_delay)
            jitter = delay * random.uniform(0.5, 1.0)
            log.warning("[%s] API transient error (attempt %d/%d): %s — retrying in %.1fs",
                        request_id, attempt, retries, exc, jitter)
            await asyncio.sleep(jitter)


async def _safe_api_call(request_id: str, func, fallback=None) -> any:
    """Like _api_call but returns *fallback* on unrecoverable failure instead of raising."""
    try:
        return await _api_call(request_id, func)
    except Exception as exc:
        log.error("[%s] API call exhausted all retries: %s", request_id, exc)
        return fallback


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
    rid = state["request_id"]
    last_user = _text(next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    ))

    log.info("[%s] CLASSIFY | question: %.120s  tools=%s",
             rid, last_user, bool(state.get("tools")))

    base_prompt = (
        'Classify the user request as "simple" or "complex". '
        "Answer with ONLY that one word.\n\n"
        "simple — answerable with a single fact or a one-shot reply:\n"
        "  • greetings, small talk\n"
        '  • single-fact lookups ("capital of France", "who wrote Hamlet")\n'
        "  • direct conversions, basic translation, short definitions\n"
        '  • short creative tasks ("write a haiku about rain")\n\n'
        "complex — benefits from step-by-step reasoning AND does NOT need external data:\n"
        "  • math/word problems, probability, combinatorics, logic puzzles\n"
        "  • problems where the obvious answer is often wrong (counterintuitive)\n"
        "  • pure code/algorithm analysis from text alone (no files to read)\n"
        "  • planning, comparison, trade-off analysis from given info\n"
        "  • anything requiring enumeration of cases or multi-step inference\n"
    )

    if state.get("tools"):
        tool_note = (
            "\nIMPORTANT — tools are available to the assistant in this request.\n"
            "If the request requires looking up external information (files on disk,\n"
            "running commands, web searches, current data, listing/reading anything),\n"
            "answer SIMPLE — the assistant will use tools directly. Pure reasoning\n"
            "without external data → COMPLEX. When in doubt with tools present,\n"
            "prefer SIMPLE.\n"
        )
    else:
        tool_note = "\nWhen in doubt, answer complex.\n"

    examples = (
        "\nExamples:\n"
        "  Request: Hi, how are you?                                     -> simple\n"
        "  Request: Translate 'good morning' to Japanese                 -> simple\n"
        "  Request: What is the capital of Brazil?                       -> simple\n"
        "  Request: List the files in /tmp                               -> simple  (needs tool)\n"
        "  Request: Read main.py and tell me what it does                -> simple  (needs tool)\n"
        "  Request: Maria has two kids. At least one is a boy born on a  -> complex\n"
        "           Tuesday. What is the probability both are boys?\n"
        "  Request: Prove that sqrt(2) is irrational                     -> complex\n"
        "  Request: Plan a 5-day trip to Tokyo with a $2000 budget       -> complex\n\n"
        f"Request: {last_user}\n\nAnswer:"
    )

    prompt = base_prompt + tool_note + examples

    client = _client()
    response = await _api_call(rid, lambda: client.chat.completions.create(
        model=state["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ))

    raw = (response.choices[0].message.content or "").strip().lower()
    # Bias pro lado seguro: se o modelo não disser "simple" explicitamente, trata
    # como complex. Custo de errar pra complex = latência; pra simple = resposta ruim.
    if "complex" in raw:
        complexity = "complex"
    elif "simple" in raw:
        complexity = "simple"
    else:
        complexity = "complex"
    log.info("[%s] CLASSIFY | raw=%r  ->  route=%s", rid, raw, complexity.upper())
    return {**state, "complexity": complexity}


async def direct_response_node(state: GraphState) -> GraphState:
    """Stream Ollama response directly for simple requests and all tool-call requests."""
    rid = state["request_id"]
    has_tools = bool(state.get("tools"))
    log.info("[%s] DIRECT | stream=%s tools=%s", rid, state["do_stream"], has_tools)
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
        try:
            stream = await _api_call(rid, lambda: client.chat.completions.create(**kwargs, stream=True))
            async for chunk in stream:
                await _push(rid, chunk.model_dump())
            await _close(rid)
            log.info("[%s] DIRECT | stream complete", rid)
        except Exception as exc:
            log.error("[%s] DIRECT | stream error: %s", rid, exc)
            await _push(rid, {"error": f"Streaming failed: {exc}"})
            await _close(rid)
        return {**state, "final_response": "", "tool_calls": None, "finish_reason": "stop"}
    else:
        response = await _safe_api_call(rid, lambda: client.chat.completions.create(**kwargs))
        if response is None:
            log.error("[%s] DIRECT | API completely failed, returning empty", rid)
            return {**state, "final_response": "", "tool_calls": None, "finish_reason": "stop"}

        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [tc.model_dump() for tc in choice.message.tool_calls]
            log.info("[%s] DIRECT | finish_reason=tool_calls  calls=%s",
                     rid, [tc["function"]["name"] for tc in tool_calls])
        else:
            log.info("[%s] DIRECT | finish_reason=stop  chars=%d",
                     rid, len(choice.message.content or ""))
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
    rid = state["request_id"]
    client = _client()
    thoughts = state["thoughts"]
    iteration = state["iterations"]

    log.info("[%s] REASONING | iteration=%d", rid, iteration + 1)

    if not thoughts:
        reasoning_instruction = (
            "You are a precise and thorough assistant. "
            "Before answering, reason step by step inside <thinking> tags, "
            "then write your final answer after them. "
            "Be thorough and consider edge cases."
        )
        # Templates de alguns modelos (Gemma 4, etc.) exigem exatamente um
        # system message no índice 0. Se o cliente já mandou system(s),
        # mesclamos todos com nossa instrução em vez de prepender outro.
        existing_messages = list(state["messages"])
        existing_systems = [m for m in existing_messages if m.get("role") == "system"]
        non_system       = [m for m in existing_messages if m.get("role") != "system"]

        parts: list[str] = []
        for m in existing_systems:
            c = _text(m.get("content", ""))
            if c:
                parts.append(c)
        parts.append(reasoning_instruction)
        merged_system = "\n\n".join(parts)

        msgs = [{"role": "system", "content": merged_system}, *non_system]
    else:
        critique = state.get("critique", "The response can be improved.")
        log.info("[%s] REASONING | applying critique: %.200s", rid, critique)
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

    response = await _api_call(rid, lambda: client.chat.completions.create(
        model=state["model"],
        messages=msgs,
        temperature=state["temperature"],
    ))

    thought = (response.choices[0].message.content or "").strip()
    log.info("[%s] REASONING | iteration=%d complete  chars=%d  preview: %.100s",
             rid, iteration + 1, len(thought), thought.replace("\n", " "))
    return {
        **state,
        "thoughts": [*thoughts, thought],
        "iterations": iteration + 1,
        "final_response": thought,
    }


async def evaluate_node(state: GraphState) -> GraphState:
    """Evaluate the current response and return a specific, actionable critique."""
    rid = state["request_id"]
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

    response = await _api_call(rid, lambda: client.chat.completions.create(
        model=state["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ))

    raw = (response.choices[0].message.content or "").strip()
    quality = "good" if "VERDICT: GOOD" in raw.upper() else "needs_improvement"

    log.info("[%s] EVALUATE | verdict=%s", rid, quality.upper())

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

        log.info("[%s] EVALUATE | critique: %s", rid, critique)

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
