import asyncio
import time
import uuid

from openai import AsyncOpenAI

from core.config import settings
from graph.state import GraphState

# Global registry: request_id -> asyncio.Queue
streaming_registry: dict[str, asyncio.Queue] = {}


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
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )

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
    return {**state, "complexity": complexity}


async def direct_response_node(state: GraphState) -> GraphState:
    """Stream Ollama response directly for simple requests and all tool-call requests."""
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
        return {**state, "final_response": "", "tool_calls": None, "finish_reason": "stop"}
    else:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [tc.model_dump() for tc in choice.message.tool_calls]
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

    if not thoughts:
        # First attempt: instruct the model to think step by step
        system_content = (
            "You are a precise and thorough assistant. "
            "Before answering, reason step by step inside <thinking> tags, "
            "then write your final answer after them. "
            "Be thorough and consider edge cases."
        )
        msgs = [{"role": "system", "content": system_content}, *state["messages"]]
    else:
        # Subsequent attempts: build a real conversation so the model sees its own mistake
        critique = state.get("critique", "The response can be improved.")
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
    return {
        **state,
        "thoughts": [*thoughts, thought],
        "iterations": iteration + 1,
        "final_response": thought,
    }


async def evaluate_node(state: GraphState) -> GraphState:
    """Evaluate the current response and return a specific, actionable critique."""
    client = _client()

    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )

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

    return {**state, "quality": quality, "critique": critique}


async def stream_final_node(state: GraphState) -> GraphState:
    """Stream the best reasoning result to the user."""
    client = _client()
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Build synthesis prompt that instructs the model to present the answer clearly
    best_thought = state["final_response"]
    last_user = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    )

    system_content = (
        "You have already reasoned through the problem carefully. "
        "Now deliver a clean, direct final answer — no meta-commentary, no 'based on my analysis'. "
        "Just the answer."
    )

    synthesis_messages = [
        {"role": "system", "content": system_content},
        *state["messages"][:-1],
        {
            "role": "user",
            "content": (
                f"{last_user}\n\n"
                f"[Your internal analysis: {best_thought}]\n\n"
                "Final answer:"
            ),
        },
    ]

    if state["do_stream"]:
        stream = await client.chat.completions.create(
            model=state["model"],
            messages=synthesis_messages,
            temperature=state["temperature"],
            stream=True,
        )
        full_content = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_content.append(delta)
                await _push(state["request_id"], _sse_chunk(delta, state["model"], response_id))
        await _push(state["request_id"], _sse_chunk("", state["model"], response_id, finish=True))
        await _close(state["request_id"])
    else:
        response = await client.chat.completions.create(
            model=state["model"],
            messages=synthesis_messages,
            temperature=state["temperature"],
        )
        full_content = [response.choices[0].message.content or ""]

    return {**state, "final_response": "".join(full_content)}
