from typing import Any, TypedDict


class GraphState(TypedDict):
    # Original request
    messages: list[dict]
    model: str
    temperature: float
    do_stream: bool

    # Tool calling
    tools: list[dict] | None
    tool_choice: Any | None  # str or dict

    # Routing
    complexity: str  # "simple" | "complex"
    quality: str     # "good" | "needs_improvement"

    # Reasoning loop
    iterations: int
    max_iterations: int
    thoughts: list[str]
    critique: str            # specific feedback from evaluator
    final_response: str
    tool_calls: list[dict] | None
    finish_reason: str       # "stop" | "tool_calls" | etc.

    # Streaming channel (request_id key into global queue registry)
    request_id: str
