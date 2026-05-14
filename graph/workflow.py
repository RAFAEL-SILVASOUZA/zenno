from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    classify_node,
    direct_response_node,
    evaluate_node,
    reasoning_step_node,
    stream_final_node,
)
from graph.state import GraphState


def _route_after_classify(state: GraphState) -> str:
    # When tools are present the agent relies on tool-use cycles for its own
    # intelligence. The reasoning loop works on text only (no tool calls),
    # so forcing it here would block the agent from accessing its tools.
    if state.get("tools"):
        return "direct"
    return "direct" if state["complexity"] == "simple" else "reasoning"


def _route_after_evaluate(state: GraphState) -> str:
    if state["iterations"] >= state["max_iterations"]:
        return "stream_final"
    if state["quality"] == "good":
        return "stream_final"
    return "reasoning"


def build_workflow() -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("classify", classify_node)
    builder.add_node("direct", direct_response_node)
    builder.add_node("reasoning", reasoning_step_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("stream_final", stream_final_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", _route_after_classify)
    builder.add_edge("direct", END)
    builder.add_edge("reasoning", "evaluate")
    builder.add_conditional_edges("evaluate", _route_after_evaluate)
    builder.add_edge("stream_final", END)

    return builder.compile()


workflow = build_workflow()
