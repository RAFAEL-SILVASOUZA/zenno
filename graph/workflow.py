from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    classify_node,
    direct_response_node,
    evaluate_node,
    reasoning_step_node,
    stream_final_node,
    verify_node,
)
from graph.state import GraphState


def _entry_route(state: GraphState) -> str:
    # Sem tools: caminho normal — classifica.
    if not state.get("tools"):
        return "classify"

    # Com tools, mas o último turno NÃO é do usuário — estamos no meio de um
    # loop agentico (resposta de tool ou continuação de tool_call). Manda direto
    # pra não quebrar o fluxo do agente.
    messages = state.get("messages") or []
    last_role = messages[-1].get("role") if messages else None
    if last_role != "user":
        return "direct"

    # Turno fresh do usuário, mesmo com tools disponíveis: vale classificar.
    # O classify_node sabe que tools existem e roteia "simple" pra tarefas que
    # precisam de tool, "complex" só pra raciocínio puro.
    return "classify"


def _route_after_classify(state: GraphState) -> str:
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
    builder.add_node("verify", verify_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("stream_final", stream_final_node)

    builder.add_conditional_edges(START, _entry_route)
    builder.add_conditional_edges("classify", _route_after_classify)
    builder.add_edge("direct", END)
    builder.add_edge("reasoning", "verify")
    builder.add_edge("verify", "evaluate")
    builder.add_conditional_edges("evaluate", _route_after_evaluate)
    builder.add_edge("stream_final", END)

    return builder.compile()


workflow = build_workflow()
