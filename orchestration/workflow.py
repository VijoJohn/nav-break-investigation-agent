from langgraph.graph import StateGraph

def investigation_workflow():

    graph = StateGraph(dict)

    graph.add_node("break_detection")
    graph.add_node("investigation")
    graph.add_node("reflection")

    graph.set_entry_point("break_detection")

    graph.add_edge("break_detection", "investigation")
    graph.add_edge("investigation", "reflection")

    return graph.compile()
