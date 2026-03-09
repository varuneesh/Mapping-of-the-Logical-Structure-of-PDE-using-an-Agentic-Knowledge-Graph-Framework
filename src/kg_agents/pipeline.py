from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from IPython.display import Image, display

from kg_agents.extraction import loader
from kg_agents.extraction import entity_extraction_agent
from kg_agents.extraction import relation_extraction_agent
from kg_agents.extraction import supervisor
from kg_agents.extraction import validator
from kg_agents.alignment import alignment
from kg_agents.consistency import consistency
from kg_agents.graph import graph_builder

class GraphState(TypedDict):

    chunk_id: str
    chunk_heading: str
    primary_chunk: str
    context_chunks: List[str]

    ontology_loader: Any
    document_memory: Dict

    entities: List[Dict]
    relationships: List[Dict]

    validation_status: str
    validation_errors: List[str]

    entity_errors: List[str]
    relation_errors: List[str]

    retry_count: int
    retry_feedback: str

    next_step: str

    agent_trace: List[str]
    
    consistent_entities: List[Dict]
    consistent_relationships: List[Dict]

    entity_agent: Any
    relation_agent: Any
    supervisor: Any
    validator: Any
    alignment_agent: Any
    consistency_agent: Any
    graph_builder: Any


def supervisor_node(state: GraphState):
    
    print("\n----- SUPERVISOR NODE -----")
    sup = state["supervisor"]
    state = sup.run(state)
    state["agent_trace"].append("supervisor")

    return state


def entity_extraction_node(state: GraphState):

    print("\n----- ENTITY EXTRACTION -----")
    print("Chunk ID:", state["chunk_id"])

    agent = state["entity_agent"]
    state = agent.run(state)
    print("Entities Extracted:", len(state["entities"]))
    for e in state["entities"]:
        print("  ", e["name"], "→", e["type"], "| confidence:", e["confidence"])
    state["agent_trace"].append("entity_extraction")

    return state

def relation_extraction_node(state: GraphState):

    print("\n----- RELATION EXTRACTION -----")
    agent = state["relation_agent"]
    state = agent.run(state)
    print("Relations Extracted:", len(state["relationships"]))

    for r in state["relationships"]:
        print(
            "  ",
            r["source"],
            "→",
            r["relation"],
            "→",
            r["target"],
            "| confidence:",
            r["confidence"]
        )

    state["agent_trace"].append("relation_extraction")

    return state


def validator_node(state: GraphState):

    print("\n----- VALIDATOR -----")
    val = state["validator"]
    state = val.validate(state)
    state["agent_trace"].append("validator")

    return state

def alignment_agent_node(state: GraphState):
    print("\n----- ALIGNMENT AGENT -----")
    align=state["alignment_agent"]
    state = align.run(state)
    state["agent_trace"].append("alignment_agent")

    return state

def consistency_agent_node(state: GraphState):
    print("\n----- CONSISTENCY AGENT -----")
    consistency_agent = state["consistency_agent"]
    state = consistency_agent.run(state)
    state["agent_trace"].append("consistency_agent")

    return state

def graph_builder_node(state: GraphState):
    print("\n----- GRAPH BUILDER -----")
    print(state)
    graph_builder = state["graph_builder"]
    state = graph_builder.run(state)
    state["agent_trace"].append("graph_builder")

    return state

def route_from_supervisor(state: GraphState):

    print("\nRouting to:", state["next_step"])

    return state["next_step"]


def build_pipeline():

    graph = StateGraph(GraphState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("relation_extraction", relation_extraction_node)
    graph.add_node("validator", validator_node)
    graph.add_node("alignment_agent", alignment_agent_node)
    graph.add_node("consistency_agent", consistency_agent_node)
    graph.add_node("graph_builder", graph_builder_node)

    graph.set_entry_point("supervisor")

    graph.add_edge("entity_extraction", "relation_extraction")

    graph.add_edge("relation_extraction", "validator")

    graph.add_edge("validator", "supervisor")
    graph.add_edge("alignment_agent", "consistency_agent")
    graph.add_edge("consistency_agent", "graph_builder")
    graph.add_edge("graph_builder", END)

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "entity_extraction": "entity_extraction",
            "retry_entity_extraction": "entity_extraction",
            "retry_relation_extraction": "relation_extraction",
            "alignment_agent": "alignment_agent"
        }
    )

    Graph = graph.compile()
    display(Image(Graph.get_graph(xray=True).draw_mermaid_png()))

    return Graph


def initialize_pipeline():

    ontology = loader.OntologyLoader(
        r"..\ontology_versions\core\ontology_core_v0_0.json",
        r"..\ontology_versions\extensions\ontology_extensions_v0_0.json"
    )

    Validator = validator.ExtractionValidator(ontology)
    entity_agent = entity_extraction_agent.EntityExtractionAgent(ontology)
    relation_agent = relation_extraction_agent.RelationExtractionAgent(ontology)
    Supervisor = supervisor.ExtractionSupervisor()
    alignment_agent = alignment.AlignmentAgent()
    consistency_agent = consistency.ConsistencyAgent(candidate_pool={})
    graph_builder_agent = graph_builder.GraphBuilderAgent(graph_memory={"nodes": {}, "edges": {}})
    pipeline = build_pipeline()

    return pipeline, ontology, entity_agent, relation_agent, Supervisor, Validator, alignment_agent, consistency_agent, graph_builder_agent