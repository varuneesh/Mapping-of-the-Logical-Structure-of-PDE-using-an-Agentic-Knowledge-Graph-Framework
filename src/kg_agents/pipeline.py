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
from kg_agents.coreference import coreference
# from kg_agents.utils.logger import PipelineLogger
from kg_agents.extraction.chunk_classifier import should_skip


class GraphState(TypedDict):
 
    # ── Chunk inputs ─────────────────────────────────────────────────────
    chunk_id:       str
    chunk_heading:  str
    primary_chunk:  str
    context_chunks: List[str]
    chunk_type:     str     # "content" | "bibliography" | "frontmatter" | "exercise"
 
    # ── Shared memories ──────────────────────────────────────────────────
    ontology_loader:  Any
    document_memory:  Dict
 
    # ── Coreference outputs ──────────────────────────────────────────────
    resolved_chunk:          str
    coreference_annotations: Dict
 
    # ── Extraction outputs ───────────────────────────────────────────────
    entities:       List[Dict]
    relationships:  List[Dict]
 
    # ── Validation ───────────────────────────────────────────────────────
    validation_status: str
    validation_errors: List[str]
    entity_errors:     List[str]
    relation_errors:   List[str]
 
    # ── Retry control (per-stage counters) ───────────────────────────────
    entity_retry_count:   int
    relation_retry_count: int
    retry_feedback:       str
    extraction_started:   bool   # True after first entity extraction attempt
    _rate_limited:        bool   # True when LLM hit rate limit — skip loop
 
    # ── Routing ──────────────────────────────────────────────────────────
    next_step:   str
    agent_trace: List[str]
 
    # ── Post-consistency outputs ─────────────────────────────────────────
    consistent_entities:       List[Dict]
    consistent_relationships:  List[Dict]
 
    # ── Agent handles (injected once, reused across chunks) ──────────────
    coreference_agent: Any
    entity_agent:      Any
    relation_agent:    Any
    supervisor:        Any
    validator:         Any
    alignment_agent:   Any
    consistency_agent: Any
    graph_builder:     Any
 
    # ── Logger ───────────────────────────────────────────────────────────
    logger: Any    # PipelineLogger instance — shared across all chunks
 
 
# ────────────────────────────────────────────────────────────────────────────
#  Node functions
# ────────────────────────────────────────────────────────────────────────────
 
def classifier_node(state: GraphState):
    """
    Classify chunk type before any LLM work.
    Skipped chunks go straight to END — no API calls wasted.
    """
    print("\n----- CHUNK CLASSIFIER -----")
    logger = state.get("logger")
    chunk  = {"heading": state["chunk_heading"],
              "content": state["primary_chunk"],
              "chunk_id": state["chunk_id"]}
 
    skip, chunk_type, reason = should_skip(chunk)
    state["chunk_type"] = chunk_type
 
    print(f"  Type: {chunk_type} | {reason}")
 
    if logger:
        logger.info("ChunkClassifier", "classified", {
            "chunk_id":   state["chunk_id"],
            "chunk_type": chunk_type,
            "skip":       skip,
            "reason":     reason,
        })
        if skip:
            # Close the chunk immediately — nothing will run
            logger.end_chunk(state["chunk_id"], state)
 
    state["agent_trace"].append("classifier")
    return state
 
 
def route_from_classifier(state: GraphState) -> str:
    """Skip non-content chunks entirely — go straight to END."""
    if state["chunk_type"] in ("bibliography", "frontmatter"):
        print(f"  → Skipping {state['chunk_type']} chunk.")
        return "skip"
    return "process"
 
 
def coreference_node(state: GraphState):
    print("\n----- COREFERENCE -----")
    logger = state.get("logger")
    try:
        state = state["coreference_agent"].run(state)
        if logger:
            logger.info("CoreferenceAgent", "completed", {
                "resolutions": len(state.get("coreference_annotations", {}))
            })
    except Exception as e:
        if logger:
            logger.log_exception("CoreferenceAgent", e, {"chunk_id": state["chunk_id"]})
        state["resolved_chunk"] = state["primary_chunk"]
        state["coreference_annotations"] = {}
    state["agent_trace"].append("coreference")
    return state
 
 
def supervisor_node(state: GraphState):
    print("\n----- SUPERVISOR -----")
    logger = state.get("logger")
    state  = state["supervisor"].run(state)
    if logger:
        logger.info("ExtractionSupervisor", "routed", {"next_step": state["next_step"]})
    state["agent_trace"].append("supervisor")
    return state
 
 
def entity_extraction_node(state: GraphState):
    print("\n----- ENTITY EXTRACTION -----")
    print("Chunk:", state["chunk_id"])
    logger = state.get("logger")
    if state.get("resolved_chunk"):
        state["primary_chunk"] = state["resolved_chunk"]
    state["extraction_started"] = True   # mark that extraction has been attempted
    try:
        state = state["entity_agent"].run(state)
        if logger:
            logger.info("EntityExtractionAgent", "extracted", {
                "count": len(state["entities"]),
                "names": [e["name"] for e in state["entities"]]
            })
        print(f"Extracted {len(state['entities'])} entities")
        for e in state["entities"]:
            print(f"  {e['name']} → {e['type']} ({e['confidence']})")
    except Exception as e:
        if logger:
            logger.log_exception("EntityExtractionAgent", e, {"chunk_id": state["chunk_id"]})
        state["entities"] = []
    state["agent_trace"].append("entity_extraction")
    return state
 
 
def relation_extraction_node(state: GraphState):
    print("\n----- RELATION EXTRACTION -----")
    logger = state.get("logger")
    try:
        state = state["relation_agent"].run(state)
        if logger:
            logger.info("RelationExtractionAgent", "extracted", {
                "count": len(state["relationships"]),
            })
        print(f"Extracted {len(state['relationships'])} relations")
        for r in state["relationships"]:
            print(f"  {r['source']} →[{r['relation']}]→ {r['target']} ({r['confidence']})")
    except Exception as e:
        if logger:
            logger.log_exception("RelationExtractionAgent", e, {"chunk_id": state["chunk_id"]})
        state["relationships"] = []
    state["agent_trace"].append("relation_extraction")
    return state
 
 
def validator_node(state: GraphState):
    print("\n----- VALIDATOR -----")
    logger = state.get("logger")
    state  = state["validator"].validate(state)
    print(f"Validation: {state['validation_status']}")
    if logger:
        if state["validation_status"] == "invalid":
            logger.log("ExtractionValidator", "validation_failed", {
                "entity_errors":   state["entity_errors"],
                "relation_errors": state["relation_errors"],
            }, level="WARNING")
        else:
            logger.info("ExtractionValidator", "validation_passed", {})
    state["agent_trace"].append("validator")
    return state
 
 
def alignment_agent_node(state: GraphState):
    print("\n----- ALIGNMENT -----")
    logger = state.get("logger")
    try:
        state = state["alignment_agent"].run(state)
        if logger:
            logger.info("AlignmentAgent", "completed", {
                "canonical_entities": len(state["entities"])
            })
    except Exception as e:
        if logger:
            logger.log_exception("AlignmentAgent", e, {"chunk_id": state["chunk_id"]})
    state["agent_trace"].append("alignment_agent")
    return state
 
 
def consistency_agent_node(state: GraphState):
    print("\n----- CONSISTENCY -----")
    logger = state.get("logger")
    state  = state["consistency_agent"].run(state)
    if logger:
        logger.info("ConsistencyAgent", "filtered", {
            "consistent_entities":  len(state["consistent_entities"]),
            "consistent_relations": len(state["consistent_relationships"]),
        })
    state["agent_trace"].append("consistency_agent")
    return state
 
 
def graph_builder_node(state: GraphState):
    print("\n----- GRAPH BUILDER -----")
    logger = state.get("logger")
    gb     = state["graph_builder"]
    state  = gb.run(state)
    if logger:
        logger.info("GraphBuilderAgent", "updated", {
            "total_nodes": len(gb.graph["nodes"]),
            "total_edges": len(gb.graph["edges"]),
        })
        logger.end_chunk(state["chunk_id"], state)
    state["agent_trace"].append("graph_builder")
    return state
 
 
def route_from_supervisor(state: GraphState):
    print("Routing →", state["next_step"])
    return state["next_step"]
 
 
# ────────────────────────────────────────────────────────────────────────────
#  Pipeline construction
# ────────────────────────────────────────────────────────────────────────────
 
def build_pipeline():
 
    graph = StateGraph(GraphState)
 
    graph.add_node("classifier",         classifier_node)
    graph.add_node("coreference",        coreference_node)
    graph.add_node("supervisor",         supervisor_node)
    graph.add_node("entity_extraction",  entity_extraction_node)
    graph.add_node("relation_extraction",relation_extraction_node)
    graph.add_node("validator",          validator_node)
    graph.add_node("alignment_agent",    alignment_agent_node)
    graph.add_node("consistency_agent",  consistency_agent_node)
    graph.add_node("graph_builder",      graph_builder_node)
 
    # Classifier runs first — skips non-content chunks immediately
    graph.set_entry_point("classifier")
    graph.add_conditional_edges(
        "classifier",
        route_from_classifier,
        {
            "skip":    END,
            "process": "coreference",
        }
    )
    graph.add_edge("coreference",         "supervisor")
 
    graph.add_edge("entity_extraction",   "relation_extraction")
    graph.add_edge("relation_extraction", "validator")
    graph.add_edge("validator",           "supervisor")
    graph.add_edge("alignment_agent",     "consistency_agent")
    graph.add_edge("consistency_agent",   "graph_builder")
    graph.add_edge("graph_builder",       END)
 
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "entity_extraction":       "entity_extraction",
            "retry_entity_extraction": "entity_extraction",
            "retry_relation_extraction":"relation_extraction",
            "alignment_agent":         "alignment_agent",
        }
    )
 
    Graph = graph.compile()
    print(display(Image(Graph.get_graph(xray=True).draw_mermaid_png())))
    return Graph
 
 
# ────────────────────────────────────────────────────────────────────────────
#  Initialisation
# ────────────────────────────────────────────────────────────────────────────
 
def initialize_pipeline():
 
    CORE_ONTOLOGY = r"..\ontology_versions\core\ontology_core_v0_0.json"
    EXTENSIONS    = r"..\ontology_versions\extensions\ontology_extensions_v0_0.json"
    CANDIDATES    = r"..\ontology_versions\candidates\ontology_candidates.json"
 
    ontology = loader.OntologyLoader(CORE_ONTOLOGY, EXTENSIONS)
 
    coref_agent       = coreference.CoreferenceAgent()
    entity_agent      = entity_extraction_agent.EntityExtractionAgent(ontology)
    relation_agent    = relation_extraction_agent.RelationExtractionAgent(ontology)
    Supervisor        = supervisor.ExtractionSupervisor()
    Validator         = validator.ExtractionValidator(ontology)
    alignment_agent   = alignment.AlignmentAgent()
    consistency_agent = consistency.ConsistencyAgent(candidates_path=CANDIDATES)
    graph_builder_agent = graph_builder.GraphBuilderAgent(
        graph_memory={"nodes": {}, "edges": {}}
    )
 
    pipeline = build_pipeline()
 
    return (
        pipeline, ontology,
        coref_agent,
        entity_agent, relation_agent,
        Supervisor, Validator,
        alignment_agent, consistency_agent, graph_builder_agent,
    )
 
 
def make_chunk_state(
    chunk: Dict,
    context_chunks: List[str],
    document_memory: Dict,
    coref_agent,
    entity_agent,
    relation_agent,
    sup,
    val,
    align,
    consist,
    gb,
    logger=None,
) -> GraphState:
    """
    Build a clean, fully-initialised state dict for a single chunk.
    Call this in your notebook instead of constructing the dict manually.
    Every field is reset to its correct default so nothing leaks between chunks.
    """
    # Register this chunk with the logger so timing and problems are tracked
    if logger:
        logger.start_chunk(chunk["chunk_id"], chunk.get("heading", ""))
 
    return GraphState(
        # Chunk inputs
        chunk_id       = chunk["chunk_id"],
        chunk_heading  = chunk.get("heading", ""),
        primary_chunk  = chunk["content"],
        context_chunks = context_chunks,
        chunk_type     = "",   # filled by classifier_node
 
        # Shared memories
        ontology_loader  = None,
        document_memory  = document_memory,
 
        # Coreference — populated by coreference_node
        resolved_chunk          = "",
        coreference_annotations = {},
 
        # Extraction outputs — empty until agents run
        entities      = [],
        relationships = [],
 
        # Validation — reset every chunk
        validation_status = "",
        validation_errors = [],
        entity_errors     = [],
        relation_errors   = [],
 
        # Retry counters — MUST be reset per chunk
        entity_retry_count   = 0,
        relation_retry_count = 0,
        retry_feedback       = "",
        extraction_started   = False,
        _rate_limited        = False,
 
        # Routing
        next_step   = "",
        agent_trace = [],
 
        # Post-consistency
        consistent_entities      = [],
        consistent_relationships = [],
 
        # Agent handles
        coreference_agent = coref_agent,
        entity_agent      = entity_agent,
        relation_agent    = relation_agent,
        supervisor        = sup,
        validator         = val,
        alignment_agent   = align,
        consistency_agent = consist,
        graph_builder     = gb,
 
        # Logger — shared across all chunks, None is safe (agents check)
        logger = logger,
    )

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