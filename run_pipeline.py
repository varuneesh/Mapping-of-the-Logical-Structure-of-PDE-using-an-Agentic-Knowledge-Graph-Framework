"""
run_pipeline.py

Complete end-to-end KG construction pipeline.

Usage:
    python run_pipeline.py

Or import and call run_corpus() from your notebook.

Pipeline flow per document:
    chunks → [coref → supervisor → entity → relation → validator
              → alignment → consistency → graph_builder] per chunk
    → inter-chunk relation extraction (after each document)

Ontology evolution (triggered during ingestion):
    proposer fires → human review → apply → reclassification
    → continue with updated ontology

After all documents:
    final proposer run → final reclassification
    → final inter-chunk extraction across full corpus
    → save graph + export
"""
# ── Imports ──────────────────────────────────────────────────────────────────
from kg_agents.extraction.loader               import OntologyLoader
from kg_agents.extraction.entity_extraction_agent  import EntityExtractionAgent
from kg_agents.extraction.relation_extraction_agent import RelationExtractionAgent
from kg_agents.extraction.supervisor           import ExtractionSupervisor
from kg_agents.extraction.validator            import ExtractionValidator
from kg_agents.alignment.alignment             import AlignmentAgent
from kg_agents.consistency.consistency         import ConsistencyAgent
from kg_agents.graph.graph_builder             import GraphBuilderAgent
from kg_agents.coreference.coreference         import CoreferenceAgent
from kg_agents.proposer.proposer               import OntologyProposerAgent
from kg_agents.proposer.reclassification       import ReclassificationPass
from kg_agents.proposer.inter_chunk_relation_extractor import InterChunkRelationExtractor
from kg_agents.utils.logger                    import PipelineLogger
from kg_agents.pipeline                        import build_pipeline, make_chunk_state
import json
# import os
from pathlib import Path
from datetime import datetime
import numpy as np

# ── Adjust these paths to match your project layout ─────────────────────────
ROOT_DIR       = Path.cwd()  # project root

CORE_ONTOLOGY  = ROOT_DIR / "ontology_versions" / "core"     / "ontology_core_v0_0.json"
EXTENSIONS     = ROOT_DIR / "ontology_versions" / "extensions"/ "ontology_extensions_v0_0.json"
CANDIDATES     = ROOT_DIR / "ontology_versions" / "candidates"/ "ontology_candidates.json"
ONTOLOGY_DIR   = ROOT_DIR / "ontology_versions" / "extensions"
PROPOSALS_DIR  = ROOT_DIR / "ontology_versions" / "proposals"
CHUNKS_DIR     = ROOT_DIR / "data" / "chunks"
GRAPH_OUT      = ROOT_DIR / "data" / "graph_memory.json"
LOG_DIR        = ROOT_DIR / "logs"
INDEX_PATH = ROOT_DIR / "data" / "entity_index"
DOCUMENT_MEMORY_PATH = ROOT_DIR / "data" / "document_memory.json"


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
 
 
# ── Inter-chunk extraction trigger config ────────────────────────────────────
# Run inter-chunk extraction every N content chunks during ingestion.
# Also runs automatically after every ontology update (reclassification
# may add new nodes that now have enough co-occurrence for inter-chunk edges).
INTER_CHUNK_CADENCE = 30   # run every 30 content chunks
 
 
def run_inter_chunk(
    chunks_json_path:   str,
    graph_builder_agent,
    ontology,
    logger,
    doc_id: str = "",
    reason: str = "",
) -> dict:
    """
    Run inter-chunk relation extraction and return the report.
    Safe to call multiple times — existing edges are merged, not duplicated.
    """
    print(f"\n[InterChunk] Running{f' ({reason})' if reason else ''}...")
    extractor = InterChunkRelationExtractor(chunks_json_path)
    report = extractor.run(
        graph_memory    = graph_builder_agent.graph,
        ontology_loader = ontology,
        logger          = logger,
        doc_id          = doc_id,
    )
    print(f"[InterChunk] +{report['relations_inserted']} edges "
          f"from {report['pairs_evaluated']} pairs evaluated.")
    return report
 
 
def load_chunks(chunks_json_path: str):
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def get_context_chunks(chunks: list, index: int, window: int = 2) -> list:
    """Return up to `window` preceding chunk texts as context."""
    start = max(0, index - window)
    return [c["content"] for c in chunks[start:index]]
 
 
def save_graph(graph_memory: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_memory, f, indent=2)
    print(f"\n[Pipeline] Graph saved → {path}")
    
def _to_jsonable(obj):
    """
    Recursively convert numpy objects to JSON-serializable Python objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    return obj

def load_document_memory(path: Path) -> dict:
    if not path.exists():
        print(f"[Pipeline] No existing document memory found at {DOCUMENT_MEMORY_PATH} — starting fresh.")
        return {"entities": {}}

    with open(path, "r", encoding="utf-8") as f:
        document_memory = json.load(f)

    entities = document_memory.get("entities", {})
    print(f"[Pipeline] Loaded document memory from {DOCUMENT_MEMORY_PATH}")
    for name, info in entities.items():
        emb = info.get("embedding")
        if emb is not None and not isinstance(emb, np.ndarray):
            info["embedding"] = np.array(emb, dtype=float)

    return document_memory


def save_document_memory(document_memory: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_memory = _to_jsonable(document_memory)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable_memory, f, indent=2, ensure_ascii=False)

    print(f"\n[Pipeline] Document memory saved → {path}")
 
def reload_agents_ontology(ontology, entity_agent, relation_agent, validator):
    """Push updated ontology to all agents that depend on it."""
    entity_agent.ontology   = ontology
    relation_agent.ontology = ontology
    validator.ontology      = ontology
 
 
def run_ontology_update(
    proposer,
    reclass,
    consistency_agent,
    graph_builder_agent,
    document_memory,
    ontology,
    entity_agent,
    relation_agent,
    validator,
    logger,
    chunks_json_path: str = "",   # needed for inter-chunk after reclassification
    run_cli_review: bool = True,
) -> OntologyLoader:
    """
    Full ontology update cycle:
      propose → review → apply → reclassify → inter-chunk → reload
    Returns the updated OntologyLoader.
    """
    print("\n" + "="*60)
    print("ONTOLOGY UPDATE TRIGGERED")
    print("="*60)
 
    # 1. Generate proposals
    proposals_path = proposer.run(
        candidate_pool=consistency_agent.candidate_pool,
        graph_memory=graph_builder_agent.graph,
        ontology_loader=ontology,
        document_memory=document_memory,
        logger=logger,
    )
 
    if proposals_path is None:
        print("[OntologyUpdate] No proposals generated — continuing.")
        return ontology
 
    # 2. Human review (or auto-accept)
    if run_cli_review:
        proposer.review_cli(str(proposals_path))
    else:
        # Auto-accept all proposals — no human review
        proposals = proposer.load_proposals(str(proposals_path))
        for section in ("entity_proposals", "relation_proposals"):
            for p in proposals[section]:
                if p["status"] == "pending":
                    p["status"]      = "accepted"
                    p["reviewed_at"] = datetime.now().isoformat()
                    p["auto_accepted"] = True
        proposer._save_proposals(proposals, str(proposals_path))
        n_accepted = sum(
            1 for s in ("entity_proposals", "relation_proposals")
            for p in proposals[s] if p["status"] == "accepted"
        )
        print(f"[OntologyUpdate] Auto-accepted {n_accepted} proposal(s).")
 
    # 3. Apply accepted proposals → write new extensions file
    #    Cluster members are inserted into graph INSIDE apply_accepted_proposals.
    result = proposer.apply_accepted_proposals(
        proposals_path=str(proposals_path), ontology_loader=ontology,
        candidate_pool=consistency_agent.candidate_pool,
        graph_builder_agent=graph_builder_agent,
        document_memory=document_memory,
        logger=logger,
    )

    if result is None:
        print("[OntologyUpdate] No proposals accepted — continuing.")
        return ontology

    new_ext_path, cluster_map = result

    # 4. Load updated ontology
    new_ontology = OntologyLoader(str(CORE_ONTOLOGY), str(new_ext_path))

    # 4b. Retype existing graph nodes to new subclasses (embedding-based)
    reclass.reclassify_graph_nodes(
        str(new_ext_path),
        graph_builder_agent=graph_builder_agent,
        document_memory=document_memory,
        logger=logger,
    )

    # 4c. Reclassify any NEW_RELATION candidates that match new relation types
    reclass.reclassify_relations(
        str(new_ext_path),
        graph_builder_agent=graph_builder_agent,
        consistency_agent=consistency_agent,
        ontology_loader=new_ontology,
        logger=logger,
    )

    # 4d. Rescue relations stranded because endpoints were just added to graph
    reclass.rescue_stranded_relations(
        graph_builder_agent=graph_builder_agent,
        consistency_agent=consistency_agent,
        ontology_loader=new_ontology,
        logger=logger,
    )

    # 4e. Recompute salience for all nodes after graph changes
    graph_builder_agent.recompute_all_salience()
 
    # Rescue relations that were stranded because their endpoints
    # were NEW_TYPE and have now been reclassified into the graph.
    reclass.rescue_stranded_relations(
        graph_builder_agent=graph_builder_agent,
        consistency_agent=consistency_agent,
        ontology_loader=new_ontology,
        logger=logger,
    )
 
    # 5. Run inter-chunk extraction immediately after reclassification.
    #    New nodes from reclassification weren't in the graph during the
    #    last inter-chunk pass — run now so they get edges right away.
    if chunks_json_path:
        run_inter_chunk(
            chunks_json_path    = chunks_json_path,
            graph_builder_agent = graph_builder_agent,
            ontology            = new_ontology,
            logger              = logger,
            reason              = "post-reclassification",
        )
 
    # 6. Reload all agents with updated ontology
    reload_agents_ontology(new_ontology, entity_agent, relation_agent, validator)
 
    print("="*60)
    print("ONTOLOGY UPDATE COMPLETE")
    print("="*60 + "\n")
 
    return new_ontology
 
 
# ────────────────────────────────────────────────────────────────────────────
#  Per-document ingestion
# ────────────────────────────────────────────────────────────────────────────
 
def ingest_document(
    chunks_json_path:   str,
    pipeline,
    ontology,
    entity_agent,
    relation_agent,
    supervisor,
    validator,
    alignment_agent,
    consistency_agent,
    graph_builder_agent,
    coref_agent,
    proposer,
    reclass,
    document_memory:    dict,
    logger:             PipelineLogger,
    run_cli_review:     bool = True,
) -> OntologyLoader:
    """
    Ingest a single document (chunks JSON) end-to-end.
    Returns the (possibly updated) OntologyLoader.
    """
    chunks   = load_chunks(chunks_json_path)
    doc_name = Path(chunks_json_path).stem.replace("_chunks", "")
 
    print(f"\n{'='*60}")
    print(f"INGESTING: {doc_name}  ({len(chunks)} chunks)")
    print(f"{'='*60}")
 
    content_chunks_since_interchunk = 0   # tracks cadence trigger
 
    for i, chunk in enumerate(chunks):
 
        print(f"\n[{i+1}/{len(chunks)}] Chunk: {chunk['chunk_id']}")
 
        context = get_context_chunks(chunks, i)
 
        state = make_chunk_state(
            chunk            = chunk,
            context_chunks   = context,
            document_memory  = document_memory,
            coref_agent      = coref_agent,
            entity_agent     = entity_agent,
            relation_agent   = relation_agent,
            sup              = supervisor,
            val              = validator,
            align            = alignment_agent,
            consist          = consistency_agent,
            gb               = graph_builder_agent,
            logger           = logger,
        )
 
        # Run the LangGraph pipeline for this chunk
        try:
            final_state = pipeline.invoke(state)
            # Count content chunks (classifier didn't skip them)
            if final_state.get("chunk_type", "content") == "content":
                content_chunks_since_interchunk += 1
        except Exception as e:
            print(f"  [Pipeline] ERROR on chunk {chunk['chunk_id']}: {e}")
            if logger:
                logger.log_exception("Pipeline", e, {
                    "chunk_id": chunk["chunk_id"]
                })
            continue
 
        proposer.tick()
 
        # ── Ontology update check ─────────────────────────────────────────
        should_run, reason = proposer.should_run(
            consistency_agent.candidate_pool
        )
        if should_run:
            old_ontology = ontology
            ontology = run_ontology_update(
                proposer            = proposer,
                reclass             = reclass,
                consistency_agent   = consistency_agent,
                graph_builder_agent = graph_builder_agent,
                document_memory     = document_memory,
                ontology            = ontology,
                entity_agent        = entity_agent,
                relation_agent      = relation_agent,
                validator           = validator,
                logger              = logger,
                chunks_json_path    = chunks_json_path,
                run_cli_review      = run_cli_review,
            )
            # Only reset cadence if ontology actually changed
            # (run_ontology_update returns a NEW ontology object on change)
            if ontology is not old_ontology:
                content_chunks_since_interchunk = 0
 
        # ── Periodic inter-chunk trigger ──────────────────────────────────
        elif content_chunks_since_interchunk >= INTER_CHUNK_CADENCE:
            run_inter_chunk(
                chunks_json_path    = chunks_json_path,
                graph_builder_agent = graph_builder_agent,
                ontology            = ontology,
                logger              = logger,
                doc_id              = doc_name,
                reason              = f"cadence ({INTER_CHUNK_CADENCE} content chunks)",
            )
            content_chunks_since_interchunk = 0
 
        # ── Periodic graph save (every 25 chunks) ────────────────────────
        if (i + 1) % 25 == 0:
            save_graph(graph_builder_agent.graph, GRAPH_OUT)
            print(f"  [Checkpoint] Graph saved at chunk {i+1}.")
 
    # ── Final inter-chunk pass for this document ──────────────────────────
    # Catches any co-occurrences not yet covered by periodic passes.
    run_inter_chunk(
        chunks_json_path    = chunks_json_path,
        graph_builder_agent = graph_builder_agent,
        ontology            = ontology,
        logger              = logger,
        doc_id              = doc_name,
        reason              = "end of document",
    )
 
    return ontology
 
 
# ────────────────────────────────────────────────────────────────────────────
#  Main corpus run
# ────────────────────────────────────────────────────────────────────────────
 
def run_corpus(
    chunks_files:   list,
    run_name:       str  = "kg_run",
    run_cli_review: bool = True,
):
    """
    Run the full pipeline over a list of chunk JSON files.
 
    Parameters
    ----------
    chunks_files : list[str]
        Ordered list of paths to chunks JSON files.
        e.g. ["data/chunks/ascher_greif_chunks.json",
               "data/chunks/leveque_chunks.json"]
    run_name : str
        Identifier for this run — used in log filenames.
    run_cli_review : bool
        If True, pause for CLI review when ontology proposals are generated.
        Set to False for fully automated runs (proposals are written to disk
        but not reviewed interactively).
    """
 
    # ── Initialise shared state ───────────────────────────────────────────
    logger   = PipelineLogger(run_name, log_dir=str(LOG_DIR))
    ontology = OntologyLoader(str(CORE_ONTOLOGY), str(EXTENSIONS))
 
    entity_agent      = EntityExtractionAgent(ontology)
    relation_agent    = RelationExtractionAgent(ontology)
    supervisor        = ExtractionSupervisor()
    validator         = ExtractionValidator(ontology)
    alignment_agent   = AlignmentAgent(index_path = str(INDEX_PATH))
    consistency_agent = ConsistencyAgent(candidates_path=str(CANDIDATES),
                                        ontology_loader=ontology)
    graph_builder_agent = GraphBuilderAgent(graph_memory={"nodes": {}, "edges": []})
    coref_agent       = CoreferenceAgent()
 
    proposer = OntologyProposerAgent(
        ontology_dir  = str(ONTOLOGY_DIR),
        proposals_dir = str(PROPOSALS_DIR),
    )
    reclass = ReclassificationPass(
        core_path       = str(CORE_ONTOLOGY),
        candidates_path = str(CANDIDATES),
    )
 
    pipeline        = build_pipeline()
    
    document_memory = load_document_memory(DOCUMENT_MEMORY_PATH)
    
    # if Path(DOCUMENT_MEMORY_PATH).exists():
    #     with open(DOCUMENT_MEMORY_PATH, "r", encoding="utf-8") as f:
    #         document_memory = json.load(f)
    #     print(f"[Pipeline] Loaded document memory from {DOCUMENT_MEMORY_PATH}")
    # else:
    #     document_memory = {"entities": {}}
    #     print(f"[Pipeline] No existing document memory found at {DOCUMENT_MEMORY_PATH} — starting fresh.")

    print(f"\n{'='*60}")
    print(f"KG CONSTRUCTION PIPELINE — {run_name}")
    print(f"Documents : {len(chunks_files)}")
    print(f"Log       : {logger.jsonl_path}")
    print(f"{'='*60}\n")
 
    # ── Ingest each document ──────────────────────────────────────────────
    for chunks_path in chunks_files:
 
        if not Path(chunks_path).exists():
            print(f"[Pipeline] WARNING: {chunks_path} not found — skipping.")
            continue
 
        ontology = ingest_document(
            chunks_json_path    = chunks_path,
            pipeline            = pipeline,
            ontology            = ontology,
            entity_agent        = entity_agent,
            relation_agent      = relation_agent,
            supervisor          = supervisor,
            validator           = validator,
            alignment_agent     = alignment_agent,
            consistency_agent   = consistency_agent,
            graph_builder_agent = graph_builder_agent,
            coref_agent         = coref_agent,
            proposer            = proposer,
            reclass             = reclass,
            document_memory     = document_memory,
            logger              = logger,
            run_cli_review      = run_cli_review,
        )
 
    # ── Final ontology update (catch remaining candidates) ────────────────
    print("\n" + "="*60)
    print("FINAL ONTOLOGY UPDATE PASS")
    print("="*60)
 
    # Force final proposer run regardless of thresholds
    final_proposals = proposer.run(
        candidate_pool=consistency_agent.candidate_pool,
        graph_memory=graph_builder_agent.graph,
        ontology_loader=ontology,
        document_memory=document_memory,
        logger=logger,
    )
    if final_proposals:
        if run_cli_review:
            proposer.review_cli(str(final_proposals))
        else:
            proposals = proposer.load_proposals(str(final_proposals))
            for section in ("entity_proposals", "relation_proposals"):
                for p in proposals[section]:
                    if p["status"] == "pending":
                        p["status"]        = "accepted"
                        p["reviewed_at"]   = datetime.now().isoformat()
                        p["auto_accepted"] = True
            proposer._save_proposals(proposals, str(final_proposals))
            print("[Final] Auto-accepted all remaining proposals.")
 
        final_result = proposer.apply_accepted_proposals(
            str(final_proposals), ontology_loader=ontology,
            candidate_pool=consistency_agent.candidate_pool,
            graph_builder_agent=graph_builder_agent,
            document_memory=document_memory,
            logger=logger,
        )
        if final_result:
            new_ext, cluster_map = final_result
            ontology = OntologyLoader(str(CORE_ONTOLOGY), str(new_ext))
            reclass.reclassify_graph_nodes(
                str(new_ext),
                graph_builder_agent=graph_builder_agent,
                document_memory=document_memory,
                logger=logger,
            )
            reclass.reclassify_relations(
                str(new_ext), graph_builder_agent=graph_builder_agent, consistency_agent=consistency_agent,
                ontology_loader=ontology, logger=logger
            )
            reclass.rescue_stranded_relations(
                graph_builder_agent=graph_builder_agent, consistency_agent=consistency_agent, ontology_loader=ontology, logger=logger
            )
            graph_builder_agent.recompute_all_salience()
            reload_agents_ontology(ontology, entity_agent, relation_agent, validator)
 
    # ── Final inter-chunk extraction across full corpus ───────────────────
    # Only makes sense if multiple documents were ingested —
    # finds cross-document relations between shared high-salience concepts.
    if len(chunks_files) > 1:
        print("\n[Pipeline] Running final inter-chunk extraction across corpus...")
        for chunks_path in chunks_files:
            if not Path(chunks_path).exists():
                continue
            extractor = InterChunkRelationExtractor(chunks_path)
            extractor.run(
                graph_memory    = graph_builder_agent.graph,
                ontology_loader = ontology,
                logger          = logger,
                doc_id          = Path(chunks_path).stem,
            )
 
    # ── Final salience recompute ──────────────────────────────────────────
    graph_builder_agent.recompute_all_salience()
 
    # ── Save final graph ──────────────────────────────────────────────────
    save_graph(graph_builder_agent.graph, GRAPH_OUT)
    alignment_agent.save_index()
    print(f"[Pipeline] Entity index saved — {len(alignment_agent.index.names)} entities")
    save_document_memory(document_memory, DOCUMENT_MEMORY_PATH)
 
    # ── Save candidate pool ───────────────────────────────────────────────
    consistency_agent.save_to_disk()
 
    # ── Logger summary ────────────────────────────────────────────────────
    logger.save_summary()
    logger.print_problem_chunks()
    logger.close()
 
    # ── Print top nodes by salience ───────────────────────────────────────
    print("\n[Pipeline] Top 15 nodes by salience:")
    top = graph_builder_agent.top_nodes_by_salience(15)
    for rank, node in enumerate(top, 1):
        print(f"  {rank:2}. {node['name']:<45} "
              f"salience={node['salience']:<6}  type={node['type']}")
 
    print("\n[Pipeline] DONE.")
    print(f"  Graph nodes : {len(graph_builder_agent.graph['nodes'])}")
    print(f"  Graph edges : {len(graph_builder_agent.graph['edges'])}")
    print(f"  Graph saved : {GRAPH_OUT}")
 
    return graph_builder_agent.graph, ontology


# ────────────────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── List your chunk files here ────────────────────────────────────────
    CHUNKS_FILES = [
        str(CHUNKS_DIR / "A First Course in Numerical Methods – Ascher & Greif_chunks.json"),
        # str(CHUNKS_DIR / "leveque_chunks.json"),
        # str(CHUNKS_DIR / "trefethen_bau_chunks.json"),
    ]

    graph, ontology = run_corpus(
        chunks_files   = CHUNKS_FILES,
        run_name       = "ascher_greif_run_1_auto_run",
        run_cli_review = False,    # set False to skip interactive review
    )