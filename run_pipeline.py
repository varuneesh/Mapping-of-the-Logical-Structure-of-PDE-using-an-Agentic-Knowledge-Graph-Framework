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
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT_DIR       = Path.cwd() 

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

INTER_CHUNK_CADENCE = 30   
 
 
def run_inter_chunk(
    chunks_json_path:   str,
    graph_builder_agent,
    ontology,
    logger,
    doc_id: str = "",
    reason: str = "",
) -> dict:

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

    start = max(0, index - window)
    return [c["content"] for c in chunks[start:index]]
 
 
def save_graph(graph_memory: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_memory, f, indent=2)
    print(f"\n[Pipeline] Graph saved → {path}")
    
def _to_jsonable(obj):

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
    chunks_json_path: str = "", 
    run_cli_review: bool = True,
) -> OntologyLoader:

    print("\n" + "="*60)
    print("ONTOLOGY UPDATE TRIGGERED")
    print("="*60)
 
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
 
    if run_cli_review:
        proposer.review_cli(str(proposals_path))
    else:
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

    new_ontology = OntologyLoader(str(CORE_ONTOLOGY), str(new_ext_path))

    reclass.reclassify_graph_nodes(
        str(new_ext_path),
        graph_builder_agent=graph_builder_agent,
        document_memory=document_memory,
        logger=logger,
    )

    reclass.reclassify_relations(
        str(new_ext_path),
        graph_builder_agent=graph_builder_agent,
        consistency_agent=consistency_agent,
        ontology_loader=new_ontology,
        logger=logger,
    )

    reclass.rescue_stranded_relations(
        graph_builder_agent=graph_builder_agent,
        consistency_agent=consistency_agent,
        ontology_loader=new_ontology,
        logger=logger,
    )

    graph_builder_agent.recompute_all_salience()
 
    reclass.rescue_stranded_relations(
        graph_builder_agent=graph_builder_agent,
        consistency_agent=consistency_agent,
        ontology_loader=new_ontology,
        logger=logger,
    )
 
    if chunks_json_path:
        run_inter_chunk(
            chunks_json_path    = chunks_json_path,
            graph_builder_agent = graph_builder_agent,
            ontology            = new_ontology,
            logger              = logger,
            reason              = "post-reclassification",
        )

    reload_agents_ontology(new_ontology, entity_agent, relation_agent, validator)
 
    print("="*60)
    print("ONTOLOGY UPDATE COMPLETE")
    print("="*60 + "\n")
 
    return new_ontology
 

 
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

    chunks   = load_chunks(chunks_json_path)
    doc_name = Path(chunks_json_path).stem.replace("_chunks", "")
 
    print(f"\n{'='*60}")
    print(f"INGESTING: {doc_name}  ({len(chunks)} chunks)")
    print(f"{'='*60}")
 
    content_chunks_since_interchunk = 0   
 
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
 
        try:
            final_state = pipeline.invoke(state)
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
            if ontology is not old_ontology:
                content_chunks_since_interchunk = 0

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

        if (i + 1) % 25 == 0:
            save_graph(graph_builder_agent.graph, GRAPH_OUT)
            print(f"  [Checkpoint] Graph saved at chunk {i+1}.")
 
    run_inter_chunk(
        chunks_json_path    = chunks_json_path,
        graph_builder_agent = graph_builder_agent,
        ontology            = ontology,
        logger              = logger,
        doc_id              = doc_name,
        reason              = "end of document",
    )
 
    return ontology
 
 
def run_corpus(
    chunks_files:   list,
    run_name:       str  = "kg_run",
    run_cli_review: bool = True,
):

    logger   = PipelineLogger(run_name, log_dir=str(LOG_DIR))
    ontology = OntologyLoader(str(CORE_ONTOLOGY), str(EXTENSIONS))
 
    entity_agent      = EntityExtractionAgent(ontology)
    relation_agent    = RelationExtractionAgent(ontology)
    supervisor        = ExtractionSupervisor()
    validator         = ExtractionValidator(ontology)
    alignment_agent   = AlignmentAgent(index_path = str(INDEX_PATH))
    consistency_agent = ConsistencyAgent(candidates_path=str(CANDIDATES),
                                        ontology_loader=ontology)

    if GRAPH_OUT.exists():
        with open(GRAPH_OUT, "r", encoding="utf-8") as f:
            existing_graph = json.load(f)
        print(f"[Pipeline] Loaded existing graph — "
            f"{len(existing_graph['nodes'])} nodes, "
            f"{len(existing_graph['edges'])} edges")
    else:
        existing_graph = {"nodes": {}, "edges": []}
        print("[Pipeline] No existing graph found — starting fresh.")

    graph_builder_agent = GraphBuilderAgent(graph_memory=existing_graph)
    
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

    print(f"\n{'='*60}")
    print(f"KG CONSTRUCTION PIPELINE — {run_name}")
    print(f"Documents : {len(chunks_files)}")
    print(f"Log       : {logger.jsonl_path}")
    print(f"{'='*60}\n")

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

    print("\n" + "="*60)
    print("FINAL ONTOLOGY UPDATE PASS")
    print("="*60)
 
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
 
    graph_builder_agent.recompute_all_salience()

    save_graph(graph_builder_agent.graph, GRAPH_OUT)
    alignment_agent.save_index()
    print(f"[Pipeline] Entity index saved — {len(alignment_agent.index.names)} entities")
    save_document_memory(document_memory, DOCUMENT_MEMORY_PATH)

    consistency_agent.save_to_disk()

    logger.save_summary()
    logger.print_problem_chunks()
    logger.close()
 
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


if __name__ == "__main__":

    CHUNKS_FILES = [
        str(CHUNKS_DIR / "Morton_Numerical Solution of PDE_chunks.json"),
    ]

    graph, ontology = run_corpus(
        chunks_files   = CHUNKS_FILES,
        run_name       = "morton_pde_run_1_auto_run",
        run_cli_review = False,   
    )