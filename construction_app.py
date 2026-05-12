"""
construction_app.py — KG Construction Interface
Upload PDF/JSON → pipeline runs → graph visualized → Neo4j export
Usage: streamlit run construction_app.py
"""
import streamlit as st
import json, sys, os, math
from pathlib import Path
from collections import Counter
from datetime import datetime
import numpy as np

ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from kg_agents.extraction.loader import OntologyLoader
from kg_agents.extraction.entity_extraction_agent import EntityExtractionAgent
from kg_agents.extraction.relation_extraction_agent import RelationExtractionAgent
from kg_agents.extraction.supervisor import ExtractionSupervisor
from kg_agents.extraction.validator import ExtractionValidator
from kg_agents.extraction.chunk_classifier import classify_chunk
from kg_agents.alignment.alignment import AlignmentAgent
from kg_agents.consistency.consistency import ConsistencyAgent
from kg_agents.graph.graph_builder import GraphBuilderAgent
from kg_agents.coreference.coreference import CoreferenceAgent
from kg_agents.proposer.proposer import OntologyProposerAgent
from kg_agents.proposer.reclassification import ReclassificationPass
from kg_agents.proposer.inter_chunk_relation_extractor import InterChunkRelationExtractor
from kg_agents.utils.logger import PipelineLogger
from kg_agents.pipeline import build_pipeline, make_chunk_state

CORE = ROOT_DIR/"ontology_versions"/"core"/"ontology_core_v0_0.json"
EXTS = ROOT_DIR/"ontology_versions"/"extensions"/"ontology_extensions_v0_0.json"
CANDS = ROOT_DIR/"ontology_versions"/"candidates"/"ontology_candidates.json"
ONT_DIR = ROOT_DIR/"ontology_versions"/"extensions"
PROP_DIR = ROOT_DIR/"ontology_versions"/"proposals"
CHUNKS_DIR = ROOT_DIR/"data"/"chunks"
GRAPH_OUT = ROOT_DIR/"data"/"graph_memory.json"
LOG_DIR = ROOT_DIR/"logs"
IDX_PATH = ROOT_DIR/"data"/"entity_index"
DMEM_PATH = ROOT_DIR/"data"/"document_memory.json"

st.set_page_config(page_title="KG Construction",page_icon="🧠",layout="wide",initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
:root{--bg:#fafaf8;--surface:#fff;--ink:#111827;--muted:#6b7280;--border:#e5e7eb;--blue:#2563eb;--green:#059669;--amber:#d97706;--red:#dc2626;--purple:#7c3aed;}
.stApp{background:var(--bg);font-family:'Inter',sans-serif;}
.hero{background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:14px;padding:2rem 2.5rem;margin-bottom:1.8rem;}
.hero h1{font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:500;color:#f1f5f9;margin:0 0 .3rem;}
.hero p{color:#94a3b8;font-size:.85rem;margin:0;}
.scard{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem;text-align:center;}
.snum{font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;color:var(--blue);}
.slbl{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-top:.1rem;}
.echip{display:inline-block;padding:.18rem .5rem;border-radius:5px;font-size:.7rem;font-family:'Space Mono',monospace;margin:.1rem;border:1px solid var(--border);}
.c-m{background:#eff6ff;color:#1d4ed8;} .c-t{background:#fef2f2;color:#991b1b;}
.c-p{background:#fffbeb;color:#92400e;} .c-e{background:#fdf2f8;color:#9d174d;}
.c-s{background:#ecfdf5;color:#065f46;} .c-d{background:#f9fafb;color:#6b7280;}
.c-n{background:#fef3c7;color:#92400e;}
.rrow{font-family:'Space Mono',monospace;font-size:.7rem;padding:.2rem 0;}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>
""",unsafe_allow_html=True)

TC={"NumericalMethod":"c-m","Theorem":"c-t","TheoreticalProperty":"c-p","ErrorConcept":"c-e",
    "MathematicalStructure":"c-s","ComputationalStructure":"c-s","FloatingPointConcept":"c-p","NEW_TYPE":"c-n"}

def ec(n,t): return f'<span class="echip {TC.get(t,"c-d")}">{n} <small>({t})</small></span>'

def lj(p):
    if Path(p).exists(): return json.load(open(p))
    return None

def sj(d,p):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    json.dump(d,open(p,"w"),indent=2,default=str)


def save_graph(graph_memory: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_memory, f, indent=2)
    print(f"[Streamlit] Graph saved → {path}")

def get_context_chunks(chunks: list, index: int, window: int = 2) -> list:
    start = max(0, index - window)
    return [c.get("content", "") for c in chunks[start:index]]


def _to_jsonable(obj):
    """Recursively convert numpy objects to JSON-serializable Python objects."""
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

def load_chunks(chunks_json_path: str):
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_document_memory(path: Path) -> dict:
    if not path.exists():
        return {"entities": {}}
    with open(path, "r", encoding="utf-8") as f:
        document_memory = json.load(f)
    entities = document_memory.get("entities", {})
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

def reload_agents_ontology(ontology, entity_agent, relation_agent, validator):
    entity_agent.ontology = ontology
    relation_agent.ontology = ontology
    validator.ontology = ontology

def run_inter_chunk(
    chunks_json_path: str,
    graph_builder_agent,
    ontology,
    logger,
    doc_id: str = "",
    reason: str = "",
) -> dict:
    extractor = InterChunkRelationExtractor(chunks_json_path)
    report = extractor.run(
        graph_memory=graph_builder_agent.graph,
        ontology_loader=ontology,
        logger=logger,
        doc_id=doc_id,
    )
    return report

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
    proposals_path = proposer.run(
        candidate_pool=consistency_agent.candidate_pool,
        graph_memory=graph_builder_agent.graph,
        ontology_loader=ontology,
        document_memory=document_memory,
        logger=logger,
    )

    if proposals_path is None:
        return ontology

    if run_cli_review:
        proposer.review_cli(str(proposals_path))
    else:
        proposals = proposer.load_proposals(str(proposals_path))
        for section in ("entity_proposals", "relation_proposals"):
            for p in proposals[section]:
                if p["status"] == "pending":
                    p["status"] = "accepted"
                    p["reviewed_at"] = datetime.now().isoformat()
                    p["auto_accepted"] = True
        proposer._save_proposals(proposals, str(proposals_path))

    result = proposer.apply_accepted_proposals(
        proposals_path=str(proposals_path),
        ontology_loader=ontology,
        candidate_pool=consistency_agent.candidate_pool,
        graph_builder_agent=graph_builder_agent,
        document_memory=document_memory,
        logger=logger,
    )

    if result is None:
        return ontology

    new_ext_path, _cluster_map = result
    new_ontology = OntologyLoader(str(CORE), str(new_ext_path))

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
    if chunks_json_path:
        run_inter_chunk(
            chunks_json_path=chunks_json_path,
            graph_builder_agent=graph_builder_agent,
            ontology=new_ontology,
            logger=logger,
            reason="post-reclassification",
        )
    reload_agents_ontology(new_ontology, entity_agent, relation_agent, validator)
    return new_ontology

def process_chunks_like_run_pipeline(
    chunks_json_path: str,
    ontology: OntologyLoader,
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
    document_memory: dict,
    logger: PipelineLogger,
    status,
    prog,
    detail,
    auto_accept_proposals: bool = True,
    export_to_neo4j: bool = True,
):
    chunks = load_chunks(chunks_json_path)
    doc_name = Path(chunks_json_path).stem.replace("_chunks", "")

    pipeline = build_pipeline()
    stats = {"c": 0, "e": 0, "r": 0, "n": len(graph_builder_agent.graph.get("nodes", {})), "ed": len(graph_builder_agent.graph.get("edges", [])), "p": 0, "s": 0}
    content_chunks_since_interchunk = 0

    status.write(f"📘 {doc_name} — {len(chunks)} chunks")

    for i, ch in enumerate(chunks):
        cid = ch["chunk_id"]
        prog.progress((i + 1) / len(chunks))

        chunk_type, _signals = classify_chunk(ch)
        if chunk_type != "content":
            stats["s"] += 1
            status.write(f"⏭ {i+1}/{len(chunks)} skip")
            continue

        stats["c"] += 1
        status.write(f"⚙️ {i+1}/{len(chunks)} — {ch.get('heading', '')[:45]}")

        context_chunks = get_context_chunks(chunks, i, window=2)
        state = make_chunk_state(
            chunk=ch,
            context_chunks=context_chunks,
            document_memory=document_memory,
            coref_agent=coref_agent,
            entity_agent=entity_agent,
            relation_agent=relation_agent,
            sup=supervisor,
            val=validator,
            align=alignment_agent,
            consist=consistency_agent,
            gb=graph_builder_agent,
            logger=logger,
        )

        try:
            res = pipeline.invoke(state)
            if res.get("chunk_type", "content") == "content":
                content_chunks_since_interchunk += 1
            stats["e"] += len(res.get("consistent_entities", []))
            stats["r"] += len(res.get("consistent_relationships", []))
            stats["n"] = len(graph_builder_agent.graph["nodes"])
            stats["ed"] = len(graph_builder_agent.graph["edges"])
            with detail.container():
                ents = res.get("consistent_entities", [])
                rels = res.get("consistent_relationships", [])
                if ents:
                    st.markdown(" ".join(ec(e["name"], e["type"]) for e in ents[:8]), unsafe_allow_html=True)
                if rels:
                    for r in rels[:4]:
                        st.markdown(f'<div class="rrow">{r["source"]} →[{r["relation"]}]→ {r["target"]}</div>', unsafe_allow_html=True)
        except Exception as ex:
            status.write(f"❌ {i+1}: {str(ex)[:80]}")
            logger.warning("Pipeline", "exception", {"chunk_id": cid, "error": str(ex)})
            continue

        proposer.tick()

        should_run, _reason = proposer.should_run(consistency_agent.candidate_pool)
        if should_run:
            status.write("💡 Ontology update triggered...")
            old_ontology = ontology
            ontology = run_ontology_update(
                proposer=proposer,
                reclass=reclass,
                consistency_agent=consistency_agent,
                graph_builder_agent=graph_builder_agent,
                document_memory=document_memory,
                ontology=ontology,
                entity_agent=entity_agent,
                relation_agent=relation_agent,
                validator=validator,
                logger=logger,
                chunks_json_path=chunks_json_path,
                run_cli_review=False,
            )
            if ontology is not old_ontology:
                stats["p"] += 1
                content_chunks_since_interchunk = 0

        elif content_chunks_since_interchunk >= 30:
            run_inter_chunk(
                chunks_json_path=chunks_json_path,
                graph_builder_agent=graph_builder_agent,
                ontology=ontology,
                logger=logger,
                doc_id=doc_name,
                reason=f"cadence ({30} content chunks)",
            )
            content_chunks_since_interchunk = 0

        if (i + 1) % 25 == 0:
            save_graph(graph_builder_agent.graph, GRAPH_OUT)

    run_inter_chunk(
        chunks_json_path=chunks_json_path,
        graph_builder_agent=graph_builder_agent,
        ontology=ontology,
        logger=logger,
        doc_id=doc_name,
        reason="end of document",
    )

    # Final ontology pass, matching run_pipeline.py
    final_proposals = proposer.run(
        candidate_pool=consistency_agent.candidate_pool,
        graph_memory=graph_builder_agent.graph,
        ontology_loader=ontology,
        document_memory=document_memory,
        logger=logger,
    )
    if final_proposals:
        proposals = proposer.load_proposals(str(final_proposals))
        for section in ("entity_proposals", "relation_proposals"):
            for p in proposals[section]:
                if p["status"] == "pending":
                    p["status"] = "accepted"
                    p["reviewed_at"] = datetime.now().isoformat()
                    p["auto_accepted"] = True
        proposer._save_proposals(proposals, str(final_proposals))

        final_result = proposer.apply_accepted_proposals(
            str(final_proposals),
            ontology_loader=ontology,
            candidate_pool=consistency_agent.candidate_pool,
            graph_builder_agent=graph_builder_agent,
            document_memory=document_memory,
            logger=logger,
        )
        if final_result:
            new_ext, _cluster_map = final_result
            ontology = OntologyLoader(str(CORE), str(new_ext))
            reclass.reclassify_graph_nodes(
                str(new_ext),
                graph_builder_agent=graph_builder_agent,
                document_memory=document_memory,
                logger=logger,
            )
            reclass.reclassify_relations(
                str(new_ext),
                graph_builder_agent=graph_builder_agent,
                consistency_agent=consistency_agent,
                ontology_loader=ontology,
                logger=logger,
            )
            reclass.rescue_stranded_relations(
                graph_builder_agent=graph_builder_agent,
                consistency_agent=consistency_agent,
                ontology_loader=ontology,
                logger=logger,
            )
            graph_builder_agent.recompute_all_salience()
            reload_agents_ontology(ontology, entity_agent, relation_agent, validator)

    graph_builder_agent.recompute_all_salience()
    save_graph(graph_builder_agent.graph, GRAPH_OUT)
    alignment_agent.save_index()
    save_document_memory(document_memory, DMEM_PATH)
    consistency_agent.save_to_disk()

    if export_to_neo4j:
        try:
            from graph_neo4j import Neo4jExporter
            ex = Neo4jExporter()
            ex.export(graph_builder_agent.graph, incremental=True)
        except Exception as e:
            logger.warning("Neo4j", "export_failed", {"error": str(e)})

    return ontology, stats, graph_builder_agent.graph, document_memory

def graph_html(graph,w=900,h=550):
    N=[{"id":n,"type":d.get("type","?"),"sal":d.get("salience",0.5),"src":len(d.get("sources",[]))} for n,d in graph.get("nodes",{}).items()]
    L=[{"source":e["source"],"target":e["target"],"rel":e["relation"]} for e in graph.get("edges",[])]
    C={"NumericalMethod":"#2563eb","ProblemType":"#7c3aed","Theorem":"#dc2626","TheoreticalProperty":"#d97706","ErrorConcept":"#db2777","Definition":"#0891b2","MathematicalStructure":"#059669","ComputationalStructure":"#4f46e5","FloatingPointConcept":"#ea580c","MathematicalObject":"#6b7280"}
    return f"""<div id="gv" style="width:100%;height:{h}px;background:#0f172a;border-radius:12px;overflow:hidden;position:relative;">
    <div style="position:absolute;top:8px;left:8px;z-index:10;font-family:monospace;font-size:8px;">
    {''.join(f'<div style="display:flex;align-items:center;gap:4px;margin:1px 0;"><div style="width:7px;height:7px;border-radius:50%;background:{c};"></div><span style="color:#94a3b8;">{t}</span></div>' for t,c in C.items() if any(n['type']==t for n in N))}
    </div><svg id="gs" width="100%" height="100%"></svg></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script>(function(){{const N={json.dumps(N)},L={json.dumps(L)},C={json.dumps(C)},W={w},H={h},p=30;
    const s=d3.select('#gs').attr('viewBox',[0,0,W,H]);
    s.append('defs').append('marker').attr('id','a').attr('viewBox','0 -5 10 10').attr('refX',22).attr('refY',0).attr('markerWidth',5).attr('markerHeight',5).attr('orient','auto').append('path').attr('d','M0,-5L10,0L0,5').attr('fill','#475569');
    const sim=d3.forceSimulation(N).force('link',d3.forceLink(L).id(d=>d.id).distance(85)).force('charge',d3.forceManyBody().strength(-180)).force('center',d3.forceCenter(W/2,H/2)).force('collision',d3.forceCollide().radius(18)).force('x',d3.forceX(W/2).strength(0.06)).force('y',d3.forceY(H/2).strength(0.06));
    const lk=s.append('g').selectAll('line').data(L).join('line').attr('stroke','#334155').attr('stroke-width',1).attr('marker-end','url(#a)');
    const ll=s.append('g').selectAll('text').data(L).join('text').text(d=>d.rel).attr('font-size','5.5px').attr('font-family','monospace').attr('fill','#64748b').attr('text-anchor','middle');
    const nd=s.append('g').selectAll('circle').data(N).join('circle').attr('r',d=>Math.max(4,Math.min(13,d.sal*4))).attr('fill',d=>C[d.type]||'#6b7280').attr('stroke','#1e293b').attr('stroke-width',1).call(d3.drag().on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}}).on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y;}}).on('end',(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}));
    const nl=s.append('g').selectAll('text').data(N).join('text').text(d=>d.id.length>18?d.id.substring(0,16)+'..':d.id).attr('font-size','6.5px').attr('font-family','Inter,sans-serif').attr('fill','#cbd5e1').attr('dx',9).attr('dy',3);
    nd.append('title').text(d=>d.id+' ('+d.type+')\\nSalience:'+d.sal.toFixed(3));
    sim.on('tick',()=>{{N.forEach(d=>{{d.x=Math.max(p,Math.min(W-p,d.x));d.y=Math.max(p,Math.min(H-p,d.y));}});lk.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);ll.attr('x',d=>(d.source.x+d.target.x)/2).attr('y',d=>(d.source.y+d.target.y)/2);nd.attr('cx',d=>d.x).attr('cy',d=>d.y);nl.attr('x',d=>d.x).attr('y',d=>d.y);}});}})();</script>"""

# Session
for k,v in [("pstate","idle"),("chunks",[]),("stats",{"c":0,"e":0,"r":0,"n":0,"ed":0,"p":0,"s":0}),
            ("graph",lj(str(GRAPH_OUT)) or {"nodes":{},"edges":[]})]:
    if k not in st.session_state: st.session_state[k]=v

st.markdown('<div class="hero"><h1>Knowledge Graph Construction</h1><p>Upload PDF or chunks JSON → pipeline extracts entities and relations → graph built and exported to Neo4j</p></div>',unsafe_allow_html=True)

t1,t2,t3=st.tabs(["📄 Ingest","🕸️ Graph","📊 Stats"])

with t1:
    cu,cm=st.columns([1,2])
    with cu:
        st.markdown("### Upload")
        up=st.file_uploader("PDF or Chunks JSON",type=["pdf","json"])
        if up and up.name.endswith(".json"): st.success(f"📄 {up.name} — chunks file")
        elif up:
            st.success(f"📄 {up.name} — PDF")
            st.markdown("**Mathpix credentials**")
            mx_id=st.text_input("App ID",key="mxid")
            mx_key=st.text_input("App Key",type="password",key="mxkey")

        aa=st.checkbox("Auto-accept proposals",True)
        export_to_neo4j=st.checkbox("Export to Neo4j",True)
        run_btn=up and st.button("🚀 Run Pipeline",type="primary",use_container_width=True)

        st.markdown("---")
        st.markdown("### Pipeline Stages")
        st.markdown("""
1. **PDF → LaTeX** — Mathpix preserves math notation
2. **Chunking** — Section-based with heading metadata
3. **Coreference** — Resolves implicit references
4. **Entity Extraction** — Ontology-typed concepts
5. **Relation Extraction** — Domain-range validated
6. **Alignment** — Embedding-based deduplication
7. **Graph Construction** — Incremental with salience
8. **Ontology Evolution** — Evidence-driven proposals
        """)

        st.markdown("---")
        st.markdown("### Ingested Documents")
        g_check = lj(str(GRAPH_OUT)) or {"nodes":{},"edges":[]}
        if not g_check.get("nodes"):
            st.caption("No documents processed yet.")
        else:
            ds=set()
            for d in g_check["nodes"].values():
                for s in d.get("sources",[]):
                    ds.add(s.rsplit("_chunk_",1)[0] if "_chunk_" in s else s)
            for did in sorted(ds):
                nc=sum(1 for nd in g_check["nodes"].values()
                       if any(did in s for s in nd.get("sources",[])))
                st.markdown(f"""<div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;
                    padding:.6rem .9rem;margin:.3rem 0;font-size:.82rem;">
                    📘 <strong>{did.split('/')[-1]}</strong><br/>
                    <span style="color:#6b7280;">{nc} entities · {len(g_check['nodes'])} total nodes</span>
                </div>""",unsafe_allow_html=True)

    with cm:
        st.markdown("### Monitor")
        if st.session_state.pstate=="idle" and not run_btn:
            st.info("Upload a document and click Run Pipeline.")
        elif run_btn or st.session_state.pstate=="running":
            # Keep the existing UI, but run the same document flow as run_pipeline.py.
            if st.session_state.pstate != "running":
                st.session_state.pstate = "running"

            # PDF path: save to disk, run convert_pdf_to_latex + chunker
            if not up.name.endswith(".json"):
                import tempfile
                pdf_bytes = up.read()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_pdf_path = tmp.name

                latex_out_dir = ROOT_DIR / "data" / "latex_outputs"
                chunks_out_dir = ROOT_DIR / "data" / "chunks"

                with st.status("📄 Converting PDF → LaTeX...", expanded=True) as conv_status:
                    try:
                        from kg_agents.ingestion.pdf_to_latex import convert_pdf_to_latex
                        conv_status.write("⏳ Calling Mathpix API (may take 1-3 min)...")
                        latex_file = convert_pdf_to_latex(tmp_pdf_path, str(latex_out_dir))
                        conv_status.write(f"✅ LaTeX saved to {latex_file}")
                    except Exception as e:
                        st.error(f"PDF conversion failed: {e}")
                        st.stop()

                with st.status("✂️ Chunking LaTeX...", expanded=False) as ch_status:
                    try:
                        from kg_agents.ingestion.chunker import chunk_latex_document
                        file_path = None
                        for folder in os.listdir(latex_file):
                            folder_path = os.path.join(latex_file, folder)
                            candidate = os.path.join(folder_path, folder + ".tex")
                            if os.path.isfile(candidate):
                                file_path = candidate
                                break

                        if not file_path:
                            st.error("No .tex file found in extracted LaTeX output.")
                            st.stop()

                        doc_name = Path(up.name).stem
                        proper_dir = ROOT_DIR / "data" / "latex_outputs" / doc_name
                        proper_dir.mkdir(parents=True, exist_ok=True)
                        proper_tex = proper_dir / f"{doc_name}.tex"

                        import shutil
                        shutil.copy(file_path, proper_tex)
                        file_path = str(proper_tex)

                        if os.path.isfile(file_path):
                            chunks = chunk_latex_document(
                                latex_path=file_path,
                                output_dir=str(chunks_out_dir),
                                max_chars=4000
                            )
                            chunks_file = str(chunks_out_dir / f"{Path(file_path).stem}_chunks.json")
                            ch_status.update(label=f"✅ {len(chunks)} chunks created", state="complete")
                    except Exception as e:
                        st.error(f"Chunking failed: {e}")
                        st.stop()
            else:
                chunks = json.load(up)
                CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
                chunks_file = str(CHUNKS_DIR / up.name)
                sj(chunks, chunks_file)

            ont = OntologyLoader(str(CORE), str(EXTS))
            eg = lj(str(GRAPH_OUT)) or {"nodes": {}, "edges": []}
            gb = GraphBuilderAgent(graph_memory=eg)
            cs = ConsistencyAgent(candidates_path=str(CANDS), ontology_loader=ont)
            ea = EntityExtractionAgent(ontology_loader=ont)
            ra = RelationExtractionAgent(ontology_loader=ont)
            va = ExtractionValidator(ontology_loader=ont)
            su = ExtractionSupervisor()
            al = AlignmentAgent(index_path=str(IDX_PATH))
            co = CoreferenceAgent()
            pr = OntologyProposerAgent(ontology_dir=str(ONT_DIR), proposals_dir=str(PROP_DIR))
            rc = ReclassificationPass(core_path=str(CORE), candidates_path=str(CANDS))
            dm = load_document_memory(DMEM_PATH)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            lg = PipelineLogger(f"app_{datetime.now():%Y%m%d_%H%M%S}", str(LOG_DIR))

            prog = st.progress(0)
            status = st.status("Running...", expanded=True)
            detail = st.empty()

            try:
                ont, S, graph, dm = process_chunks_like_run_pipeline(
                    chunks_json_path=chunks_file,
                    ontology=ont,
                    entity_agent=ea,
                    relation_agent=ra,
                    supervisor=su,
                    validator=va,
                    alignment_agent=al,
                    consistency_agent=cs,
                    graph_builder_agent=gb,
                    coref_agent=co,
                    proposer=pr,
                    reclass=rc,
                    document_memory=dm,
                    logger=lg,
                    status=status,
                    prog=prog,
                    detail=detail,
                    auto_accept_proposals=aa,
                    export_to_neo4j=export_to_neo4j,
                )
                st.session_state.stats = S
                st.session_state.graph = graph
                st.session_state.pstate = "complete"
            except Exception as ex:
                status.write(f"❌ Pipeline failed: {ex}")
                lg.warning("App", "pipeline_failed", {"error": str(ex)})
                st.session_state.pstate = "idle"
                st.stop()

            prog.progress(1.0)
            status.update(label=f"✅ Done — {S['n']} nodes, {S['ed']} edges", state="complete", expanded=False)

        elif st.session_state.pstate=="complete":
            S=st.session_state.stats
            st.markdown("### Results")
            cols=st.columns(6)
            for c,l,v in zip(cols,["Chunks","Entities","Relations","Nodes","Edges","Proposals"],
                             [S["c"],S["e"],S["r"],S["n"],S["ed"],S["p"]]):
                with c: st.markdown(f'<div class="scard"><div class="snum">{v}</div><div class="slbl">{l}</div></div>',unsafe_allow_html=True)
            if st.button("Process another"): st.session_state.pstate="idle"; st.rerun()

with t2:
    g=st.session_state.graph
    if not g.get("nodes"):
        dg=lj(str(GRAPH_OUT))
        if dg and dg.get("nodes"): g=dg; st.session_state.graph=g
    if not g.get("nodes"): st.info("No graph. Process a document first.")
    else:
        st.markdown(f"### Knowledge Graph — {len(g['nodes'])} nodes · {len(g['edges'])} edges")
        c1,c2,c3=st.columns(3)
        at=sorted(set(d["type"] for d in g["nodes"].values()));ar=sorted(set(e["relation"] for e in g["edges"]))
        with c1: st_=st.multiselect("Types",at,default=at)
        with c2: sr_=st.multiselect("Relations",ar,default=ar)
        with c3: ms_=st.slider("Min salience",0.0,3.0,0.0,0.1)
        fn={n:d for n,d in g["nodes"].items() if d["type"] in st_ and d.get("salience",0)>=ms_}
        fe=[e for e in g["edges"] if e["source"] in fn and e["target"] in fn and e["relation"] in sr_]
        st.components.v1.html(graph_html({"nodes":fn,"edges":fe}),height=580,scrolling=False)
        tc1,tc2=st.columns(2)
        with tc1:
            st.markdown("#### Nodes")
            nd=[{"Entity":n,"Type":d["type"],"Salience":round(d.get("salience",0),3),"Sources":len(d.get("sources",[]))} for n,d in sorted(fn.items(),key=lambda x:-x[1].get("salience",0))]
            if nd: st.dataframe(nd,use_container_width=True,hide_index=True)
        with tc2:
            st.markdown("#### Edges")
            ed=[{"Source":e["source"],"Relation":e["relation"],"Target":e["target"]} for e in fe]
            if ed: st.dataframe(ed,use_container_width=True,hide_index=True)

with t3:
    g=st.session_state.graph
    if not g.get("nodes"):
        dg=lj(str(GRAPH_OUT))
        if dg: g=dg
    if not g.get("nodes"): st.info("No data.")
    else:
        st.markdown("### Statistics")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("#### Entity Types")
            for t,c in Counter(d["type"] for d in g["nodes"].values()).most_common():
                p=int(c/len(g["nodes"])*100)
                st.markdown(f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;"><div style="width:120px;font-size:.75rem;">{t}</div><div style="flex:1;height:16px;background:#f1f5f9;border-radius:3px;overflow:hidden;"><div style="width:{p}%;height:100%;background:#2563eb;border-radius:3px;"></div></div><div style="width:25px;text-align:right;font-family:monospace;font-size:.7rem;">{c}</div></div>',unsafe_allow_html=True)
        with c2:
            st.markdown("#### Relation Types")
            for t,c in Counter(e["relation"] for e in g["edges"]).most_common():
                p=int(c/max(len(g["edges"]),1)*100)
                st.markdown(f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;"><div style="width:120px;font-size:.75rem;">{t}</div><div style="flex:1;height:16px;background:#f1f5f9;border-radius:3px;overflow:hidden;"><div style="width:{p}%;height:100%;background:#059669;border-radius:3px;"></div></div><div style="width:25px;text-align:right;font-family:monospace;font-size:.7rem;">{c}</div></div>',unsafe_allow_html=True)
        st.markdown("---")
        cf=[(n,d) for n,d in g["nodes"].items() if d.get("type_conflict")]
        iso=sum(1 for n in g["nodes"] if not any(e["source"]==n or e["target"]==n for e in g["edges"]))
        ad=len(g["edges"])*2/max(len(g["nodes"]),1)
        hc=st.columns(4)
        for col,(l,v) in zip(hc,[("Avg Degree",f"{ad:.1f}"),("Isolated",iso),("Conflicts",len(cf)),("Inter-Chunk",sum(1 for e in g["edges"] if e.get("inter_chunk")))]):
            with col: st.markdown(f'<div class="scard"><div class="snum">{v}</div><div class="slbl">{l}</div></div>',unsafe_allow_html=True)
        if cf:
            st.markdown("#### Type Conflicts")
            for n,d in cf: st.warning(f"**{n}**: {d.get('observed_types',[])}")
        st.markdown("#### Documents")
        ds=set()
        for d in g["nodes"].values():
            for s in d.get("sources",[]):
                ds.add(s.rsplit("_chunk_",1)[0] if "_chunk_" in s else s)
        for d in sorted(ds):
            nc=sum(1 for nd in g["nodes"].values() if any(d in s for s in nd.get("sources",[])))
            st.markdown(f"📘 **{d}** — {nc} entities")