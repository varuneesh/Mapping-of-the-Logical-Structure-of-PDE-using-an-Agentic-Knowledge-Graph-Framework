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
        if up and up.name.endswith(".json"): st.success(f"📄 {up.name}")
        elif up: st.success(f"📄 {up.name} (PDF)")
        aa=st.checkbox("Auto-accept proposals",True)
        ne=st.checkbox("Export to Neo4j",True)
        run_btn=up and st.button("🚀 Run Pipeline",type="primary",use_container_width=True)

    with cm:
        st.markdown("### Monitor")
        if st.session_state.pstate=="idle" and not run_btn:
            st.info("Upload a document to begin.")
        elif run_btn or st.session_state.pstate=="running":
            if up.name.endswith(".json"):
                chunks=json.load(up)
                CHUNKS_DIR.mkdir(parents=True,exist_ok=True)
                sj(chunks,str(CHUNKS_DIR/up.name))
                chunks_file=str(CHUNKS_DIR/up.name)
            else:
                st.warning("PDF support requires Mathpix. Upload chunks JSON."); st.stop()

            ont=OntologyLoader(str(CORE),str(EXTS))
            eg=lj(str(GRAPH_OUT)) or {"nodes":{},"edges":[]}
            gb=GraphBuilderAgent(graph_memory=eg)
            cs=ConsistencyAgent(candidates_path=str(CANDS),ontology_loader=ont)
            ea=EntityExtractionAgent(ontology=ont)
            ra=RelationExtractionAgent(ontology=ont)
            va=ExtractionValidator(ontology=ont)
            su=ExtractionSupervisor()
            al=AlignmentAgent(index_path=str(IDX_PATH))
            co=CoreferenceAgent()
            pr=OntologyProposerAgent(ontology_dir=str(ONT_DIR),proposals_dir=str(PROP_DIR))
            rc=ReclassificationPass(core_path=str(CORE),candidates_path=str(CANDS))
            dm=lj(str(DMEM_PATH)) or {"entities":{}}
            LOG_DIR.mkdir(parents=True,exist_ok=True)
            lg=PipelineLogger(str(LOG_DIR),f"app_{datetime.now():%Y%m%d_%H%M%S}")
            pipe=build_pipeline(ea,ra,va,su,al,cs,gb,co)

            prog=st.progress(0)
            status=st.status("Running...",expanded=True)
            detail=st.empty()
            S={"c":0,"e":0,"r":0,"n":0,"ed":0,"p":0,"s":0}

            for i,ch in enumerate(chunks):
                cid=ch["chunk_id"];prog.progress((i+1)/len(chunks))
                ct,_=classify_chunk(ch.get("heading",""),ch.get("content",""))
                if ct!="content": S["s"]+=1; status.write(f"⏭ {i+1}/{len(chunks)} skip"); continue
                S["c"]+=1
                status.write(f"⚙️ {i+1}/{len(chunks)} — {ch.get('heading','')[:45]}")
                state=make_chunk_state(chunk_id=cid,primary_chunk=ch.get("content",""),
                    chunk_heading=ch.get("heading",""),context_chunks=[],
                    document_memory=dm,ontology=ont,logger=lg)
                try:
                    res=pipe.invoke(state)
                    ne_=len(res.get("consistent_entities",[]));nr_=len(res.get("consistent_relationships",[]))
                    S["e"]+=ne_
                    S["r"]+=nr_
                    S["n"]=len(gb.graph["nodes"])
                    S["ed"]=len(gb.graph["edges"])
                    with detail.container():
                        ents=res.get("consistent_entities",[]);rels=res.get("consistent_relationships",[])
                        if ents: 
                            st.markdown(" ".join(ec(e["name"],e["type"]) for e in ents[:8]),unsafe_allow_html=True)
                        if rels:
                            for r in rels[:4]: 
                                st.markdown(f'<div class="rrow">{r["source"]} →[{r["relation"]}]→ {r["target"]}</div>',unsafe_allow_html=True)
                except Exception as ex:
                    status.write(f"❌ {i+1}: {str(ex)[:80]}")
                    lg.warning("Pipeline","exception",{"chunk_id":cid,"error":str(ex)})

                sr,_=pr.should_run(cs.candidate_pool)
                if sr:
                    status.write("💡 Proposer triggered...")
                    pp=pr.run(cs.candidate_pool,gb.graph,ont,dm,lg)
                    if pp:
                        props=pr.load_proposals(str(pp))
                        for sec in ("entity_proposals","relation_proposals"):
                            for p in props[sec]:
                                if p["status"]=="pending": 
                                    p["status"]="accepted"
                                    p["reviewed_at"]=datetime.now().isoformat()
                        pr._save_proposals(props,str(pp))
                        r2=pr.apply_accepted_proposals(str(pp),ont,cs.candidate_pool,gb,dm,lg)
                        if r2:
                            ne_,cm_=r2
                            ont=OntologyLoader(str(CORE),str(ne_))
                            ea.ontology=ont
                            ra.ontology=ont
                            va.ontology=ont
                            cs.ontology_loader=ont
                            rc.reclassify_graph_nodes(str(ne_),gb,dm,lg)
                            rc.rescue_stranded_relations(gb,cs,ont,lg)
                            gb.recompute_all_salience()
                            S["p"]+=1
                            status.write("✅ Ontology extended")

            prog.progress(1.0)
            status.update(label=f"✅ Done — {S['n']} nodes, {S['ed']} edges",state="complete",expanded=False)

            sj(gb.graph,str(GRAPH_OUT));st.session_state.graph=gb.graph
            import numpy as np
            dms={k:{kk:(vv.tolist() if isinstance(vv,np.ndarray) else vv) for kk,vv in v.items()} for k,v in dm.get("entities",{}).items()}
            sj({"entities":dms},str(DMEM_PATH))
            al.save_index();cs.save_to_disk()

            if ne:
                status.write("🔄 Exporting to Neo4j...")
                try:
                    from graph_neo4j import Neo4jExporter
                    ex=Neo4jExporter()
                    r3=ex.export(gb.graph,incremental=True)
                    st.success(f"Neo4j: {r3['nodes_written']} nodes, {r3['edges_written']} edges")
                except Exception as e: 
                    st.warning(f"Neo4j failed: {e}")

            st.session_state.pstate="complete";st.session_state.stats=S

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