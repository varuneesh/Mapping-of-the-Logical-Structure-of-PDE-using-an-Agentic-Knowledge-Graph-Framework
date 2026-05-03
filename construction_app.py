"""
knowledge_graph_demo.py

Professional demonstration interface for the Multi-Agent Knowledge Graph
Construction Pipeline. Designed for thesis presentation and evaluation demos.

Usage:
    streamlit run knowledge_graph_demo.py

Features:
    - PDF upload → LaTeX conversion → Chunking
    - Live pipeline execution with per-agent visualization
    - Real-time graph growth animation
    - Full knowledge graph explorer (current + historical)
    - Pipeline statistics dashboard
"""

import streamlit as st
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# ── Path setup ────────────────────────────────────────────────────────────
ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KG Construction Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────────────────── */
.stApp {
    font-family: 'Source Sans 3', sans-serif;
}

/* ── Hero banner ────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid #1e293b;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 70%, rgba(59,130,246,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 30%, rgba(139,92,246,0.06) 0%, transparent 50%);
}
.hero h1 {
    font-family: 'Instrument Serif', serif;
    font-size: 2.4rem;
    color: #f1f5f9;
    margin: 0 0 0.5rem 0;
    font-weight: 400;
    position: relative;
}
.hero p {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0;
    position: relative;
}

/* ── Pipeline stage cards ───────────────────────────────── */
.stage-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.stage-card.active {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
}
.stage-card.complete {
    border-left: 4px solid #10b981;
}
.stage-card.error {
    border-left: 4px solid #ef4444;
}
.stage-card .stage-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.5rem;
}
.stage-card .stage-icon {
    font-size: 1.4rem;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f1f5f9;
    border-radius: 8px;
}
.stage-card .stage-name {
    font-weight: 600;
    font-size: 0.95rem;
    color: #1e293b;
}
.stage-card .stage-detail {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #64748b;
    line-height: 1.6;
    padding-left: 2.8rem;
}

/* ── Metrics row ────────────────────────────────────────── */
.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #1e293b;
}
.metric-box .metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Entity/relation chips ──────────────────────────────── */
.chip {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.15rem;
}
.chip-entity { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
.chip-relation { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.chip-new { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
.chip-error { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

/* ── Progress indicator ─────────────────────────────────── */
.progress-bar {
    height: 4px;
    background: #e2e8f0;
    border-radius: 2px;
    margin: 1rem 0;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    border-radius: 2px;
    transition: width 0.5s ease;
}

/* ── Sidebar styling ────────────────────────────────────── */
.doc-list-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
}
.doc-list-item:hover {
    border-color: #3b82f6;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def load_graph(path: str) -> dict:
    """Load graph_memory.json or return empty graph."""
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"nodes": {}, "edges": []}


def render_metric(label: str, value, delta=None):
    """Render a single metric box."""
    delta_html = ""
    if delta is not None:
        color = "#10b981" if delta >= 0 else "#ef4444"
        delta_html = f'<span style="color:{color};font-size:0.7rem;">{"+" if delta>=0 else ""}{delta}</span>'
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{value} {delta_html}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_stage_card(icon, name, status, details="", key=""):
    """Render a pipeline stage card."""
    status_class = {
        "active": "active",
        "complete": "complete",
        "error": "error",
        "waiting": ""
    }.get(status, "")

    status_indicator = {
        "active": "⏳ Running...",
        "complete": "✅ Complete",
        "error": "❌ Error",
        "waiting": "⏸ Waiting"
    }.get(status, "")

    st.markdown(f"""
    <div class="stage-card {status_class}">
        <div class="stage-header">
            <div class="stage-icon">{icon}</div>
            <div class="stage-name">{name}</div>
            <div style="margin-left:auto;font-size:0.75rem;color:#64748b;">{status_indicator}</div>
        </div>
        <div class="stage-detail">{details}</div>
    </div>
    """, unsafe_allow_html=True)


def render_entity_chips(entities: list):
    """Render entity chips."""
    chips = ""
    for e in entities[:12]:
        etype = e.get("type", "?")
        css = "chip-new" if etype == "NEW_TYPE" else "chip-entity"
        chips += f'<span class="chip {css}">{e["name"]} <small>({etype})</small></span> '
    if len(entities) > 12:
        chips += f'<span class="chip" style="background:#f1f5f9;color:#64748b;">+{len(entities)-12} more</span>'
    st.markdown(chips, unsafe_allow_html=True)


def render_relation_chips(relations: list):
    """Render relation chips."""
    chips = ""
    for r in relations[:8]:
        rel_type = r.get("relation", "?")
        css = "chip-new" if rel_type == "NEW_RELATION" else "chip-relation"
        chips += f'<span class="chip {css}">{r["source"]} →[{rel_type}]→ {r["target"]}</span> '
    if len(relations) > 8:
        chips += f'<span class="chip" style="background:#f1f5f9;color:#64748b;">+{len(relations)-8} more</span>'
    st.markdown(chips, unsafe_allow_html=True)


def build_graph_html(graph: dict) -> str:
    """Build an interactive force-directed graph visualization using D3."""
    nodes_data = []
    for name, data in graph.get("nodes", {}).items():
        nodes_data.append({
            "id": name,
            "type": data.get("type", "Unknown"),
            "salience": data.get("salience", 0),
            "sources": len(data.get("sources", [])),
        })

    edges_data = []
    for edge in graph.get("edges", []):
        edges_data.append({
            "source": edge["source"],
            "target": edge["target"],
            "relation": edge["relation"],
        })

    type_colors = {
        "NumericalMethod": "#3b82f6",
        "ProblemType": "#8b5cf6",
        "Theorem": "#ef4444",
        "TheoreticalProperty": "#f59e0b",
        "ErrorConcept": "#ec4899",
        "Definition": "#06b6d4",
        "MathematicalStructure": "#10b981",
        "ComputationalStructure": "#6366f1",
        "FloatingPointConcept": "#f97316",
        "MathematicalObject": "#64748b",
    }

    return f"""
    <div id="graph-container" style="width:100%;height:600px;background:#0f172a;border-radius:12px;position:relative;overflow:hidden;">
        <div id="graph-legend" style="position:absolute;top:12px;left:12px;z-index:10;font-family:'JetBrains Mono',monospace;font-size:0.65rem;">
            {''.join(f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;"><div style="width:10px;height:10px;border-radius:50%;background:{c};"></div><span style="color:#94a3b8;">{t}</span></div>' for t, c in type_colors.items() if any(n['type']==t for n in nodes_data))}
        </div>
        <svg id="graph-svg" width="100%" height="100%"></svg>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script>
    (function() {{
        const nodes = {json.dumps(nodes_data)};
        const links = {json.dumps(edges_data)};
        const colors = {json.dumps(type_colors)};

        const container = document.getElementById('graph-container');
        const svg = d3.select('#graph-svg');
        const width = container.clientWidth;
        const height = 600;

        svg.attr('viewBox', [0, 0, width, height]);

        // Arrow markers
        svg.append('defs').selectAll('marker')
            .data(['arrow'])
            .join('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#475569');

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(120))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));

        const link = svg.append('g')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', '#334155')
            .attr('stroke-width', 1.5)
            .attr('marker-end', 'url(#arrow)');

        const linkLabel = svg.append('g')
            .selectAll('text')
            .data(links)
            .join('text')
            .text(d => d.relation)
            .attr('font-size', '7px')
            .attr('font-family', "'JetBrains Mono', monospace")
            .attr('fill', '#64748b')
            .attr('text-anchor', 'middle');

        const node = svg.append('g')
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', d => Math.max(6, Math.min(16, d.salience * 5)))
            .attr('fill', d => colors[d.type] || '#64748b')
            .attr('stroke', '#1e293b')
            .attr('stroke-width', 1.5)
            .call(d3.drag()
                .on('start', (e, d) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
                .on('drag', (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
                .on('end', (e, d) => {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }})
            );

        const label = svg.append('g')
            .selectAll('text')
            .data(nodes)
            .join('text')
            .text(d => d.id.length > 20 ? d.id.substring(0, 18) + '...' : d.id)
            .attr('font-size', '8px')
            .attr('font-family', "'Source Sans 3', sans-serif")
            .attr('fill', '#cbd5e1')
            .attr('dx', 12)
            .attr('dy', 4);

        // Tooltip
        node.append('title').text(d => d.id + ' (' + d.type + ')\\nSalience: ' + d.salience.toFixed(3) + '\\nSources: ' + d.sources);

        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            linkLabel
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2);
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        }});
    }})();
    </script>
    """


# ═══════════════════════════════════════════════════════════════════════════
#  Session state initialization
# ═══════════════════════════════════════════════════════════════════════════

if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = "idle"   # idle, uploading, chunking, running, complete
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "current_chunk_idx" not in st.session_state:
    st.session_state.current_chunk_idx = 0
if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = {}
if "graph" not in st.session_state:
    st.session_state.graph = load_graph(str(ROOT_DIR / "data" / "graph_memory.json"))
if "processing_log" not in st.session_state:
    st.session_state.processing_log = []
if "stats" not in st.session_state:
    st.session_state.stats = {
        "entities_extracted": 0,
        "relations_extracted": 0,
        "entities_in_graph": 0,
        "edges_in_graph": 0,
        "chunks_processed": 0,
        "ontology_proposals": 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main layout
# ═══════════════════════════════════════════════════════════════════════════

# Hero banner
st.markdown("""
<div class="hero">
    <h1>Multi-Agent Knowledge Graph Construction</h1>
    <p>Upload a mathematical textbook PDF → watch the pipeline extract structured knowledge → explore the resulting graph</p>
</div>
""", unsafe_allow_html=True)

# ── Navigation tabs ───────────────────────────────────────────────────────
tab_upload, tab_pipeline, tab_graph, tab_stats = st.tabs([
    "📄 Upload & Ingest",
    "⚙️ Pipeline Monitor",
    "🕸️ Knowledge Graph",
    "📊 Statistics"
])


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1: Upload & Ingest
# ═══════════════════════════════════════════════════════════════════════════
with tab_upload:
    col_upload, col_preview = st.columns([1, 1.5])

    with col_upload:
        st.markdown("### Upload PDF")
        uploaded_file = st.file_uploader(
            "Drop a mathematical textbook PDF here",
            type=["pdf"],
            help="The PDF will be converted to LaTeX using Mathpix, then chunked by section."
        )

        if uploaded_file:
            st.success(f"📄 {uploaded_file.name} ({uploaded_file.size // 1024} KB)")

            mathpix_key = st.text_input("Mathpix App Key", type="password",
                                        help="Required for PDF → LaTeX conversion")
            mathpix_id  = st.text_input("Mathpix App ID",
                                        help="Your Mathpix application ID")

            st.markdown("---")

            col_a, col_b = st.columns(2)
            with col_a:
                auto_accept = st.checkbox("Auto-accept ontology proposals", value=True,
                                          help="Skip human review for faster processing")
            with col_b:
                run_interchunk = st.checkbox("Inter-chunk extraction", value=True)

            if st.button("🚀 Start Pipeline", type="primary", use_container_width=True):
                st.session_state.pipeline_state = "uploading"
                st.info("Pipeline started! Switch to the Pipeline Monitor tab to watch progress.")

    with col_preview:
        st.markdown("### Pipeline Overview")
        st.markdown("""
        The pipeline processes your PDF through **seven stages**:

        1. **PDF → LaTeX** — Mathpix API preserves mathematical notation
        2. **Chunking** — Section-based splitting with heading metadata
        3. **Coreference Resolution** — Resolves "this method", "it converges"
        4. **Entity Extraction** — Named concepts with ontology types
        5. **Relation Extraction** — Domain-range validated relationships
        6. **Alignment & Consistency** — Deduplication and quality filtering
        7. **Graph Construction** — Incremental insertion with salience scoring

        Plus **ontology evolution**: new concept types are proposed, reviewed,
        and incorporated during ingestion.
        """)

        # Show previously processed documents
        st.markdown("### Previously Processed")
        graph = st.session_state.graph
        if graph["nodes"]:
            st.markdown(f"Current graph: **{len(graph['nodes'])} nodes**, **{len(graph['edges'])} edges**")

            # Extract unique doc_ids from node sources
            doc_ids = set()
            for n, d in graph["nodes"].items():
                for src in d.get("sources", []):
                    doc_id = src.rsplit("_chunk_", 1)[0] if "_chunk_" in src else src
                    doc_ids.add(doc_id)

            for doc_id in sorted(doc_ids):
                node_count = sum(1 for n, d in graph["nodes"].items()
                                 if any(doc_id in s for s in d.get("sources", [])))
                st.markdown(f"""
                <div class="doc-list-item">
                    📘 <strong>{doc_id}</strong><br/>
                    <small style="color:#64748b;">{node_count} entities contributed</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No documents processed yet. Upload a PDF to start.")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2: Pipeline Monitor
# ═══════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    if st.session_state.pipeline_state == "idle":
        st.info("Upload a PDF in the first tab to start the pipeline.")
    else:
        # Progress bar
        total   = max(len(st.session_state.chunks), 1)
        current = st.session_state.current_chunk_idx
        pct     = min(100, int(current / total * 100))

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#64748b;">
                Chunk {current}/{total}
            </span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#64748b;">
                {pct}%
            </span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div>
        """, unsafe_allow_html=True)

        # Metrics row
        s = st.session_state.stats
        mcols = st.columns(6)
        with mcols[0]:
            render_metric("Chunks", s["chunks_processed"])
        with mcols[1]:
            render_metric("Entities", s["entities_extracted"])
        with mcols[2]:
            render_metric("Relations", s["relations_extracted"])
        with mcols[3]:
            render_metric("Graph Nodes", s["entities_in_graph"])
        with mcols[4]:
            render_metric("Graph Edges", s["edges_in_graph"])
        with mcols[5]:
            render_metric("Proposals", s["ontology_proposals"])

        st.markdown("---")

        # Pipeline stages - left: stages, right: current chunk detail
        col_stages, col_detail = st.columns([1, 2])

        with col_stages:
            st.markdown("#### Agent Pipeline")
            outputs = st.session_state.agent_outputs

            stages = [
                ("🏷️", "Chunk Classifier", "classifier"),
                ("🔗", "Coreference Agent", "coreference"),
                ("🧬", "Entity Extraction", "entity"),
                ("🔀", "Relation Extraction", "relation"),
                ("✅", "Validation", "validator"),
                ("🎯", "Alignment", "alignment"),
                ("⚖️", "Consistency", "consistency"),
                ("🕸️", "Graph Builder", "graph_builder"),
                ("💡", "Ontology Proposer", "proposer"),
            ]

            for icon, name, key in stages:
                if key in outputs:
                    data   = outputs[key]
                    status = data.get("status", "complete")
                    detail = data.get("summary", "")
                    render_stage_card(icon, name, status, detail)
                else:
                    render_stage_card(icon, name, "waiting")

        with col_detail:
            st.markdown("#### Current Chunk")

            if "current_chunk" in outputs:
                chunk = outputs["current_chunk"]
                st.markdown(f"**Heading:** {chunk.get('heading', 'N/A')}")
                with st.expander("Chunk content", expanded=False):
                    st.text(chunk.get("content", "")[:1000])

            if "entity" in outputs and outputs["entity"].get("entities"):
                st.markdown("**Extracted Entities:**")
                render_entity_chips(outputs["entity"]["entities"])

            if "relation" in outputs and outputs["relation"].get("relations"):
                st.markdown("**Extracted Relations:**")
                render_relation_chips(outputs["relation"]["relations"])

            if "alignment" in outputs:
                alias_map = outputs["alignment"].get("alias_map", {})
                merges = {k: v for k, v in alias_map.items() if k != v}
                if merges:
                    st.markdown("**Alignment Merges:**")
                    for orig, canon in merges.items():
                        st.markdown(f'<span class="chip chip-entity">{orig} → {canon}</span>',
                                    unsafe_allow_html=True)

            if "proposer" in outputs and outputs["proposer"].get("proposals"):
                st.markdown("**Ontology Proposals:**")
                for prop in outputs["proposer"]["proposals"]:
                    st.markdown(f"""
                    <div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;
                         padding:0.8rem;margin:0.3rem 0;">
                        <strong>{prop.get('proposed_class_name', '?')}</strong>
                        <small> (parent: {prop.get('parent_class', '?')})</small><br/>
                        <small style="color:#64748b;">{prop.get('description', '')}</small>
                    </div>
                    """, unsafe_allow_html=True)

        # Processing log
        with st.expander("📋 Processing Log", expanded=False):
            for entry in st.session_state.processing_log[-50:]:
                ts    = entry.get("time", "")
                agent = entry.get("agent", "")
                msg   = entry.get("message", "")
                st.markdown(f"`{ts}` **{agent}** — {msg}")


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3: Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════
with tab_graph:
    graph = st.session_state.graph

    if not graph["nodes"]:
        st.info("The knowledge graph is empty. Process a document to populate it.")
    else:
        # Graph stats
        st.markdown("### Knowledge Graph Explorer")
        st.markdown(f"**{len(graph['nodes'])} nodes** · **{len(graph['edges'])} edges** · "
                    f"across {len(set(src.rsplit('_chunk_',1)[0] for n,d in graph['nodes'].items() for src in d.get('sources',[])))} document(s)")

        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        all_types = sorted(set(d["type"] for d in graph["nodes"].values()))
        all_rels  = sorted(set(e["relation"] for e in graph["edges"]))

        with col_f1:
            selected_types = st.multiselect("Filter by entity type", all_types, default=all_types)
        with col_f2:
            selected_rels = st.multiselect("Filter by relation", all_rels, default=all_rels)
        with col_f3:
            min_salience = st.slider("Min salience", 0.0, 3.0, 0.0, 0.1)

        # Filter graph
        filtered_nodes = {
            name: data for name, data in graph["nodes"].items()
            if data["type"] in selected_types and data.get("salience", 0) >= min_salience
        }
        filtered_edges = [
            e for e in graph["edges"]
            if e["source"] in filtered_nodes
            and e["target"] in filtered_nodes
            and e["relation"] in selected_rels
        ]

        filtered_graph = {"nodes": filtered_nodes, "edges": filtered_edges}

        # Render interactive graph
        st.components.v1.html(build_graph_html(filtered_graph), height=630, scrolling=False)

        # Node and edge tables
        st.markdown("---")
        col_nodes, col_edges = st.columns(2)

        with col_nodes:
            st.markdown("#### Nodes (sorted by salience)")
            node_data = []
            for name, data in sorted(filtered_nodes.items(),
                                      key=lambda x: x[1].get("salience", 0), reverse=True):
                node_data.append({
                    "Entity": name,
                    "Type": data["type"],
                    "Salience": round(data.get("salience", 0), 3),
                    "Sources": len(data.get("sources", [])),
                })
            if node_data:
                st.dataframe(node_data, use_container_width=True, hide_index=True)

        with col_edges:
            st.markdown("#### Edges")
            edge_data = []
            for e in filtered_edges:
                edge_data.append({
                    "Source": e["source"],
                    "Relation": e["relation"],
                    "Target": e["target"],
                    "Confidence": round(sum(e.get("confidence_scores", [])) /
                                        max(len(e.get("confidence_scores", [])), 1), 3),
                })
            if edge_data:
                st.dataframe(edge_data, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4: Statistics
# ═══════════════════════════════════════════════════════════════════════════
with tab_stats:
    graph = st.session_state.graph

    if not graph["nodes"]:
        st.info("No data yet. Process a document to see statistics.")
    else:
        st.markdown("### Pipeline Statistics")

        # Type distribution
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### Entity Type Distribution")
            type_counts = Counter(d["type"] for d in graph["nodes"].values())
            for t, c in type_counts.most_common():
                pct = int(c / len(graph["nodes"]) * 100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
                    <div style="width:120px;font-size:0.8rem;color:#475569;">{t}</div>
                    <div style="flex:1;height:20px;background:#f1f5f9;border-radius:4px;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:#3b82f6;border-radius:4px;"></div>
                    </div>
                    <div style="width:40px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.75rem;">{c}</div>
                </div>
                """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("#### Relation Type Distribution")
            rel_counts = Counter(e["relation"] for e in graph["edges"])
            for t, c in rel_counts.most_common():
                pct = int(c / len(graph["edges"]) * 100)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
                    <div style="width:120px;font-size:0.8rem;color:#475569;">{t}</div>
                    <div style="flex:1;height:20px;background:#f1f5f9;border-radius:4px;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:#10b981;border-radius:4px;"></div>
                    </div>
                    <div style="width:40px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:0.75rem;">{c}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Type conflicts
        conflicts = [(n, d) for n, d in graph["nodes"].items() if d.get("type_conflict")]
        if conflicts:
            st.markdown("#### Type Conflicts")
            for name, data in conflicts:
                st.warning(f"**{name}**: observed types = {data.get('observed_types', [])}")

        # Graph health metrics
        st.markdown("#### Graph Health")
        total_nodes = len(graph["nodes"])
        total_edges = len(graph["edges"])
        avg_degree  = total_edges * 2 / max(total_nodes, 1)
        isolated    = sum(1 for n in graph["nodes"]
                         if not any(e["source"] == n or e["target"] == n for e in graph["edges"]))

        hcols = st.columns(4)
        with hcols[0]:
            render_metric("Avg Degree", f"{avg_degree:.1f}")
        with hcols[1]:
            render_metric("Isolated Nodes", isolated)
        with hcols[2]:
            render_metric("Type Conflicts", len(conflicts))
        with hcols[3]:
            ic_edges = sum(1 for e in graph["edges"] if e.get("inter_chunk"))
            render_metric("Inter-Chunk Edges", ic_edges)