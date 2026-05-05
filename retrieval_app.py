"""
retrieval_app.py — Professional query interface using Neo4j-backed retrieval.
Usage: streamlit run retrieval_app.py
"""
import streamlit as st
# import json
import time
# import os
import sys
from pathlib import Path
# from collections import Counter

ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from kg_agents.retrieval.retrieval import GraphGuidedRetriever  # noqa: E402

st.set_page_config(page_title="Graph-Guided Retrieval", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600;700&display=swap');
:root { --ink:#1a1a2e; --paper:#faf9f6; --accent:#2563eb; --accent-soft:#dbeafe;
         --sage:#166534; --sage-soft:#dcfce7; --warm:#92400e; --warm-soft:#fef3c7;
         --rose:#9f1239; --rose-soft:#ffe4e6; --border:#e5e1d8; --muted:#6b7280;
         --surface:#ffffff; }
.stApp { background-color: var(--paper); font-family: 'Outfit', sans-serif; }
.app-header { padding: 2rem 0 1.5rem; border-bottom: 2px solid var(--ink); margin-bottom: 2rem; }
.app-header h1 { font-family: 'Newsreader', serif; font-size: 2.6rem; font-weight: 400;
    color: var(--ink); margin: 0; letter-spacing: -0.02em; }
.app-header .subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.4rem; }
.answer-card { background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--accent); border-radius: 8px; padding: 1.8rem 2rem;
    margin: 1.5rem 0; font-family: 'Newsreader', serif; font-size: 1.05rem;
    line-height: 1.8; color: var(--ink); }
.cypher-box { background: #1e293b; color: #e2e8f0; border-radius: 8px; padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; line-height: 1.6;
    margin: 0.8rem 0; overflow-x: auto; }
.provenance-section { background: #f8f7f4; border: 1px solid var(--border); border-radius: 8px;
    padding: 1.2rem 1.5rem; margin: 0.8rem 0; }
.provenance-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.6rem; }
.entity-tag { display: inline-block; padding: 0.25rem 0.7rem; border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; margin: 0.2rem; font-weight: 500; }
.tag-method { background: var(--accent-soft); color: var(--accent); }
.tag-theorem { background: var(--rose-soft); color: var(--rose); }
.tag-property { background: var(--warm-soft); color: var(--warm); }
.tag-error { background: #fce7f3; color: #9d174d; }
.tag-structure { background: var(--sage-soft); color: var(--sage); }
.tag-default { background: #f1f5f9; color: #475569; }
.metric-pill { display: inline-flex; align-items: center; gap: 0.4rem; background: #f1f5f9;
    border: 1px solid var(--border); border-radius: 20px; padding: 0.3rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; margin: 0.2rem; }
.metric-pill .metric-num { font-weight: 600; color: var(--accent); }
.trace-step { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--muted);
    padding: 0.3rem 0; border-left: 2px solid var(--border); padding-left: 1rem; margin: 0.2rem 0; }
.chunk-heading { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--accent); }
.chunk-content { font-family: 'Newsreader', serif; font-size: 0.88rem; line-height: 1.7; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
""", unsafe_allow_html=True)

TYPE_TAG = {"NumericalMethod":"tag-method","Theorem":"tag-theorem",
    "TheoreticalProperty":"tag-property","ErrorConcept":"tag-error",
    "MathematicalStructure":"tag-structure","ComputationalStructure":"tag-structure",
    "FloatingPointConcept":"tag-property"}

def render_tag(name, etype):
    return f'<span class="entity-tag {TYPE_TAG.get(etype,"tag-default")}">{name}</span>'

def render_latex(text, css_class="answer-card"):
    import html as h
    safe = h.escape(text).replace("&#x27;","'").replace("&amp;","&").replace("\n","<br/>")
    uid = f"ltx_{hash(text)%99999}"
    return f"""<div class="{css_class}" id="{uid}">{safe}</div>
    <script>setTimeout(function(){{var e=document.getElementById('{uid}');
    if(e&&typeof renderMathInElement!=='undefined'){{renderMathInElement(e,{{
    delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}},
    {{left:'\\\\(',right:'\\\\)',display:false}},{{left:'\\\\[',right:'\\\\]',display:true}}],
    throwOnError:false}});}}}},500);</script>"""

CHUNKS_DIR = ROOT_DIR / "data" / "chunks"

@st.cache_resource
def init_retriever():
    return GraphGuidedRetriever(chunks_dir=str(CHUNKS_DIR))

retriever = init_retriever()

if "query_history" not in st.session_state: 
    st.session_state.query_history = []
if "current_result" not in st.session_state: 
    st.session_state.current_result = None

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Knowledge Base")
    stats = retriever.get_graph_stats()
    st.markdown(f"**{stats.get('nodes',0)}** entities · **{stats.get('edges',0)}** relationships")
    st.markdown("---")
    st.markdown("### Example Queries")
    examples = [
        "What is the bisection method and how does it work?",
        "Compare Newton's method with the secant method.",
        "What properties does Gaussian elimination have?",
        "How is LU decomposition related to Gaussian elimination?",
        "What is the condition number and why does it matter?",
        "Explain roundoff error in floating point arithmetic.",
        "What methods solve nonlinear equations?",
        "How does the Taylor series relate to numerical approximation?",
        "What is the role of pivoting in numerical linear algebra?",
        "What are iterative methods for linear systems?",
    ]
    for q in examples:
        if st.button(q, key=f"eq_{hash(q)}", use_container_width=True):
            st.session_state.pending_query = q
    st.markdown("---")
    if st.session_state.query_history:
        st.markdown("### History")
        for i, h in enumerate(reversed(st.session_state.query_history[-10:])):
            if st.button(f"↩ {h[:45]}...", key=f"h_{i}", use_container_width=True):
                st.session_state.pending_query = h

# ── Main ─────────────────────────────────────────────────────────────────
st.markdown("""<div class="app-header"><h1>Graph-Guided Retrieval</h1>
<div class="subtitle">Ask questions about numerical methods · answers grounded in textbook content via Neo4j</div></div>""",
            unsafe_allow_html=True)

pending = st.session_state.pop("pending_query", None)
query_input = st.text_input("Ask a question", value=pending or "",
    placeholder="e.g., How does Newton's method achieve quadratic convergence?",
    label_visibility="collapsed")

c1, c2 = st.columns([1, 5])
with c1: 
    search = st.button("🔍 Search", type="primary", use_container_width=True)
with c2:
    if st.button("Clear"):
        st.session_state.current_result = None 
        st.rerun()

# ── Execute ──────────────────────────────────────────────────────────────
if search and query_input.strip():
    t0 = time.time()
    icons = {"cypher_generation":"🧠","graph_query":"🕸️","chunk_collection":"📄",
             "embedding_supplement":"🔗","fallback_search":"⚠️","answer_generation":"✍️"}
    status = st.status("🔍 Searching knowledge graph...", expanded=True)
    def cb(stage, msg): status.write(f"{icons.get(stage,'⏳')} {msg}")

    result = retriever.query(query_input.strip(), status_callback=cb)
    elapsed = time.time() - t0
    status.update(label=f"✅ Done in {elapsed:.1f}s — {len(result.retrieved_chunks)} passages",
                  state="complete", expanded=False)
    st.session_state.current_result = result
    st.session_state.query_elapsed = elapsed
    if query_input not in st.session_state.query_history:
        st.session_state.query_history.append(query_input)

# ── Results ──────────────────────────────────────────────────────────────
result = st.session_state.current_result
if result:
    elapsed = st.session_state.get("query_elapsed", 0)

    # Metrics
    st.markdown(f"""<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin:1rem 0;">
        <span class="metric-pill"><span class="metric-num">{len(result.identified_nodes)}</span> nodes matched</span>
        <span class="metric-pill"><span class="metric-num">{len(result.retrieved_chunks)}</span> passages</span>
        <span class="metric-pill"><span class="metric-num">{result.confidence:.0%}</span> confidence</span>
        <span class="metric-pill"><span class="metric-num">{elapsed:.1f}s</span> elapsed</span>
    </div>""", unsafe_allow_html=True)

    # Retrieval mode banner
    mode = result.retrieval_mode
    if mode == "graph":
        st.markdown('<div style="background:#dcfce7;border:1px solid #86efac;border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:#166534;margin:0.5rem 0;"><strong>Graph-Guided Retrieval</strong> — answer grounded in knowledge graph traversal via Cypher.</div>', unsafe_allow_html=True)
    elif mode == "hybrid":
        st.markdown('<div style="background:#dbeafe;border:1px solid #93c5fd;border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:#1e40af;margin:0.5rem 0;"><strong>Hybrid Retrieval</strong> — graph traversal + embedding supplement.</div>', unsafe_allow_html=True)
    elif mode == "embedding_only":
        st.markdown('<div style="background:#fef3c7;border:1px solid #fcd34d;border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:#92400e;margin:0.5rem 0;"><strong>⚠ Embedding-Only Retrieval</strong> — the knowledge graph did not contain entities relevant to this query. Falling back to standard vector similarity search. This may indicate that the relevant concepts were not captured during graph construction, or that the query involves topics outside the ingested material.</div>', unsafe_allow_html=True)

    # Answer with LaTeX
    st.components.v1.html(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"/>'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>'
        + render_latex(result.answer),
        height=max(300, len(result.answer) // 4), scrolling=True)

    # Two columns: left = provenance, right = cypher
    cl, cr = st.columns([1.2, 1])

    with cl:
        if result.identified_nodes:
            st.markdown('<div class="provenance-section"><div class="provenance-label">Matched Graph Entities</div>', unsafe_allow_html=True)
            st.markdown(" ".join(render_tag(n["name"], n.get("primary_type","?")) for n in result.identified_nodes) + "</div>", unsafe_allow_html=True)

    with cr:
        if result.cypher_query:
            st.markdown('<div class="provenance-label">Generated Cypher Query</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cypher-box">{result.cypher_query}</div>', unsafe_allow_html=True)

    # Source passages
    st.markdown("---")
    st.markdown('<div class="provenance-label">Source Passages</div>', unsafe_allow_html=True)
    for i, chunk in enumerate(result.retrieved_chunks):
        heading = chunk.get("heading", "Untitled")
        content = chunk.get("content", "")
        score = chunk.get("score", 0)
        cid = chunk.get("chunk_id", "")
        display_id = cid.split("_chunk_")[-1] if "_chunk_" in cid else cid

        with st.expander(f"📄 Passage {i+1} — {heading} (chunk {display_id})", expanded=(i < 2)):
            st.components.v1.html(
                '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css"/>'
                '<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>'
                '<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>'
                + render_latex(f"Section: {heading} · relevance: {score:.3f}\n\n{content}", "chunk-content"),
                height=max(200, len(content) // 5), scrolling=True)

    # Reasoning trace
    with st.expander("🔬 Reasoning Trace", expanded=False):
        for step in result.reasoning_trace:
            st.markdown(f'<div class="trace-step">{step}</div>', unsafe_allow_html=True)

elif not search:
    st.markdown("""<div style="text-align:center;padding:4rem 2rem;color:#94a3b8;">
        <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
        <div style="font-family:'Newsreader',serif;font-size:1.4rem;color:#6b7280;">
            Ask a question about numerical methods</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;margin-top:0.8rem;">
            Queries are translated to Cypher · Neo4j finds relevant entities · source passages generate the answer</div>
    </div>""", unsafe_allow_html=True)