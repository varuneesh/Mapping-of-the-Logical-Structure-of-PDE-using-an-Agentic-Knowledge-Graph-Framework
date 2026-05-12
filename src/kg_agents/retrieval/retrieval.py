"""
retrieval.py — Neo4j-based Graph-Guided Retrieval Pipeline.

Query → LLM generates Cypher → Neo4j executes → Chunk IDs collected
→ Chunks retrieved → LLM generates answer with LaTeX preservation.
"""
import json
import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class RetrievalResult:
    query:              str
    answer:             str
    cypher_query:       str = ""
    identified_nodes:   List[Dict] = field(default_factory=list)
    identified_edges:   List[Dict] = field(default_factory=list)
    retrieved_chunks:   List[Dict] = field(default_factory=list)
    chunk_ids:          List[str]  = field(default_factory=list)
    confidence:         float      = 0.0
    retrieval_mode:     str        = "graph"
    reasoning_trace:    List[str]  = field(default_factory=list)

class GraphGuidedRetriever:
    def __init__(self, chunks_dir: str, model: str = "gpt-4o-mini",
                 max_chunks: int = 12, embed_model: str = "text-embedding-3-small"):
        
        self.neo4j = Neo4jGraph(url=os.getenv("NEO4J_URI"),
                                username=os.getenv("NEO4J_USERNAME"),
                                password=os.getenv("NEO4J_PASSWORD"))
        self.chunks_dir = Path(chunks_dir)
        self.max_chunks = max_chunks
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.embed_model = embed_model
        self.chunk_lookup: Dict[str, Dict] = {}
        self._load_chunks()
        self._chunk_emb_cache: Optional[Dict[str, np.ndarray]] = None
        self._schema = self._get_graph_schema()
        print(f"[Retriever] Neo4j connected. {len(self.chunk_lookup)} chunks loaded.")

    def _load_chunks(self):
        for f in self.chunks_dir.glob("*.json"):
            try:
                for c in json.load(open(f)):
                    self.chunk_lookup[c["chunk_id"]] = c
            except Exception as e:
                print(f"[Retriever] Failed to load {f}: {e}")

    def _get_graph_schema(self) -> str:
        try:
            labels = self.neo4j.query("CALL db.labels() YIELD label RETURN collect(label) AS l")[0]["l"]
            rels = self.neo4j.query("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS r")[0]["r"]
            props = self.neo4j.query("MATCH (n:Entity) WITH keys(n) AS p LIMIT 1 RETURN p")
            node_props = props[0]["p"] if props else []
            return f"""NODE LABELS: {', '.join(labels)}
                        RELATIONSHIP TYPES: {', '.join(rels)}
                        NODE PROPERTIES: {', '.join(node_props)}
                        KEY RULES:
                        - All entities have :Entity label plus type labels
                        - name property stores entity name
                        - observed_types is a list — use ANY() for type filtering:
                            WHERE ANY(t IN n.observed_types WHERE t = 'NumericalMethod')
                        - sources is a list of chunk IDs
                        - salience is a float importance score
                        - Relationship properties: confidence_scores, sources, inter_chunk, relation_name"""
        except Exception:
            return "Entity nodes with name, type, salience, sources properties."

    def query(self, user_query: str, status_callback=None) -> RetrievalResult:
        trace = []
        def _status(s, m):
            if status_callback:
                status_callback(s, m)

        _status("cypher_generation", "Generating graph query from your question...")
        trace.append("Stage 1: Generating Cypher query...")
        cypher, explanation = self._generate_cypher(user_query)
        trace.append(f"  Cypher: {cypher}")
        trace.append(f"  Explanation: {explanation}")

        _status("graph_query", "Querying knowledge graph via Neo4j...")
        trace.append("Stage 2: Executing Cypher...")
        nodes, edges = self._execute_cypher(cypher, trace)
        trace.append(f"  Retrieved {len(nodes)} nodes, {len(edges)} edges")

        if not nodes:
            _status("fallback_search", "Graph query empty — trying text search...")
            trace.append("  Trying text search fallback...")
            nodes, edges = self._text_search_fallback(user_query)
            trace.append(f"  Text search: {len(nodes)} nodes")

        _status("chunk_collection", f"Collecting passages from {len(nodes)} nodes...")
        trace.append("Stage 3: Collecting chunks...")
        graph_chunks = self._collect_chunks_from_nodes(nodes)
        trace.append(f"  Graph-guided: {len(graph_chunks)} chunks")

        emb_chunks = []
        if len(graph_chunks) < 8:
            _status("embedding_supplement", "Supplementing with embedding search...")
            emb_chunks = self._embedding_fallback(user_query, top_k=5)
            trace.append(f"  Embedding supplement: {len(emb_chunks)} chunks")

        all_chunks = self._merge_chunks(graph_chunks, emb_chunks)
        trace.append(f"  Total: {len(all_chunks)} chunks")

        mode = "embedding_only" if (not nodes and not graph_chunks) else ("hybrid" if emb_chunks else "graph")

        _status("answer_generation", f"Generating answer from {len(all_chunks)} passages...")
        trace.append("Stage 4: Generating answer...")
        answer = self._generate_answer(user_query, all_chunks, nodes, edges, trace)
        confidence = min(1.0, len(nodes) * 0.2 + len(graph_chunks) * 0.05)

        return RetrievalResult(query=user_query, answer=answer, cypher_query=cypher,
            identified_nodes=nodes, identified_edges=edges, retrieved_chunks=all_chunks,
            chunk_ids=[c["chunk_id"] for c in all_chunks], confidence=round(confidence, 3),
            retrieval_mode=mode, reasoning_trace=trace)

    def _generate_cypher(self, user_query: str) -> tuple:
        prompt = f"""You are a Cypher query generator for a mathematical knowledge graph.

SCHEMA:
{self._schema}

QUESTION: {user_query}

Generate a Cypher query to find relevant entities and relationships.
Rules:
- Use WHERE toLower(n.name) CONTAINS toLower('keyword') for name matching
- Use ANY(t IN n.observed_types WHERE t = 'Type') for type filtering
- Always RETURN n and optionally relationships
- LIMIT 20
- Use OPTIONAL MATCH for relationships

Return ONLY JSON: {{"cypher": "...", "explanation": "..."}}
No markdown fences."""

        try:
            resp = self.llm.invoke(prompt)
            c = re.sub(r"^```json\s*", "", resp.content.strip())
            c = re.sub(r"\s*```$", "", c)
            data = json.loads(c)
            return data.get("cypher", ""), data.get("explanation", "")
        except Exception:
            kws = [w for w in user_query.lower().split() if len(w) > 3][:3]
            conds = " OR ".join(f"toLower(n.name) CONTAINS '{k}'" for k in kws)
            return (f"MATCH (n:Entity) WHERE {conds} OPTIONAL MATCH (n)-[r]->(m:Entity) RETURN n, r, m LIMIT 20",
                    "Fallback keyword search")

    def _execute_cypher(self, cypher: str, trace: List[str]) -> tuple:
        for attempt in range(2):
            try:
                results = self.neo4j.query(cypher)
                nodes, edges, seen_n, seen_e = [], [], set(), set()
                for row in results:
                    for key, val in row.items():
                        if isinstance(val, dict) and "name" in val and val["name"] not in seen_n:
                            seen_n.add(val["name"])
                            nodes.append({"name": val.get("name","?"), "primary_type": val.get("primary_type","?"),
                                "salience": val.get("salience",0), "sources": val.get("sources",[]),
                                "observed_types": val.get("observed_types",[])})
                        elif isinstance(val, dict) and "relation_name" in val:
                            ek = f"{val.get('source','')}-{val.get('relation_name','')}-{val.get('target','')}"
                            if ek not in seen_e:
                                seen_e.add(ek)
                                edges.append(val)
                return nodes, edges
            except Exception as e:
                if attempt == 0:
                    trace.append(f"  Cypher failed: {e}. Retrying...")
                    cypher = self._fix_cypher(cypher, str(e))
                else:
                    trace.append(f"  Retry failed: {e}")
                    return [], []
        return [], []

    def _fix_cypher(self, cypher: str, error: str) -> str:
        try:
            resp = self.llm.invoke(f"Fix this Cypher query:\n{cypher}\nError: {error}\nReturn ONLY corrected Cypher.")
            fixed = re.sub(r"^```(?:cypher)?\s*", "", resp.content.strip())
            return re.sub(r"\s*```$", "", fixed)
        except Exception:
            return cypher

    def _text_search_fallback(self, user_query: str) -> tuple:
        kws = [w for w in user_query.lower().split() if len(w) > 3]
        if not kws: 
            return [], []
        conds = " OR ".join(f"toLower(n.name) CONTAINS '{k}'" for k in kws[:5])
        try:
            results = self.neo4j.query(
                f"MATCH (n:Entity) WHERE {conds} OPTIONAL MATCH (n)-[r]->(m:Entity) RETURN n, r, m LIMIT 20")
            nodes, seen = [], set()
            for row in results:
                for k, v in row.items():
                    if isinstance(v, dict) and "name" in v and v["name"] not in seen:
                        seen.add(v["name"])
                        nodes.append({"name": v.get("name","?"), "primary_type": v.get("primary_type","?"),
                            "salience": v.get("salience",0), "sources": v.get("sources",[]),
                            "observed_types": v.get("observed_types",[])})
            return nodes, []
        except Exception:
            return [], []

    def _collect_chunks_from_nodes(self, nodes: List[Dict]) -> List[Dict]:
        chunk_scores: Dict[str, float] = {}
        for n in nodes:
            sal = n.get("salience", 0.5)
            for cid in n.get("sources", []):
                chunk_scores[cid] = chunk_scores.get(cid, 0) + sal
        ranked = sorted(chunk_scores.items(), key=lambda x: -x[1])[:self.max_chunks]
        return [{"chunk_id": cid, "heading": self.chunk_lookup.get(cid, {}).get("heading", ""),
                 "content": self.chunk_lookup.get(cid, {}).get("content", ""), "score": sc}
                for cid, sc in ranked if cid in self.chunk_lookup]

    def _embedding_fallback(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            q_emb = np.array(openai_client.embeddings.create(model=self.embed_model, input=query).data[0].embedding)
            if self._chunk_emb_cache is None:
                self._chunk_emb_cache = {}
                items = [(cid, c) for cid, c in self.chunk_lookup.items()
                         if len(f"{c.get('heading','')} {c.get('content','')}".strip()) > 50]
                for i in range(0, len(items), 100):
                    batch = items[i:i+100]
                    texts = [f"{c.get('heading','')} {c.get('content','')}"[:500] for _, c in batch]
                    resp = openai_client.embeddings.create(model=self.embed_model, input=texts)
                    for j, (cid, _) in enumerate(batch):
                        self._chunk_emb_cache[cid] = np.array(resp.data[j].embedding)
            results = sorted([(cid, float(np.dot(q_emb, emb))) for cid, emb in self._chunk_emb_cache.items()],
                             key=lambda x: -x[1])
            return [{"chunk_id": cid, "heading": self.chunk_lookup.get(cid,{}).get("heading",""),
                     "content": self.chunk_lookup.get(cid,{}).get("content",""), "score": sim}
                    for cid, sim in results[:top_k] if cid in self.chunk_lookup]
        except Exception as e:
            print(f"[Retriever] Embedding fallback failed: {e}")
            return []

    def _merge_chunks(self, g: List[Dict], e: List[Dict]) -> List[Dict]:
        seen = {c["chunk_id"] for c in g}
        merged = list(g)
        for c in e:
            if c["chunk_id"] not in seen and len(merged) < self.max_chunks:
                merged.append(c)
                seen.add(c["chunk_id"])
        return merged

    def _generate_answer(self, query, chunks, nodes, edges, trace) -> str:
        if not chunks:
            return "I could not find relevant information to answer this question."
        ctx = "\n\n---\n\n".join(f"[Passage {i+1}] Section: {c.get('heading','')}\n{c.get('content','')}"
                                  for i, c in enumerate(chunks))
        gi = ""
        if nodes:
            gi = "\nRELEVANT ENTITIES:\n" + "\n".join(
                f"  - {n['name']} ({n.get('primary_type','?')}, salience={n.get('salience',0):.2f})"
                for n in nodes[:15]) + "\n"
        prompt = f"""You are a mathematical tutor answering from textbook passages ONLY.

QUESTION: {query}
{gi}
PASSAGES:
{ctx}

INSTRUCTIONS:
1. Answer from passages ONLY. No external knowledge.
2. PRESERVE LaTeX: use $...$ inline and $$...$$ display.
3. If passages lack information, say what is known and what is missing.
4. List passage numbers used at the end.

ANSWER:"""
        try:
            answer = self.llm.invoke(prompt).content.strip()
            trace.append(f"  Answer: {len(answer)} chars")
            return answer
        except Exception as e:
            trace.append(f"  Failed: {e}")
            return f"Error: {e}"

    def get_graph_stats(self) -> Dict:
        try:
            r = self.neo4j.query("MATCH (n:Entity) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS edges")
            return r[0] if r else {"nodes": 0, "edges": 0}
        except Exception:
            return {"nodes": 0, "edges": 0}