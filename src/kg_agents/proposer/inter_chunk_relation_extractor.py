"""
inter_chunk_relation_extractor.py

Second-pass relation extraction that runs after a full document is ingested.

Problem it solves:
    Chunk-level extraction only sees a 4000-character window. Mathematical
    arguments often span chapters — a method introduced in chunk 5 may have
    its stability proven in chunk 47. These cross-chunk relations are invisible
    to the per-chunk pipeline.

How it works:
    1. Identify high-salience node pairs that:
         - both have salience >= MIN_SALIENCE
         - share at least MIN_COOCCURRENCE source chunks
         - have NO existing direct edge between them
    2. For each candidate pair, collect the chunks where both appear.
    3. Run a targeted LLM call: "given these two entities and these passages,
       does a mathematical relation exist between them?"
    4. If yes, insert the new edge into the graph with provenance.

Filters to keep LLM calls manageable:
    - Only pairs where both nodes exceed salience threshold
    - Only pairs with >= MIN_COOCCURRENCE shared source chunks
    - Cap total LLM calls per run at MAX_PAIRS
    - Skip pairs already connected by any edge (direct or via one hop
      if SKIP_INDIRECT is True)
"""

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional#, Any
from langchain_openai import ChatOpenAI


# ── Thresholds (all tunable) ─────────────────────────────────────────────────
MIN_SALIENCE      = 1.0    # both nodes must exceed this salience
MIN_COOCCURRENCE  = 2      # shared source chunks required
MAX_PAIRS         = 50     # hard cap on LLM calls per run
MIN_CONFIDENCE    = 0.6    # minimum confidence to insert an extracted relation
SKIP_INDIRECT     = False  # if True, also skip pairs connected via one hop


# ── Prompt ───────────────────────────────────────────────────────────────────

_RELATION_PROMPT = """\
You are a mathematical knowledge extraction specialist.

Two mathematical entities have been identified as co-occurring across multiple
sections of a numerical methods textbook. Your task is to determine whether a
meaningful mathematical relationship exists between them based on the provided
text passages.

ENTITY A: <<ENTITY_A>> (type: <<TYPE_A>>)
ENTITY B: <<ENTITY_B>> (type: <<TYPE_B>>)

ONTOLOGY RELATIONS AVAILABLE:
<<ONTOLOGY_RELATIONS>>

TEXT PASSAGES (chunks where both entities appear):
<<PASSAGES>>

TASK
----
1. Read the passages carefully.
2. Determine whether a direct mathematical relationship between ENTITY A and
   ENTITY B is supported by the text.
3. If yes, choose the best matching relation from ONTOLOGY RELATIONS.
   If no relation fits, use NEW_RELATION.
4. If no meaningful relation is supported by the text, set relation to null.

RULES
-----
- Only extract relations explicitly or strongly implied by the passages.
- Do NOT hallucinate relations not supported by the text.
- Do NOT force a relation just because the entities co-occur.
- Source and target can be in either direction — choose whichever the
  text supports.
- Confidence: 0.9 explicitly stated, 0.75 strongly implied, 0.5 moderately
  implied. Do not return confidence below 0.5.

OUTPUT — valid JSON only, no markdown fences:
{
  "relation_found": true,
  "source": "<<ENTITY_A>> or <<ENTITY_B>>",
  "relation": "relation_name or NEW_RELATION or null",
  "target": "<<ENTITY_B>> or <<ENTITY_A>>",
  "confidence": 0.0,
  "justification": "one sentence explaining the relation"
}
"""


class InterChunkRelationExtractor:
    """
    Extracts relations between high-salience entities that span multiple chunks.

    Parameters
    ----------
    chunks_path : str | Path
        Path to the document's chunks JSON file — used to retrieve passage text.
    model : str
        Groq model for relation extraction.
    """

    def __init__(
        self,
        chunks_path: str,
        model: str = "gpt-4o-mini",
    ):
        self.chunks_path = Path(chunks_path)
        self.llm         = ChatOpenAI(model=model, temperature=0)
        self._chunk_map: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    #  Main entry                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        graph_memory:    Dict,
        ontology_loader,
        logger=None,
        doc_id: str = "",
    ) -> Dict:
        """
        Run the full inter-chunk relation extraction pass.

        Parameters
        ----------
        graph_memory : dict
            Live graph memory from GraphBuilderAgent.
        ontology_loader : OntologyLoader
            Current ontology (post-reclassification).
        logger : PipelineLogger | None
        doc_id : str
            Document identifier for logging.

        Returns
        -------
        report dict with keys: pairs_evaluated, relations_inserted, skipped
        """
        agent = "InterChunkRelationExtractor"

        print(f"\n[InterChunkRelation] Starting second-pass relation extraction"
              f"{f' for {doc_id}' if doc_id else ''}...")

        # Load chunk texts into memory
        self._load_chunks()

        if logger:
            logger.info(agent, "started", {
                "doc_id":      doc_id,
                "total_nodes": len(graph_memory["nodes"]),
                "total_edges": len(graph_memory["edges"]),
            })

        # ── Step 1: identify candidate pairs ─────────────────────────────
        pairs = self._find_candidate_pairs(graph_memory)

        print(f"[InterChunkRelation] {len(pairs)} candidate pair(s) identified "
              f"(cap: {MAX_PAIRS}).")

        if not pairs:
            print("[InterChunkRelation] No candidate pairs — skipping.")
            return {"pairs_evaluated": 0, "relations_inserted": 0, "skipped": 0}

        # ── Step 2: extract relations via LLM ────────────────────────────
        inserted = 0
        skipped  = 0
        ontology_relations = ontology_loader.get_all_relations()

        for name_a, name_b, shared_chunks in pairs:

            result = self._extract_relation(
                name_a, name_b,
                shared_chunks,
                graph_memory,
                ontology_relations,
            )

            if result is None:
                skipped += 1
                continue

            if not result.get("relation_found") or not result.get("relation"):
                skipped += 1
                continue

            if result.get("relation") == "null" or result.get("confidence", 0) < MIN_CONFIDENCE:
                skipped += 1
                continue

            # Validate source/target are actual node names
            source = result.get("source", "").strip()
            target = result.get("target", "").strip()

            if source not in graph_memory["nodes"] or \
               target not in graph_memory["nodes"]:
                skipped += 1
                continue

            if source == target:
                skipped += 1
                continue

            # Domain-range validation — skip if relation type violates ontology
            # ── Relation validation ───────────────────────────────
            relation_type = result["relation"]

            # Reject NEW_RELATION directly
            if relation_type == "NEW_RELATION":
                skipped += 1
                continue

            # Reject unknown relations
            if not ontology_loader.relation_exists(relation_type):
                skipped += 1
                continue

            # Validate domain/range
            source_type = graph_memory["nodes"].get(source, {}).get("type")
            target_type = graph_memory["nodes"].get(target, {}).get("type")

            if source_type and target_type:
                if not ontology_loader.validate_domain_range(
                    relation_type,
                    source_type,
                    target_type,
                ):
                    skipped += 1
                    continue

            # Insert edge into graph
            self._insert_edge(
                graph_memory  = graph_memory,
                source        = source,
                relation      = result["relation"],
                target        = target,
                confidence    = result["confidence"],
                shared_chunks = shared_chunks,
                justification = result.get("justification", ""),
            )

            inserted += 1
            print(f"  ✓ {source} →[{result['relation']}]→ {target} "
                  f"(conf: {result['confidence']}, "
                  f"shared chunks: {len(shared_chunks)})")

            if logger:
                logger.info(agent, "relation_inserted", {
                    "source":    source,
                    "relation":  result["relation"],
                    "target":    target,
                    "confidence": result["confidence"],
                    "shared_chunks": len(shared_chunks),
                })

        report = {
            "pairs_evaluated":    len(pairs),
            "relations_inserted": inserted,
            "skipped":            skipped,
        }

        print(f"\n[InterChunkRelation] Complete — "
              f"{inserted} relation(s) inserted, "
              f"{skipped} pair(s) skipped.")

        if logger:
            logger.info(agent, "completed", report)

        return report

    # ------------------------------------------------------------------ #
    #  Candidate pair identification                                       #
    # ------------------------------------------------------------------ #

    def _find_candidate_pairs(
        self, graph_memory: Dict
    ) -> List[Tuple[str, str, List[str]]]:
        """
        Find pairs of high-salience nodes with shared source chunks
        and no existing direct edge.

        Returns list of (name_a, name_b, shared_chunk_ids), capped at MAX_PAIRS.
        """
        nodes = graph_memory["nodes"]
        edges = graph_memory["edges"]

        # Filter to high-salience nodes only
        salient_nodes = {
            name: data for name, data in nodes.items()
            if data.get("salience", 0.0) >= MIN_SALIENCE
        }

        # Build existing edge set for fast lookup (both directions)
        existing_edges: set = set()
        for edge in edges:
            existing_edges.add((edge["source"], edge["target"]))
            existing_edges.add((edge["target"], edge["source"]))

        # Optionally build one-hop neighbours for indirect skip
        one_hop: Dict[str, set] = {}
        if SKIP_INDIRECT:
            for edge in edges:
                one_hop.setdefault(edge["source"], set()).add(edge["target"])
                one_hop.setdefault(edge["target"], set()).add(edge["source"])

        candidates = []

        for name_a, name_b in combinations(salient_nodes.keys(), 2):

            # Skip if already directly connected
            if (name_a, name_b) in existing_edges:
                continue

            # Skip if indirectly connected (one hop) — optional
            if SKIP_INDIRECT:
                neighbours_a = one_hop.get(name_a, set())
                neighbours_b = one_hop.get(name_b, set())
                if neighbours_a & neighbours_b:
                    continue

            # Find shared source chunks
            sources_a = set(salient_nodes[name_a].get("sources", []))
            sources_b = set(salient_nodes[name_b].get("sources", []))
            shared    = list(sources_a & sources_b)

            if len(shared) < MIN_COOCCURRENCE:
                continue

            candidates.append((name_a, name_b, shared))

        # Sort by number of shared chunks descending — most evidence first
        candidates.sort(key=lambda x: len(x[2]), reverse=True)

        return candidates[:MAX_PAIRS]

    # ------------------------------------------------------------------ #
    #  LLM relation extraction                                            #
    # ------------------------------------------------------------------ #

    def _extract_relation(
        self,
        name_a:            str,
        name_b:            str,
        shared_chunks:     List[str],
        graph_memory:      Dict,
        ontology_relations: List[str],
    ) -> Optional[Dict]:
        """
        Ask the LLM whether a relation exists between two entities
        given their shared source passages.
        """
        nodes  = graph_memory["nodes"]
        type_a = nodes.get(name_a, {}).get("type", "MathematicalObject")
        type_b = nodes.get(name_b, {}).get("type", "MathematicalObject")

        # Collect passage text for shared chunks (up to 3 to keep prompt size)
        passages = []
        for chunk_id in shared_chunks[:3]:
            text = self._chunk_map.get(chunk_id, "")
            if text:
                passages.append(f"[{chunk_id}]\n{text}")

        if not passages:
            return None

        passages_text = "\n\n---\n\n".join(passages)

        prompt = (
            _RELATION_PROMPT
            .replace("<<ENTITY_A>>",        name_a)
            .replace("<<TYPE_A>>",          type_a)
            .replace("<<ENTITY_B>>",        name_b)
            .replace("<<TYPE_B>>",          type_b)
            .replace("<<ONTOLOGY_RELATIONS>>", str(ontology_relations))
            .replace("<<PASSAGES>>",        passages_text)
        )

        try:
            response = self.llm.invoke(prompt)
            content  = response.content.strip()
            content  = re.sub(r"^```json\s*", "", content)
            content  = re.sub(r"\s*```$",     "", content)
            return json.loads(content)

        except Exception as e:
            print(f"  [InterChunkRelation] LLM error for "
                  f"'{name_a}' ↔ '{name_b}': {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Edge insertion                                                      #
    # ------------------------------------------------------------------ #

    def _insert_edge(
        self,
        graph_memory:  Dict,
        source:        str,
        relation:      str,
        target:        str,
        confidence:    float,
        shared_chunks: List[str],
        justification: str = "",
    ) -> None:
        """
        Insert a new inter-chunk edge into the graph.
        Merges with existing edge if identical triple already exists.
        """
        edges = graph_memory["edges"]

        for edge in edges:
            if (edge["source"]   == source
                    and edge["relation"] == relation
                    and edge["target"]   == target):
                # Merge evidence into existing edge
                edge["confidence_scores"].append(confidence)
                for src in shared_chunks:
                    if src not in edge["sources"]:
                        edge["sources"].append(src)
                edge["inter_chunk"] = True
                return

        # New edge
        edges.append({
            "source":            source,
            "relation":          relation,
            "target":            target,
            "confidence_scores": [confidence],
            "sources":           shared_chunks,
            "inter_chunk":       True,       # flag for provenance
            "justification":     justification,
        })

    # ------------------------------------------------------------------ #
    #  Chunk loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_chunks(self) -> None:
        """Load chunk texts from the JSON file into self._chunk_map."""
        if self._chunk_map:
            return   # already loaded

        if not self.chunks_path.exists():
            print(f"  [InterChunkRelation] Warning: chunks file not found "
                  f"at {self.chunks_path}")
            return

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        for chunk in chunks:
            self._chunk_map[chunk["chunk_id"]] = chunk.get("content", "")

        print(f"[InterChunkRelation] Loaded {len(self._chunk_map)} chunks "
              f"from {self.chunks_path.name}")