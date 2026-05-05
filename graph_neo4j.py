"""
neo4j_export.py

Exports the knowledge graph from graph_memory.json to Neo4j.
Uses langchain_neo4j for connection and raw Cypher for writes.

Handles:
  - Node creation/update with MERGE (no duplicates across runs)
  - Multiple Neo4j labels per node (Entity + ontology type + all observed types)
  - Relationship creation/update with MERGE
  - Sources array deduplication on merge
  - All metadata preserved: salience, confidence_scores, type_conflict,
    observed_types, sources, inter_chunk flag

Usage:
    from neo4j_export import Neo4jExporter
    exporter = Neo4jExporter()
    exporter.export(graph_memory)       # full export
    exporter.export(graph_memory, incremental=True)  # merge only

Or standalone:
    python neo4j_export.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()


class Neo4jExporter:
    """
    Exports graph_memory dict to Neo4j using langchain_neo4j.

    Node schema in Neo4j:
        (:Entity:NumericalMethod {
            name: "Gaussian elimination",
            salience: 1.82,
            confidence_scores: [0.9, 0.9, 0.9],
            sources: ["chunk_96", "chunk_97", ...],
            type_conflict: false,
            observed_types: ["NumericalMethod"],
            primary_type: "NumericalMethod"
        })

    Relationship schema:
        (:Entity)-[:HAS_PROPERTY {
            confidence_scores: [0.9],
            sources: ["chunk_98"],
            inter_chunk: false
        }]->(:Entity)
    """

    def __init__(self):

        neo4j_url      = os.getenv("NEO4J_URI")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        print(neo4j_url, neo4j_username, neo4j_password)

        if not all([neo4j_url, neo4j_username, neo4j_password]):
            raise ValueError(
                "Missing Neo4j credentials. Set NEO4J_URL, NEO4J_USERNAME, "
                "NEO4J_PASSWORD in your .env file."
            )

        self.graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password,
        )
        print(f"[Neo4j] Connected to {neo4j_url}")

    # ── Main export ───────────────────────────────────────────────────────

    def export(self, graph_memory: Dict, incremental: bool = True) -> Dict:
        """
        Export graph_memory to Neo4j.

        Parameters
        ----------
        graph_memory : dict
            The graph dict with "nodes" and "edges" keys.
        incremental : bool
            If True (default), uses MERGE — updates existing nodes/edges
            and creates new ones. Sources arrays are unioned, not replaced.
            If False, clears the database first then creates everything fresh.

        Returns
        -------
        dict with counts of nodes and edges written.
        """
        nodes = graph_memory.get("nodes", {})
        edges = graph_memory.get("edges", [])

        if not incremental:
            print("[Neo4j] Clearing existing graph...")
            self.graph.query("MATCH (n) DETACH DELETE n")

        # Create indexes for fast MERGE lookups
        self._ensure_indexes()

        # Export nodes
        node_count = self._export_nodes(nodes)

        # Export edges
        edge_count = self._export_edges(edges)

        # Refresh schema so langchain_neo4j sees the updated graph
        self.graph.refresh_schema()

        print(f"\n[Neo4j] Export complete — {node_count} nodes, {edge_count} edges")
        return {"nodes_written": node_count, "edges_written": edge_count}

    # ── Index creation ────────────────────────────────────────────────────

    def _ensure_indexes(self):
        """Create indexes for efficient MERGE operations."""
        try:
            self.graph.query(
                "CREATE INDEX entity_name IF NOT EXISTS "
                "FOR (n:Entity) ON (n.name)"
            )
            print("[Neo4j] Index on Entity.name ensured.")
        except Exception as e:
            # Index may already exist
            print(f"[Neo4j] Index note: {e}")

    # ── Node export ───────────────────────────────────────────────────────

    def _export_nodes(self, nodes: Dict) -> int:
        """
        Export all nodes using MERGE on name.
        Each node gets:
          - :Entity label (always)
          - :PrimaryType label (e.g. :NumericalMethod)
          - :ObservedType labels for ALL observed types
          - All properties from graph_memory
        """
        count = 0

        for name, data in nodes.items():
            primary_type   = data.get("type", "MathematicalObject")
            observed_types = data.get("observed_types", [primary_type])
            sources        = data.get("sources", [])
            conf_scores    = data.get("confidence_scores", [])
            salience       = data.get("salience", 0.0)
            type_conflict  = data.get("type_conflict", False)

            # Build label string: Entity + primary type + all observed types
            all_labels = list(set(["Entity", primary_type] + observed_types))
            # Neo4j labels can't have spaces — sanitize
            all_labels = [l.replace(" ", "_") for l in all_labels if l]  # noqa: E741
            label_str  = ":".join(all_labels)

            # MERGE on name, then SET all properties
            # For sources: union existing + new using APOC-free approach
            cypher = f"""
            MERGE (n:Entity {{name: $name}})
            SET n:{label_str}
            SET n.primary_type      = $primary_type
            SET n.salience          = $salience
            SET n.type_conflict     = $type_conflict
            SET n.observed_types    = $observed_types
            WITH n
            SET n.confidence_scores = $conf_scores
            WITH n
            // Union sources: combine existing + new, remove duplicates
            WITH n, coalesce(n.sources, []) AS old_sources
            WITH n, old_sources + [s IN $sources WHERE NOT s IN old_sources] AS merged
            SET n.sources = merged
            """

            try:
                self.graph.query(cypher, params={
                    "name":           name,
                    "primary_type":   primary_type,
                    "salience":       salience,
                    "type_conflict":  type_conflict,
                    "observed_types": observed_types,
                    "conf_scores":    [float(c) for c in conf_scores],
                    "sources":        sources,
                })
                count += 1

                if count % 50 == 0:
                    print(f"  [Neo4j] {count} nodes written...")

            except Exception as e:
                print(f"  [Neo4j] Failed to write node '{name}': {e}")

        print(f"[Neo4j] {count} nodes exported.")
        return count

    # ── Edge export ───────────────────────────────────────────────────────

    def _export_edges(self, edges: List[Dict]) -> int:
        """
        Export all edges using MERGE on (source, relation, target).
        Relationship types are converted to UPPER_SNAKE_CASE for Neo4j convention.
        """
        count = 0

        for edge in edges:
            source   = edge.get("source", "")
            target   = edge.get("target", "")
            relation = edge.get("relation", "UNKNOWN")
            sources  = edge.get("sources", [])
            conf     = edge.get("confidence_scores", [])
            ic_flag  = edge.get("inter_chunk", False)

            # Convert relation to Neo4j convention: has_property → HAS_PROPERTY
            neo4j_rel = relation.upper().replace(" ", "_")

            # MERGE relationship — uses dynamic relationship type via APOC-free approach
            # Since Cypher doesn't support parameterized relationship types in MERGE,
            # we use string formatting for the relationship type (safe because it
            # comes from our ontology, not user input)
            cypher = f"""
            MATCH (s:Entity {{name: $source}})
            MATCH (t:Entity {{name: $target}})
            MERGE (s)-[r:`{neo4j_rel}`]->(t)
            SET r.relation_name = $relation_name
            SET r.inter_chunk   = $inter_chunk
            SET r.confidence_scores = $conf_scores
            WITH r, coalesce(r.sources, []) AS old_sources
            WITH r, old_sources + [s IN $sources WHERE NOT s IN old_sources] AS merged
            SET r.sources = merged
            """

            try:
                self.graph.query(cypher, params={
                    "source":        source,
                    "target":        target,
                    "relation_name": relation,
                    "inter_chunk":   ic_flag,
                    "conf_scores":   [float(c) for c in conf],
                    "sources":       sources,
                })
                count += 1

                if count % 50 == 0:
                    print(f"  [Neo4j] {count} edges written...")

            except Exception as e:
                print(f"  [Neo4j] Failed to write edge "
                      f"'{source}' →[{relation}]→ '{target}': {e}")

        print(f"[Neo4j] {count} edges exported.")
        return count

    # ── Utility methods ───────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get current Neo4j graph statistics."""
        result = self.graph.query("""
            MATCH (n:Entity)
            WITH count(n) AS nodes
            MATCH ()-[r]->()
            RETURN nodes, count(r) AS edges
        """)
        if result:
            return {"nodes": result[0]["nodes"], "edges": result[0]["edges"]}
        return {"nodes": 0, "edges": 0}

    def clear_all(self):
        """Delete everything in Neo4j. Use with caution."""
        self.graph.query("MATCH (n) DETACH DELETE n")
        print("[Neo4j] Database cleared.")


# ── Standalone usage ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    graph_path = sys.argv[1] if len(sys.argv) > 1 else "data/graph_memory.json"

    if not Path(graph_path).exists():
        print(f"Graph file not found: {graph_path}")
        sys.exit(1)

    with open(graph_path) as f:
        graph_memory = json.load(f)

    print(f"Loaded graph: {len(graph_memory['nodes'])} nodes, "
          f"{len(graph_memory['edges'])} edges")

    exporter = Neo4jExporter()
    exporter.export(graph_memory, incremental=True)