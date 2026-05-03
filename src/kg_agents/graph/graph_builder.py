import math


class GraphBuilderAgent:
    """
    Incrementally builds a persistent knowledge graph.

    Key fixes vs original:
    - Nodes store confidence_scores (per spec) — was silently dropped before.
    - Node type conflicts are tracked explicitly: if a canonical entity is
      inserted with type A in chunk 1 and type B in chunk 5, both types are
      recorded and the conflict is flagged so the Ontology Proposer / human
      reviewer can resolve it later.
    """

    def __init__(self, graph_memory: dict):
        self.graph = graph_memory

    def run(self, state: dict) -> dict:

        chunk_id = state["chunk_id"]

        for entity in state["consistent_entities"]:
            self._add_node(entity, chunk_id)

        for rel in state["consistent_relationships"]:
            self._add_edge(rel, chunk_id)

        print(f"Graph — nodes: {len(self.graph['nodes'])} | "
              f"edges: {len(self.graph['edges'])}")

        return state

    # ------------------------------------------------------------------ #

    def _compute_salience(self, confidence_scores: list, source_count: int) -> float:
        """
        Salience = mean_confidence × log(1 + source_count)

        Intuition:
          - mean_confidence reflects extraction reliability
          - log(1 + source_count) rewards entities seen across many chunks
            while dampening the effect of very high counts so a node seen
            200 times isn't absurdly dominant over one seen 40 times

        Examples:
          2 sources,  mean_conf=0.88 → 0.88 × log(3)  ≈ 0.97
          40 sources, mean_conf=0.88 → 0.88 × log(41) ≈ 3.27
        """
        if not confidence_scores:
            return 0.0
        mean_conf = sum(confidence_scores) / len(confidence_scores)
        return round(mean_conf * math.log(1 + source_count), 4)

    def recompute_all_salience(self) -> None:
        """
        Recompute salience for every node in the graph.
        Call this after reclassification or any bulk graph update
        to ensure scores reflect the latest evidence.
        """
        for name, node in self.graph["nodes"].items():
            node["salience"] = self._compute_salience(
                node.get("confidence_scores", []),
                len(node.get("sources", []))
            )
        print(f"Salience recomputed for {len(self.graph['nodes'])} nodes.")

    def top_nodes_by_salience(self, n: int = 20) -> list:
        """
        Return the top-n nodes ranked by salience score.
        Useful for notebook inspection and Ontology Proposer prioritisation.
        """
        ranked = sorted(
            self.graph["nodes"].items(),
            key=lambda kv: kv[1].get("salience", 0.0),
            reverse=True
        )
        return [
            {"name": name, "salience": data["salience"], "type": data["type"]}
            for name, data in ranked[:n]
        ]

    def _add_node(self, entity: dict, chunk_id: str) -> None:

        name       = entity["name"]
        etype      = entity["type"]
        confidence = entity["confidence"]
        nodes      = self.graph["nodes"]

        if name not in nodes:
            nodes[name] = {
                "type":              etype,
                "sources":           [chunk_id],
                "confidence_scores": [confidence],
                "type_conflict":     False,
                "observed_types":    [etype],
                "salience":          0.0,   # computed below
            }
        else:
            node = nodes[name]
            node["confidence_scores"].append(confidence)
            if chunk_id not in node["sources"]:
                node["sources"].append(chunk_id)

            # Track type conflicts — don't silently overwrite
            if etype not in node["observed_types"]:
                node["observed_types"].append(etype)
                node["type_conflict"] = True
                print(f"  [type conflict] '{name}': "
                      f"previously '{node['type']}', now seen as '{etype}'")

        # Recompute salience after every update so it always reflects
        # the latest evidence — cheap operation, always consistent
        node = nodes[name]
        node["salience"] = self._compute_salience(
            node["confidence_scores"],
            len(node["sources"])
        )

    def _add_edge(self, rel: dict, chunk_id: str) -> None:

        edges = self.graph["edges"]

        for edge in edges:
            if (
                edge["source"]       == rel["source"]
                and edge["relation"] == rel["relation"]
                and edge["target"]   == rel["target"]
            ):
                edge["confidence_scores"].append(rel["confidence"])
                if chunk_id not in edge["sources"]:
                    edge["sources"].append(chunk_id)
                return

        edges.append({
            "source":            rel["source"],
            "relation":          rel["relation"],
            "target":            rel["target"],
            "confidence_scores": [rel["confidence"]],
            "sources":           [chunk_id],
        })