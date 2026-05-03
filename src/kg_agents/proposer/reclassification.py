"""
reclassification.py

Reclassification pass — triggered immediately after OntologyProposer writes
a new ontology_extensions file.

Responsibilities:
  1. Scan the candidate pool for entities whose suggested_types fuzzy-match
     any newly accepted class.
  2. Reclassify matched entities with their proper ontology type.
  3. Run them through consistency checks (they now pass since type != NEW_TYPE).
  4. Insert/update them in the graph via GraphBuilderAgent.
  5. Update document_memory so future chunks see the correct type.
  6. Remove reclassified entities from the candidate pool.
  7. Reload the OntologyLoader so the running pipeline uses updated ontology.
  8. Recompute salience for all affected nodes.

Fuzzy matching strategy:
  - Normalise both strings (lowercase, strip spaces, remove underscores/hyphens)
  - Check if either string contains the other as a substring
  - Compute character-level overlap ratio as a fallback
  - Threshold: overlap_ratio >= 0.75 OR substring containment
  This catches: "Discretization" → "DiscretizationScheme",
                "stabilityCondition" → "StabilityCondition",
                "error estimator" → "ErrorEstimator"
"""

# import json
from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import numpy as np

from kg_agents.extraction.loader import OntologyLoader


# ── Fuzzy match threshold ────────────────────────────────────────────────────
FUZZY_THRESHOLD = 0.75   # SequenceMatcher ratio — tunable


def _normalise(s: str) -> str:
    """Lowercase, strip, remove spaces/underscores/hyphens for comparison."""
    return s.lower().replace(" ", "").replace("_", "").replace("-", "").strip()


def _fuzzy_match(suggested: str, class_name: str) -> Tuple[bool, float]:
    """
    Return (matched: bool, score: float).

    Two conditions — either is sufficient:
      1. Substring containment (normalised): one contains the other.
         e.g. "discretization" in "discretizationscheme" → True
      2. SequenceMatcher ratio >= FUZZY_THRESHOLD.
         e.g. "stabilityCondition" vs "StabilityCondition" → 1.0
    """
    a = _normalise(suggested)
    b = _normalise(class_name)

    # Substring containment
    if a in b or b in a:
        return True, 1.0

    # Character-level similarity
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= FUZZY_THRESHOLD, ratio


class ReclassificationPass:
    """
    Reclassifies candidate pool entities against newly accepted ontology classes.

    Parameters
    ----------
    core_path : str | Path
        Path to ontology_core_v0_0.json (never changes).
    candidates_path : str | Path
        Path to ontology_candidates.json (candidate pool on disk).
    min_confidence : float
        Minimum average confidence for a reclassified entity to enter the graph.
    """

    def __init__(
        self,
        core_path:        str,
        candidates_path:  str,
        min_confidence:   float = 0.35,
    ):
        self.core_path        = Path(core_path)
        self.candidates_path  = Path(candidates_path)
        self.min_confidence   = min_confidence

    # ------------------------------------------------------------------ #
    #  Main entry                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        new_extensions_path: str,
        graph_builder_agent,
        consistency_agent,
        document_memory:     Dict,
        cluster_map:         Dict = None,
        logger=None,
    ) -> Tuple[OntologyLoader, Dict]:
        """
        Execute the full reclassification pass.

        Parameters
        ----------
        new_extensions_path : str
            Path to the newly written ontology_extensions_vX_Y.json.
        graph_builder_agent : GraphBuilderAgent
        consistency_agent : ConsistencyAgent
        document_memory : dict
        cluster_map : dict | None
            {class_name: [member_names]} from apply_accepted_proposals.
            Used as primary matching strategy — cluster members are
            reclassified directly without needing fuzzy name matching.
        logger : PipelineLogger | None
        """
        agent = "ReclassificationPass"

        # ── Step 1: reload OntologyLoader with new extensions ────────────
        new_ontology = OntologyLoader(
            str(self.core_path),
            new_extensions_path
        )
        new_classes = list(
            new_ontology.extensions.get("subclasses", {}).keys()
        )

        print(f"\n[Reclassification] New classes available: {new_classes}")

        if logger:
            logger.info(agent, "started", {
                "new_classes":       new_classes,
                "extensions_path":   new_extensions_path,
                "candidate_pool_size": len(
                    consistency_agent.candidate_pool.get("entities", {})
                )
            })

        # ── Step 2: scan candidate pool for matches ──────────────────────
        matched = self._find_matches(
            consistency_agent.candidate_pool.get("entities", {}),
            new_classes,
            cluster_map or {},
        )

        print(f"[Reclassification] {len(matched)} candidate(s) matched "
              f"to new classes.")

        if not matched:
            if logger:
                logger.info(agent, "no_matches", {})
            return new_ontology, {"reclassified": [], "failed": []}

        # ── Step 3: reclassify, consistency-check, insert into graph ─────
        reclassified = []
        failed       = []

        for name, class_name, score, cand_data in matched:

            result = self._reclassify_entity(
                name         = name,
                class_name   = class_name,
                cand_data    = cand_data,
                graph_builder_agent = graph_builder_agent,
                consistency_agent   = consistency_agent,
                document_memory     = document_memory,
                new_ontology        = new_ontology,
            )

            if result["success"]:
                reclassified.append(result)
                print(f"  ✓ '{name}' → {class_name} "
                      f"(match score: {score:.2f}, "
                      f"inserted into graph)")
            else:
                failed.append(result)
                print(f"  ✗ '{name}' → {class_name} "
                      f"FAILED: {result['reason']}")

        # ── Step 4: recompute salience for all affected nodes ────────────
        graph_builder_agent.recompute_all_salience()

        # ── Step 5: persist updated candidate pool ───────────────────────
        consistency_agent.save_to_disk()

        # ── Step 6: log and report ────────────────────────────────────────
        report = {
            "new_classes":    new_classes,
            "total_matched":  len(matched),
            "reclassified":   reclassified,
            "failed":         failed,
        }

        if logger:
            logger.info(agent, "completed", {
                "reclassified": len(reclassified),
                "failed":       len(failed),
                "new_classes":  new_classes,
            })

        print(f"\n[Reclassification] Complete — "
              f"{len(reclassified)} inserted, {len(failed)} failed.")
        print(f"[Reclassification] OntologyLoader reloaded with "
              f"{len(new_classes)} new class(es).")

        return new_ontology, report

    # ------------------------------------------------------------------ #
    #  Candidate matching                                                  #
    # ------------------------------------------------------------------ #

    def _find_matches(
        self,
        entities:    Dict,
        new_classes: List[str],
        cluster_map: Dict = None,
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        THREE matching strategies in priority order:
          0. Cluster membership (cluster_map) — primary for cluster-based proposer
          1. Entity NAME vs class name — fuzzy string match
          2. suggested_types vs class name — fallback
        """
        cluster_map = cluster_map or {}

        # Build reverse map: entity_name → class_name
        entity_to_class: Dict[str, str] = {}
        for class_name, members in cluster_map.items():
            for member in members:
                entity_to_class[member] = class_name

        matches = []

        for name, data in entities.items():

            best_class = None
            best_score = 0.0

            # ── Strategy 0: cluster membership ───────────────────────────
            if name in entity_to_class:
                mapped_class = entity_to_class[name]
                if mapped_class in new_classes:
                    best_class = mapped_class
                    best_score = 1.0
                    print(f"  [reclass] cluster match: '{name}' → {best_class}")

            # ── Strategy 1: entity NAME against class name ────────────────
            if not best_class:
                for class_name in new_classes:
                    matched, score = _fuzzy_match(name, class_name)
                    if matched and score > best_score:
                        best_score = score
                        best_class = class_name

            # ── Strategy 2: suggested_types against class name ────────────
            if not best_class:
                suggested_types = data.get("suggested_types", [])
                for suggested in suggested_types:
                    for class_name in new_classes:
                        matched, score = _fuzzy_match(suggested, class_name)
                        if matched and score > best_score:
                            best_score = score
                            best_class = class_name

            if best_class:
                matches.append((name, best_class, best_score, data))

        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    # ------------------------------------------------------------------ #
    #  Single entity reclassification                                     #
    # ------------------------------------------------------------------ #

    def _reclassify_entity(
        self,
        name:               str,
        class_name:         str,
        cand_data:          Dict,
        graph_builder_agent,
        consistency_agent,
        document_memory:    Dict,
        new_ontology,
    ) -> Dict:
        """
        Reclassify one entity: consistency check → graph insert →
        document_memory update → remove from candidate pool.
        """
        scores  = cand_data.get("confidence_scores", [])
        sources = cand_data.get("sources", [])

        avg_conf = sum(scores) / len(scores) if scores else 0.0

        # ── Confidence gate ───────────────────────────────────────────────
        if avg_conf < self.min_confidence:
            return {
                "success": False,
                "name":    name,
                "class":   class_name,
                "reason":  f"avg_confidence {avg_conf:.3f} < {self.min_confidence}"
            }

        # ── Build a synthetic entity object for graph insertion ──────────
        # Use best (highest) confidence from accumulated scores
        best_conf = max(scores) if scores else avg_conf
        entity = {
            "entity_id":          f"reclass_{name}",
            "name":               name,
            "type":               class_name,
            "suggested_type":     None,
            "source_chunk_id":    sources[0] if sources else "reclassification",
            "evidence_chunk_ids": sources,
            "confidence":         best_conf,
        }

        # ── Insert into graph (GraphBuilderAgent handles merge logic) ────
        # Use the most recent source chunk as the insertion chunk_id
        chunk_id = sources[-1] if sources else "reclassification_pass"

        try:
            graph_builder_agent._add_node(entity, chunk_id)

            # Set full provenance — all source chunks and all confidence scores
            node = graph_builder_agent.graph["nodes"].get(name, {})
            node["sources"] = list(set(node.get("sources", [])) | set(sources))
            node["confidence_scores"] = list(scores)

        except Exception as e:
            return {
                "success": False,
                "name":    name,
                "class":   class_name,
                "reason":  f"graph insertion error: {e}"
            }

        # ── Update document_memory ────────────────────────────────────────
        mem_entities = document_memory.setdefault("entities", {})

        if name not in mem_entities:
            mem_entities[name] = {
                "type":              class_name,
                "confidence_scores": scores,
                "sources":           sources,
            }
        else:
            # Update type — reclassification supersedes old NEW_TYPE label
            mem_entities[name]["type"] = class_name
            for src in sources:
                if src not in mem_entities[name]["sources"]:
                    mem_entities[name]["sources"].append(src)

        # ── Remove from candidate pool ────────────────────────────────────
        pool = consistency_agent.candidate_pool.get("entities", {})
        if name in pool:
            del pool[name]

        return {
            "success":      True,
            "name":         name,
            "class":        class_name,
            "avg_conf":     round(avg_conf, 3),
            "source_count": len(sources),
        }

    # ------------------------------------------------------------------ #
    #  Graph node subclass reclassification                                #
    # ------------------------------------------------------------------ #
    
    def _is_subclass_of(
        self,
        child: str,
        parent: str,
        subclasses: Dict,
    ) -> bool:
        """
        Recursively check whether `child`
        is a subclass of `parent`.

        subclasses format:
        {
            "FloatingPointConcept": {
                "parent": "ComputationalStructure"
            },
            ...
        }
        """

        if child == parent:
            return True

        visited = set()

        while child in subclasses and child not in visited:
            visited.add(child)

            child_parent = subclasses[child].get(
                "parent",
                "MathematicalObject"
            )

            if child_parent == parent:
                return True

            child = child_parent

        return False

    def reclassify_graph_nodes(
        self,
        new_extensions_path: str,
        graph_builder_agent,
        document_memory:     Dict,
        logger=None,
    ) -> List[Dict]:
        """
        After new subclasses are accepted, scan existing graph nodes to check
        if any should be retyped to the new subclass.

        For example: if 'FloatingPointConcept' is accepted with parent
        'ComputationalStructure', scan all graph nodes currently typed
        'ComputationalStructure' and check (via embedding similarity) if
        they are semantically closer to the new subclass.

        Uses embeddings from document_memory for comparison.
        """
        agent = "ReclassificationPass"

        new_ontology = OntologyLoader(str(self.core_path), new_extensions_path)
        new_subclasses = new_ontology.extensions.get("subclasses", {})

        if not new_subclasses:
            return []

        graph_nodes = graph_builder_agent.graph.get("nodes", {})
        entities_mem = document_memory.get("entities", {})
        retyped = []

        for class_name, class_info in new_subclasses.items():
            parent_class = class_info.get("parent", "MathematicalObject")

            # Find all graph nodes currently typed as the parent class
            parent_nodes = [
                (name, data)
                for name, data in graph_nodes.items()
                if self._is_subclass_of(
                    data.get("type", ""),
                    parent_class,
                    new_subclasses,
                )
            ]

            if not parent_nodes:
                continue

            # Get embedding for the class name itself from any cluster member
            # that exists in document_memory
            # class_emb = None
            # class_norm = class_name.lower().replace(" ", "").replace("_", "")
            
            # ── Build subclass embedding from cluster member centroid ──────────
            cluster_members = class_info.get("cluster_members", [])

            member_embeddings = []

            for member in cluster_members:
                ent_info = entities_mem.get(member)

                if not ent_info:
                    continue

                emb = ent_info.get("embedding")

                if emb is None:
                    continue

                if not isinstance(emb, np.ndarray):
                    emb = np.array(emb, dtype=float)

                # normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                member_embeddings.append(emb)

            # No usable embeddings
            if not member_embeddings:
                continue

            # centroid embedding for subclass
            class_emb = np.mean(member_embeddings, axis=0)

            # normalize centroid
            norm = np.linalg.norm(class_emb)
            if norm > 0:
                class_emb = class_emb / norm

            # for ent_name, ent_info in entities_mem.items():
            #     ent_norm = ent_name.lower().replace(" ", "").replace("_", "")
            #     if ent_norm == class_norm or class_norm in ent_norm or ent_norm in class_norm:
            #         emb = ent_info.get("embedding")
            #         if emb is not None:
            #             if not isinstance(emb, np.ndarray):
            #                 emb = np.array(emb, dtype=float)
            #             class_emb = emb
            #             break

            # if class_emb is None:
            #     continue

            # Normalise
            # norm = np.linalg.norm(class_emb)
            # if norm > 0:
            #     class_emb = class_emb / norm

            # Check each parent-typed node
            for node_name, node_data in parent_nodes:
                node_mem = entities_mem.get(node_name, {})
                node_emb = node_mem.get("embedding")

                if node_emb is None:
                    continue
                if not isinstance(node_emb, np.ndarray):
                    node_emb = np.array(node_emb, dtype=float)

                norm = np.linalg.norm(node_emb)
                if norm > 0:
                    node_emb = node_emb / norm

                similarity = float(np.dot(class_emb, node_emb))

                if similarity >= 0.70:
                    # Retype this node to the new subclass
                    old_type = node_data["type"]
                    node_data["type"] = class_name
                    node_data["reclassified_from"] = old_type

                    # Update document_memory too
                    if node_name in entities_mem:
                        entities_mem[node_name]["type"] = class_name

                    retyped.append({
                        "name":       node_name,
                        "old_type":   old_type,
                        "new_type":   class_name,
                        "similarity": round(similarity, 3),
                    })
                    print(f"  ✓ graph node '{node_name}': "
                          f"{old_type} → {class_name} (sim={similarity:.3f})")

        if logger and retyped:
            logger.info(agent, "graph_nodes_retyped", {"count": len(retyped)})

        print(f"[Reclassification] Retyped {len(retyped)} existing graph node(s).")
        return retyped

    # ------------------------------------------------------------------ #
    #  Relation reclassification                                           #
    # ------------------------------------------------------------------ #

    def reclassify_relations(
        self,
        new_extensions_path: str,
        graph_builder_agent,
        consistency_agent,
        ontology_loader=None,
        logger=None,
    ) -> List[Dict]:
        """
        After new relation types are accepted, find NEW_RELATION candidates
        whose suggested_relations fuzzy-match a new relation type and insert
        them into the graph.
        """
        agent = "ReclassificationPass"

        new_ontology  = OntologyLoader(str(self.core_path), new_extensions_path)
        new_relations = list(
            new_ontology.extensions.get("relation_extensions", {}).keys()
        )

        # Use passed ontology_loader for validation, fall back to new_ontology
        onto_for_validation = ontology_loader or new_ontology

        if not new_relations:
            return []

        print(f"\n[Reclassification] New relation types: {new_relations}")

        graph_nodes    = set(graph_builder_agent.graph.get("nodes", {}).keys())
        pool_relations = consistency_agent.candidate_pool.get("relations", [])
        inserted = []
        to_remove = []

        for rel in pool_relations:
            # Match suggested_relations against new types, NOT literal "NEW_RELATION"
            suggested = rel.get("suggested_relations", [])
            if not suggested:
                suggested = [rel.get("relation", "")]

            best_match = None
            best_score = 0.0

            for suggestion in suggested:
                for new_rel in new_relations:
                    matched, score = _fuzzy_match(suggestion, new_rel)
                    if matched and score > best_score:
                        best_score = score
                        best_match = new_rel

            if not best_match:
                continue

            scores   = rel.get("confidence_scores", [])
            avg_conf = sum(scores) / len(scores) if scores else 0.0

            if avg_conf < self.min_confidence:
                continue

            # Endpoint existence check
            if rel["source"] not in graph_nodes or rel["target"] not in graph_nodes:
                continue

            # Domain-range validation
            source_type = graph_builder_agent.graph["nodes"].get(
                rel["source"], {}
            ).get("type")
            target_type = graph_builder_agent.graph["nodes"].get(
                rel["target"], {}
            ).get("type")

            if source_type and target_type:
                if not onto_for_validation.validate_domain_range(
                    best_match, source_type, target_type
                ):
                    continue

            # Insert edge — single call, then set full sources
            synthetic_rel = {
                "source":     rel["source"],
                "relation":   best_match,
                "target":     rel["target"],
                "confidence": max(scores) if scores else avg_conf,
            }
            sources = rel.get("sources", [])

            try:
                graph_builder_agent._add_edge(
                    synthetic_rel,
                    sources[0] if sources else "reclassification"
                )
                for edge in graph_builder_agent.graph["edges"]:
                    if (edge["source"] == synthetic_rel["source"]
                            and edge["relation"] == synthetic_rel["relation"]
                            and edge["target"] == synthetic_rel["target"]):
                        edge["sources"] = list(set(edge["sources"]) | set(sources))
                        break

                inserted.append({
                    "source":   rel["source"],
                    "relation": best_match,
                    "target":   rel["target"],
                    "score":    round(best_score, 3),
                })
                to_remove.append(rel)
                print(f"  ✓ relation '{rel.get('relation','')}' → '{best_match}' inserted")

            except Exception as e:
                print(f"  ✗ relation insertion failed: {e}")
                if logger:
                    logger.warning(agent, "relation_insertion_failed", {
                        "relation": rel.get("relation", ""), "error": str(e)
                    })

        for rel in to_remove:
            consistency_agent.candidate_pool["relations"].remove(rel)

        consistency_agent.save_to_disk()

        if logger:
            logger.info(agent, "relations_reclassified", {"count": len(inserted)})

        return inserted

    # ------------------------------------------------------------------ #
    #  Stranded relation rescue                                            #
    # ------------------------------------------------------------------ #

    def rescue_stranded_relations(
        self,
        graph_builder_agent,
        consistency_agent,
        ontology_loader,
        logger=None,
    ) -> List[Dict]:
        """
        After entity reclassification adds new nodes to the graph, scan the
        relation candidate pool for relations that PREVIOUSLY failed because
        one or both endpoints were missing from the graph. If both endpoints
        now exist AND the relation type is valid in the ontology, insert the
        edge directly.

        This rescues relations that were stranded in the pool not because the
        relation type was unknown (NEW_RELATION), but because one of the
        entities was NEW_TYPE and hadn't been reclassified yet.

        Call this AFTER run() and reclassify_relations() complete, so all
        new entity nodes and relation types are already in the graph/ontology.
        """
        agent = "ReclassificationPass"

        graph_nodes    = set(graph_builder_agent.graph.get("nodes", {}).keys())
        pool_relations = consistency_agent.candidate_pool.get("relations", [])

        if not pool_relations or not graph_nodes:
            return []

        rescued   = []
        to_remove = []

        for rel in pool_relations:
            source   = rel.get("source", "")
            target   = rel.get("target", "")
            relation = rel.get("relation", "")

            # Both endpoints must now exist in the graph
            if source not in graph_nodes or target not in graph_nodes:
                continue

            # The relation type must be valid in the ontology
            # (either an existing type, or skip NEW_RELATION — those are
            # handled by reclassify_relations separately)
            if relation == "NEW_RELATION":
                continue

            if not ontology_loader.relation_exists(relation):
                continue

            # Confidence check
            scores   = rel.get("confidence_scores", [])
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            if avg_conf < self.min_confidence:
                continue

            # Domain-range validation against ontology
            source_type = graph_builder_agent.graph["nodes"].get(
                source, {}
            ).get("type")
            target_type = graph_builder_agent.graph["nodes"].get(
                target, {}
            ).get("type")

            if source_type and target_type:
                if not ontology_loader.validate_domain_range(
                    relation, source_type, target_type
                ):
                    continue

            # All checks passed — insert edge
            best_conf = max(scores) if scores else avg_conf
            sources   = rel.get("sources", [])
            chunk_id  = sources[-1] if sources else "rescue_pass"

            synthetic_rel = {
                "source":     source,
                "relation":   relation,
                "target":     target,
                "confidence": best_conf,
            }

            try:
                graph_builder_agent._add_edge(synthetic_rel, chunk_id)
                # Add all source chunks for full provenance
                for src in sources:
                    if src != chunk_id:
                        graph_builder_agent._add_edge(synthetic_rel, src)

                rescued.append({
                    "source":     source,
                    "relation":   relation,
                    "target":     target,
                    "avg_conf":   round(avg_conf, 3),
                    "source_count": len(sources),
                })
                to_remove.append(rel)
                print(f"  ✓ rescued: {source} →[{relation}]→ {target} "
                      f"(avg_conf={avg_conf:.3f})")

            except Exception as e:
                print(f"  ✗ rescue failed: {source} →[{relation}]→ {target}: {e}")
                if logger:
                    logger.warning(agent, "rescue_failed", {
                        "source": source, "relation": relation,
                        "target": target, "error": str(e)
                    })

        # Remove rescued relations from pool
        for rel in to_remove:
            if rel in consistency_agent.candidate_pool["relations"]:
                consistency_agent.candidate_pool["relations"].remove(rel)

        if rescued:
            consistency_agent.save_to_disk()

        print(f"[Reclassification] Rescued {len(rescued)} stranded relation(s) "
              f"from candidate pool.")

        if logger:
            logger.info(agent, "relations_rescued", {"count": len(rescued)})

        return rescued