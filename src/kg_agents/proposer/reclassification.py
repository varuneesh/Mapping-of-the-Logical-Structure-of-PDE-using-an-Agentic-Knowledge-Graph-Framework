from pathlib import Path
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import numpy as np

from kg_agents.extraction.loader import OntologyLoader

FUZZY_THRESHOLD = 0.75   


def _normalise(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "").replace("-", "").strip()


def _fuzzy_match(suggested: str, class_name: str) -> Tuple[bool, float]:

    a = _normalise(suggested)
    b = _normalise(class_name)

    if a in b or b in a:
        return True, 1.0

    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= FUZZY_THRESHOLD, ratio


class ReclassificationPass:

    def __init__(
        self,
        core_path:        str,
        candidates_path:  str,
        min_confidence:   float = 0.35,
    ):
        self.core_path        = Path(core_path)
        self.candidates_path  = Path(candidates_path)
        self.min_confidence   = min_confidence


    def run(
        self,
        new_extensions_path: str,
        graph_builder_agent,
        consistency_agent,
        document_memory:     Dict,
        cluster_map:         Dict = None,
        logger=None,
    ) -> Tuple[OntologyLoader, Dict]:
        
        agent = "ReclassificationPass"

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

        graph_builder_agent.recompute_all_salience()

        consistency_agent.save_to_disk()

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


    def _find_matches(
        self,
        entities:    Dict,
        new_classes: List[str],
        cluster_map: Dict = None,
    ) -> List[Tuple[str, str, float, Dict]]:

        cluster_map = cluster_map or {}

        entity_to_class: Dict[str, str] = {}
        for class_name, members in cluster_map.items():
            for member in members:
                entity_to_class[member] = class_name

        matches = []

        for name, data in entities.items():

            best_class = None
            best_score = 0.0

            if name in entity_to_class:
                mapped_class = entity_to_class[name]
                if mapped_class in new_classes:
                    best_class = mapped_class
                    best_score = 1.0
                    print(f"  [reclass] cluster match: '{name}' → {best_class}")

            if not best_class:
                for class_name in new_classes:
                    matched, score = _fuzzy_match(name, class_name)
                    if matched and score > best_score:
                        best_score = score
                        best_class = class_name

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
        
        scores  = cand_data.get("confidence_scores", [])
        sources = cand_data.get("sources", [])

        avg_conf = sum(scores) / len(scores) if scores else 0.0

        if avg_conf < self.min_confidence:
            return {
                "success": False,
                "name":    name,
                "class":   class_name,
                "reason":  f"avg_confidence {avg_conf:.3f} < {self.min_confidence}"
            }

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

        chunk_id = sources[-1] if sources else "reclassification_pass"

        try:
            graph_builder_agent._add_node(entity, chunk_id)

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

        mem_entities = document_memory.setdefault("entities", {})

        if name not in mem_entities:
            mem_entities[name] = {
                "type":              class_name,
                "confidence_scores": scores,
                "sources":           sources,
            }
        else:
            mem_entities[name]["type"] = class_name
            for src in sources:
                if src not in mem_entities[name]["sources"]:
                    mem_entities[name]["sources"].append(src)

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
    
    def _is_subclass_of(
        self,
        child: str,
        parent: str,
        subclasses: Dict,
    ) -> bool:

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

                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm

                member_embeddings.append(emb)

            if not member_embeddings:
                continue

            class_emb = np.mean(member_embeddings, axis=0)

            norm = np.linalg.norm(class_emb)
            if norm > 0:
                class_emb = class_emb / norm

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

                    old_type = node_data["type"]
                    node_data["type"] = class_name
                    node_data["reclassified_from"] = old_type

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


    def reclassify_relations(
        self,
        new_extensions_path: str,
        graph_builder_agent,
        consistency_agent,
        ontology_loader=None,
        logger=None,
    ) -> List[Dict]:
        
        agent = "ReclassificationPass"

        new_ontology  = OntologyLoader(str(self.core_path), new_extensions_path)
        new_relations = list(
            new_ontology.extensions.get("relation_extensions", {}).keys()
        )

        onto_for_validation = ontology_loader or new_ontology

        if not new_relations:
            return []

        print(f"\n[Reclassification] New relation types: {new_relations}")

        graph_nodes    = set(graph_builder_agent.graph.get("nodes", {}).keys())
        pool_relations = consistency_agent.candidate_pool.get("relations", [])
        inserted = []
        to_remove = []

        for rel in pool_relations:
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

            if rel["source"] not in graph_nodes or rel["target"] not in graph_nodes:
                continue

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


    def rescue_stranded_relations(
        self,
        graph_builder_agent,
        consistency_agent,
        ontology_loader,
        logger=None,
    ) -> List[Dict]:

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

            if source not in graph_nodes or target not in graph_nodes:
                continue

            if relation == "NEW_RELATION":
                continue

            if not ontology_loader.relation_exists(relation):
                continue

            scores   = rel.get("confidence_scores", [])
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            if avg_conf < self.min_confidence:
                continue

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