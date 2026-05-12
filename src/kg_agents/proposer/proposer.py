import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import numpy as np

from langchain_openai import ChatOpenAI


# ENTITY_COUNT_THRESHOLD    = 5      
RELATION_COUNT_THRESHOLD  = 5      
MIN_SINGLETON_COUNT = 7          
MIN_CLUSTER_COUNT    = 5         
MIN_CLUSTER_SIZE     = 3        
CLUSTER_SIM_THRESHOLD = 0.7
MIN_AVG_CONFIDENCE        = 0.6   
TYPE_CONSISTENCY_RATIO    = 0.2   
# MIN_SALIENCE              = 0.5   
POOL_SIZE_TRIGGER         = 30     
CHUNK_COUNT_FALLBACK      = 50    

_STOPLIST = {
    "method", "scheme", "algorithm", "approach", "procedure",
    "technique", "framework", "system", "model", "process",
    "it", "this", "that", "they", "the", "a", "an",
    "result", "value", "number", "way", "form", "type",
    "case", "example", "problem", "solution", "term"
}


class OntologyProposerAgent:

    def __init__(
        self,
        ontology_dir:  str,
        proposals_dir: str,
        model: str = "gpt-4o-mini",
    ):
        self.ontology_dir  = Path(ontology_dir)
        self.proposals_dir = Path(proposals_dir)
        self.proposals_dir.mkdir(parents=True, exist_ok=True)

        self.llm = ChatOpenAI(model=model, temperature=0)
        self._chunk_counter = 0
        self._last_pool_entity_count = 0 


    def tick(self) -> None:
        """Call once per processed chunk to advance the fallback counter."""
        self._chunk_counter += 1

    def should_run(self, candidate_pool: Dict) -> Tuple[bool, str]:
        """
        Decide whether to run the proposer now.
        Returns (True, reason) or (False, "").
        """
        entities = candidate_pool.get("entities", {})
        current_count = len(entities)

        if current_count <= self._last_pool_entity_count and current_count > 0:
            return False, ""


        if len(entities) >= POOL_SIZE_TRIGGER:
            return True, f"pool size={len(entities)} >= {POOL_SIZE_TRIGGER}"

        if self._chunk_counter > 0 and self._chunk_counter % CHUNK_COUNT_FALLBACK == 0:
            return True, f"chunk fallback at chunk {self._chunk_counter}"

        return False, ""


    def run(
        self,
        candidate_pool: Dict,
        graph_memory:   Dict,
        ontology_loader,
        document_memory: Dict = None,
        logger=None,
    ) -> Path:

        agent = "OntologyProposerAgent"
        if logger:
            logger.info(agent, "run_started", {
                "entity_candidates": len(candidate_pool.get("entities", {})),
                "relation_candidates": len(candidate_pool.get("relations", []))
            })


        entity_candidates = self._select_entity_candidates(
            candidate_pool.get("entities", {}),
            graph_memory,
            ontology_loader,
        )

        relation_candidates = self._select_relation_candidates(
            candidate_pool.get("relations", []),
            ontology_loader,
            graph_memory,
            entity_candidates,
        )

        print(f"\n[OntologyProposer] Selected {len(entity_candidates)} entity "
              f"candidate(s) and {len(relation_candidates)} relation candidate(s) "
              f"for proposal.")

        if not entity_candidates and not relation_candidates:
            print("[OntologyProposer] No candidates meet thresholds — skipping.")
            if logger:
                logger.info(agent, "no_candidates", {})
            return None

        entity_proposals   = self._generate_entity_proposals(
            entity_candidates, ontology_loader,
            document_memory or {},
            logger,
        )
        relation_proposals = self._generate_relation_proposals(
            relation_candidates, ontology_loader, logger
        )

        proposals_path = self._write_proposals(
            entity_proposals, relation_proposals, ontology_loader
        )

        if logger:
            logger.info(agent, "proposals_written", {
                "path": str(proposals_path),
                "entity_proposals":   len(entity_proposals),
                "relation_proposals": len(relation_proposals),
            })

        print(f"[OntologyProposer] Proposals written → {proposals_path}")
        print(f"  Entity proposals  : {len(entity_proposals)}")
        print(f"  Relation proposals: {len(relation_proposals)}")

        self._last_pool_entity_count = len(candidate_pool.get("entities", {}))

        return proposals_path


    def _select_entity_candidates(
        self,
        entities:        Dict,
        graph_memory:    Dict,
        ontology_loader,
    ) -> List[Dict]:
        
        selected = []

        from difflib import SequenceMatcher
        existing_classes = ontology_loader.get_all_classes()
        existing_norm = {
            c: c.lower().replace("_", "").replace("-", "").replace(" ", "")
            for c in existing_classes
        }

        for name, data in entities.items():

            if name.lower().strip() in _STOPLIST or len(name.strip()) <= 2:
                continue

            scores = data.get("confidence_scores", [])
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            if avg_conf < MIN_AVG_CONFIDENCE:
                continue

            suggested = data.get("suggested_types", [])
            if suggested:
                def _norm_type(s):
                    return s.lower().replace(" ", "").replace("_", "").replace("-", "").strip()

                suggested_norm = [_norm_type(s) for s in suggested if s]
                if suggested_norm:
                    most_common = max(set(suggested_norm), key=suggested_norm.count)
                    consistency = suggested_norm.count(most_common) / len(suggested_norm)
                    if consistency < TYPE_CONSISTENCY_RATIO:
                        continue
  
                    for s in suggested:
                        if s and _norm_type(s) == most_common:
                            dominant_suggestion = s
                            break
                    else:
                        dominant_suggestion = most_common
                else:
                    dominant_suggestion = None
            else:
                dominant_suggestion = None

            if ontology_loader.class_exists(name):
                continue

            name_norm = name.lower().replace("_", "").replace("-", "").replace(" ", "")
            already_exists = False
            for class_name, class_norm in existing_norm.items():
                if name_norm == class_norm:
                    already_exists = True
                    break
                if name_norm in class_norm or class_norm in name_norm:
                    already_exists = True
                    break
                if SequenceMatcher(None, name_norm, class_norm).ratio() >= 0.8:
                    already_exists = True
                    break
            if already_exists:
                continue

            graph_node = graph_memory.get("nodes", {}).get(name, {})
            salience   = graph_node.get("salience", 0.0)

            selected.append({
                "name":               name,
                "count":              data["count"],
                "avg_confidence":     round(avg_conf, 3),
                "dominant_suggestion": dominant_suggestion,
                "suggested_types":    suggested,
                "sources":            data.get("sources", [])[:5],
                "salience":           salience,
            })

        selected.sort(key=lambda x: (x["salience"], x["count"]), reverse=True)
        return selected

    def _select_relation_candidates(
        self,
        relations:          List[Dict],
        ontology_loader,
        graph_memory:       Dict = None,
        entity_candidates:  List[Dict] = None,
    ) -> List[Dict]:

        selected = []

        known_names = set()
        if graph_memory:
            known_names.update(graph_memory.get("nodes", {}).keys())
        if entity_candidates:
            known_names.update(c["name"] for c in entity_candidates)

        for rel in relations:
            if rel["count"] < RELATION_COUNT_THRESHOLD:
                continue

            scores   = rel.get("confidence_scores", [])
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            if avg_conf < MIN_AVG_CONFIDENCE:
                continue

            if ontology_loader.relation_exists(rel["relation"]):
                continue

            if known_names:
                source_known = rel["source"] in known_names
                target_known = rel["target"] in known_names
                if not (source_known and target_known):
                    continue

            selected.append({
                "source":              rel["source"],
                "relation":            rel["relation"],
                "target":              rel["target"],
                "count":               rel["count"],
                "avg_confidence":      round(avg_conf, 3),
                "sources":             rel.get("sources", [])[:5],
                "suggested_relations": rel.get("suggested_relations", []),
            })

        selected.sort(key=lambda x: x["count"], reverse=True)
        return selected
    
    
    def _should_propose_cluster(self, cluster: List[Dict]) -> bool:
        size = len(cluster)
        total_count = sum(c["count"] for c in cluster)
        avg_conf = sum(c["avg_confidence"] for c in cluster) / size

        if size == 1:
            return total_count >= MIN_SINGLETON_COUNT and avg_conf >= MIN_AVG_CONFIDENCE
        
        if size == 2:
            return total_count >= (MIN_CLUSTER_COUNT + 1) and avg_conf >= MIN_AVG_CONFIDENCE

        if size >= MIN_CLUSTER_SIZE:
            return total_count >= MIN_CLUSTER_COUNT and avg_conf >= MIN_AVG_CONFIDENCE

        return False


    def _cluster_candidates(
        self,
        candidates: List[Dict],
        document_memory: Dict,
        threshold: float = 0.7,
    ) -> List[List[Dict]]:

        if not candidates:
            return []

        entities_mem = document_memory.get("entities", {})

        cand_embeddings = {}
        for cand in candidates:
            name = cand["name"]
            mem_entry = entities_mem.get(name, {})
            emb = mem_entry.get("embedding")
            if emb is not None:
                if not isinstance(emb, np.ndarray):
                    emb = np.array(emb, dtype=float)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    cand_embeddings[name] = emb / norm

        has_emb = [c for c in candidates if c["name"] in cand_embeddings]
        no_emb  = [c for c in candidates if c["name"] not in cand_embeddings]

        clusters: List[List[Dict]] = []
        centroids: List[np.ndarray] = []

        for cand in has_emb:
            emb = cand_embeddings[cand["name"]]

            best_idx   = -1
            best_score = 0.0

            for i, centroid in enumerate(centroids):
                score = float(np.dot(emb, centroid))
                if score > best_score:
                    best_score = score
                    best_idx   = i

            if best_score >= threshold and best_idx >= 0:
                clusters[best_idx].append(cand)
                n = len(clusters[best_idx])
                centroids[best_idx] = (centroids[best_idx] * (n - 1) + emb) / n
                norm = np.linalg.norm(centroids[best_idx])
                if norm > 0:
                    centroids[best_idx] /= norm
            else:
                clusters.append([cand])
                centroids.append(emb.copy())

        for cand in no_emb:
            clusters.append([cand])

        clusters.sort(key=lambda cl: sum(c["count"] for c in cl), reverse=True)

        for i, cl in enumerate(clusters):
            names = [c["name"] for c in cl]
            total = sum(c["count"] for c in cl)
            print(f"  Cluster {i+1}: {names} (total count: {total})")

        return clusters

    
    def _generate_entity_proposals(
    self,
    candidates: List[Dict],
    ontology_loader,
    document_memory: Dict,
    logger=None,
    ) -> List[Dict]:
        existing_classes = ontology_loader.get_all_classes()

        clust = self._cluster_candidates(candidates, document_memory)

        clusters = [cl for cl in clust if self._should_propose_cluster(cl)]

        proposals = []
        for cluster in clusters:
            try:
                member_payload = []
                total_count = 0
                total_conf = 0.0

                for c in cluster:
                    member_payload.append({
                        "name": c["name"],
                        "count": c["count"],
                        "avg_confidence": c["avg_confidence"],
                        "dominant_suggestion": c.get("dominant_suggestion"),
                        "suggested_types": c.get("suggested_types", []),
                        "sources": c.get("sources", []),
                    })
                    total_count += c["count"]
                    total_conf += c["avg_confidence"]

                cluster_info = {
                    "members": member_payload,
                    "cluster_size": len(cluster),
                    "cluster_total_count": total_count,
                    "cluster_avg_confidence": round(total_conf / len(cluster), 3)
                }

                proposal = self._llm_propose_class(
                    cluster_info,
                    existing_classes,
                )
                
                proposal["candidate"] = cluster_info
                proposal["proposal_id"] = str(uuid4())
                proposal["proposal_type"] = "entity_class"
                proposal["status"] = "pending"
                proposal["cluster_members"] = [m["name"] for m in member_payload]

                proposals.append(proposal)

            except Exception as e:
                print(
                    f"  [OntologyProposer] Failed to generate proposal "
                    f"for cluster {[c['name'] for c in cluster]}: {e}"
                )
                if logger:
                    logger.warning("OntologyProposerAgent", "proposal_generation_failed", {
                        "cluster": [c["name"] for c in cluster],
                        "error": str(e),
                    })

        return proposals
    

    def _generate_relation_proposals(
        self,
        candidates:      List[Dict],
        ontology_loader,
        logger=None,
    ) -> List[Dict]:
        existing_relations = ontology_loader.get_all_relations()
        existing_classes   = ontology_loader.get_all_classes()
        proposals = []

        for cand in candidates:
            try:
                proposal = self._llm_propose_relation(
                    cand, existing_relations, existing_classes
                )
                proposal["proposal_id"]   = str(uuid4())
                proposal["proposal_type"] = "relation"
                proposal["status"]        = "pending"
                proposal["candidate"]     = cand
                proposals.append(proposal)

            except Exception as e:
                print(f"  [OntologyProposer] Failed relation proposal "
                      f"for '{cand['relation']}': {e}")

        return proposals

    def _llm_propose_class(
    self,
    cluster_info: Dict,
    existing_classes: List[str],
) -> Dict:

        members = cluster_info["members"]

        prompt = f"""
    You are an ontology engineer working on a numerical methods knowledge graph.

    A semantic cluster of related extracted entities has been identified.

    CLUSTER SUMMARY
    ---------------
    Cluster size: {cluster_info['cluster_size']}
    Total count: {cluster_info['cluster_total_count']}
    Average confidence across cluster: {cluster_info['cluster_avg_confidence']}

    CLUSTER MEMBERS
    ---------------
    {json.dumps(members, indent=2)}

    EXISTING ONTOLOGY CLASSES
    -------------------------
    {json.dumps(existing_classes, indent=2)}

    TASK
    ----
    Propose ONE ontology class that best represents the ENTIRE cluster.

    The class must:
    1. Be broad enough to cover all current cluster members.
    2. Also be broad enough to cover future variants that belong to the same concept family.
    3. Avoid overfitting to one member name.
    4. Avoid being too generic.
    5. Fit naturally into a numerical methods ontology.

    Important:
    - Use the cluster as the semantic unit.
    - Do NOT base the class name on only one member.
    - Prefer a conceptual family name over a surface-form name.

    Examples:
    - A cluster containing "single precision", "double precision",
    "machine precision", "machine epsilon" should not become
    "DoublePrecision". A better class might be "FloatingPointConcept".
    - A cluster containing "Jacobi method" and "Gauss-Seidel method"
    should not become "JacobiMethod". A better class might be
    "IterativeLinearSolver".

    OUTPUT RULES
    ------------
    1. Class name must be concise CamelCase.
    2. Parent class must come from EXISTING ONTOLOGY CLASSES.
    3. Write a one-sentence description.
    4. Justify why this needs a new class rather than an existing class.
    5. Output ONLY valid JSON.
    6. No markdown fences, no commentary.

    OUTPUT SCHEMA
    --------------
    {{
    "proposed_class_name": "...",
    "parent_class": "...",
    "description": "...",
    "justification": "..."
    }}
    """

        response = self.llm.invoke(prompt)
        content = response.content.strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)

    def _llm_propose_relation(
        self,
        candidate:          Dict,
        existing_relations: List[str],
        existing_classes:   List[str],
    ) -> Dict:

        prompt = f"""\
You are an ontology engineer working on a numerical methods knowledge graph.

A new relation type has been observed repeatedly in the literature.

CANDIDATE RELATION:
  Source entity       : {candidate['source']}
  Relation label      : {candidate['relation']}
  Target entity       : {candidate['target']}
  Occurrences         : {candidate['count']}
  Avg confidence      : {candidate['avg_confidence']}
  Suggested relations : {candidate.get('suggested_relations', [])}

EXISTING RELATION TYPES:
{json.dumps(existing_relations, indent=2)}

EXISTING CLASSES (for domain/range):
{json.dumps(existing_classes, indent=2)}

TASK
----
Propose a formal relation type for this observed relation.

Rules:
1. Relation name must be snake_case verb phrase.
   e.g. "stabilizes", "extends", "approximates_order_of"
2. Consider the suggested relation names provided — use one if it is
   appropriate, or propose a better name if none fits well.
3. Specify domain and range from EXISTING CLASSES.
4. Write a one-sentence description.
5. Output ONLY valid JSON — no markdown fences.

OUTPUT SCHEMA:
{{
  "proposed_relation_name": "...",
  "domain":                 "...",
  "range":                  "...",
  "description":            "..."
}}
"""
        response = self.llm.invoke(prompt)
        content  = response.content.strip()
        content  = re.sub(r"^```json\s*", "", content)
        content  = re.sub(r"\s*```$",     "", content)
        return json.loads(content)


    def _write_proposals(
        self,
        entity_proposals:   List[Dict],
        relation_proposals: List[Dict],
        ontology_loader,
    ) -> Path:

        existing = sorted(self.proposals_dir.glob("proposals_v*.json"))
        next_version = len(existing) + 1

        path = self.proposals_dir / f"proposals_v{next_version:03d}.json"

        payload = {
            "metadata": {
                "created_at":        datetime.now().isoformat(),
                "proposals_version": next_version,
                "parent_ontology":   ontology_loader.core.get(
                    "ontology_metadata", {}
                ).get("ontology_version", "0.0"),
                "status":            "pending_review",
            },
            "entity_proposals":   entity_proposals,
            "relation_proposals": relation_proposals,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return path

    def load_proposals(self, proposals_path: str) -> Dict:
        with open(proposals_path, "r") as f:
            return json.load(f)

    def _save_proposals(self, proposals: Dict, proposals_path: str) -> None:
        with open(proposals_path, "w") as f:
            json.dump(proposals, f, indent=2)

    def review_cli(self, proposals_path: str) -> None:
        proposals = self.load_proposals(proposals_path)
        all_props = (
            proposals["entity_proposals"] +
            proposals["relation_proposals"]
        )
        pending = [p for p in all_props if p["status"] == "pending"]

        if not pending:
            print("No pending proposals to review.")
            return

        print(f"\n{'='*60}")
        print(f"ONTOLOGY PROPOSER — {len(pending)} proposal(s) to review")
        print(f"{'='*60}\n")

        for i, prop in enumerate(pending, 1):
            self._print_proposal(prop, i, len(pending))

            while True:
                choice = input(
                    "\n  [A]ccept  [R]eject  [M]odify class name  > "
                ).strip().upper()

                if choice == "A":
                    prop["status"]      = "accepted"
                    prop["reviewed_at"] = datetime.now().isoformat()
                    print(f"  ✓ Accepted: {prop.get('proposed_class_name') or prop.get('proposed_relation_name')}")
                    break

                elif choice == "R":
                    reason = input("  Reason (optional): ").strip()
                    prop["status"]        = "rejected"
                    prop["reject_reason"] = reason
                    prop["reviewed_at"]   = datetime.now().isoformat()
                    print("  ✗ Rejected.")
                    break

                elif choice == "M":
                    if prop["proposal_type"] == "entity_class":
                        new_name   = input("  New class name (CamelCase): ").strip()
                        new_parent = input("  Parent class: ").strip()
                        prop["proposed_class_name"] = new_name
                        prop["parent_class"]        = new_parent
                    else:
                        new_name = input("  New relation name (snake_case): ").strip()
                        prop["proposed_relation_name"] = new_name
                    prop["status"]      = "accepted"
                    prop["modified"]    = True
                    prop["reviewed_at"] = datetime.now().isoformat()
                    print("  ✓ Modified and accepted.")
                    break

                else:
                    print("  Please enter A, R, or M.")

            self._save_proposals(proposals, proposals_path)

        print(f"\n[OntologyProposer] Review complete. "
              f"Results saved → {proposals_path}")

    def _print_proposal(self, prop: Dict, index: int, total: int) -> None:

        print(f"\n── Proposal {index}/{total} "
              f"[{prop['proposal_type']}] ──────────────────────")

        if prop["proposal_type"] == "entity_class":
            cluster = prop["candidate"]
            print(f"  Cluster size       : {cluster['cluster_size']}")
            print(f"  Total evidence     : {cluster['cluster_total_count']}")
            print(f"  Avg confidence     : {cluster['cluster_avg_confidence']}")
            print("  Cluster members:")
            all_sources = []
            for member in cluster["members"]:
                print(
                    f"    - {member['name']} "
                    f"(count={member['count']}, "
                    f"conf={member['avg_confidence']})"
                )
                all_sources.extend(member.get("sources", []))
            print("  ── LLM Proposal ──")
            print(f"  Proposed class     : {prop.get('proposed_class_name')}")
            print(f"  Parent class       : {prop.get('parent_class')}")
            print(f"  Description        : {prop.get('description')}")
            print(f"  Justification      : {prop.get('justification')}")
            print(f"  Example sources    : {list(dict.fromkeys(all_sources))[:3]}")
        else:
            cand = prop["candidate"]
            print(f"  Observed relation  : {cand['source']} → "
                  f"{cand['relation']} → {cand['target']}")
            print(f"  Occurrences        : {cand['count']}")
            print(f"  Avg confidence     : {cand['avg_confidence']}")
            print("  ── LLM Proposal ──")
            print(f"  Proposed name      : {prop.get('proposed_relation_name')}")
            print(f"  Domain → Range     : {prop.get('domain')} → {prop.get('range')}")
            print(f"  Description        : {prop.get('description')}")
            print(f"  Suggested names    : {cand.get('suggested_relations', [])}")
            print(f"  Example sources    : {cand.get('sources', [])[:3]}")

    def get_pending_proposals(self, proposals_path: str) -> List[Dict]:

        proposals = self.load_proposals(proposals_path)
        return [
            p for p in (
                proposals["entity_proposals"] +
                proposals["relation_proposals"]
            )
            if p["status"] == "pending"
        ]

    def accept_proposal(self, proposal_id: str, proposals_path: str) -> None:

        self._update_proposal_status(
            proposal_id, proposals_path,
            {"status": "accepted", "reviewed_at": datetime.now().isoformat()}
        )

    def reject_proposal(
        self, proposal_id: str, proposals_path: str, reason: str = ""
    ) -> None:

        self._update_proposal_status(
            proposal_id, proposals_path,
            {"status": "rejected", "reject_reason": reason,
             "reviewed_at": datetime.now().isoformat()}
        )

    def modify_and_accept(
        self,
        proposal_id:    str,
        proposals_path: str,
        new_name:       str,
        new_parent:     str = None,
    ) -> None:

        updates = {
            "status":     "accepted",
            "modified":   True,
            "reviewed_at": datetime.now().isoformat(),
        }
        proposals = self.load_proposals(proposals_path)
        for section in ("entity_proposals", "relation_proposals"):
            for p in proposals[section]:
                if p["proposal_id"] == proposal_id:
                    if p["proposal_type"] == "entity_class":
                        p["proposed_class_name"] = new_name
                        if new_parent:
                            p["parent_class"] = new_parent
                    else:
                        p["proposed_relation_name"] = new_name
                    p.update(updates)
        self._save_proposals(proposals, proposals_path)

    def _update_proposal_status(
        self, proposal_id: str, proposals_path: str, updates: Dict
    ) -> None:
        proposals = self.load_proposals(proposals_path)
        for section in ("entity_proposals", "relation_proposals"):
            for p in proposals[section]:
                if p["proposal_id"] == proposal_id:
                    p.update(updates)
        self._save_proposals(proposals, proposals_path)


    def apply_accepted_proposals(
        self,
        proposals_path:     str,
        ontology_loader,
        candidate_pool:     Dict = None,
        graph_builder_agent = None,
        document_memory:    Dict = None,
        logger=None,
    ) -> Optional[Tuple[Path, Dict]]:

        proposals = self.load_proposals(proposals_path)
        accepted_entities  = [
            p for p in proposals["entity_proposals"]
            if p["status"] == "accepted"
        ]
        accepted_relations = [
            p for p in proposals["relation_proposals"]
            if p["status"] == "accepted"
        ]

        if not accepted_entities and not accepted_relations:
            print("[OntologyProposer] No accepted proposals — nothing to apply.")
            return None

        new_subclasses = {}
        for p in accepted_entities:
            class_name = p["proposed_class_name"]
            if class_name in new_subclasses:
                new_subclasses[class_name]["evidence_count"] += p["candidate"]["cluster_total_count"]
                continue
            new_subclasses[class_name] = {
                "parent":         p.get("parent_class", "MathematicalObject"),
                "description":    p.get("description", ""),
                "added_by":       "OntologyProposer",
                "added_at":       p.get("reviewed_at", datetime.now().isoformat()),
                "evidence_count": p["candidate"]["cluster_total_count"],
                "proposal_id":    p["proposal_id"],
            }

        new_relations = {}
        for p in accepted_relations:
            rel_name = p["proposed_relation_name"]
            if rel_name in new_relations:
                new_relations[rel_name]["evidence_count"] += p["candidate"]["count"]
                continue
            new_relations[rel_name] = {
                "domain":         p.get("domain", "MathematicalObject"),
                "range":          p.get("range",  "MathematicalObject"),
                "description":    p.get("description", ""),
                "added_by":       "OntologyProposer",
                "added_at":       p.get("reviewed_at", datetime.now().isoformat()),
                "evidence_count": p["candidate"]["count"],
                "proposal_id":    p["proposal_id"],
            }

        cluster_map: Dict[str, List[str]] = {}

        if candidate_pool is not None:
            pool_entities = candidate_pool.get("entities", {})

            for p in accepted_entities:
                class_name   = p["proposed_class_name"]
                member_names = p.get("cluster_members", [])
                cluster_map.setdefault(class_name, []).extend(
                    m for m in member_names if m not in cluster_map.get(class_name, [])
                )

                for member_name in member_names:
                    cand_data = pool_entities.get(member_name)

                    if cand_data:
                        scores  = cand_data.get("confidence_scores", [])
                        sources = cand_data.get("sources", [])
                        avg_conf = sum(scores) / len(scores) if scores else 0.0
                        best_conf = max(scores) if scores else avg_conf

                        if graph_builder_agent is not None and sources:
                            entity = {
                                "entity_id":          f"proposed_{member_name}",
                                "name":               member_name,
                                "type":               class_name,
                                "suggested_type":     None,
                                "source_chunk_id":    sources[0],
                                "evidence_chunk_ids": sources,
                                "confidence":         best_conf,
                            }
                            try:
                                graph_builder_agent._add_node(entity, sources[-1])
                                node = graph_builder_agent.graph["nodes"].get(member_name, {})
                                node["sources"]           = list(set(node.get("sources", [])) | set(sources))
                                node["confidence_scores"] = list(scores)
                                print(f"  [graph insert] '{member_name}' → {class_name}")
                            except Exception as e:
                                print(f"  [graph insert FAILED] '{member_name}': {e}")

                        if document_memory is not None:
                            mem = document_memory.setdefault("entities", {})
                            if member_name not in mem:
                                mem[member_name] = {
                                    "type":              class_name,
                                    "confidence_scores": scores,
                                    "sources":           sources,
                                }
                            else:
                                mem[member_name]["type"] = class_name

                        del pool_entities[member_name]
                        print(f"  [pool cleanup] removed '{member_name}' → {class_name}")

            pool_relations = candidate_pool.get("relations", [])
            for p in accepted_relations:
                cand = p["candidate"]
                pool_relations[:] = [
                    r for r in pool_relations
                    if not (r["source"] == cand["source"]
                            and r["relation"] == cand["relation"]
                            and r["target"] == cand["target"])
                ]

        existing = ontology_loader.extensions
        merged_subclasses = {**existing.get("subclasses", {}), **new_subclasses}
        merged_relations  = {**existing.get("relation_extensions", {}), **new_relations}

        cur_ver = existing.get("extension_metadata", {}).get("extension_version", "0.0")
        major, minor = cur_ver.split(".")
        new_ver = f"{major}.{int(minor) + 1}"

        new_ext_path = (
            self.ontology_dir /
            f"ontology_extensions_v{new_ver.replace('.', '_')}.json"
        )

        new_extensions = {
            "extension_metadata": {
                "extension_version":   new_ver,
                "parent_core_version": ontology_loader.core.get(
                    "ontology_metadata", {}
                ).get("ontology_version", "0.0"),
                "description": (
                    f"Extensions added by OntologyProposer "
                    f"from proposals_v{proposals['metadata']['proposals_version']:03d}"
                ),
                "created_at": datetime.now().isoformat(),
            },
            "subclasses":          merged_subclasses,
            "relation_extensions": merged_relations,
        }

        with open(new_ext_path, "w") as f:
            json.dump(new_extensions, f, indent=2)

        proposals["metadata"]["status"]            = "applied"
        proposals["metadata"]["applied_at"]        = datetime.now().isoformat()
        proposals["metadata"]["extension_version"] = new_ver
        self._save_proposals(proposals, proposals_path)

        print(f"\n[OntologyProposer] New extensions written → {new_ext_path}")
        print(f"  New classes   : {len(new_subclasses)}")
        print(f"  New relations : {len(new_relations)}")

        if logger:
            logger.info("OntologyProposerAgent", "extensions_applied", {
                "path":          str(new_ext_path),
                "new_classes":   list(new_subclasses.keys()),
                "new_relations": list(new_relations.keys()),
                "version":       new_ver,
            })

        return new_ext_path, cluster_map