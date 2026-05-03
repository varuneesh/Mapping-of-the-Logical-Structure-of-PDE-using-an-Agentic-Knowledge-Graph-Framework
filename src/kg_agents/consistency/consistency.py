import json
from pathlib import Path


CANDIDATES_PATH = (
    Path(__file__).parent.parent
    / "ontology_versions" / "candidates" / "ontology_candidates.json"
)


class ConsistencyAgent:
    """
    Filters low-confidence / NEW_TYPE / NEW_RELATION items and routes them
    to the candidate pool for Ontology Proposer review.

    Key fixes vs original:
    - candidate_pool always initialised with {"entities": {}, "relations": []}
      so _store_entity_candidate / _store_relation_candidate never KeyError.
    - Candidate pool is loaded from and saved to ontology_candidates.json so
      evidence accumulates across sessions.
    - suggested_type from NEW_TYPE entities is captured and stored in the pool
      so the Ontology Proposer has a concrete type hint to work with.
    """

    def __init__(self, candidate_pool=None, candidates_path: str = None,
                 ontology_loader=None):

        self.entity_threshold   = 0.35
        self.relation_threshold = 0.45
        self.ontology_loader    = ontology_loader
        self.candidates_path    = (
            Path(candidates_path) if candidates_path else CANDIDATES_PATH
        )

        if candidate_pool is not None:
            self.candidate_pool = {
                "entities":  candidate_pool.get("entities", {}),
                "relations": candidate_pool.get("relations", []),
            }
        else:
            self.candidate_pool = self._load_from_disk()

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _load_from_disk(self) -> dict:
        if self.candidates_path.exists():
            with open(self.candidates_path, "r") as f:
                data = json.load(f)
            return {
                "entities":  data.get("candidate_types", {}),
                "relations": data.get("candidate_relations", []),
            }
        return {"entities": {}, "relations": []}

    def save_to_disk(self) -> None:
        self.candidates_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "description": "Candidate types and relations for Ontology Proposer."
            },
            "candidate_types":     self.candidate_pool["entities"],
            "candidate_relations": self.candidate_pool["relations"],
        }
        with open(self.candidates_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Candidate pool saved → {self.candidates_path}")

    # ------------------------------------------------------------------ #
    #  Main run                                                            #
    # ------------------------------------------------------------------ #

    def run(self, state: dict) -> dict:

        entities  = state["entities"]
        relations = state["relationships"]

        consistent_entities  = []
        consistent_relations = []

        for e in entities:
            if self._entity_is_valid(e):
                consistent_entities.append(e)
            else:
                self._store_entity_candidate(e, state["chunk_id"])

        for r in relations:
            if self._relation_is_valid(r, consistent_entities):
                consistent_relations.append(r)
            else:
                self._store_relation_candidate(r, state["chunk_id"])

        state["consistent_entities"]      = consistent_entities
        state["consistent_relationships"] = consistent_relations

        print(f"Consistent — entities: {len(consistent_entities)} | "
              f"relations: {len(consistent_relations)}")
        print(f"Candidate pool — entities: {len(self.candidate_pool['entities'])} | "
              f"relations: {len(self.candidate_pool['relations'])}")

        self.save_to_disk()
        return state

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def _entity_is_valid(self, entity: dict) -> bool:
        if entity["confidence"] < self.entity_threshold:
            return False
        if entity["type"] == "NEW_TYPE":
            return False
        return True

    def _relation_is_valid(self, rel: dict, entities: list) -> bool:
        if rel["confidence"] < self.relation_threshold:
            return False
        if rel["relation"] == "NEW_RELATION":
            return False
        # Check relation type exists in ontology
        if (self.ontology_loader
                and not self.ontology_loader.relation_exists(rel["relation"])):
            return False
        valid_names = {e["name"] for e in entities}
        if rel["source"] not in valid_names or rel["target"] not in valid_names:
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Candidate pool storage                                              #
    # ------------------------------------------------------------------ #

    def _store_entity_candidate(self, entity: dict, chunk_id: str) -> None:
        name     = entity["name"]
        pool     = self.candidate_pool["entities"]
        suggested = entity.get("suggested_type")   # captured from LLM output

        if name not in pool:
            pool[name] = {
                "count":             1,
                "confidence_scores": [entity["confidence"]],
                "types":             [entity["type"]],
                "suggested_types":   [suggested] if suggested else [],
                "sources":           [chunk_id],
            }
        else:
            pool[name]["count"] += 1
            pool[name]["confidence_scores"].append(entity["confidence"])
            if entity["type"] not in pool[name]["types"]:
                pool[name]["types"].append(entity["type"])
            if suggested and suggested not in pool[name].get("suggested_types", []):
                pool[name].setdefault("suggested_types", []).append(suggested)
            pool[name]["sources"].append(chunk_id)

    def _store_relation_candidate(self, rel: dict, chunk_id: str) -> None:
        pool = self.candidate_pool["relations"]
        suggested = rel.get("suggested_relation")

        for candidate in pool:
            if (
                candidate["source"]   == rel["source"]
                and candidate["relation"] == rel["relation"]
                and candidate["target"]   == rel["target"]
            ):
                candidate["count"] += 1
                candidate["confidence_scores"].append(rel["confidence"])
                candidate["sources"].append(chunk_id)
                if suggested and suggested not in candidate.get("suggested_relations", []):
                    candidate.setdefault("suggested_relations", []).append(suggested)
                return

        pool.append({
            "source":              rel["source"],
            "relation":            rel["relation"],
            "target":              rel["target"],
            "count":               1,
            "confidence_scores":   [rel["confidence"]],
            "sources":             [chunk_id],
            "suggested_relations": [suggested] if suggested else [],
        })