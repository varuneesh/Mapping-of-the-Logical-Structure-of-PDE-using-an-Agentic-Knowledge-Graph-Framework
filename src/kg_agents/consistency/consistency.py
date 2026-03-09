class ConsistencyAgent:

    def __init__(self, candidate_pool):

        self.entity_threshold = 0.35
        self.relation_threshold = 0.45

        self.candidate_pool = candidate_pool
        
    def run(self, state):

        entities = state["entities"]
        relations = state["relationships"]

        consistent_entities = []
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

        state["consistent_entities"] = consistent_entities
        state["consistent_relationships"] = consistent_relations
        
        print(f"Candidate Pool: {self.candidate_pool}")
        print(state)

        return state
    
    def _entity_is_valid(self, entity):

        if entity["confidence"] < self.entity_threshold:
            return False

        if entity["type"] == "NEW_TYPE":
            return False

        return True
    
    def _relation_is_valid(self, rel, entities):

        if rel["confidence"] < self.relation_threshold:
            return False

        if rel["relation"] == "NEW_RELATION":
            return False

        valid_entities = {e["name"] for e in entities}

        if rel["source"] not in valid_entities:
            return False

        if rel["target"] not in valid_entities:
            return False

        return True
    
    def _store_entity_candidate(self, entity, chunk_id):

        name = entity["name"]

        pool = self.candidate_pool["entities"]

        if name not in pool:

            pool[name] = {
                "count": 1,
                "confidence_scores": [entity["confidence"]],
                "types": [entity["type"]],
                "sources": [chunk_id]
            }

        else:

            pool[name]["count"] += 1
            pool[name]["confidence_scores"].append(entity["confidence"])
            pool[name]["sources"].append(chunk_id)
            
    def _store_relation_candidate(self, rel, chunk_id):

        entry = {
            "source": rel["source"],
            "relation": rel["relation"],
            "target": rel["target"]
        }

        for candidate in self.candidate_pool["relations"]:

            if (
                candidate["source"] == entry["source"]
                and candidate["relation"] == entry["relation"]
                and candidate["target"] == entry["target"]
            ):

                candidate["count"] += 1
                candidate["confidence_scores"].append(rel["confidence"])
                candidate["sources"].append(chunk_id)

                return

        self.candidate_pool["relations"].append({
            "source": entry["source"],
            "relation": entry["relation"],
            "target": entry["target"],
            "count": 1,
            "confidence_scores": [rel["confidence"]],
            "sources": [chunk_id]
        })