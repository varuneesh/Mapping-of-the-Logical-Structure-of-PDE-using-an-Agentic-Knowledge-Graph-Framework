import numpy as np
from typing import Dict, Any
from openai import OpenAI

from kg_agents.alignment.entity_index import EntityIndex
from kg_agents.config.settings import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

class AlignmentAgent:

    def __init__(self):

        self.index = EntityIndex()

    def run(self, state: Dict[str, Any]):

        entities = state["entities"]
        relationships = state["relationships"]
        memory = state["document_memory"]

        alias_map = {}

        for entity in entities:

            canonical = self._align_entity(entity, state)

            alias_map[entity["name"]] = canonical
            print(f"Mapping {entity['name']} → {canonical}")

        # rewrite entities
        state["entities"] = self._canonicalize_entities(
            entities,
            alias_map
        )

        # rewrite relationships
        state["relationships"] = self._rewrite_relationships(
            relationships,
            alias_map
        )

        # update document memory
        self._update_memory(
            state["entities"],
            memory,
            state["chunk_id"],
            state
        )

        print("Alignment complete.")
        print("Alias map:", alias_map)

        return state
    
    def _align_entity(self, entity, state):

        name = entity["name"]
        entity_type = entity["type"]

        embedding = self._generate_embedding(entity, state)

        candidates = self.index.search(
            embedding,
            entity_type
        )

        if not candidates:
            return name

        best = candidates[0]

        if best["score"] > 0.9:

            print(f"Alias detected: {name} → {best['name']}")
            return best["name"]

        if best["score"] > 0.75:

            if self._llm_verify(name, best["name"]):

                print(f"LLM confirmed alias: {name} → {best['name']}")
                return best["name"]

        return name
    
    def _generate_embedding(self, entity, state):

        text = f"""
            Entity: {entity['name']}
            Section: {state.get("chunk_heading", "")}
            Context: {state["primary_chunk"][:300]}
            """

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        return np.array(response.data[0].embedding)
    
    def _llm_verify(self, entity_a, entity_b):

        prompt = f"""
    Are the following two entities referring to the same concept?

    Entity A: {entity_a}
    Entity B: {entity_b}

    Answer only YES or NO.
    """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content.strip()

        return answer.upper().startswith("YES")
    
    def _canonicalize_entities(self, entities, alias_map):

        groups = {}

        for e in entities:

            canonical = alias_map[e["name"]]

            if canonical not in groups:
                groups[canonical] = []

            groups[canonical].append(e)

        canonical_entities = []

        for canonical, members in groups.items():

            # choose representative entity
            best = max(
                members,
                key=lambda x: x["confidence"]
            )

            new_entity = best.copy()
            new_entity["name"] = canonical

            canonical_entities.append(new_entity)

        return canonical_entities
    
    def _rewrite_relationships(self, relationships, alias_map):

        rewritten = []

        for r in relationships:

            new_r = r.copy()

            new_r["source"] = alias_map.get(
                r["source"],
                r["source"]
            )

            new_r["target"] = alias_map.get(
                r["target"],
                r["target"]
            )

            rewritten.append(new_r)

        return rewritten
    
    def _update_memory(self, entities, memory, chunk_id, state):

        if "entities" not in memory:
            memory["entities"] = {}

        for entity in entities:

            name = entity["name"]
            entity_type = entity["type"]

            embedding = self._generate_embedding(entity, state)
            print(memory)

            if name not in memory["entities"]:
                
                memory["entities"][name] = {
                    "type": entity_type,
                    "confidence_scores": [entity["confidence"]],
                    "sources": [chunk_id]
                }
                
                self.index.add_entity(
                    name,
                    embedding,
                    entity_type
                )

            else:
                
                memory["entities"][name]["confidence_scores"].append(
                    entity["confidence"]
                )

                memory["entities"][name]["sources"].append(
                    chunk_id
                )