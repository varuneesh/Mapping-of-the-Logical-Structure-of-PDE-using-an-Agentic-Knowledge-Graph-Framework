import numpy as np
from typing import Dict, Any, Optional
from openai import OpenAI
from langchain_openai import ChatOpenAI

from kg_agents.alignment.entity_index import EntityIndex
from kg_agents.config.settings import OPENAI_API_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY)


class AlignmentAgent:

    def __init__(self, index_path: str = None, verify_model: str = None):
        self.index_path = index_path
        
        if index_path:
            self.index = EntityIndex.load(index_path)
        else:
            self.index = EntityIndex()
        
        self._verify_model = verify_model or "gpt-4o-mini"
        self._verify_llm   = ChatOpenAI(model=self._verify_model, temperature=0)

    def save_index(self) -> None:
        if self.index_path:
            self.index.save(self.index_path)


    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:

        entities      = state["entities"]
        relationships = state["relationships"]
        memory        = state["document_memory"]

        embedding_cache: Dict[str, np.ndarray] = {}
        for entity in entities:
            embedding_cache[entity["name"]] = self._generate_embedding(entity, state)

        alias_map: Dict[str, str] = {}
        for entity in entities:
            embedding = embedding_cache[entity["name"]]
            canonical = self._align_entity(entity, embedding)
            alias_map[entity["name"]] = canonical
            print(f"  Mapping: {entity['name']} → {canonical}")

        state["entities"]      = self._canonicalize_entities(entities, alias_map)
        state["relationships"] = self._rewrite_relationships(relationships, alias_map)

        self._update_memory(
            state["entities"],
            memory,
            state["chunk_id"],
            embedding_cache,
            alias_map,
        )

        print("Alignment complete.")
        print("Alias map:", alias_map)
        return state


    def _align_entity(self, entity: dict, embedding: np.ndarray) -> str:

        name        = entity["name"]
        norm_name = self._normalise(name)
        for existing_name in self.index.names:
            if self._normalise(existing_name) == norm_name:
                if existing_name != name:
                    print(f"  [align] string-norm match: '{name}' → '{existing_name}'")
                return existing_name

        candidates = self.index.search(embedding, top_k=10)

        if not candidates:
            return name

        scored = []
        for c in candidates:
            if c["name"] == name:
                continue

            emb_score = c["score"]
            overlap   = self._token_overlap_score(name, c["name"])

            final_score = 0.9 * emb_score + 0.1 * overlap

            scored.append({
                "name": c["name"],
                "score": final_score,
                "emb": emb_score,
                "overlap": overlap
            })

        if not scored:
            return name

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        
        if best["score"] > 0.80:
            return best["name"]

        good_candidates = [c for c in scored if c["score"] > 0.6]

        if good_candidates:
            if len(good_candidates) >= 3:
                candidates_for_llm = good_candidates[:3]
            else:
                candidates_for_llm = good_candidates

            chosen = self._llm_select_best(name, candidates_for_llm)
            if chosen:
                return chosen

        return name

    @staticmethod
    def _normalise(s: str) -> str:

        import re
        s = s.lower().strip()
        s = re.sub(r"['\-]", " ", s)  
        s = re.sub(r"[^a-z0-9 ]", "", s)  
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    
    def _token_overlap_score(self, a: str, b: str) -> float:
        a_set = set(self._normalise(a).split())
        b_set = set(self._normalise(b).split())

        if not a_set or not b_set:
            return 0.0

        return len(a_set & b_set) / len(a_set | b_set)


    def _generate_embedding(self, entity: dict, state: Dict[str, Any]) -> np.ndarray:

        name    = self._normalise(entity["name"])
        heading = state.get("chunk_heading", "")
        chunk   = state.get("primary_chunk", "")

        name_text = f"Entity: {name}"
        context   = self._extract_local_context(name, chunk)

        context_text = (
            f"Entity: {name}\n"
            f"Section: {heading}\n"
            f"Context: {context}"
        )

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[name_text, context_text]
        )

        name_emb = np.array(response.data[0].embedding)
        ctx_emb  = np.array(response.data[1].embedding)

        combined = 0.65 * name_emb + 0.35 * ctx_emb
        combined /= np.linalg.norm(combined)

        return combined

    def _extract_local_context(self, name: str, chunk: str) -> str:
        if not chunk:
            return ""

        lines = chunk.split("\n")
        name_lower = name.lower()

        for i, line in enumerate(lines):
            if name_lower in line.lower():
                start = max(0, i - 1)
                end   = min(len(lines), i + 2)
                return "\n".join(lines[start:end])[:400]

        return chunk[:300]
    

    def _llm_select_best(self, name: str, candidates: list) -> Optional[str]:

        options = "\n".join(
            [f"{i+1}. {c['name']}" for i, c in enumerate(candidates)]
        )

        prompt = f"""
        Entity: {name}

        Which of the following refers to the SAME mathematical concept as Entity?

        {options}

        Return ONLY:
        - a number (1,2,3)
        - or NONE
        """

        response = self._verify_llm.invoke(prompt)
        ans = response.content.strip().upper()

        if "NONE" in ans:
            return None

        import re
        match = re.search(r"\d+", ans)
        if not match:
            return None

        idx = int(match.group()) - 1

        if 0 <= idx < len(candidates):
            return candidates[idx]["name"]

        return None


    def _canonicalize_entities(
        self, entities: list, alias_map: Dict[str, str]
    ) -> list:
        
        groups: Dict[str, list] = {}
        for e in entities:
            canonical = alias_map[e["name"]]
            groups.setdefault(canonical, []).append(e)

        canonical_entities = []
        for canonical, members in groups.items():
            best = max(members, key=lambda x: x["confidence"])
            new_entity = best.copy()
            new_entity["name"] = canonical
            canonical_entities.append(new_entity)

        return canonical_entities

    def _rewrite_relationships(
        self, relationships: list, alias_map: Dict[str, str]
    ) -> list:
        rewritten = []
        for r in relationships:
            new_r          = r.copy()
            new_r["source"] = alias_map.get(r["source"], r["source"])
            new_r["target"] = alias_map.get(r["target"], r["target"])
            # Drop self-loops that may appear after canonicalization
            if new_r["source"] == new_r["target"]:
                print(f"  [alignment] dropped self-loop after canonicalization: "
                      f"{new_r['source']} → {new_r['relation']}")
                continue
            rewritten.append(new_r)
        return rewritten

    def _update_memory(
        self,
        entities: list,
        memory: dict,
        chunk_id: str,
        embedding_cache: Dict[str, np.ndarray],
        alias_map: Dict[str, str],
    ) -> None:

        memory.setdefault("entities", {})

        for entity in entities:
            name        = entity["name"]
            entity_type = entity["type"]

            original_name = next(
                (orig for orig, canon in alias_map.items() if canon == name),
                name
            )
            embedding = embedding_cache.get(original_name)

            if name not in memory["entities"]:
                memory["entities"][name] = {
                    "type":              entity_type,
                    "confidence_scores": [entity["confidence"]],
                    "sources":           [chunk_id],
                    "embedding":         embedding,
                    "count":             1
                }
                if embedding is not None:
                    self.index.add_entity(name, embedding, entity_type)
            else:
                node = memory["entities"][name]
                node["confidence_scores"].append(entity["confidence"])
                node["count"] += 1
            
                if chunk_id not in node["sources"]:
                    node["sources"].append(chunk_id)
                    
                if embedding is not None:
                    old = node["embedding"]
                    count = node["count"]

                    new_emb = (old * (count - 1) + embedding) / count
                    new_emb /= np.linalg.norm(new_emb)

                    node["embedding"] = new_emb
                    self.index.add_entity(name, new_emb)                  