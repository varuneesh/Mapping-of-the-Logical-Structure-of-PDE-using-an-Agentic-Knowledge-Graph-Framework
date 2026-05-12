import numpy as np
from typing import List, Dict

import json
from pathlib import Path


class EntityIndex:

    def __init__(self):
        self.names: List[str] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}

    def add_entity(self, name: str, embedding: np.ndarray, entity_type=None):

        if embedding is None:
            return

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if name not in self.embeddings:
            self.names.append(name)
            self.embeddings[name] = embedding
            self.counts[name] = 1
            return

        count = self.counts[name]
        current = self.embeddings[name]

        new_emb = (current * count + embedding) / (count + 1)

        norm = np.linalg.norm(new_emb)
        if norm > 0:
            new_emb = new_emb / norm

        self.embeddings[name] = new_emb
        self.counts[name] += 1

    def search(self, query_embedding: np.ndarray, top_k: int = 10):

        if not self.names:
            return []

        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        results = []

        for name in self.names:
            emb = self.embeddings[name]

            score = float(np.dot(query_embedding, emb))

            results.append({
                "name": name,
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.embeddings:
            np.savez(
                path / "embeddings.npz",
                **{name.replace("/", "_"): emb   
                for name, emb in self.embeddings.items()}
            )
        
        metadata = {
            "names":  self.names,
            "counts": self.counts,
            "key_map": {name: name.replace("/", "_") for name in self.names}
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[EntityIndex] Saved {len(self.names)} entities to {path}")

    @classmethod
    def load(cls, path: str) -> "EntityIndex":
        path = Path(path)
        index = cls()
        
        metadata_path  = path / "metadata.json"
        embeddings_path = path / "embeddings.npz"
        
        if not metadata_path.exists() or not embeddings_path.exists():
            print(f"[EntityIndex] No existing index at {path} — starting fresh.")
            return index
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        archive  = np.load(embeddings_path)
        key_map  = metadata["key_map"]
        
        index.names  = metadata["names"]
        index.counts = metadata["counts"]
        index.embeddings = {
            name: archive[key_map[name]]
            for name in index.names
            if key_map[name] in archive
        }
        
        print(f"[EntityIndex] Loaded {len(index.names)} entities from {path}")
        return index