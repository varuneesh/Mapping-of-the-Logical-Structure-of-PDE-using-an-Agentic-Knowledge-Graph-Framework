import numpy as np
from typing import List, Dict

import json
from pathlib import Path


class EntityIndex:
    """
    Lightweight in-memory index for entity embeddings.

    Responsibilities:
    - Maintain a centroid embedding per canonical entity
    - Support fast cosine similarity search
    """

    def __init__(self):
        self.names: List[str] = []
        self.embeddings: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}

    # =========================
    # ADD / UPDATE (CENTROID)
    # =========================
    def add_entity(self, name: str, embedding: np.ndarray, entity_type=None):
        """
        Add or update entity embedding using running centroid.
        """

        if embedding is None:
            return

        # Ensure normalized input
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # ---- New entity ----
        if name not in self.embeddings:
            self.names.append(name)
            self.embeddings[name] = embedding
            self.counts[name] = 1
            return

        # ---- Update centroid ----
        count = self.counts[name]
        current = self.embeddings[name]

        new_emb = (current * count + embedding) / (count + 1)

        # Normalize centroid
        norm = np.linalg.norm(new_emb)
        if norm > 0:
            new_emb = new_emb / norm

        self.embeddings[name] = new_emb
        self.counts[name] += 1

    # =========================
    # SEARCH
    # =========================
    def search(self, query_embedding: np.ndarray, top_k: int = 10):
        """
        Return top_k most similar entities using cosine similarity.
        """

        if not self.names:
            return []

        # Normalize query (important)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        results = []

        for name in self.names:
            emb = self.embeddings[name]

            # Dot product = cosine similarity (since normalized)
            score = float(np.dot(query_embedding, emb))

            results.append({
                "name": name,
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def save(self, path: str) -> None:
        """Persist index to disk — embeddings as .npz, metadata as .json."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings as numpy archive
        if self.embeddings:
            np.savez(
                path / "embeddings.npz",
                **{name.replace("/", "_"): emb   # npz keys can't have slashes
                for name, emb in self.embeddings.items()}
            )
        
        # Save names, counts, and name→key mapping as JSON
        metadata = {
            "names":  self.names,
            "counts": self.counts,
            # Map original name → sanitised npz key
            "key_map": {name: name.replace("/", "_") for name in self.names}
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[EntityIndex] Saved {len(self.names)} entities to {path}")

    @classmethod
    def load(cls, path: str) -> "EntityIndex":
        """Load index from disk. Returns empty index if path doesn't exist."""
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