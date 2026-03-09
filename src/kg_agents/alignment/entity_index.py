import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EntityIndex:

    def __init__(self):

        self.embeddings = []
        self.names = []
        self.types = []

    def add_entity(self, name, embedding, entity_type):

        self.names.append(name)
        self.embeddings.append(embedding)
        self.types.append(entity_type)

    def search(self, embedding, entity_type, top_k=3):
        
        print(self.names,self.types,self.embeddings)

        if len(self.embeddings) == 0:
            return []

        # filter by type
        filtered = [
            i for i, t in enumerate(self.types)
            if t == entity_type
        ]

        if not filtered:
            return []

        emb_matrix = np.array(self.embeddings)[filtered]

        sims = cosine_similarity(
            [embedding],
            emb_matrix
        )[0]

        ranked = sorted(
            zip(filtered, sims),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for idx, score in ranked[:top_k]:

            results.append({
                "name": self.names[idx],
                "score": float(score)
            })

        return results