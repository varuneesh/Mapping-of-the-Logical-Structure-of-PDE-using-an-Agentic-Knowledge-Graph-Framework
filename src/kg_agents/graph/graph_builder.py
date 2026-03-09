class GraphBuilderAgent:

    def __init__(self, graph_memory):

        self.graph = graph_memory
        
    def run(self, state):

        entities = state["consistent_entities"]
        relations = state["consistent_relationships"]

        for entity in entities:

            self._add_node(entity, state["chunk_id"])

        for rel in relations:

            self._add_edge(rel, state["chunk_id"])
            
        print(self.graph)

        return state
    
    def _add_node(self, entity, chunk_id):

        name = entity["name"]
        entity_type = entity["type"]

        nodes = self.graph["nodes"]

        if name not in nodes:

            nodes[name] = {
                "type": entity_type,
                "sources": [chunk_id]
            }

        else:

            if chunk_id not in nodes[name]["sources"]:
                nodes[name]["sources"].append(chunk_id)
                
    def _add_edge(self, rel, chunk_id):

        edges = self.graph["edges"]

        for edge in edges:

            if (
                edge["source"] == rel["source"]
                and edge["relation"] == rel["relation"]
                and edge["target"] == rel["target"]
            ):

                edge["confidence_scores"].append(rel["confidence"])

                if chunk_id not in edge["sources"]:
                    edge["sources"].append(chunk_id)

                return

        edges.append({

            "source": rel["source"],
            "relation": rel["relation"],
            "target": rel["target"],
            "confidence_scores": [rel["confidence"]],
            "sources": [chunk_id]

        })