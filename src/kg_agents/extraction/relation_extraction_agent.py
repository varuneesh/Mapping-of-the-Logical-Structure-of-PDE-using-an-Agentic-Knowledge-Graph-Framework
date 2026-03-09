# import json
# import re
from uuid import uuid4
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from kg_agents.extraction.schemas import RelationExtractionOutput
# from kg_agents.config.settings import GROQ_API_KEY


class RelationExtractionAgent:

    def __init__(self, ontology_loader):
        self.ontology = ontology_loader
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=RelationExtractionOutput)

        self.prompt = ChatPromptTemplate.from_template(
        """
You are a mathematical relation extraction agent.

Domain: Numerical Methods and PDEs.

TASK:
Extract relationships between entities.

INPUTS
------
{format_instructions}

PRIMARY_CHUNK:
{primary_chunk}

CONTEXT_CHUNKS:
{context_chunks}

ENTITY_LIST:
{entity_list}

DOCUMENT_MEMORY:
{document_memory}

ONTOLOGY_RELATIONS:
{ontology_relations}

RETRY_FEEDBACK:
{retry_feedback}

RULES
-----

1. Extract relations ONLY between entities in ENTITY_LIST.
2. Relations must follow ONTOLOGY_RELATIONS.
3. If a relation does not exist in ONTOLOGY_RELATIONS, label NEW_RELATION instead of inventing a relation type.
4. If a relation does not exist between entities, do NOT hallucinate relations.
    There is no need to force a relation if the text does not support it.
5. Output STRICT JSON.
6. DOCUMENT_MEMORY contains entities previously extracted from this document.
   Use it only to resolve references such as "this method" or "the scheme".
7. RETRY_FEEDBACK appears only if a previous extraction had errors.
   If present, correct the extraction according to the feedback.
8. If a relation clearly exists but is not present in ONTOLOGY_RELATIONS,
    label it NEW_RELATION and propose it in proposed_relations.
    Do NOT invent relations that are not supported by the text.
9. Ensure that the relation respects the ontology domain and range.
10. Source and target must exactly match entity names from ENTITY_LIST.
11. Confidence must be a number between 0 and 1.
12. If the text contains definitions or explanations rather than clear ontology relations, do NOT force them as relations.

Guideline:
0.9 – explicitly mentioned concept
0.75 – strongly implied concept
0.5 – moderately implied concept
0.2 – weak evidence

OUTPUT SCHEMA
-------------

{{
  "relationships": [
    {{
      "source": "...",
      "relation": "...",
      "target": "...",
      "confidence": 0.0
    }}
  ]
}}
"""
        )

    def run(self, state: Dict[str, Any]):

        entities = [e["name"] for e in state["entities"]]
        document_memory = state.get("document_memory", {})
        retry_feedback = state.get("retry_feedback", "")
        format_instructions = self.parser.get_format_instructions()

        prompt = self.prompt.format(
            format_instructions=format_instructions,
            primary_chunk=state["primary_chunk"],
            context_chunks=state["context_chunks"],
            entity_list=entities,
            document_memory=document_memory,
            ontology_relations=self.ontology.get_all_relations(),
            retry_feedback=retry_feedback
        )

        response = self.llm.invoke(prompt)
        print(response.content)

        # # Extract JSON block
        # matches = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", response.content)

        # if matches:
        #     json_str = matches[-1]  # take the final corrected JSON
        # else:
        #     raise ValueError("No JSON block found")

        # data = json.loads(json_str)
        
        data = self.parser.parse(response.content)

        relations = []

        for rel in data["relationships"]:
            relations.append({
                "relation_id": str(uuid4()),
                "source": rel["source"],
                "relation": rel["relation"],
                "target": rel["target"],
                "source_chunk_id": state["chunk_id"],
                "evidence_chunk_ids": [state["chunk_id"]],
                "confidence": rel["confidence"]
            })

        state["relationships"] = relations

        return state