# import json
# import re
from uuid import uuid4
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from kg_agents.extraction.schemas import EntityExtractionOutput
# from kg_agents.config.settings import GROQ_API_KEY


class EntityExtractionAgent:

    def __init__(self, ontology_loader):
        self.ontology = ontology_loader
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=EntityExtractionOutput)

        self.prompt = ChatPromptTemplate.from_template(
        """
You are a mathematical knowledge extraction agent.

Domain: Numerical Methods and Partial Differential Equations.

TASK:
Extract entities from the PRIMARY_CHUNK.

INPUTS
------
{format_instructions}

PRIMARY_CHUNK:
{primary_chunk}

CONTEXT_CHUNKS:
{context_chunks}

DOCUMENT_MEMORY:
{document_memory}

ONTOLOGY_CLASSES:
{ontology_classes}

RETRY_FEEDBACK:
{retry_feedback}

ENTITY SELECTION RULES
----------------------

Extract ONLY meaningful mathematical concepts such as:

- numerical methods
- problem types
- theorems
- mathematical properties
- discretization schemes
- operators
- equations

Generic nouns without specific meaning should NOT be extracted.

RULES
-----
1. Extract entities appearing in PRIMARY_CHUNK.
2. CONTEXT_CHUNKS are only for resolving references.
3. Classify each entity into an ontology class.
4. If an entity does not belong to any ontology class, label its type as NEW_TYPE.
    Do not attempt to invent new ontology classes.
5. Do NOT hallucinate entities.
6. Keep entity names EXACTLY as written.
7. Output STRICT JSON only.
8. DOCUMENT_MEMORY contains entities previously extracted from this document.
   Use it only to resolve references such as "this method" or "the scheme".
9. RETRY_FEEDBACK appears only if a previous extraction had errors.
   If present, correct the extraction according to the feedback.
10. Confidence must be a number between 0 and 1.

Guideline:
0.9 – explicitly mentioned concept
0.75 – strongly implied concept
0.5 – moderately implied concept
0.2 – weak evidence

OUTPUT SCHEMA
-------------
{{
  "entities": [
    {{
      "name": "...",
      "type": "...",
      "confidence": 0.0
    }}
  ]
}}
"""
        )

    def run(self, state: Dict[str, Any]):

        primary_chunk = state["primary_chunk"]
        context_chunks = state["context_chunks"]
        document_memory = state.get("document_memory", {})
        retry_feedback = state.get("retry_feedback", "")

        classes = self.ontology.get_all_classes()
        format_instructions = self.parser.get_format_instructions()

        prompt = self.prompt.format(
            format_instructions=format_instructions,
            primary_chunk=primary_chunk,
            context_chunks=context_chunks,
            document_memory=document_memory,
            ontology_classes=classes,
            retry_feedback=retry_feedback
        )

        # response = self.llm.invoke(prompt)
        # # print(response)
        # print(response.content)

        # # Extract JSON block
        # matches = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", response.content)

        # if matches:
        #     json_str = matches[-1]  # take the final corrected JSON
        # else:
        #     raise ValueError("No JSON block found")

        # data = json.loads(json_str)
        
        response = self.llm.invoke(prompt)
        print(response.content)
        data = self.parser.parse(response.content)

        entities = []

        for ent in data["entities"]:
            entities.append({
                "entity_id": str(uuid4()),
                "name": ent["name"],
                "type": ent["type"],
                "source_chunk_id": state["chunk_id"],
                "evidence_chunk_ids": [state["chunk_id"]],
                "confidence": ent["confidence"]
            })

        state["entities"] = entities

        return state