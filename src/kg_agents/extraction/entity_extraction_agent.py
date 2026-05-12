from uuid import uuid4
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from kg_agents.extraction.schemas import EntityExtractionOutput
from kg_agents.utils.llm_utils import invoke_with_backoff


ENTITY_PROMPT = """\
You are a mathematical knowledge extraction agent.
Domain: Numerical Methods and Partial Differential Equations.

TASK
----
Extract entities from the PRIMARY_CHUNK.

CHUNK HEADING (section this chunk belongs to):
<<CHUNK_HEADING>>

PRIMARY_CHUNK:
<<PRIMARY_CHUNK>>

CONTEXT_CHUNKS:
<<CONTEXT_CHUNKS>>

DOCUMENT_MEMORY:
<<DOCUMENT_MEMORY>>

ONTOLOGY_CLASSES:
<<ONTOLOGY_CLASSES>>

RETRY_FEEDBACK:
<<RETRY_FEEDBACK>>

ENTITY SELECTION RULES
-----------------------
Extract ONLY specific, named mathematical concepts such as:
  - Named numerical methods  (e.g. "Forward Euler method", "Runge-Kutta method")
  - Specific problem types   (e.g. "initial value problem", "boundary value problem")
  - Named theorems / lemmas  (e.g. "Intermediate Value Theorem", "Lax equivalence theorem")
  - Specific properties      (e.g. "A-stability", "second-order convergence")
  - Named operators / spaces / functions (e.g. "Laplacian operator", "Sobolev space")
  - Named algorithms         (e.g. "Gaussian elimination", "Jacobi iteration")
  - Specific defined mathematical objects / concepts

DO NOT extract:
  - Single letters used as variables: f, x, h, n, c, a, b, s, u, v, t, k, m
  - Mathematical notation fragments: C[a,b], O(h^q), f'(x), h^2, e=O(h)
  - Generic nouns: "method", "scheme", "approach", "problem", "equation", "result"
  - Chapter or book titles, author names, institution names
  - Vague subject areas: "Numerical Methods", "PDEs", "Linear Algebra"
  - Table values, numerical constants, or computed results

HEADING RULE (important):
  The CHUNK HEADING often names the primary mathematical concept being introduced.
  If the heading names a theorem, lemma, definition, method, or algorithm,
  extract it as an entity even if the chunk body contains only its formal statement.
  Examples:
    Heading "Intermediate Value Theorem" → extract "Intermediate Value Theorem" as Theorem
    Heading "Big-O and Theta Notation" → extract "Big-O notation" and "Theta notation" as Definition

RULES
-----
1. Extract entities appearing in PRIMARY_CHUNK only.
2. CONTEXT_CHUNKS and DOCUMENT_MEMORY are for resolving references only.
3. Classify each entity using ONTOLOGY_CLASSES.
4. Unknown types → set type to NEW_TYPE and suggested_type to a short label.
5. Do NOT hallucinate entities not in the text or heading.
6. Keep entity names as written in source — use full descriptive names, not symbols.
7. If RETRY_FEEDBACK is non-empty, correct accordingly.

CONFIDENCE CALIBRATION — follow these examples precisely:
  0.90 — "The Forward Euler method is defined as y_{n+1} = y_n + h*f(t_n, y_n)"
          → "Forward Euler method" at 0.90 (explicitly named and defined)
  0.75 — "This explicit scheme suffers from stability restrictions"
          → a specific scheme previously named in context at 0.75
  0.50 — "One can show that the truncation error is O(h^2)"
          → "truncation error" at 0.50 (mentioned but not named or defined)
  0.20 — "...similar methods exist for stiff problems..."
          → weak implied reference — prefer to OMIT

When in doubt: ask whether the entity is explicitly named in THIS chunk (0.90)
or only implied/referenced (0.75). NEVER assign 0.9 to a generic concept,
subject area, single variable letter, or notation fragment.

OUTPUT — return ONLY valid JSON, no markdown fences, no commentary:
{
  "entities": [
    {
      "name": "...",
      "type": "...",
      "suggested_type": null,
      "confidence": 0.0
    }
  ]
}
If no entities qualify, return: {"entities": []}
"""


class EntityExtractionAgent:

    def __init__(self, ontology_loader):
        self.ontology = ontology_loader
        self.llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser   = JsonOutputParser(pydantic_object=EntityExtractionOutput)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:

        logger   = state.get("logger")
        chunk_id = state.get("chunk_id", "")

        prompt = (
            ENTITY_PROMPT
            .replace("<<CHUNK_HEADING>>",    str(state.get("chunk_heading", "")))
            .replace("<<PRIMARY_CHUNK>>",    str(state["primary_chunk"]))
            .replace("<<CONTEXT_CHUNKS>>",   str(state.get("context_chunks", [])))
            .replace("<<DOCUMENT_MEMORY>>",  str(state.get("document_memory", {})))
            .replace("<<ONTOLOGY_CLASSES>>", str(self.ontology.get_all_classes()))
            .replace("<<RETRY_FEEDBACK>>",   str(state.get("retry_feedback", "")))
        )

        response = invoke_with_backoff(
            self.llm, prompt,
            logger=logger, agent="EntityExtractionAgent", chunk_id=chunk_id
        )

        if response is None:
            state["entities"] = []
            state["_rate_limited"] = True
            return state

        print(response.content)
        data = self.parser.parse(response.content)

        entities = []
        for ent in data["entities"]:
            entities.append({
                "entity_id":          str(uuid4()),
                "name":               ent["name"],
                "type":               ent["type"],
                "suggested_type":     ent.get("suggested_type"),
                "source_chunk_id":    chunk_id,
                "evidence_chunk_ids": [chunk_id],
                "confidence":         ent["confidence"],
            })

        state["entities"] = entities
        return state