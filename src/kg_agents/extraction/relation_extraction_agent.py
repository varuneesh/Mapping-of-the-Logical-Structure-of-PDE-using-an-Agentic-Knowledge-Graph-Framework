from uuid import uuid4
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from kg_agents.extraction.schemas import RelationExtractionOutput
from kg_agents.utils.llm_utils import invoke_with_backoff


RELATION_PROMPT = """\
You are a mathematical relation extraction agent.
Domain: Numerical Methods and PDEs.

TASK
----
Extract relationships between the entities listed in ENTITY_LIST.

CHUNK HEADING (section this chunk belongs to):
<<CHUNK_HEADING>>

PRIMARY_CHUNK:
<<PRIMARY_CHUNK>>

CONTEXT_CHUNKS:
<<CONTEXT_CHUNKS>>

ENTITY_LIST:
<<ENTITY_LIST>>

DOCUMENT_MEMORY:
<<DOCUMENT_MEMORY>>

ONTOLOGY_RELATIONS (name: SOURCE_TYPE → TARGET_TYPE):
<<ONTOLOGY_RELATIONS_WITH_TYPES>>

RETRY_FEEDBACK:
<<RETRY_FEEDBACK>>

RULES
-----
1.  Extract relations ONLY between entities in ENTITY_LIST.
    Source and target must EXACTLY match names from ENTITY_LIST.
2.  Each relation has a fixed direction shown above as SOURCE → TARGET.
    NEVER reverse the direction. Examples:
      CORRECT:   Forward Euler method  --[solves]-->  Initial Value Problem
      WRONG:     Initial Value Problem --[solves]-->  Forward Euler method

      CORRECT:   truncation error --[quantifies]--> Runge-Kutta method
      WRONG:     Runge-Kutta method --[quantifies]--> truncation error

      CORRECT:   Runge-Kutta method --[has_error_analysis]--> truncation error
      WRONG:     truncation error --[has_error_analysis]--> Runge-Kutta method

      CORRECT:   forward difference --[approximates]--> derivative operator
      WRONG:     forward difference --[approximates]--> euler method

3.  Only use a relation if the source type matches SOURCE_TYPE and
    the target type matches TARGET_TYPE (or their subclasses).
4.  If a clear relation exists but no type fits, use NEW_RELATION.
    This is IMPORTANT — do NOT force an existing relation when the fit is poor.
    Using NEW_RELATION is better than using a wrong relation.
    Examples of when to use NEW_RELATION:
      - "Method A generalizes Method B"     → NEW_RELATION, suggested_relation: "generalizes"
      - "Property X is necessary for Y"     → NEW_RELATION, suggested_relation: "necessary_for"
      - "Algorithm A reduces to Algorithm B" → NEW_RELATION, suggested_relation: "reduces_to"
    NEW_RELATION candidates are collected and reviewed for ontology extension.
5.  Do NOT hallucinate relations. If the text does not support a relation,
    return an empty list — this is correct and expected.
6.  Do NOT output self-loops (source == target).
7.  If RETRY_FEEDBACK is non-empty, correct the previous extraction.

VERIFICATION — before outputting each relation, check:
  1. Look up the relation in ONTOLOGY_RELATIONS above
  2. Check: does the SOURCE entity's type match the required SOURCE_TYPE?
  3. Check: does the TARGET entity's type match the required TARGET_TYPE?
  4. If either check fails, do NOT output that relation — try a different
     relation type, or use NEW_RELATION, or omit entirely.

COMMON MISTAKES TO AVOID:
  - applies_to requires NumericalMethod → ProblemType.
    Do NOT use applies_to when the target is a Theorem. Use 'requires' or 'uses' instead.
  - has_error_analysis requires NumericalMethod → ErrorConcept.
    The SOURCE must be a method, not an error. If you want to say an error
    measures a method, use 'quantifies' (ErrorConcept → NumericalMethod).
  - quantifies requires ErrorConcept → NumericalMethod.
    The TARGET must be a method, not another error or a theorem.
  - When unsure, 'uses' and 'depends_on' accept MathematicalObject on both sides,
    so they work for almost any entity type combination.

CONFIDENCE CALIBRATION — follow these examples precisely:
  0.90 — "The Runge-Kutta method solves the initial value problem y'=f(t,y)"
          → Runge-Kutta method --[solves]--> Initial Value Problem at 0.90
  0.75 — "This method is applied to parabolic equations throughout"
          → [some method] --[applies_to]--> [parabolic PDE] at 0.75
  0.50 — "Stability is an important property for time-stepping schemes"
          → [some method] --[has_property]--> Stability at 0.50
  0.20 — Weak implied relation not directly stated — prefer to OMIT

OUTPUT — return ONLY valid JSON, no markdown fences, no commentary:
{
  "relationships": [
    {
      "source": "...",
      "relation": "...",
      "suggested_relation": null,
      "target": "...",
      "confidence": 0.0
    }
  ]
}
If no relations exist, return: {"relationships": []}
"""


def _format_relations_with_types(ontology_loader) -> str:
    """
    Format ontology relations with domain→range so the LLM sees
    the direction constraint explicitly, not just a list of names.
    Merges core and extension relations.
    """
    lines = []
    for name, data in ontology_loader.relations.items():
        domain = data.get("domain", "?")
        range_ = data.get("range",  "?")
        desc   = data.get("description", "")
        lines.append(f"  {name}: {domain} → {range_}   # {desc}")
    return "\n".join(lines)


class RelationExtractionAgent:

    def __init__(self, ontology_loader):
        self.ontology = ontology_loader
        self.llm      = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser   = JsonOutputParser(pydantic_object=RelationExtractionOutput)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:

        logger   = state.get("logger")
        chunk_id = state.get("chunk_id", "")

        entity_names = [e["name"] for e in state["entities"]]

        # No entities → no point calling the LLM
        if not entity_names:
            state["relationships"] = []
            return state

        relations_with_types = _format_relations_with_types(self.ontology)

        prompt = (
            RELATION_PROMPT
            .replace("<<CHUNK_HEADING>>",            str(state.get("chunk_heading", "")))
            .replace("<<PRIMARY_CHUNK>>",            str(state["primary_chunk"]))
            .replace("<<CONTEXT_CHUNKS>>",           str(state.get("context_chunks", [])))
            .replace("<<ENTITY_LIST>>",              str(entity_names))
            .replace("<<DOCUMENT_MEMORY>>",          str(state.get("document_memory", {})))
            .replace("<<ONTOLOGY_RELATIONS_WITH_TYPES>>", relations_with_types)
            .replace("<<RETRY_FEEDBACK>>",           str(state.get("retry_feedback", "")))
        )

        response = invoke_with_backoff(
            self.llm, prompt,
            logger=logger, agent="RelationExtractionAgent", chunk_id=chunk_id
        )

        # Rate limit exhausted — return empty, do NOT loop
        if response is None:
            state["relationships"] = []
            state["_rate_limited"] = True
            return state

        print(response.content)
        data = self.parser.parse(response.content)

        # Build set of ontology class names for hallucination detection
        ontology_classes = set(self.ontology.get_all_classes())
        
        entities  = state.get("entities", [])
        # Entity names from current chunk
        current_entity_names = {e["name"]: e["type"] for e in entities}

        # Entity names seen in any previous chunk of this document
        memory_entity_names = {
            name: info["type"]
            for name, info in state.get("document_memory", {})
                                   .get("entities", {}).items()
        }

        # Union: an entity is "known" if it's in the current chunk OR memory
        all_known_entities = {**memory_entity_names, **current_entity_names}

        relations = []
        for rel in data["relationships"]:

            # Filter: self-loops
            if rel["source"] == rel["target"]:
                continue

            # Filter: LLM hallucinated an ontology TYPE NAME as an entity name
            # e.g. source="NumericalMethod" or target="MathematicalObject"
            if rel["source"] in ontology_classes or rel["target"] in ontology_classes:
                print(f"  [filtered] type-as-entity: "
                      f"{rel['source']} →[{rel['relation']}]→ {rel['target']}")
                continue
            
            source_type = all_known_entities.get(rel["source"])
            target_type = all_known_entities.get(rel["target"])
            
            domain=self.ontology.relations.get(rel["relation"], {}).get("domain")
            range_ =self.ontology.relations.get(rel["relation"], {}).get("range")
            
            if domain==target_type and range_==source_type:
                print("Direction reversed — flipping source and target")
                rel["source"], rel["target"] = rel["target"], rel["source"]
            

            # Fix: if relation type doesn't exist in ontology and isn't
            # NEW_RELATION, convert it to NEW_RELATION with the original
            # type as suggested_relation. This catches 'defines', 'avoids',
            # 'forms' etc. that the LLM invents.
            relation_type       = rel["relation"]
            suggested_relation  = rel.get("suggested_relation")

            if (relation_type != "NEW_RELATION"
                    and not self.ontology.relation_exists(relation_type)):
                print(f"  [auto-convert] unknown relation '{relation_type}' "
                      f"→ NEW_RELATION (suggested: '{relation_type}')")
                suggested_relation = relation_type
                relation_type      = "NEW_RELATION"

            relations.append({
                "relation_id":        str(uuid4()),
                "source":             rel["source"],
                "relation":           relation_type,
                "suggested_relation": suggested_relation,
                "target":             rel["target"],
                "source_chunk_id":    chunk_id,
                "evidence_chunk_ids": [chunk_id],
                "confidence":         rel["confidence"],
            })

        state["relationships"] = relations
        return state