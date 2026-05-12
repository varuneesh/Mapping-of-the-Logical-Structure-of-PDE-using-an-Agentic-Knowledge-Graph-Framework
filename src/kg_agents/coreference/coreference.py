import re
import json
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI

_TYPE_COMPATIBILITY: Dict[str, set] = {
    "method_reference": {
        "NumericalMethod", "MathematicalObject"
    },
    "problem_reference": {
        "ProblemType", "MathematicalObject"
    },
    "property_reference": {
        "TheoreticalProperty", "MathematicalObject"
    },
    "equation_reference": {
        "ProblemType", "MathematicalStructure", "MathematicalObject"
    },
    "general_reference": {
        # Uncategorised reference — accept any known ontology type
        "NumericalMethod", "ProblemType", "TheoreticalProperty",
        "ErrorConcept", "MathematicalStructure", "ComputationalStructure",
        "Theorem", "Definition", "MathematicalObject"
    },
}

_METHOD_NOUNS   = {"method", "scheme", "approach", "algorithm", "procedure",
                   "discretization", "solver", "technique", "framework",
                   "estimator", "approximation", "formulation"}
_PROBLEM_NOUNS  = {"equation", "problem", "system", "formulation"}
_PROPERTY_NOUNS = {"stability", "convergence", "consistency", "property",
                   "condition", "estimate", "bound", "order", "rate"}



_SWEEP_PROMPT = """\
You are a coreference resolution specialist for mathematical textbooks.

TASK
----
Identify ALL references in CHUNK that refer to a previously introduced
mathematical concept without naming it explicitly.

Examples of such references:
  - Demonstratives : "this method", "the above scheme", "the proposed approach"
  - Pronouns       : "it converges", "its stability", "they satisfy"
  - Label refs     : "method (3.4)", "scheme (FE)", "(2.1)"
  - Possessives    : "the error of the method", "its convergence rate"
  - Implicit subj  : "We apply it to ...", "The error satisfies ..."

CHUNK HEADING (section context):
<<CHUNK_HEADING>>

CHUNK:
<<CHUNK>>

KNOWN ENTITIES (from previous chunks of this document):
<<KNOWN_ENTITIES>>

CONTEXT (immediately preceding chunks):
<<CONTEXT>>

RULES
-----
1. For each reference span you find, suggest the most likely antecedent
   from KNOWN ENTITIES or CONTEXT or CHUNK_HEADING.
2. Only suggest resolutions to entities that genuinely appear in
   KNOWN ENTITIES or CONTEXT or CHUNK_HEADING — do NOT hallucinate entity names.
3. If you cannot find a plausible antecedent, set resolved_name to null.
4. Classify each reference using one of:
     method_reference, problem_reference, property_reference,
     equation_reference, general_reference
5. Return ONLY valid JSON — no markdown fences, no commentary.

OUTPUT SCHEMA:
{
  "references": [
    {
      "span": "the exact text span as it appears in CHUNK",
      "reference_class": "method_reference",
      "resolved_name": "Forward Euler method",
      "confidence": 0.9
    }
  ]
}

If no coreferences are found, return: {"references": []}
"""


class CoreferenceAgent:

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        min_confidence: float = 0.7,
    ):
        self.llm            = ChatOpenAI(model=model, temperature=0)
        self.min_confidence = min_confidence


    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:

        chunk_text      = state["primary_chunk"]
        context_chunks  = state.get("context_chunks", [])
        document_memory = state.get("document_memory", {})
        chunk_heading   = state.get("chunk_heading", "")

        known_entities  = self._build_known_entities(document_memory)
        context_surface = self._extract_surface_names(context_chunks)
        all_known_names = set(known_entities.keys()) | context_surface

        raw_references = self._llm_sweep(
            chunk_text, known_entities, context_chunks,
            chunk_heading
        )

        print(f"  [coref] LLM detected {len(raw_references)} candidate reference(s)")

        validated: List[Dict] = []

        for ref in raw_references:
            result = self._validate(ref, all_known_names, known_entities)
            if result:
                validated.append(result)
            else:
                print(f"  [coref] dropped '{ref.get('span')}' → "
                      f"'{ref.get('resolved_name')}' (failed validation)")

        print(f"  [coref] {len(validated)} resolution(s) passed validation")

        annotations: Dict[str, str] = {}
        resolved_text = chunk_text

        for ref in validated:
            span     = ref["span"]
            resolved = ref["resolved_name"]

            if resolved and resolved.lower() != span.lower():
                resolved_text        = self._safe_replace(resolved_text, span, resolved)
                annotations[span]    = resolved
                print(f"  [coref] '{span}' → '{resolved}'")

        state["resolved_chunk"]          = resolved_text
        state["coreference_annotations"] = annotations

        print(f"Coreference complete: {len(annotations)} substitution(s) applied")
        return state


    def _llm_sweep(
        self,
        chunk_text: str,
        known_entities: Dict[str, str],
        context_chunks: List[str],
        chunk_heading: str = "",
    ) -> List[Dict]:
        entity_summary = "\n".join(
            f"  - {name} ({etype})"
            for name, etype in known_entities.items()
        ) or "  (none yet)"

        context_preview = "\n---\n".join(
            c for c in context_chunks[-3:]  
        ) or "(none)"

        prompt = (
            _SWEEP_PROMPT
            .replace("<<CHUNK_HEADING>>",  chunk_heading)
            .replace("<<CHUNK>>",          chunk_text)      
            .replace("<<KNOWN_ENTITIES>>", entity_summary)
            .replace("<<CONTEXT>>",        context_preview)
        )

        try:
            response = self.llm.invoke(prompt)
            content  = response.content.strip()

            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"\s*```$",     "", content)

            data = json.loads(content)
            return data.get("references", [])

        except (json.JSONDecodeError, Exception) as e:
            print(f"  [coref] LLM sweep parse error: {e} — skipping coreference")
            return []

    def _validate(
        self,
        ref: Dict,
        all_known_names: set,
        known_entities: Dict[str, str],
    ) -> Optional[Dict]:
        span         = ref.get("span", "").strip()
        resolved     = ref.get("resolved_name")
        ref_class    = ref.get("reference_class", "general_reference")
        confidence   = ref.get("confidence", 0.0)

        if not span or not resolved:
            return None

        if confidence < self.min_confidence:
            return None

        if resolved not in all_known_names:
            return None

        resolved_type = known_entities.get(resolved)

        if resolved_type is not None:
            allowed_types = _TYPE_COMPATIBILITY.get(
                ref_class,
                _TYPE_COMPATIBILITY["general_reference"]
            )
            if ref_class != "general_reference" and \
               resolved_type not in allowed_types:
                return None

        return ref


    def _build_known_entities(
        self, document_memory: Dict
    ) -> Dict[str, str]:
        mem = document_memory.get("entities", {})
        ranked = sorted(
            mem.items(),
            key=lambda kv: len(kv[1].get("sources", [])),
            reverse=True,
        )
        return {name: info.get("type", "MathematicalObject")
                for name, info in ranked}

    def _extract_surface_names(self, context_chunks: List[str]) -> set:
        
        pattern = re.compile(
            r"\b([A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){0,3}?\s+)?"
            r"(method|scheme|equation|operator|algorithm|formula|"
            r"solver|discretization|estimator|problem|theorem|lemma)\b",
            re.IGNORECASE
        )
        names = set()
        for chunk in context_chunks:
            for m in pattern.finditer(chunk):
                phrase = m.group(0).strip()
                if len(phrase) > 4:
                    names.add(phrase)
        return names


    def _safe_replace(self, text: str, span: str, resolved: str) -> str:
        
        escaped = re.escape(span)
        pattern = re.compile(
            r"(?<!\\)\b" + escaped + r"\b",
            re.IGNORECASE
        )
        return pattern.sub(resolved, text, count=1)