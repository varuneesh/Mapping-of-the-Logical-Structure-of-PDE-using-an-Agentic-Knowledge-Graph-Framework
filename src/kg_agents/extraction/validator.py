class ExtractionValidator:
    """
    Validates extracted entities and relationships against the ontology.

    Key fixes vs original:
    - Relations whose source/target exist in document_memory (from a previous
      chunk) are no longer flagged as errors.  Cross-chunk relations are valid
      and should not trigger retries.
    - Self-loops (source == target) are always flagged as errors.
    - Domain-range validation now benefits from the fixed loader.py
      _get_ancestors() walk, so subclass types are handled correctly.
    """

    def __init__(self, ontology_loader):
        self.ontology = ontology_loader

    def validate(self, state: dict) -> dict:

        entities  = state.get("entities", [])
        relations = state.get("relationships", [])

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

        entity_errors   = []
        relation_errors = []

        # ── Validate entities ─────────────────────────────────────────────
        for e in entities:

            if "name" not in e or "type" not in e:
                entity_errors.append(f"Malformed entity object: {e}")
                continue

            if not isinstance(e.get("confidence"), (int, float)) or \
               not (0.0 <= e["confidence"] <= 1.0):
                entity_errors.append(
                    f"Invalid confidence for entity '{e['name']}': {e.get('confidence')}"
                )

            if e["type"] != "NEW_TYPE" and not self.ontology.class_exists(e["type"]):
                entity_errors.append(
                    f"Unknown ontology class '{e['type']}' for entity '{e['name']}'"
                )

        # ── Validate relations ────────────────────────────────────────────
        for r in relations:

            # Self-loop guard (belt-and-suspenders — agent also filters these)
            if r["source"] == r["target"]:
                relation_errors.append(
                    f"Self-loop detected: '{r['source']}' → '{r['relation']}'"
                )
                continue

            # Source must be known (current chunk OR document memory)
            if r["source"] not in all_known_entities:
                relation_errors.append(
                    f"Unknown source entity '{r['source']}' "
                    f"(not in current chunk or document memory)"
                )

            # Target must be known
            if r["target"] not in all_known_entities:
                relation_errors.append(
                    f"Unknown target entity '{r['target']}' "
                    f"(not in current chunk or document memory)"
                )

            if not isinstance(r.get("confidence"), (int, float)) or \
               not (0.0 <= r["confidence"] <= 1.0):
                relation_errors.append(
                    f"Invalid confidence for relation "
                    f"'{r['source']} → {r['relation']} → {r['target']}'"
                )

            if r["relation"] != "NEW_RELATION":

                if not self.ontology.relation_exists(r["relation"]):
                    relation_errors.append(
                        f"Unknown relation type '{r['relation']}'"
                    )
                else:
                    # Use the union dict so cross-chunk types are resolved
                    source_type = all_known_entities.get(r["source"])
                    target_type = all_known_entities.get(r["target"])

                    if source_type and target_type:
                        valid = self.ontology.validate_domain_range(
                            r["relation"], source_type, target_type
                        )
                        if not valid:
                            relation_errors.append(
                                f"Domain-range violation: "
                                f"'{r['relation']}' with "
                                f"source type '{source_type}' and "
                                f"target type '{target_type}'"
                            )

        # ── Write back ────────────────────────────────────────────────────
        state["entity_errors"]    = entity_errors
        state["relation_errors"]  = relation_errors
        state["validation_errors"] = entity_errors + relation_errors
        state["validation_status"] = "valid" if not (entity_errors + relation_errors) \
                                              else "invalid"

        if entity_errors:
            print(f"  Entity errors: {entity_errors}")
        if relation_errors:
            print(f"  Relation errors: {relation_errors}")

        return state