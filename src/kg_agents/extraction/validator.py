# import json

# from email import errors


class ExtractionValidator:

    def __init__(self, ontology_loader):
        self.ontology = ontology_loader

    def validate(self, state):

        entities = state.get("entities", [])
        relations = state.get("relationships", [])

        entity_errors = []
        relation_errors = []

        entity_names = {e["name"]: e["type"] for e in entities}

        # Validate entities
        for e in entities:

            if "name" not in e or "type" not in e:
                entity_errors.append(f"Malformed entity: {e}")

            if not (0 <= e["confidence"] <= 1):
                entity_errors.append(f"Invalid confidence: {e['name']}")

            if e["type"] != "NEW_TYPE":
                if not self.ontology.class_exists(e["type"]):
                    entity_errors.append(f"Unknown entity type: {e['type']}")

        # Validate relations
        for r in relations:

            if r["source"] not in entity_names:
                relation_errors.append(f"Unknown source entity: {r['source']}")

            if r["target"] not in entity_names:
                relation_errors.append(f"Unknown target entity: {r['target']}")

            if not (0 <= r["confidence"] <= 1):
                relation_errors.append("Invalid relation confidence")

            if r["relation"] != "NEW_RELATION":

                if not self.ontology.relation_exists(r["relation"]):
                    relation_errors.append(f"Unknown relation: {r['relation']}")

                else:
                    source_type = entity_names[r["source"]]
                    target_type = entity_names[r["target"]]

                    valid = self.ontology.validate_domain_range(
                        r["relation"],
                        source_type,
                        target_type
                    )

                    if not valid:
                        relation_errors.append(
                            f"Domain-range violation: {r['relation']}"
                        )

        state["entity_errors"] = entity_errors
        state["relation_errors"] = relation_errors

        all_errors = entity_errors + relation_errors

        state["validation_errors"] = all_errors

        if len(all_errors) == 0:
            state["validation_status"] = "valid"
        else:
            state["validation_status"] = "invalid"

        return state