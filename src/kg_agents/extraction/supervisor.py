class ExtractionSupervisor:
    """
    Controls extraction flow and retry logic.

    Key fixes:
    - Uses extraction_started flag instead of `if not entities` for first-entry
      detection — prevents infinite loop when LLM legitimately returns no entities.
    - Reads _rate_limited flag (now properly declared in GraphState) to
      immediately route to alignment without looping.
    - Empty entity list after extraction has started → go to alignment,
      not back to extraction.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def run(self, state: dict) -> dict:

        extraction_started = state.get("extraction_started", False)
        entities           = state.get("entities", [])
        validation_status  = state.get("validation_status")
        entity_errors      = state.get("entity_errors", [])
        relation_errors    = state.get("relation_errors", [])

        entity_retry_count   = state.get("entity_retry_count", 0)
        relation_retry_count = state.get("relation_retry_count", 0)

        print(f"Entity retries used: {entity_retry_count}/{self.max_retries}")
        print(f"Relation retries used: {relation_retry_count}/{self.max_retries}")

        # ── RATE LIMITED — skip to alignment immediately ──────────────────
        # _rate_limited is now in GraphState so LangGraph preserves it
        if state.get("_rate_limited"):
            print("Rate limited — routing to alignment with empty extraction.")
            state["retry_feedback"] = ""
            state["_rate_limited"]  = False
            state["next_step"]      = "alignment_agent"
            return state

        # ── FIRST ENTRY — extraction not yet attempted ────────────────────
        if not extraction_started:
            print("Starting entity extraction.")
            state["retry_feedback"] = ""
            state["next_step"]      = "entity_extraction"
            return state

        # ── EXTRACTION DONE, NO ENTITIES — move on, don't loop ───────────
        # LLM returned empty list: either genuinely empty chunk or dense
        # notation the model couldn't parse. Either way, looping won't help.
        if extraction_started and not entities and not entity_errors:
            print("Extraction returned no entities — moving to alignment.")
            state["retry_feedback"] = ""
            state["next_step"]      = "alignment_agent"
            return state

        # ── VALID EXTRACTION ──────────────────────────────────────────────
        if validation_status == "valid":
            print("Extraction valid — proceeding to alignment.")
            state["retry_feedback"] = ""
            state["next_step"]      = "alignment_agent"
            return state

        # ── ENTITY ERRORS ─────────────────────────────────────────────────
        if entity_errors:
            if entity_retry_count >= self.max_retries:
                print("Entity max retries exceeded — moving on with partial entities.")
                state["entity_errors"]      = []
                state["entity_retry_count"] = 0
                state["retry_feedback"]     = ""
                state["next_step"]          = "alignment_agent"
            else:
                print(f"Retrying entity extraction "
                      f"(attempt {entity_retry_count + 1}/{self.max_retries}).")
                state["retry_feedback"]     = self._build_feedback(entity_errors)
                state["entity_retry_count"] = entity_retry_count + 1
                state["next_step"]          = "retry_entity_extraction"
            return state

        # ── RELATION ERRORS ───────────────────────────────────────────────
        if relation_errors:

            # Check if ALL errors are caused by NEW_TYPE entities.
            # If so, retrying is pointless — the entity types won't change.
            # Keep the relations and move to alignment; the consistency agent
            # will route them to the candidate pool.
            all_new_type = all(
                "NEW_TYPE" in err for err in relation_errors
            )

            if all_new_type:
                print("All relation errors involve NEW_TYPE entities — "
                      "skipping retry, keeping relations for candidate pool.")
                state["relation_errors"]      = []
                state["relation_retry_count"] = 0
                state["retry_feedback"]       = ""
                # DO NOT drop relationships — let consistency agent pool them
                state["next_step"]            = "alignment_agent"

            elif relation_retry_count >= self.max_retries:
                print("Relation max retries exceeded.")

                # Separate salvageable relations (NEW_TYPE issues) from
                # structurally broken ones (unknown entity names).
                # NEW_TYPE relations go to alignment → consistency → pool.
                # Unknown-entity relations are dropped — the entity name
                # is wrong and pooling won't help.
                current_rels = state.get("relationships", [])
                unknown_entity_names = set()
                for err in relation_errors:
                    if "Unknown" in err and "entity" in err:
                        # Extract the entity name from the error message
                        # Format: "Unknown target entity 'X' (not in ...)"
                        import re
                        m = re.search(r"entity '([^']+)'", err)
                        if m:
                            unknown_entity_names.add(m.group(1))

                if unknown_entity_names:
                    kept = [r for r in current_rels
                            if r["source"] not in unknown_entity_names
                            and r["target"] not in unknown_entity_names]
                    dropped = len(current_rels) - len(kept)
                    if dropped:
                        print(f"  Dropped {dropped} relation(s) with unknown entities: "
                              f"{unknown_entity_names}")
                    state["relationships"] = kept

                state["relation_errors"]      = []
                state["relation_retry_count"] = 0
                state["retry_feedback"]       = ""
                state["next_step"]            = "alignment_agent"

            else:
                print(f"Retrying relation extraction "
                      f"(attempt {relation_retry_count + 1}/{self.max_retries}).")
                # Filter out NEW_TYPE errors from feedback — only send
                # errors the LLM can actually fix
                fixable_errors = [
                    e for e in relation_errors if "NEW_TYPE" not in e
                ]
                if fixable_errors:
                    state["retry_feedback"] = self._build_feedback(
                        fixable_errors, state.get("entities", [])
                    )
                    state["relation_retry_count"] = relation_retry_count + 1
                    state["next_step"]            = "retry_relation_extraction"
                else:
                    # Only NEW_TYPE errors remain — skip retry
                    print("  No fixable errors remain — skipping retry.")
                    state["relation_errors"]      = []
                    state["retry_feedback"]       = ""
                    state["next_step"]            = "alignment_agent"

            return state

        # ── FALLBACK ──────────────────────────────────────────────────────
        state["retry_feedback"] = ""
        state["next_step"]      = "alignment_agent"
        return state

    def _build_feedback(self, errors: list, entities: list = None) -> str:
        """
        Build actionable retry feedback that tells the LLM not just what
        was wrong but what the correct constraints are.
        """
        lines = [
            "Previous extraction produced invalid results.",
            "",
            "Errors:"
        ]
        for e in errors:
            lines.append(f"  - {e}")

        lines.append("")
        lines.append("HOW TO FIX:")
        lines.append("  - Domain-range violations mean the source or target entity type")
        lines.append("    does not match what the relation requires.")
        lines.append("  - Check the ONTOLOGY_RELATIONS list above — each relation shows")
        lines.append("    SOURCE_TYPE → TARGET_TYPE. Your source and target must match.")
        lines.append("  - If no valid relation exists, use NEW_RELATION with a suggested name.")
        lines.append("  - If the relation direction is wrong, swap source and target.")
        lines.append("  - If you cannot fix the error, omit the problematic relation entirely.")
        lines.append("")
        lines.append("Please correct the extraction while respecting ontology constraints.")
        return "\n".join(lines)