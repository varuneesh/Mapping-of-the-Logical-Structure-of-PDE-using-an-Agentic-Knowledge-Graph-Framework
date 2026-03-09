class ExtractionSupervisor:

    def __init__(self, max_retries=2):
        self.max_retries = max_retries

    def run(self, state):
        
        retry_count = state.get("retry_count", 0)

        entities = state.get("entities", [])
        validation_status = state.get("validation_status")

        entity_errors = state.get("entity_errors", [])
        relation_errors = state.get("relation_errors", [])

        print("Retry Count:", retry_count)

        # FIRST ENTRY INTO PIPELINE
        if not entities:

            print("No entities yet. Starting entity extraction.")

            state["next_step"] = "entity_extraction"

            return state


        # VALID EXTRACTION
        if validation_status == "valid":

            # print("Extraction valid. Updating memory.")

            # self._update_memory(state)

            state["next_step"] = "alignment_agent"

            return state


        # RETRIES EXCEEDED
        if retry_count >= self.max_retries:

            print("Max retries exceeded. Dropping relations.")

            state["relationships"] = []

            # self._update_memory(state)

            state["next_step"] = "alignment_agent"

            return state


        # ENTITY ERRORS
        if entity_errors:

            print("Retrying entity extraction.")

            state["next_step"] = "retry_entity_extraction"

            state["retry_feedback"] = self._generate_feedback(entity_errors)

            state["retry_count"] = retry_count + 1

            return state


        # RELATION ERRORS
        if relation_errors:

            print("Retrying relation extraction.")

            state["next_step"] = "retry_relation_extraction"

            state["retry_feedback"] = self._generate_feedback(relation_errors)

            state["retry_count"] = retry_count + 1

            return state


        state["next_step"] = "alignment_agent"

        return state


    # def _update_memory(self, state):

    #     memory = state["document_memory"]

    #     for e in state["entities"]:
    #         memory.setdefault("entities", set()).add(e["name"])


    def _generate_feedback(self, errors):

        feedback = "Previous extraction produced invalid results.\n\nErrors:\n"

        for e in errors:
            feedback += f"- {e}\n"

        feedback += "\nPlease correct the extraction while respecting ontology constraints."

        return feedback