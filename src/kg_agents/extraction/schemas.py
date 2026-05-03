from pydantic import BaseModel, Field
from typing import List, Optional


class Entity(BaseModel):
    name: str = Field(description="Entity name exactly as written in the text")
    type: str = Field(
        description=(
            "Ontology class of the entity. "
            "Use NEW_TYPE if no ontology class fits."
        )
    )
    suggested_type: Optional[str] = Field(
        default=None,
        description=(
            "Only populated when type is NEW_TYPE. "
            "A short descriptive label for what class this entity belongs to "
            "(e.g. 'DiscretizationScheme', 'SolverAlgorithm'). "
            "Leave null otherwise."
        )
    )
    confidence: float = Field(description="Confidence score between 0 and 1")


class EntityExtractionOutput(BaseModel):
    entities: List[Entity]


class Relationship(BaseModel):
    source: str = Field(description="Source entity name exactly as in ENTITY_LIST")
    relation: str = Field(
        description=(
            "Relation type from ONTOLOGY_RELATIONS. "
            "Use NEW_RELATION if no ontology relation fits."
        )
    )
    suggested_relation: Optional[str] = Field(
        default=None,
        description=(
            "Only populated when relation is NEW_RELATION. "
            "A short snake_case verb phrase for the proposed relation "
            "(e.g. 'generalizes', 'is_variant_of', 'reduces_to'). "
            "Leave null otherwise."
        )
    )
    target: str = Field(description="Target entity name exactly as in ENTITY_LIST")
    confidence: float = Field(description="Confidence score between 0 and 1")


class RelationExtractionOutput(BaseModel):
    relationships: List[Relationship]