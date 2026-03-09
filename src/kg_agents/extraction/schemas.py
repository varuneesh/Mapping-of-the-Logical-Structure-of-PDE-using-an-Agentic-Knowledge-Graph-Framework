from pydantic import BaseModel, Field
from typing import List


class Entity(BaseModel):
    name: str = Field(description="Entity name exactly as written in the text")
    type: str = Field(description="Ontology class of the entity")
    confidence: float = Field(description="Confidence score between 0 and 1")


class EntityExtractionOutput(BaseModel):
    entities: List[Entity]


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    relation: str = Field(description="Relation type")
    target: str = Field(description="Target entity name")
    confidence: float = Field(description="Confidence score between 0 and 1")


class RelationExtractionOutput(BaseModel):
    relationships: List[Relationship]