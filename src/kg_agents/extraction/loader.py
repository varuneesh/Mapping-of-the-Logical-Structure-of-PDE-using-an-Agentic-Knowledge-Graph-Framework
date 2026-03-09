import json
from pathlib import Path


class OntologyLoader:
    def __init__(self, core_path: str, extension_path: str):
        self.core_path = Path(core_path)
        self.extension_path = Path(extension_path)

        self.core = self._load_json(self.core_path)
        self.extensions = self._load_json(self.extension_path)

        self.classes = self._merge_classes()
        self.relations = self._merge_relations()

    def _load_json(self, path: Path):
        with open(path, "r") as f:
            return json.load(f)

    def _merge_classes(self):
        merged = {}

        # Core classes
        for name, data in self.core["core_classes"].items():
            merged[name] = data

        # Extension subclasses
        for name, data in self.extensions["subclasses"].items():
            merged[name] = data

        return merged

    def _merge_relations(self):
        merged = {}

        # Core relations
        for name, data in self.core["core_relations"].items():
            merged[name] = data

        # Extension relations
        for name, data in self.extensions["relation_extensions"].items():
            merged[name] = data

        return merged

    def get_all_classes(self):
        return list(self.classes.keys())

    def get_all_relations(self):
        return list(self.relations.keys())

    def class_exists(self, class_name: str) -> bool:
        return class_name in self.classes

    def relation_exists(self, relation_name: str) -> bool:
        return relation_name in self.relations

    def validate_domain_range(self, relation: str, source_type: str, target_type: str) -> bool:
        if relation not in self.relations:
            return False

        domain = self.relations[relation]["domain"]
        range_ = self.relations[relation]["range"]
        
        if source_type == domain and target_type == range_:
            return True

        # allow subclass relationships
        source_parent = self.classes.get(source_type, {}).get("parent")
        target_parent = self.classes.get(target_type, {}).get("parent")

        return source_parent == domain and target_parent == range_