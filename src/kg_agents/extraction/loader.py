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

        for name, data in self.core["core_classes"].items():
            merged[name] = data

        for name, data in self.extensions["subclasses"].items():
            merged[name] = data

        return merged

    def _merge_relations(self):
        merged = {}

        for name, data in self.core["core_relations"].items():
            merged[name] = data

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

    def _get_ancestors(self, class_name: str) -> set:

        ancestors = set()
        current = class_name

        while current:
            ancestors.add(current)
            parent = self.classes.get(current, {}).get("parent")
            if parent and parent != current:
                current = parent
            else:
                break

        return ancestors

    def validate_domain_range(
        self, relation: str, source_type: str, target_type: str
    ) -> bool:

        if relation not in self.relations:
            return False

        required_domain = self.relations[relation]["domain"]
        required_range = self.relations[relation]["range"]

        source_ancestors = self._get_ancestors(source_type)
        target_ancestors = self._get_ancestors(target_type)

        domain_ok = required_domain in source_ancestors
        range_ok = required_range in target_ancestors

        return domain_ok and range_ok