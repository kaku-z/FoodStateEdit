"""Canonical scene graph used by all material-specific solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_MATERIALS = {
    "rigid",
    "strand",
    "granular",
    "liquid",
    "soft_body",
    "appearance",
    "vapor",
}


@dataclass
class FoodObject:
    object_id: str
    material: str
    state: dict[str, Any] = field(default_factory=dict)
    semantic_class: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FoodObject":
        return cls(
            object_id=str(value["id"]),
            material=str(value["material"]),
            semantic_class=value.get("semantic_class"),
            state=dict(value.get("state", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.object_id,
            "material": self.material,
            "state": self.state,
        }
        if self.semantic_class is not None:
            value["semantic_class"] = self.semantic_class
        return value


@dataclass
class Scene:
    scene_id: str
    dish_type: str
    objects: list[FoodObject]
    relations: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "foodstateedit.scene.v0.1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scene":
        scene = cls(
            scene_id=str(value["scene_id"]),
            dish_type=str(value["dish_type"]),
            objects=[FoodObject.from_dict(item) for item in value.get("objects", [])],
            relations=[dict(item) for item in value.get("relations", [])],
            actions=[dict(item) for item in value.get("actions", [])],
            constraints=[dict(item) for item in value.get("constraints", [])],
            schema_version=str(value.get("schema_version", "foodstateedit.scene.v0.1")),
            metadata=dict(value.get("metadata", {})),
        )
        scene.validate()
        return scene

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "dish_type": self.dish_type,
            "metadata": self.metadata,
            "objects": [item.to_dict() for item in self.objects],
            "relations": self.relations,
            "actions": self.actions,
            "constraints": self.constraints,
        }

    def object_map(self) -> dict[str, FoodObject]:
        return {item.object_id: item for item in self.objects}

    def get_object(self, object_id: str) -> FoodObject:
        try:
            return self.object_map()[object_id]
        except KeyError as error:
            raise ValueError(f"unknown object: {object_id}") from error

    def validate(self) -> None:
        if not self.objects:
            raise ValueError("scene must contain at least one object")
        object_ids = [item.object_id for item in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("object ids must be unique")
        for item in self.objects:
            if item.material not in SUPPORTED_MATERIALS:
                raise ValueError(
                    f"unsupported material {item.material!r} for {item.object_id}"
                )
        known = set(object_ids)
        for relation in self.relations:
            for key in ("subject", "object"):
                if key in relation and relation[key] not in known:
                    raise ValueError(
                        f"relation references unknown {key}: {relation[key]}"
                    )
