"""Tool registry (Phase 6).

Each tool declares: name, description, JSON-Schema arguments, risk_tier
(read_only | state_changing | destructive), requires_confirmation, sandbox
(how it is isolated) and scope_check (callable enforcing configs/scope.yaml).

Names are plain snake_case verbs kept easy to port to the Ayanami-AI `run`
(ReAct) allowlist. This module defines tools only; nothing here executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

RISK_TIERS = ("read_only", "state_changing", "destructive")


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    arguments: dict[str, Any]  # JSON-Schema (subset: object/properties/required/enum)
    risk_tier: str
    requires_confirmation: bool
    sandbox: str
    scope_check: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None

    def __post_init__(self) -> None:
        if self.risk_tier not in RISK_TIERS:
            raise ValueError(f"bad risk_tier {self.risk_tier!r}")
        if self.risk_tier == "destructive" and not self.requires_confirmation:
            raise ValueError(f"destructive tool {self.name!r} must require confirmation")


def no_scope_needed(args: dict[str, Any], scope: dict[str, Any]) -> bool:
    """Scope check for tools that touch no external asset (local reads)."""
    return True


def asset_in_scope(asset_key: str) -> Callable[[dict[str, Any], dict[str, Any]], bool]:
    """Build a scope check requiring args[asset_key] to be a listed asset."""

    def check(args: dict[str, Any], scope: dict[str, Any]) -> bool:
        target = args.get(asset_key)
        assets = scope.get("assets", {})
        return isinstance(target, str) and target in assets

    check.__name__ = f"asset_in_scope({asset_key})"
    return check


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def by_tier(self, tier: str) -> list[str]:
        return sorted(t.name for t in self._tools.values() if t.risk_tier == tier)


def validate_args(schema: dict[str, Any], args: Any) -> list[str]:
    """Minimal JSON-Schema validation (object/string/integer/boolean/array,
    required, properties, enum, additionalProperties). Returns error list."""
    errors: list[str] = []
    if not isinstance(schema, dict):
        return ["schema is not a dict"]
    want = schema.get("type", "object")

    def check(node: Any, sch: dict, path: str) -> None:
        t = sch.get("type")
        if t == "object":
            if not isinstance(node, dict):
                errors.append(f"{path}: expected object")
                return
            for req in sch.get("required", []):
                if req not in node:
                    errors.append(f"{path}: missing required {req!r}")
            props = sch.get("properties", {})
            for k, v in node.items():
                if k in props:
                    check(v, props[k], f"{path}.{k}")
                elif sch.get("additionalProperties") is False:
                    errors.append(f"{path}: unexpected property {k!r}")
        elif t == "string":
            if not isinstance(node, str):
                errors.append(f"{path}: expected string")
            elif "enum" in sch and node not in sch["enum"]:
                errors.append(f"{path}: {node!r} not in enum")
        elif t == "integer":
            if not isinstance(node, int) or isinstance(node, bool):
                errors.append(f"{path}: expected integer")
        elif t == "boolean":
            if not isinstance(node, bool):
                errors.append(f"{path}: expected boolean")
        elif t == "array":
            if not isinstance(node, list):
                errors.append(f"{path}: expected array")
            else:
                for i, item in enumerate(node):
                    check(item, sch.get("items", {}), f"{path}[{i}]")

    if want != "object":
        errors.append("top-level tool schema must be type object")
        return errors
    check(args, schema, "args")
    return errors
