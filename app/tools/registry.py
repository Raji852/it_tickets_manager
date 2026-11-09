from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.config import REGISTRIES_DIR


class ToolRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (REGISTRIES_DIR / "tools.yaml")
        with self.path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self._tools: dict[str, dict[str, Any]] = {
            t["name"]: t for t in raw.get("tools", [])
        }

    def get(self, name: str) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Tool not found in registry: {name}")
        return self._tools[name]

    def approval_required(self, name: str) -> bool:
        return bool(self.get(name).get("approval_required", False))

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools.values())


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    return ToolRegistry()
