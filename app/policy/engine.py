from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.config import REGISTRIES_DIR


class PolicyHit(BaseModel):
    needs_approval: bool = False
    reason: str = ""
    matched_policy_id: str | None = None


class PolicyRule(BaseModel):
    id: str
    description: str = ""
    approval_required: bool = False
    match_keywords: list[str] = Field(default_factory=list)
    match_categories: list[str] = Field(default_factory=list)


class PolicyEngine:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (REGISTRIES_DIR / "policies.yaml")
        with self.path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self.rules = [PolicyRule.model_validate(r) for r in raw.get("policies", [])]

    def evaluate(self, query: str, intent: str = "") -> PolicyHit:
        text = (query or "").lower()
        intent_l = (intent or "").lower()

        for rule in self.rules:
            keyword_hit = any(k.lower() in text for k in rule.match_keywords)
            category_ok = (
                not rule.match_categories
                or intent_l in [c.lower() for c in rule.match_categories]
            )
            if keyword_hit and category_ok and rule.approval_required:
                return PolicyHit(
                    needs_approval=True,
                    reason=f"Policy '{rule.id}': {rule.description}",
                    matched_policy_id=rule.id,
                )
        return PolicyHit(needs_approval=False)


@lru_cache(maxsize=1)
def get_policy_engine() -> PolicyEngine:
    return PolicyEngine()
