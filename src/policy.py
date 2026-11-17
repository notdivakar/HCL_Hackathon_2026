"""
Loads config/policy.json — the single place for segment thresholds, strategies,
reminder stages, contact rules, blocked language and footers.
Every module reads policy from here so the rules stay in one file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "config" / "policy.json"


@lru_cache(maxsize=1)
def load_policy(path: str | Path = POLICY_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reminder_stage_for(days_overdue: int, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pick the reminder stage whose day range contains days_overdue."""
    policy = policy or load_policy()
    days = max(0, int(days_overdue or 0))
    for stage in policy["reminder_stages"]:
        if stage["min_days_overdue"] <= days <= stage["max_days_overdue"]:
            return stage
    return policy["reminder_stages"][-1]
