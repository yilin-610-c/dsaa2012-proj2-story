from __future__ import annotations

from typing import Any


def diff_payload_fields(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    _diff_value(before, after, path="$", diffs=diffs)
    return diffs


def _diff_value(before: Any, after: Any, *, path: str, diffs: list[dict[str, Any]]) -> None:
    if isinstance(before, dict) and isinstance(after, dict):
        keys = sorted(set(before) | set(after))
        for key in keys:
            _diff_value(before.get(key), after.get(key), path=f"{path}.{key}", diffs=diffs)
        return
    if isinstance(before, list) and isinstance(after, list):
        max_len = max(len(before), len(after))
        for index in range(max_len):
            old = before[index] if index < len(before) else None
            new = after[index] if index < len(after) else None
            _diff_value(old, new, path=f"{path}[{index}]", diffs=diffs)
        return
    if before != after:
        diffs.append({"path": path, "before": before, "after": after})
