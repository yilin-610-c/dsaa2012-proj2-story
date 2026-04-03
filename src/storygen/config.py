from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_config(config_path: str | Path, profile: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_config = load_yaml(config_path)
    profiles = raw_config.get("runtime_profiles", {})
    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown runtime profile '{profile}'. Available profiles: {available}")

    base_config = {key: value for key, value in raw_config.items() if key != "runtime_profiles"}
    resolved = _deep_merge(base_config, profiles[profile])
    resolved.setdefault("runtime", {})
    resolved["runtime"]["profile"] = profile

    for dotted_key, value in (overrides or {}).items():
        if value is None:
            continue
        cursor = resolved
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    return resolved
