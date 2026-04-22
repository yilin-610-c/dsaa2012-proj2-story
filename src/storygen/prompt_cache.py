from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from storygen.types import Story


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_to_jsonable(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_prompt_cache_key(story: Story, prompt_config: dict[str, Any]) -> str:
    llm_config = prompt_config.get("llm", {})
    relevant_prompt_config = {
        "pipeline": prompt_config.get("pipeline", "rule_based"),
        "style_prompt": prompt_config.get("style_prompt", ""),
        "negative_prompt": prompt_config.get("negative_prompt", ""),
        "generation_max_words": prompt_config.get("generation_max_words"),
        "generation_max_chars": prompt_config.get("generation_max_chars"),
        "scoring_max_words": prompt_config.get("scoring_max_words"),
        "scoring_max_chars": prompt_config.get("scoring_max_chars"),
        "llm": {
            "provider": llm_config.get("provider"),
            "model": llm_config.get("model"),
            "schema_version": llm_config.get("schema_version"),
            "builder_version": llm_config.get("builder_version"),
        },
    }
    payload = {
        "raw_story_text": story.raw_text,
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "index": scene.index,
                "clean_text": scene.clean_text,
                "entities": scene.entities,
            }
            for scene in story.scenes
        ],
        "prompt_config": relevant_prompt_config,
    }
    return stable_hash(payload)


class PromptCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for_key(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def load(self, cache_key: str) -> dict[str, Any] | None:
        path = self.path_for_key(cache_key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, cache_key: str, payload: dict[str, Any]) -> Path:
        path = self.path_for_key(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2, ensure_ascii=True, sort_keys=True)
        return path


def build_cache_record(
    *,
    cache_key: str,
    request_metadata: dict[str, Any],
    raw_response: str,
    parsed_response: dict[str, Any],
    validated_output: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "request_metadata": request_metadata,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "validated_output": validated_output,
    }
