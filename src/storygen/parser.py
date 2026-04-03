from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from storygen.types import Scene, Story

SCENE_PATTERN = re.compile(r"^\[(SCENE-\d+)\]\s*(.*)$", re.DOTALL)
ENTITY_PATTERN = re.compile(r"<([^<>]+)>")


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _clean_scene_text(raw_text: str) -> str:
    return _normalize_whitespace(ENTITY_PATTERN.sub(r"\1", raw_text))


def parse_story_file(path: str | Path) -> Story:
    source_path = Path(path)
    raw_text = source_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw_text.split("[SEP]") if block.strip()]

    scenes: list[Scene] = []
    entity_to_scene_ids: dict[str, list[str]] = defaultdict(list)

    for index, block in enumerate(blocks):
        match = SCENE_PATTERN.match(block)
        if not match:
            raise ValueError(f"Invalid scene block at index {index}: {block!r}")

        scene_id, scene_body = match.groups()
        scene_body = scene_body.strip()
        entities = ENTITY_PATTERN.findall(scene_body)
        clean_text = _clean_scene_text(scene_body)

        scene = Scene(
            scene_id=scene_id,
            index=index,
            raw_text=scene_body,
            clean_text=clean_text,
            entities=entities,
        )
        scenes.append(scene)

        for entity in dict.fromkeys(entities):
            entity_to_scene_ids[entity].append(scene_id)

    if not scenes:
        raise ValueError(f"No scenes found in {source_path}")

    entity_counts = Counter(entity for scene in scenes for entity in dict.fromkeys(scene.entities))
    all_entities = sorted(entity_counts)
    recurring_entities = sorted(entity for entity, count in entity_counts.items() if count > 1)

    return Story(
        source_path=str(source_path),
        raw_text=raw_text,
        scenes=scenes,
        all_entities=all_entities,
        recurring_entities=recurring_entities,
        entity_to_scene_ids={key: value for key, value in sorted(entity_to_scene_ids.items())},
    )
