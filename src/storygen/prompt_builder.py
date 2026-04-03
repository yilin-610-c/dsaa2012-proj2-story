from __future__ import annotations

from typing import Any

from storygen.types import PromptSpec, Scene, Story


class PromptBuilder:
    """Rule-based prompt builder with explicit prompt components."""

    def __init__(self, prompt_config: dict[str, Any]) -> None:
        self.prompt_config = prompt_config

    def build_story_prompts(self, story: Story) -> dict[str, PromptSpec]:
        return {scene.scene_id: self.build_prompt_for_scene(story, scene) for scene in story.scenes}

    def build_prompt_for_scene(self, story: Story, scene: Scene) -> PromptSpec:
        style_prompt = self.prompt_config.get("style_prompt", "").strip()
        relevant_recurring_entities = [entity for entity in scene.entities if entity in story.recurring_entities]
        character_prompt = self._join_entities(
            relevant_recurring_entities,
            prefix=self.prompt_config.get("character_prefix", ""),
        )
        global_context_prompt = self._join_entities(
            story.recurring_entities,
            prefix=self.prompt_config.get("global_context_prefix", ""),
        )
        local_prompt = scene.clean_text
        negative_prompt = self.prompt_config.get("negative_prompt", "").strip()
        full_prompt = self._compose_full_prompt(
            style_prompt=style_prompt,
            character_prompt=character_prompt,
            global_context_prompt=global_context_prompt,
            local_prompt=local_prompt,
        )

        return PromptSpec(
            scene_id=scene.scene_id,
            style_prompt=style_prompt,
            character_prompt=character_prompt,
            global_context_prompt=global_context_prompt,
            local_prompt=local_prompt,
            full_prompt=full_prompt,
            negative_prompt=negative_prompt,
        )

    def _compose_full_prompt(
        self,
        *,
        style_prompt: str,
        character_prompt: str,
        global_context_prompt: str,
        local_prompt: str,
    ) -> str:
        parts = [
            style_prompt,
            character_prompt,
            global_context_prompt,
            local_prompt,
            self.prompt_config.get("quality_suffix", "").strip(),
        ]
        return ", ".join(part for part in parts if part)

    @staticmethod
    def _join_entities(entities: list[str], prefix: str) -> str:
        unique_entities = list(dict.fromkeys(entity.strip() for entity in entities if entity.strip()))
        if not unique_entities:
            return ""
        entity_text = ", ".join(unique_entities)
        prefix = prefix.strip()
        return f"{prefix} {entity_text}".strip()
