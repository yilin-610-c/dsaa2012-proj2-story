from __future__ import annotations

import re
from typing import Any

from storygen.types import PromptSpec, Scene, Story

LEADING_PRONOUN_PATTERN = re.compile(r"^(she|he|they|it)\b", re.IGNORECASE)
LOCATION_PATTERN = re.compile(
    r"\b(?:in|inside|at|on|along|across|under|near|beside|by|outside|through)\s+([^.,;]+)",
    re.IGNORECASE,
)
HUMAN_HINTS = {
    "artist",
    "boy",
    "girl",
    "kid",
    "man",
    "woman",
    "person",
    "chef",
    "driver",
    "teacher",
    "student",
    "doctor",
    "runner",
}
ANIMAL_HINTS = {
    "dog",
    "cat",
    "bird",
    "horse",
    "fox",
    "wolf",
    "bear",
    "rabbit",
    "lion",
    "tiger",
    "deer",
    "fish",
}


class PromptBuilder:
    """Rule-based prompt builder with explicit prompt components."""

    def __init__(self, prompt_config: dict[str, Any]) -> None:
        self.prompt_config = prompt_config

    def build_story_prompts(self, story: Story) -> dict[str, PromptSpec]:
        story_context = self._build_story_context(story)
        return {
            scene.scene_id: self.build_prompt_for_scene(story, scene, story_context=story_context)
            for scene in story.scenes
        }

    def build_prompt_for_scene(
        self,
        story: Story,
        scene: Scene,
        story_context: dict[str, str | list[str] | None] | None = None,
    ) -> PromptSpec:
        story_context = story_context or self._build_story_context(story)
        style_prompt = self.prompt_config.get("style_prompt", "").strip()
        local_prompt = self._build_local_prompt(scene, story_context)
        character_prompt = self._build_character_prompt(story, scene, story_context)
        global_context_prompt = self._build_global_context_prompt(story, story_context)
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

    def _build_story_context(self, story: Story) -> dict[str, str | list[str] | None]:
        primary_entity = self._select_primary_entity(story)
        pronoun = self._infer_story_pronoun(story)
        subject_type = self._infer_subject_type(primary_entity, pronoun)
        location_anchor = self._infer_location_anchor(story)
        return {
            "primary_entity": primary_entity,
            "pronoun": pronoun,
            "subject_type": subject_type,
            "location_anchor": location_anchor,
            "recurring_entities": story.recurring_entities,
        }

    def _build_character_prompt(
        self,
        story: Story,
        scene: Scene,
        story_context: dict[str, str | list[str] | None],
    ) -> str:
        subject_prefix = self.prompt_config.get(
            "subject_prefix",
            self.prompt_config.get("character_prefix", ""),
        ).strip()
        primary_entity = story_context.get("primary_entity")
        scene_entities = list(dict.fromkeys(scene.entities))
        effective_entities = scene_entities or self._scene_fallback_entities(scene, story_context)
        continuity_prompt = self._continuity_prompt_for_subject_type(str(story_context.get("subject_type") or "generic"))
        parts = []

        if effective_entities:
            parts.append(self._join_entities(effective_entities, prefix=subject_prefix))
        elif primary_entity:
            parts.append(f"{subject_prefix} {primary_entity}".strip())

        recurring_in_scene = [entity for entity in effective_entities if entity in story.recurring_entities]
        if recurring_in_scene or primary_entity:
            parts.append(continuity_prompt)

        return ", ".join(part for part in parts if part)

    def _build_global_context_prompt(
        self,
        story: Story,
        story_context: dict[str, str | list[str] | None],
    ) -> str:
        parts = []
        global_context_prefix = self.prompt_config.get("global_context_prefix", "").strip()
        recurring_entities = [entity for entity in story.recurring_entities if entity.strip()]
        if recurring_entities:
            parts.append(self._join_entities(recurring_entities, prefix=global_context_prefix))

        location_anchor = str(story_context.get("location_anchor") or "").strip()
        setting_prefix = self.prompt_config.get("setting_prefix", "").strip()
        if location_anchor:
            parts.append(f"{setting_prefix} {location_anchor}".strip())

        scene_continuity_prompt = self.prompt_config.get("scene_continuity_prompt", "").strip()
        if scene_continuity_prompt:
            parts.append(scene_continuity_prompt)

        return ", ".join(part for part in parts if part)

    def _build_local_prompt(
        self,
        scene: Scene,
        story_context: dict[str, str | list[str] | None],
    ) -> str:
        local_prompt = scene.clean_text
        primary_entity = str(story_context.get("primary_entity") or "").strip()

        if self.prompt_config.get("replace_leading_pronouns", True) and primary_entity:
            local_prompt = LEADING_PRONOUN_PATTERN.sub(primary_entity, local_prompt, count=1)

        action_emphasis_prompt = self._build_action_emphasis_prompt(local_prompt)
        scene_composition_prompt = self.prompt_config.get("scene_composition_prompt", "").strip()
        local_prompt_suffix = self.prompt_config.get("local_prompt_suffix", "").strip()
        return ", ".join(
            part
            for part in [local_prompt, action_emphasis_prompt, scene_composition_prompt, local_prompt_suffix]
            if part
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

    def _scene_fallback_entities(
        self,
        scene: Scene,
        story_context: dict[str, str | list[str] | None],
    ) -> list[str]:
        primary_entity = str(story_context.get("primary_entity") or "").strip()
        if not primary_entity:
            return []
        if LEADING_PRONOUN_PATTERN.match(scene.clean_text):
            return [primary_entity]
        return []

    def _select_primary_entity(self, story: Story) -> str | None:
        if len(story.recurring_entities) == 1:
            return story.recurring_entities[0]
        if story.scenes and story.scenes[0].entities:
            return story.scenes[0].entities[0]
        if story.all_entities:
            return story.all_entities[0]
        return None

    def _infer_story_pronoun(self, story: Story) -> str | None:
        for scene in story.scenes:
            match = LEADING_PRONOUN_PATTERN.match(scene.clean_text)
            if match:
                return match.group(1).lower()
        return None

    def _infer_subject_type(self, primary_entity: str | None, pronoun: str | None) -> str:
        entity_key = (primary_entity or "").strip().lower()
        if pronoun in {"she", "he", "they"}:
            return "human"
        if pronoun == "it":
            if entity_key in ANIMAL_HINTS:
                return "animal"
            return "generic"
        if entity_key in HUMAN_HINTS:
            return "human"
        if entity_key in ANIMAL_HINTS:
            return "animal"
        if primary_entity and primary_entity[:1].isupper():
            return "human"
        return "generic"

    def _infer_location_anchor(self, story: Story) -> str:
        for scene in story.scenes:
            match = LOCATION_PATTERN.search(scene.clean_text)
            if match:
                return self._normalize_fragment(match.group(1))
        return self.prompt_config.get("default_setting_prompt", "").strip()

    def _continuity_prompt_for_subject_type(self, subject_type: str) -> str:
        if subject_type == "human":
            return self.prompt_config.get("human_identity_prompt", "").strip()
        if subject_type == "animal":
            return self.prompt_config.get("animal_identity_prompt", "").strip()
        return self.prompt_config.get("generic_identity_prompt", "").strip()

    def _build_action_emphasis_prompt(self, local_prompt: str) -> str:
        action_map = self.prompt_config.get("action_emphasis_map", {})
        normalized_text = local_prompt.lower()
        for action_key, action_phrase in sorted(action_map.items(), key=lambda item: len(item[0]), reverse=True):
            if action_key.lower() in normalized_text:
                template = self.prompt_config.get("action_emphasis_template", "").strip()
                if template:
                    return template.format(action_phrase=action_phrase, action_key=action_key)
                return str(action_phrase).strip()
        return self.prompt_config.get("default_action_prompt", "").strip()

    @staticmethod
    def _normalize_fragment(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip(" ,.;"))

    @staticmethod
    def _join_entities(entities: list[str], prefix: str) -> str:
        unique_entities = list(dict.fromkeys(entity.strip() for entity in entities if entity.strip()))
        if not unique_entities:
            return ""
        entity_text = ", ".join(unique_entities)
        prefix = prefix.strip()
        return f"{prefix} {entity_text}".strip()
