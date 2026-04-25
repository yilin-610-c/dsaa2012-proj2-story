from __future__ import annotations

import re
from typing import Any

from storygen.character_specs import build_rule_based_character_specs
from storygen.types import PromptSpec, Scene, Story

GENDER_PHRASES = {
    "male": "male man",
    "female": "female woman",
}

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
        scene_text = self._build_base_scene_text(scene, story_context)
        local_prompt = self._build_local_prompt(scene_text)
        action_prompt = self._build_action_prompt(scene_text)
        generation_prompt = self._build_generation_prompt(scene, scene_text, action_prompt, story_context)
        character_prompt = self._build_character_prompt(story, scene, story_context)
        global_context_prompt = self._build_global_context_prompt(story, story_context)
        scoring_prompt = self._build_scoring_prompt(scene, scene_text, story_context)
        negative_prompt = self.prompt_config.get("negative_prompt", "").strip()
        full_prompt = self._compose_full_prompt(
            style_prompt=style_prompt,
            character_prompt=character_prompt,
            global_context_prompt=global_context_prompt,
            local_prompt=local_prompt,
        )
        generation_prompt = self._apply_prompt_rewriter("generation", generation_prompt)
        scoring_prompt = self._apply_prompt_rewriter("scoring", scoring_prompt)
        action_prompt = self._apply_prompt_rewriter("action", action_prompt)
        full_prompt = self._apply_prompt_rewriter("full", full_prompt)

        return PromptSpec(
            scene_id=scene.scene_id,
            style_prompt=style_prompt,
            character_prompt=character_prompt,
            global_context_prompt=global_context_prompt,
            local_prompt=local_prompt,
            action_prompt=action_prompt,
            generation_prompt=generation_prompt,
            scoring_prompt=scoring_prompt,
            full_prompt=full_prompt,
            negative_prompt=negative_prompt,
        )

    def _build_story_context(self, story: Story) -> dict[str, str | list[str] | None]:
        primary_entity = self._select_primary_entity(story)
        pronoun = self._infer_story_pronoun(story)
        subject_type = self._infer_subject_type(primary_entity, pronoun)
        location_anchor = self._infer_location_anchor(story)
        character_specs = build_rule_based_character_specs(story)
        primary_character_spec = character_specs.get(primary_entity or "", {}) if isinstance(character_specs, dict) else {}
        gender_presentation = str(primary_character_spec.get("gender_presentation") or "").strip() or None
        profession_marker = str(primary_character_spec.get("profession_marker") or "").strip() or None
        return {
            "primary_entity": primary_entity,
            "pronoun": pronoun,
            "subject_type": subject_type,
            "location_anchor": location_anchor,
            "recurring_entities": story.recurring_entities,
            "gender_presentation": gender_presentation,
            "profession_marker": profession_marker,
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
        gender_phrase = self._gender_identity_phrase(story_context)
        profession_marker = str(story_context.get("profession_marker") or "").strip()
        parts = []

        if effective_entities:
            entity_text = self._join_entities(effective_entities, prefix=subject_prefix)
            if len(effective_entities) == 1:
                extra_parts = [gender_phrase, profession_marker]
                entity_text = ", ".join(part for part in [entity_text, *extra_parts] if part)
            parts.append(entity_text)
        elif primary_entity:
            primary_text = f"{subject_prefix} {primary_entity}".strip()
            extra_parts = [gender_phrase, profession_marker]
            primary_text = ", ".join(part for part in [primary_text, *extra_parts] if part)
            parts.append(primary_text)

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

    def _build_base_scene_text(
        self,
        scene: Scene,
        story_context: dict[str, str | list[str] | None],
    ) -> str:
        scene_text = scene.clean_text
        primary_entity = str(story_context.get("primary_entity") or "").strip()

        if self.prompt_config.get("replace_leading_pronouns", True) and primary_entity:
            scene_text = LEADING_PRONOUN_PATTERN.sub(primary_entity, scene_text, count=1)
        return scene_text

    def _build_local_prompt(self, scene_text: str) -> str:
        action_emphasis_prompt = self._build_action_emphasis_prompt(scene_text)
        scene_composition_prompt = self.prompt_config.get("scene_composition_prompt", "").strip()
        local_prompt_suffix = self.prompt_config.get("local_prompt_suffix", "").strip()
        return ", ".join(
            part
            for part in [scene_text, action_emphasis_prompt, scene_composition_prompt, local_prompt_suffix]
            if part
        )

    def _build_action_prompt(self, scene_text: str) -> str:
        action_text = self._extract_main_action(scene_text)
        if not action_text:
            action_text = scene_text
        return self._shorten_prompt_text(
            action_text,
            max_words=int(self.prompt_config.get("scoring_max_words", 20)),
            max_chars=int(self.prompt_config.get("scoring_max_chars", 160)),
        )

    def _build_generation_prompt(
        self,
        scene: Scene,
        scene_text: str,
        action_prompt: str,
        story_context: dict[str, str | list[str] | None],
    ) -> str:
        subject = self._extract_scoring_subject(scene, story_context)
        setting_clause = self._extract_setting_clause(scene_text)
        style_clause = ""
        global_context_clause = ""
        quality_clause = ""
        composition_clause = ""

        if self.prompt_config.get("generation_include_style", True):
            short_style = self._extract_short_style_prompt()
            if short_style:
                style_clause = f", {short_style}"

        if self.prompt_config.get("generation_include_global_context", False):
            short_global_context = self._extract_short_global_context(story_context)
            if short_global_context:
                global_context_clause = f", {short_global_context}"

        if self.prompt_config.get("generation_include_quality_suffix", False):
            short_quality = self._extract_short_quality_prompt()
            if short_quality:
                quality_clause = f", {short_quality}"

        if self.prompt_config.get("generation_include_scene_composition", False):
            short_composition = self._extract_short_scene_composition_prompt()
            if short_composition:
                composition_clause = f", {short_composition}"

        template = self.prompt_config.get(
            "generation_template",
            "{subject}, {action}{setting_clause}{style_clause}",
        ).strip()
        generation_prompt = template.format(
            subject=subject,
            action=action_prompt,
            setting_clause=setting_clause,
            style_clause=style_clause,
            global_context_clause=global_context_clause,
            quality_clause=quality_clause,
            composition_clause=composition_clause,
        ).strip(" ,")
        if global_context_clause:
            generation_prompt = ", ".join([generation_prompt, global_context_clause.strip(", ")])
        if quality_clause:
            generation_prompt = ", ".join([generation_prompt, quality_clause.strip(", ")])
        if composition_clause:
            generation_prompt = ", ".join([generation_prompt, composition_clause.strip(", ")])

        return self._shorten_prompt_text(
            generation_prompt,
            max_words=int(self.prompt_config.get("generation_max_words", 28)),
            max_chars=int(self.prompt_config.get("generation_max_chars", 220)),
        )

    def _build_scoring_prompt(
        self,
        scene: Scene,
        scene_text: str,
        story_context: dict[str, str | list[str] | None],
    ) -> str:
        subject = self._extract_scoring_subject(scene, story_context)
        action = self._extract_main_action(scene_text)
        setting_clause = self._extract_setting_clause(scene_text)
        template = self.prompt_config.get("scoring_template", "{subject}, {action}{setting_clause}").strip()
        scoring_prompt = template.format(
            subject=subject,
            action=action,
            setting_clause=setting_clause,
        ).strip(" ,")

        if self.prompt_config.get("scoring_include_global_context", False):
            global_context = str(story_context.get("location_anchor") or "").strip()
            if global_context:
                scoring_prompt = ", ".join([scoring_prompt, global_context])

        if self.prompt_config.get("scoring_include_style", False):
            style_prompt = self.prompt_config.get("style_prompt", "").strip()
            if style_prompt:
                scoring_prompt = ", ".join([scoring_prompt, style_prompt])

        return self._shorten_prompt_text(
            scoring_prompt,
            max_words=int(self.prompt_config.get("scoring_max_words", 20)),
            max_chars=int(self.prompt_config.get("scoring_max_chars", 160)),
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

    def _gender_identity_phrase(self, story_context: dict[str, str | list[str] | None]) -> str:
        gender = str(story_context.get("gender_presentation") or "").strip().lower()
        if not gender:
            return ""
        return GENDER_PHRASES.get(gender, gender)

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

    def _extract_scoring_subject(
        self,
        scene: Scene,
        story_context: dict[str, str | list[str] | None],
    ) -> str:
        effective_entities = scene.entities or self._scene_fallback_entities(scene, story_context)
        if effective_entities:
            return effective_entities[0]
        primary_entity = str(story_context.get("primary_entity") or "").strip()
        if primary_entity:
            return primary_entity
        return "main subject"

    def _extract_short_style_prompt(self) -> str:
        style_prompt = self.prompt_config.get("style_prompt", "").strip()
        if not style_prompt:
            return ""
        first_fragment = style_prompt.split(",")[0]
        return self._shorten_prompt_text(first_fragment, max_words=8, max_chars=64)

    def _extract_short_global_context(self, story_context: dict[str, str | list[str] | None]) -> str:
        location_anchor = str(story_context.get("location_anchor") or "").strip()
        if not location_anchor:
            return ""
        return self._shorten_prompt_text(location_anchor, max_words=8, max_chars=64)

    def _extract_short_quality_prompt(self) -> str:
        quality_prompt = self.prompt_config.get("quality_suffix", "").strip()
        if not quality_prompt:
            return ""
        first_fragment = quality_prompt.split(",")[0]
        return self._shorten_prompt_text(first_fragment, max_words=6, max_chars=48)

    def _extract_short_scene_composition_prompt(self) -> str:
        composition_prompt = self.prompt_config.get("scene_composition_prompt", "").strip()
        if not composition_prompt:
            return ""
        return self._shorten_prompt_text(composition_prompt, max_words=8, max_chars=64)

    def _extract_main_action(self, scene_text: str) -> str:
        action_text = scene_text
        setting_match = LOCATION_PATTERN.search(action_text)
        if setting_match:
            action_text = action_text[: setting_match.start()].strip(" ,.;")

        pronoun_match = LEADING_PRONOUN_PATTERN.match(action_text)
        if pronoun_match:
            action_text = action_text[pronoun_match.end() :].strip(" ,.;")
        else:
            words = action_text.split(maxsplit=1)
            if len(words) == 2 and words[0][:1].isupper():
                action_text = words[1].strip(" ,.;")

        return self._normalize_fragment(action_text) or self._normalize_fragment(scene_text)

    def _extract_setting_clause(self, scene_text: str) -> str:
        match = LOCATION_PATTERN.search(scene_text)
        if not match:
            return ""
        setting_phrase = self._normalize_fragment(match.group(0))
        if not setting_phrase:
            return ""
        return f", {setting_phrase}"

    def _shorten_prompt_text(self, text: str, *, max_words: int, max_chars: int) -> str:
        normalized = self._normalize_fragment(text)
        words = normalized.split()
        if max_words > 0 and len(words) > max_words:
            normalized = " ".join(words[:max_words])
        if max_chars > 0 and len(normalized) > max_chars:
            normalized = normalized[:max_chars].rstrip(" ,.;")
        return normalized

    def _apply_prompt_rewriter(self, layer_name: str, text: str) -> str:
        rewriter_config = self.prompt_config.get("rewriter", {})
        rewriter_type = str(rewriter_config.get("type", "rule_based")).strip()
        if rewriter_type in {"", "rule_based"}:
            return text
        raise ValueError(f"Unsupported prompt rewriter type for layer '{layer_name}': {rewriter_type}")

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
