from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from storygen.prompt_stack.state import PromptState

if TYPE_CHECKING:
    from storygen.prompt_builder import PromptBuilder


@runtime_checkable
class PromptPipelineStep(Protocol):
    """Single step in the modular rule pipeline (story-level or scene-level)."""

    def run(self, state: PromptState) -> None: ...


class StoryContextStep:
    """Fill `story_context` once per story (shared across scenes)."""

    def __init__(self, legacy: PromptBuilder) -> None:
        self._legacy = legacy

    def run(self, state: PromptState) -> None:
        state.story_context = self._legacy.build_story_context(state.story)


class SceneSpecsStep:
    """Materialize PromptSpec per scene using the legacy rule engine."""

    def __init__(self, legacy: PromptBuilder) -> None:
        self._legacy = legacy

    def run(self, state: PromptState) -> None:
        if not state.story_context:
            state.story_context = self._legacy.build_story_context(state.story)
        for scene in state.story.scenes:
            state.scene_specs[scene.scene_id] = self._legacy.build_prompt_for_scene(
                state.story, scene, story_context=state.story_context
            )


def default_rule_steps(legacy: PromptBuilder) -> list[PromptPipelineStep]:
    return [StoryContextStep(legacy), SceneSpecsStep(legacy)]
