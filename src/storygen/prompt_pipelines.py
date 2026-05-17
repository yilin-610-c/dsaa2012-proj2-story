from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

from storygen.character_specs import build_rule_based_character_specs
from storygen.llm_assisted_prompt_builder import LLMAssistedPromptBuilder
from storygen.llm_client import BaseLLMClient
from storygen.native_prompting import LLMDirectPromptBuilder, build_anchor_prompt_bundle
from storygen.native_prompting.types import NativePromptPayload
from storygen.prompt_stack.factory import build_rule_prompt_builder
from storygen.types import PromptBundle, Story


class BasePromptPipeline(ABC):
    @abstractmethod
    def build(self, story: Story) -> PromptBundle:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError


class RuleBasedPromptPipeline(BasePromptPipeline):
    def __init__(self, prompt_config: dict[str, Any]) -> None:
        self.prompt_config = prompt_config
        self.builder = build_rule_prompt_builder(prompt_config)
        self.last_character_specs: dict[str, dict[str, Any]] = {}

    def build(self, story: Story) -> PromptBundle:
        self.last_character_specs = build_rule_based_character_specs(story)
        return PromptBundle(
            scene_prompts=self.builder.build_story_prompts(story),
            story_prompt=None,
            metadata=self.metadata(),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "pipeline": "rule_based",
            "implemented": True,
            "prompt_builder": str(self.prompt_config.get("builder", "legacy")),
            "rewriter_type": self.prompt_config.get("rewriter", {}).get("type", "rule_based"),
            "character_specs": self.last_character_specs,
        }


class LLMAssistedPromptPipeline(BasePromptPipeline):
    def __init__(
        self,
        prompt_config: dict[str, Any],
        *,
        llm_client: BaseLLMClient | None = None,
        event_logger=None,
    ) -> None:
        self.prompt_config = prompt_config
        self.builder = LLMAssistedPromptBuilder(
            prompt_config,
            llm_client=llm_client,
            event_logger=event_logger,
        )

    def build(self, story: Story) -> PromptBundle:
        prompts = self.builder.build_story_prompts(story)
        return PromptBundle(
            scene_prompts=prompts,
            story_prompt=None,
            metadata=self.metadata(),
        )

    def metadata(self) -> dict[str, Any]:
        return self.builder.metadata()


class LLMDirectPromptPipeline(BasePromptPipeline):
    def __init__(
        self,
        prompt_config: dict[str, Any],
        *,
        llm_client: BaseLLMClient | None = None,
        event_logger=None,
    ) -> None:
        self.prompt_config = prompt_config
        self.builder = LLMDirectPromptBuilder(
            prompt_config,
            llm_client=llm_client,
            event_logger=event_logger,
        )
        self.last_bundle: PromptBundle | None = None

    def build(self, story: Story) -> PromptBundle:
        payload = self.builder.build(story)
        if "anchor" not in payload.target_backends:
            raise ValueError("prompt.pipeline=llm_direct requires prompt.llm_direct.targets to include 'anchor'")
        self.last_bundle = build_anchor_prompt_bundle(payload, story, self.prompt_config)
        self.last_bundle.metadata.update(self.metadata())
        return self.last_bundle

    def metadata(self) -> dict[str, Any]:
        return self.builder.metadata()


class ManualPayloadPromptPipeline(BasePromptPipeline):
    def __init__(self, prompt_config: dict[str, Any]) -> None:
        self.prompt_config = prompt_config
        self.manual_config = prompt_config.get("manual_payload", {})
        self.last_metadata: dict[str, Any] = {
            "pipeline": "manual_payload",
            "implemented": True,
        }

    def build(self, story: Story) -> PromptBundle:
        payload_path = self._payload_path()
        raw_payload = self._load_payload(payload_path)
        self._validate_raw_payload(raw_payload, story, payload_path)
        payload = NativePromptPayload.from_dict(raw_payload)
        bundle = build_anchor_prompt_bundle(
            payload,
            story,
            self.prompt_config,
            metadata_source="manual_payload",
        )
        self.last_metadata = {
            "pipeline": "manual_payload",
            "implemented": True,
            "payload_path": str(payload_path),
            "target_backends": list(payload.target_backends),
            "native_prompt_payload": payload.to_dict(),
        }
        bundle.metadata.update(self.last_metadata)
        return bundle

    def metadata(self) -> dict[str, Any]:
        return dict(self.last_metadata)

    def _payload_path(self) -> Path:
        path_value = self.manual_config.get("path")
        if not str(path_value or "").strip():
            raise ValueError("prompt.pipeline=manual_payload requires prompt.manual_payload.path")
        return Path(str(path_value)).expanduser()

    @staticmethod
    def _load_payload(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Manual prompt payload not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Manual prompt payload must be a JSON object: {path}")
        return payload

    @staticmethod
    def _validate_raw_payload(raw_payload: dict[str, Any], story: Story, payload_path: Path) -> None:
        targets = raw_payload.get("target_backends")
        if not isinstance(targets, list) or "anchor" not in {str(target).strip().lower() for target in targets}:
            raise ValueError(f"Manual prompt payload must include target_backends containing 'anchor': {payload_path}")

        characters = raw_payload.get("characters")
        if not isinstance(characters, list) or not characters:
            raise ValueError(f"Manual prompt payload must include at least one character: {payload_path}")
        for index, character in enumerate(characters):
            if not isinstance(character, dict):
                raise ValueError(f"Manual prompt payload characters[{index}] must be an object")
            for field in ("character_id", "subject_type", "stable_identity", "anchor_reference_prompt"):
                if not str(character.get(field, "")).strip():
                    raise ValueError(f"Manual prompt payload characters[{index}].{field} is required")

        scenes = raw_payload.get("scenes")
        if not isinstance(scenes, list):
            raise ValueError(f"Manual prompt payload must include scenes list: {payload_path}")
        expected_scene_ids = [scene.scene_id for scene in story.scenes]
        actual_scene_ids = [str(scene.get("scene_id", "")).strip() for scene in scenes if isinstance(scene, dict)]
        if actual_scene_ids != expected_scene_ids:
            raise ValueError(
                "Manual prompt payload scene ids must match the input story exactly: "
                f"expected {expected_scene_ids}, got {actual_scene_ids}"
            )
        for index, scene in enumerate(scenes):
            if not isinstance(scene, dict):
                raise ValueError(f"Manual prompt payload scenes[{index}] must be an object")
            for field in ("visible_character_ids", "identity_conditioning_subject_id", "anchor_generation_prompt", "scoring_prompt"):
                if field not in scene:
                    raise ValueError(f"Manual prompt payload scenes[{index}].{field} is required")
            if not isinstance(scene.get("visible_character_ids"), list) or not scene["visible_character_ids"]:
                raise ValueError(f"Manual prompt payload scenes[{index}].visible_character_ids must be a non-empty list")
            if not str(scene.get("anchor_generation_prompt", "")).strip():
                raise ValueError(f"Manual prompt payload scenes[{index}].anchor_generation_prompt is required")
            if not str(scene.get("scoring_prompt", "")).strip():
                raise ValueError(f"Manual prompt payload scenes[{index}].scoring_prompt is required")


ApiPromptPipeline = LLMAssistedPromptPipeline


def build_prompt_pipeline(prompt_config: dict[str, Any], *, event_logger=None) -> BasePromptPipeline:
    pipeline_type = prompt_config.get("pipeline", "rule_based")
    if pipeline_type == "rule_based":
        return RuleBasedPromptPipeline(prompt_config)
    if pipeline_type in {"llm_assisted", "api"}:
        return LLMAssistedPromptPipeline(prompt_config, event_logger=event_logger)
    if pipeline_type == "llm_direct":
        return LLMDirectPromptPipeline(prompt_config, event_logger=event_logger)
    if pipeline_type == "manual_payload":
        return ManualPayloadPromptPipeline(prompt_config)
    raise ValueError(f"Unsupported prompt pipeline: {pipeline_type}")
