import json
from pathlib import Path

from storygen.config import resolve_config
from storygen.generators import BaseSceneGenerator, BaseStoryGenerator
from storygen.pipeline import run_pipeline
from storygen.prompt_builder import PromptBuilder
from storygen.types import (
    GenerationCandidate,
    GenerationRequest,
    PanelGenerationOutput,
    PromptBundle,
    StoryGenerationRequest,
    StoryGenerationResult,
)


def _selected_scene_suffix(path: str | None) -> str | None:
    if path is None:
        return None
    parts = Path(path).parts
    if "scenes" not in parts:
        return Path(path).name
    start = parts.index("scenes")
    return str(Path(*parts[start:]))


class FakeImage:
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"fake image")


class FakeSceneGenerator(BaseSceneGenerator):
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def load(self) -> None:
        return None

    def generate_scene(self, request: GenerationRequest) -> GenerationCandidate:
        self.requests.append(request)
        return GenerationCandidate(
            scene_id=request.scene_id,
            candidate_index=request.candidate_index,
            seed=request.seed,
            prompt_spec=request.prompt_spec,
            image=FakeImage(),
            metadata={"backend": "fake", **request.extra_options},
        )


class FakeStoryGenerator(BaseStoryGenerator):
    def __init__(self) -> None:
        self.requests: list[StoryGenerationRequest] = []

    def load(self) -> None:
        return None

    def generate_story(self, request: StoryGenerationRequest) -> StoryGenerationResult:
        self.requests.append(request)
        return StoryGenerationResult(
            backend="storydiffusion_direct",
            seed=request.seed,
            panel_outputs=[
                PanelGenerationOutput(
                    scene_id=plan.scene_id,
                    panel_index=plan.scene_index,
                    prompt=plan.generation_prompt,
                    image=FakeImage(),
                    metadata={"route_hint": dict(plan.route_hint or {})},
                )
                for plan in request.scene_plans
            ],
            metadata={"scene_plan_count": len(request.scene_plans)},
        )


class FakeGuidedPromptPipeline:
    def __init__(self, prompt_config: dict) -> None:
        self.prompt_config = prompt_config
        self._metadata = {"pipeline": "llm_assisted", "implemented": True, "scene_route_hints": {}}

    def build(self, story) -> PromptBundle:
        self._metadata = {
            "pipeline": "llm_assisted",
            "implemented": True,
            "character_specs": {
                "Lily": {
                    "character_id": "Lily",
                    "age_band": "adult",
                    "gender_presentation": "woman",
                    "metadata": {"source": "llm_assisted"},
                }
            },
            "scene_route_hints": {
                "SCENE-1": {
                    "identity_conditioning_subject_id": "Lily",
                    "primary_visible_character_ids": ["Lily"],
                    "continuity_subject_ids": ["Lily"],
                    "continuity_route_hint": "text2img",
                    "llm_route_change_level": "large",
                    "route_change_level": "large",
                    "route_level_adjustment_reason": None,
                    "route_factors": {"same_subject": True, "same_setting": True},
                    "route_reason": "first scene",
                },
                "SCENE-2": {
                    "identity_conditioning_subject_id": "Lily",
                    "primary_visible_character_ids": ["Lily"],
                    "continuity_subject_ids": ["Lily"],
                    "continuity_route_hint": "img2img",
                    "llm_route_change_level": "small",
                    "route_change_level": "medium",
                    "route_level_adjustment_reason": "small_inconsistent_with_route_factors",
                    "route_factors": {"primary_action_change": True, "composition_change_needed": True},
                    "route_reason": "same subject with visible action change",
                },
            },
        }
        return PromptBundle(
            scene_prompts=PromptBuilder(self.prompt_config).build_story_prompts(story),
            metadata=self._metadata,
        )

    def metadata(self) -> dict:
        return self._metadata


class FakeSmallChangePromptPipeline(FakeGuidedPromptPipeline):
    def build(self, story) -> PromptBundle:
        self._metadata = {
            "pipeline": "llm_assisted",
            "implemented": True,
            "character_specs": {
                "Lily": {
                    "character_id": "Lily",
                    "age_band": "adult",
                    "gender_presentation": "woman",
                    "metadata": {"source": "llm_assisted"},
                }
            },
            "scene_route_hints": {
                scene.scene_id: {
                    "identity_conditioning_subject_id": "Lily",
                    "primary_visible_character_ids": ["Lily"],
                    "continuity_subject_ids": ["Lily"],
                    "continuity_route_hint": "img2img" if scene.index > 0 else "text2img",
                    "llm_route_change_level": "small" if scene.index > 0 else "large",
                    "route_change_level": "small" if scene.index > 0 else "large",
                    "route_level_adjustment_reason": None,
                    "route_factors": {
                        "same_subject": True,
                        "same_setting": True,
                        "composition_change_needed": False,
                    },
                    "route_reason": "small continuity-preserving change" if scene.index > 0 else "first scene",
                }
                for scene in story.scenes
            },
        }
        return PromptBundle(
            scene_prompts=PromptBuilder(self.prompt_config).build_story_prompts(story),
            metadata=self._metadata,
        )


class FakeTwoCharacterIdentityPromptPipeline:
    def __init__(self, prompt_config: dict, *, subject_id: str | None) -> None:
        self.prompt_config = prompt_config
        self.subject_id = subject_id
        self._metadata = {"pipeline": "llm_assisted", "implemented": True, "scene_route_hints": {}}

    def build(self, story) -> PromptBundle:
        self._metadata = {
            "pipeline": "llm_assisted",
            "implemented": True,
            "character_specs": {
                "Jack": {"character_id": "Jack", "metadata": {"source": "llm_assisted"}},
                "Sara": {"character_id": "Sara", "metadata": {"source": "llm_assisted"}},
            },
            "scene_route_hints": {
                scene.scene_id: {
                    "identity_conditioning_subject_id": self.subject_id,
                    "primary_visible_character_ids": ["Jack", "Sara"],
                    "continuity_subject_ids": [],
                    "continuity_route_hint": "text2img",
                    "llm_route_change_level": "large",
                    "route_change_level": "large",
                    "route_level_adjustment_reason": None,
                    "route_factors": {"same_subject": True, "composition_change_needed": True},
                    "route_reason": "two character test",
                }
                for scene in story.scenes
            },
        }
        return PromptBundle(
            scene_prompts=PromptBuilder(self.prompt_config).build_story_prompts(story),
            metadata=self._metadata,
        )

    def metadata(self) -> dict:
        return self._metadata


class FakeMetadataPromptPipeline:
    def __init__(self, prompt_config: dict, *, include_character_specs: bool) -> None:
        self.prompt_config = prompt_config
        self.include_character_specs = include_character_specs
        self._metadata = {"pipeline": "rule_based", "implemented": True, "scene_route_hints": {}}

    def build(self, story) -> PromptBundle:
        self._metadata = {"pipeline": "rule_based", "implemented": True, "scene_route_hints": {}}
        if self.include_character_specs:
            self._metadata["character_specs"] = {
                "Lily": {
                    "character_id": "Lily",
                    "metadata": {"source": "test"},
                }
            }
        return PromptBundle(
            scene_prompts=PromptBuilder(self.prompt_config).build_story_prompts(story),
            metadata=self._metadata,
        )

    def metadata(self) -> dict:
        return self._metadata


class FakeStoryPromptAuditPipeline:
    def __init__(self, prompt_config: dict) -> None:
        self.prompt_config = prompt_config
        self._metadata = {"pipeline": "llm_assisted", "implemented": True}

    def build(self, story) -> PromptBundle:
        prompt_specs = PromptBuilder(self.prompt_config).build_story_prompts(story)
        for scene in story.scenes:
            prompt_spec = prompt_specs[scene.scene_id]
            prompt_spec.generation_prompt = f"optimized prompt audit generation for {scene.scene_id}"
            prompt_spec.scene_consistency_prompt = f"consistent story context for {scene.scene_id}"

        self._metadata = {
            "pipeline": "llm_assisted",
            "implemented": True,
            "provider": "openai",
            "model": "gpt-4o-2024-08-06",
            "schema_version": "v1",
            "builder_version": "llm_assisted_v9",
            "character_specs": {
                "Jack": {"character_id": "Jack", "metadata": {"source": "llm_assisted"}},
                "Sara": {"character_id": "Sara", "metadata": {"source": "llm_assisted"}},
            },
            "scene_plans": {
                scene.scene_id: {
                    "identity_conditioning_subject_id": None,
                    "primary_visible_character_ids": ["Jack", "Sara"],
                    "continuity_subject_ids": ["Jack", "Sara"],
                    "continuity_route_hint": "text2img",
                    "llm_route_change_level": "large",
                    "route_change_level": "large",
                    "route_level_adjustment_reason": None,
                    "route_hint_adjustment_reason": "visible_character_change",
                    "route_factors": {"same_subject": True, "composition_change_needed": True},
                    "route_reason": "dual-primary prompt audit test",
                    "interaction_summary": "Jack and Sara talk together",
                    "spatial_relation": "Jack on the left, Sara on the right",
                    "framing": "medium two-shot, both characters visible",
                    "setting_focus": "park bench",
                    "policy": {"scene_focus_mode": "dual_primary", "visible_character_count": 2},
                }
                for scene in story.scenes
            },
        }
        self._metadata["scene_route_hints"] = {
            scene_id: {
                "identity_conditioning_subject_id": scene_plan["identity_conditioning_subject_id"],
                "primary_visible_character_ids": scene_plan["primary_visible_character_ids"],
                "continuity_subject_ids": scene_plan["continuity_subject_ids"],
                "continuity_route_hint": scene_plan["continuity_route_hint"],
                "llm_route_change_level": scene_plan["llm_route_change_level"],
                "route_change_level": scene_plan["route_change_level"],
                "route_level_adjustment_reason": scene_plan["route_level_adjustment_reason"],
                "route_hint_adjustment_reason": scene_plan["route_hint_adjustment_reason"],
                "route_factors": scene_plan["route_factors"],
                "route_reason": scene_plan["route_reason"],
            }
            for scene_id, scene_plan in self._metadata["scene_plans"].items()
        }
        return PromptBundle(scene_prompts=prompt_specs, metadata=self._metadata)

    def metadata(self) -> dict:
        return self._metadata


def test_run_pipeline_writes_minimal_logs_without_changing_summary_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: FakeSceneGenerator())
    config = resolve_config(
        "configs/base.yaml",
        "smoke_test",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "log_test",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
        },
    )

    summary = run_pipeline(config)
    run_dir = Path(summary.run_directory)

    prompt_log = run_dir / "logs" / "prompt_pipeline.json"
    prompt_bundle_log = run_dir / "logs" / "prompt_bundle.json"
    backend_log = run_dir / "logs" / "generation_backend.json"
    events_log = run_dir / "logs" / "events.jsonl"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert prompt_log.exists()
    assert prompt_bundle_log.exists()
    assert backend_log.exists()
    assert events_log.exists()
    assert manifest["prompt_pipeline"] == "rule_based"
    assert manifest["generation_backend"] == "diffusers_text2img"
    assert manifest["generation_granularity"] == "scene"
    assert len(manifest["selected_outputs"]) == len(summary.scene_results)

    events = [json.loads(line)["event"] for line in events_log.read_text(encoding="utf-8").splitlines()]
    assert "run_started" in events
    assert "prompt_pipeline_selected" in events
    assert "backend_selected" in events
    assert "generation_route_selected" in events
    assert "scene_generation_started" in events
    assert "scene_generation_completed" in events
    assert "run_completed" in events
    prompt_bundle = json.loads(prompt_bundle_log.read_text(encoding="utf-8"))
    assert prompt_bundle["character_specs"]["Lily"]["metadata"]["source"] == "rule_based"


def test_run_pipeline_logs_img2img_routes_when_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: FakeSceneGenerator())
    config = resolve_config(
        "configs/base.yaml",
        "rule_prompt_img2img",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "route_test",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
            "generation.routing.large_change_keywords": [],
        },
    )

    summary = run_pipeline(config)
    run_dir = Path(summary.run_directory)

    scene_two = json.loads((run_dir / "scenes" / "scene_002" / "scene_result.json").read_text(encoding="utf-8"))
    candidate_metadata = scene_two["candidates"][0]["metadata"]
    prompt_log = json.loads((run_dir / "logs" / "prompt_pipeline.json").read_text(encoding="utf-8"))
    prompt_bundle_log = json.loads((run_dir / "logs" / "prompt_bundle.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    route_events = [event for event in events if event["event"] == "generation_route_selected"]

    assert candidate_metadata["generation_mode"] in {"text2img", "img2img"}
    assert len(route_events) == len(summary.scene_results)
    assert route_events[0]["generation_mode"] == "text2img"
    assert route_events[1]["generation_mode"] in {"text2img", "img2img"}
    assert prompt_bundle_log["character_specs"]["Lily"]["metadata"]["source"] == "rule_based"


def test_run_pipeline_logs_llm_guided_route_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: FakeSceneGenerator())
    monkeypatch.setattr("storygen.pipeline.build_prompt_pipeline", lambda prompt, event_logger=None: FakeGuidedPromptPipeline(prompt))
    config = resolve_config(
        "configs/base.yaml",
        "llm_prompt_img2img_guided",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "guided_route_test",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
            "generation.routing.large_change_keywords": [],
        },
    )

    summary = run_pipeline(config)
    run_dir = Path(summary.run_directory)

    scene_two = json.loads((run_dir / "scenes" / "scene_002" / "scene_result.json").read_text(encoding="utf-8"))
    candidate_metadata = scene_two["candidates"][0]["metadata"]
    prompt_log = json.loads((run_dir / "logs" / "prompt_pipeline.json").read_text(encoding="utf-8"))
    prompt_bundle_log = json.loads((run_dir / "logs" / "prompt_bundle.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    route_events = [event for event in events if event["event"] == "generation_route_selected"]

    assert candidate_metadata["generation_mode"] == "text2img"
    assert candidate_metadata["route_change_level"] == "medium"
    assert candidate_metadata["continuity_subject_ids"] == ["Lily"]
    assert candidate_metadata["continuity_route_hint"] == "img2img"
    assert candidate_metadata["llm_route_change_level"] == "small"
    assert candidate_metadata["route_level_adjustment_reason"] == "small_inconsistent_with_route_factors"
    assert candidate_metadata["route_factors"] == {"primary_action_change": True, "composition_change_needed": True}
    assert candidate_metadata["img2img_strength"] is None
    assert candidate_metadata["route_reason"].startswith("llm_guided_composition_change_text2img:")
    assert prompt_log["scene_route_hints"]["SCENE-2"]["route_factors"] == {
        "primary_action_change": True,
        "composition_change_needed": True,
    }
    assert prompt_bundle_log["character_specs"]["Lily"]["metadata"]["source"] == "llm_assisted"
    assert prompt_bundle_log["scene_route_hints"]["SCENE-2"]["route_change_level"] == "medium"
    assert route_events[1]["route_change_level"] == "medium"
    assert route_events[1]["continuity_subject_ids"] == ["Lily"]
    assert route_events[1]["llm_route_change_level"] == "small"
    assert route_events[1]["generation_mode"] == "text2img"


def test_character_specs_metadata_does_not_change_downstream_generation_requests(tmp_path, monkeypatch) -> None:
    captured_requests = {}

    def run_case(run_name: str, include_character_specs: bool) -> list[GenerationRequest]:
        generator = FakeSceneGenerator()
        monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: generator)
        monkeypatch.setattr(
            "storygen.pipeline.build_prompt_pipeline",
            lambda prompt, event_logger=None: FakeMetadataPromptPipeline(
                prompt,
                include_character_specs=include_character_specs,
            ),
        )
        config = resolve_config(
            "configs/base.yaml",
            "smoke_test",
            overrides={
                "runtime.output_root": str(tmp_path),
                "runtime.run_name": run_name,
                "scoring.type": "heuristic",
                "generation.candidate_count": 1,
            },
        )
        run_pipeline(config)
        return generator.requests

    captured_requests["without"] = run_case("without_character_specs", False)
    captured_requests["with"] = run_case("with_character_specs", True)

    for request_without, request_with in zip(captured_requests["without"], captured_requests["with"], strict=True):
        assert request_with.scene_id == request_without.scene_id
        assert request_with.candidate_index == request_without.candidate_index
        assert request_with.seed == request_without.seed
        assert request_with.prompt_spec == request_without.prompt_spec
        assert request_with.reference_image_path == request_without.reference_image_path
        assert _selected_scene_suffix(request_with.previous_selected_image_path) == _selected_scene_suffix(
            request_without.previous_selected_image_path
        )
        assert request_with.extra_options == request_without.extra_options


def test_anchor_bank_disabled_does_not_create_anchor_outputs_or_change_scene_requests(tmp_path, monkeypatch) -> None:
    generator = FakeSceneGenerator()
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: generator)
    config = resolve_config(
        "configs/base.yaml",
        "smoke_test",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "anchor_disabled",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
            "generation.anchor_bank.enabled": False,
        },
    )

    summary = run_pipeline(config)
    run_dir = Path(summary.run_directory)

    assert not (run_dir / "anchors").exists()
    assert not (run_dir / "logs" / "anchor_bank.json").exists()
    assert all(request.reference_image_path is None for request in generator.requests)
    assert all("anchor_type" not in request.extra_options for request in generator.requests)


def test_story_backend_receives_prompt_audit_scene_plans(tmp_path, monkeypatch) -> None:
    story_generator = FakeStoryGenerator()
    scene_stub_generator = FakeSceneGenerator()

    def fake_build_generation_backend(model_config, runtime_config):
        if model_config.get("backend") == "storydiffusion_direct":
            return story_generator
        return scene_stub_generator

    monkeypatch.setattr("storygen.pipeline.build_generation_backend", fake_build_generation_backend)
    monkeypatch.setattr(
        "storygen.pipeline.build_prompt_pipeline",
        lambda prompt, event_logger=None: FakeStoryPromptAuditPipeline(prompt),
    )
    monkeypatch.setattr(
        "storygen.pipeline.run_anchor_bank",
        lambda **kwargs: {
            "enabled": True,
            "characters": {
                "Jack": {"anchors": {}},
                "Sara": {"anchors": {}},
            },
        },
    )
    config = resolve_config(
        "configs/base.yaml",
        "cloud_storydiffusion_debug",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "story_prompt_audit",
            "runtime.input_path": "test_set/06.txt",
            "scoring.type": "heuristic",
            "generation.identity_conditioning.fail_on_missing_anchor": False,
        },
    )

    summary = run_pipeline(config)
    run_dir = Path(summary.run_directory)
    prompt_bundle_log = json.loads((run_dir / "logs" / "prompt_bundle.json").read_text(encoding="utf-8"))
    story_scene_plans = json.loads((run_dir / "logs" / "story_scene_plans.json").read_text(encoding="utf-8"))
    story_backend_request = json.loads((run_dir / "logs" / "story_backend_request.json").read_text(encoding="utf-8"))

    assert prompt_bundle_log["builder_version"] == "llm_assisted_v9"
    assert prompt_bundle_log["scene_route_hints"]["SCENE-1"]["route_hint_adjustment_reason"] == "visible_character_change"
    prompt_audit_scene_plan = prompt_bundle_log["scene_plans"]["SCENE-1"]
    assert prompt_audit_scene_plan["interaction_summary"] == "Jack and Sara talk together"
    assert prompt_audit_scene_plan["spatial_relation"] == "Jack on the left, Sara on the right"
    assert prompt_audit_scene_plan["framing"] == "medium two-shot, both characters visible"
    assert prompt_audit_scene_plan["setting_focus"] == "park bench"

    first_scene_plan = story_scene_plans[0]
    assert first_scene_plan["generation_prompt"] == "optimized prompt audit generation for SCENE-1"
    assert first_scene_plan["prompt_spec"]["generation_prompt"] == "optimized prompt audit generation for SCENE-1"
    assert first_scene_plan["prompt_spec"]["scene_consistency_prompt"] == "consistent story context for SCENE-1"
    assert first_scene_plan["route_hint"]["route_hint_adjustment_reason"] == "visible_character_change"

    assert story_backend_request["scene_plans"][0]["generation_prompt"] == first_scene_plan["generation_prompt"]
    assert story_backend_request["scene_plans"][0]["route_hint"] == first_scene_plan["route_hint"]
    assert story_backend_request["scene_plans"][0]["prompt_spec"]["scene_consistency_prompt"] == (
        "consistent story context for SCENE-1"
    )
    assert story_generator.requests
    request = story_generator.requests[0]
    assert request.scene_plans[0].generation_prompt == "optimized prompt audit generation for SCENE-1"
    assert request.scene_plans[0].route_hint["route_hint_adjustment_reason"] == "visible_character_change"
    assert request.scene_plans[0].prompt_spec.scene_consistency_prompt == "consistent story context for SCENE-1"


def test_anchor_bank_enabled_generates_run_local_anchors_without_scene_reference_paths(tmp_path, monkeypatch) -> None:
    generator = FakeSceneGenerator()
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: generator)
    config = resolve_config(
        "configs/base.yaml",
        "llm_prompt_anchor_bank",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "anchor_enabled",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
            "prompt.pipeline": "rule_based",
            "runtime.input_path": "test_set/06.txt",
            "generation.anchor_bank.anchor_types": ["portrait", "half_body"],
        },
    )

    summary = run_pipeline(config)
    run_dir = Path(summary.run_directory)
    anchor_log = json.loads((run_dir / "logs" / "anchor_bank.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_names = [event["event"] for event in events]

    assert (run_dir / "anchors" / "Jack" / "anchor_spec.json").exists()
    assert (run_dir / "anchors" / "Jack" / "canonical_anchor.json").exists()
    assert (run_dir / "anchors" / "Sara" / "anchor_spec.json").exists()
    assert (run_dir / "anchors" / "Jack" / "portrait.png").exists()
    assert (run_dir / "anchors" / "Jack" / "half_body_cand_0.png").exists()
    assert (run_dir / "anchors" / "Jack" / "half_body_cand_1.png").exists()
    assert (run_dir / "anchors" / "Jack" / "half_body_cand_2.png").exists()
    assert (run_dir / "anchors" / "Jack" / "canonical_half_body.png").exists()
    assert list(anchor_log["characters"]) == ["Jack", "Sara"]
    assert anchor_log["characters"]["Jack"]["anchors"]["portrait"]["seed"] == 900000
    assert anchor_log["characters"]["Jack"]["anchors"]["half_body"]["candidate_count"] == 3
    assert anchor_log["characters"]["Jack"]["anchors"]["half_body"]["canonical_image_path"].endswith("canonical_half_body.png")
    assert "anchor_bank_started" in event_names
    assert "anchor_generation_started" in event_names
    assert "anchor_generation_completed" in event_names
    assert "anchor_canonical_selected" in event_names
    assert "anchor_bank_completed" in event_names

    scene_requests = [request for request in generator.requests if not request.scene_id.startswith("ANCHOR-")]
    anchor_requests = [request for request in generator.requests if request.scene_id.startswith("ANCHOR-")]
    assert len(anchor_requests) == 8
    assert len(scene_requests) == len(summary.scene_results)
    assert all(request.reference_image_path is None for request in scene_requests)
    assert all("anchor_type" not in request.extra_options for request in scene_requests)
    assert all(request.extra_options["generation_mode"] == "text2img" for request in anchor_requests)


def test_ip_adapter_text2img_profile_passes_anchor_reference_to_scene_requests(tmp_path, monkeypatch) -> None:
    generator = FakeSceneGenerator()
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: generator)
    monkeypatch.setattr("storygen.pipeline.build_prompt_pipeline", lambda prompt, event_logger=None: FakeGuidedPromptPipeline(prompt))
    config = resolve_config(
        "configs/base.yaml",
        "llm_prompt_ip_adapter_text2img",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "ip_adapter_text2img",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
        },
    )

    summary = run_pipeline(config)
    run_dir = Path(summary.run_directory)
    backend_log = json.loads((run_dir / "logs" / "generation_backend.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_names = [event["event"] for event in events]
    scene_requests = [request for request in generator.requests if not request.scene_id.startswith("ANCHOR-")]
    anchor_requests = [request for request in generator.requests if request.scene_id.startswith("ANCHOR-")]

    assert backend_log["identity_conditioning"]["enabled"] is True
    assert len(anchor_requests) == 4
    assert len(scene_requests) == len(summary.scene_results)
    assert all(request.reference_image_path for request in scene_requests)
    assert all(request.reference_image_path is None for request in anchor_requests)
    assert all(request.reference_image_path.endswith("canonical_half_body.png") for request in scene_requests)
    assert all(request.extra_options["identity_anchor_type"] == "half_body" for request in scene_requests)
    assert all(request.extra_options["identity_conditioning_enabled"] is True for request in scene_requests)
    assert "identity_anchor_selected" in event_names


def test_ip_adapter_text2img_profile_uses_explicit_identity_subject_for_multi_character_story(tmp_path, monkeypatch) -> None:
    generator = FakeSceneGenerator()
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: generator)
    monkeypatch.setattr(
        "storygen.pipeline.build_prompt_pipeline",
        lambda prompt, event_logger=None: FakeTwoCharacterIdentityPromptPipeline(prompt, subject_id="Jack"),
    )
    config = resolve_config(
        "configs/base.yaml",
        "llm_prompt_ip_adapter_text2img",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "ip_adapter_two_character_subject",
            "runtime.input_path": "test_set/06.txt",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
        },
    )

    run_pipeline(config)
    scene_requests = [request for request in generator.requests if not request.scene_id.startswith("ANCHOR-")]

    assert scene_requests
    assert all("/Jack/" in request.reference_image_path for request in scene_requests)
    assert all(request.extra_options["identity_anchor_character_id"] == "Jack" for request in scene_requests)
    assert all(request.extra_options["identity_conditioning_reason"] == "identity_subject_id" for request in scene_requests)


def test_ip_adapter_text2img_profile_skips_ambiguous_multi_character_scene_when_configured(tmp_path, monkeypatch) -> None:
    generator = FakeSceneGenerator()
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: generator)
    monkeypatch.setattr(
        "storygen.pipeline.build_prompt_pipeline",
        lambda prompt, event_logger=None: FakeTwoCharacterIdentityPromptPipeline(prompt, subject_id=None),
    )
    config = resolve_config(
        "configs/base.yaml",
        "llm_prompt_ip_adapter_text2img",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "ip_adapter_two_character_ambiguous",
            "runtime.input_path": "test_set/06.txt",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
            "generation.identity_conditioning.fail_on_missing_anchor": False,
        },
    )

    run_pipeline(config)
    scene_requests = [request for request in generator.requests if not request.scene_id.startswith("ANCHOR-")]

    assert scene_requests
    assert all(request.reference_image_path is None for request in scene_requests)
    assert all(request.extra_options["identity_conditioning_enabled"] is False for request in scene_requests)
    assert all(
        request.extra_options["identity_conditioning_reason"] == "ambiguous_or_missing_scene_character"
        for request in scene_requests
    )


def test_hybrid_identity_profile_does_not_apply_ip_adapter_to_img2img_scenes_by_default(tmp_path, monkeypatch) -> None:
    generator = FakeSceneGenerator()
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: generator)
    monkeypatch.setattr(
        "storygen.pipeline.build_prompt_pipeline",
        lambda prompt, event_logger=None: FakeSmallChangePromptPipeline(prompt),
    )
    config = resolve_config(
        "configs/base.yaml",
        "llm_prompt_hybrid_identity",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "hybrid_identity",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
            "generation.routing.large_change_keywords": [],
        },
    )

    run_pipeline(config)
    scene_requests = [request for request in generator.requests if not request.scene_id.startswith("ANCHOR-")]
    text2img_requests = [request for request in scene_requests if request.extra_options["generation_mode"] == "text2img"]
    img2img_requests = [request for request in scene_requests if request.extra_options["generation_mode"] == "img2img"]

    assert text2img_requests
    assert img2img_requests
    assert all(request.reference_image_path for request in text2img_requests)
    assert all(request.reference_image_path is None for request in img2img_requests)
    assert all(
        request.extra_options["identity_conditioning_reason"] == "generation_mode_not_enabled:img2img"
        for request in img2img_requests
    )
