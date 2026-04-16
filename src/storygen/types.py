from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Scene:
    scene_id: str
    index: int
    raw_text: str
    clean_text: str
    entities: list[str]


@dataclass(slots=True)
class Story:
    source_path: str
    raw_text: str
    scenes: list[Scene]
    all_entities: list[str]
    recurring_entities: list[str]
    entity_to_scene_ids: dict[str, list[str]]


@dataclass(slots=True)
class PromptSpec:
    scene_id: str
    style_prompt: str
    character_prompt: str
    global_context_prompt: str
    local_prompt: str
    full_prompt: str
    negative_prompt: str


@dataclass(slots=True)
class GenerationRequest:
    scene_id: str
    candidate_index: int
    seed: int
    prompt_spec: PromptSpec
    width: int
    height: int
    guidance_scale: float
    num_inference_steps: int
    reference_image_path: str | None = None
    previous_selected_image_path: str | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationCandidate:
    scene_id: str
    candidate_index: int
    seed: int
    prompt_spec: PromptSpec
    image: Any | None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CandidateScore:
    scene_id: str
    candidate_index: int
    seed: int
    score: float
    scorer_name: str
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneSelectionResult:
    scene_id: str
    selected_candidate_index: int
    selected_seed: int
    selected_image_path: str
    selected_score: CandidateScore
    candidate_scores: list[CandidateScore]
    candidate_image_paths: list[str]


@dataclass(slots=True)
class RunSummary:
    run_name: str
    runtime_profile: str
    timestamp: str
    pipeline_version: str
    model_id: str
    scorer_type: str
    scorer_config: dict[str, Any]
    git_commit_id: str | None
    input_story_path: str
    output_root: str
    run_directory: str
    base_seed: int
    candidate_count: int
    resolved_config: dict[str, Any]
    scene_results: list[SceneSelectionResult]


@dataclass(slots=True)
class RunContext:
    run_name: str
    output_root: Path
    run_directory: Path
    scenes_directory: Path
