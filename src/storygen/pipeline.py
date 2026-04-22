from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from storygen import __version__
from storygen.generators import (
    BaseSceneGenerator,
    BaseStoryGenerator,
    build_backend_metadata,
    build_generation_backend,
)
from storygen.io.results import (
    append_event,
    build_manifest,
    create_run_context,
    get_timestamp_string,
    save_candidate_image,
    save_json,
    save_resolved_config,
    save_selected_image,
    scene_directory,
)
from storygen.parser import parse_story_file
from storygen.prompt_pipelines import build_prompt_pipeline
from storygen.routing import choose_scene_route
from storygen.scoring import CLIPConsistencyScorer, HeuristicScorer
from storygen.types import GenerationRequest, PromptBundle, RunContext, RunSummary, Story, StoryGenerationRequest


def _resolve_run_name(config: dict[str, Any], explicit_run_name: str | None = None) -> str:
    if explicit_run_name:
        return explicit_run_name
    prefix = config["runtime"].get("run_name_prefix", "storygen")
    timestamp = get_timestamp_string().replace(":", "-")
    return f"{prefix}_{timestamp}"


def _seed_for_candidate(base_seed: int, scene_index: int, candidate_index: int) -> int:
    return int(base_seed + scene_index * 1000 + candidate_index)


def _get_git_commit_id(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _build_scorer(config: dict[str, Any]):
    scorer_type = config["scoring"].get("type")
    if scorer_type == "heuristic":
        return HeuristicScorer()
    if scorer_type == "clip_consistency":
        return CLIPConsistencyScorer(config["scoring"], config["runtime"])
    raise ValueError(f"Unsupported scorer type: {scorer_type}")


def _build_story_generation_request(
    story: Story,
    prompt_bundle: PromptBundle,
    config: dict[str, Any],
) -> StoryGenerationRequest:
    story_prompt = prompt_bundle.story_prompt
    if story_prompt is not None:
        character_description = story_prompt.character_description
        panel_prompts = story_prompt.panel_prompts
        num_identity_panels = story_prompt.num_identity_panels
        style_name = story_prompt.style_name
        negative_prompt = story_prompt.negative_prompt
    else:
        scene_prompts = [prompt_bundle.scene_prompts[scene.scene_id] for scene in story.scenes]
        character_description = ", ".join(
            dict.fromkeys(prompt.character_prompt for prompt in scene_prompts if prompt.character_prompt)
        )
        panel_prompts = [prompt.generation_prompt for prompt in scene_prompts]
        num_identity_panels = int(config["model"].get("num_identity_panels", 3))
        style_name = config["prompt"].get("style_name")
        negative_prompt = config["prompt"].get("negative_prompt", "")

    return StoryGenerationRequest(
        story_id=Path(story.source_path).stem,
        seed=int(config["generation"]["base_seed"]),
        character_description=character_description,
        panel_prompts=panel_prompts,
        num_identity_panels=num_identity_panels,
        style_name=style_name,
        negative_prompt=negative_prompt,
        width=int(config["model"]["width"]),
        height=int(config["model"]["height"]),
        reference_image_path=config["model"].get("reference_image_path"),
        extra_options=config["model"].get("extra_options", {}),
    )


def _run_story_backend_placeholder(
    story: Story,
    prompt_bundle: PromptBundle,
    generator: BaseStoryGenerator,
    run_context: RunContext,
    config: dict[str, Any],
) -> None:
    append_event(
        run_context,
        "placeholder_backend_attempted",
        stage="generation",
        backend=config["model"].get("backend"),
        granularity=config["model"].get("granularity"),
    )
    request = _build_story_generation_request(story, prompt_bundle, config)
    generator.generate_story(request)


def run_pipeline(config: dict[str, Any]) -> RunSummary:
    repo_root = Path(config["runtime"].get("repo_root", ".")).resolve()
    run_name = _resolve_run_name(config, config["runtime"].get("run_name"))
    run_context = create_run_context(config["runtime"]["output_root"], run_name)
    append_event(run_context, "run_started", stage="run", profile=config["runtime"]["profile"])
    save_resolved_config(run_context.run_directory / "config_resolved.yaml", config)

    story = parse_story_file(config["runtime"]["input_path"])
    prompt_pipeline = build_prompt_pipeline(
        config["prompt"],
        event_logger=lambda event, **metadata: append_event(run_context, event, stage="prompt", **metadata),
    )
    append_event(
        run_context,
        "prompt_pipeline_selected",
        stage="prompt",
        prompt_pipeline=prompt_pipeline.metadata().get("pipeline"),
    )
    prompt_bundle = prompt_pipeline.build(story)
    save_json(run_context.logs_directory / "prompt_pipeline.json", prompt_pipeline.metadata())
    save_json(run_context.logs_directory / "prompt_bundle.json", prompt_bundle.metadata)
    prompt_specs = prompt_bundle.scene_prompts
    scene_route_hints = prompt_bundle.metadata.get("scene_route_hints", {})
    generator = build_generation_backend(config["model"], config["runtime"])
    backend_metadata = build_backend_metadata(config["model"], config["runtime"])
    backend_metadata["img2img_enabled"] = bool(config.get("generation", {}).get("routing", {}).get("img2img_enabled", False))
    backend_metadata["route_policy"] = config.get("generation", {}).get("routing", {}).get("route_policy", "disabled")
    save_json(run_context.logs_directory / "generation_backend.json", backend_metadata)
    append_event(
        run_context,
        "backend_selected",
        stage="generation",
        backend=backend_metadata["backend"],
        granularity=backend_metadata["granularity"],
    )

    if isinstance(generator, BaseStoryGenerator):
        _run_story_backend_placeholder(story, prompt_bundle, generator, run_context, config)

    if not isinstance(generator, BaseSceneGenerator):
        raise TypeError(f"Expected a scene-level generator, got {type(generator).__name__}")

    scorer = _build_scorer(config)

    base_seed = int(config["generation"]["base_seed"])
    candidate_count = int(config["generation"]["candidate_count"])
    previous_results = []
    scene_results = []

    for scene in story.scenes:
        append_event(
            run_context,
            "scene_generation_started",
            stage="generation",
            scene_id=scene.scene_id,
            scene_index=scene.index,
        )
        prompt_spec = prompt_specs[scene.scene_id]
        candidate_records = []
        candidate_scores = []
        previous_selected_path = previous_results[-1].selected_image_path if previous_results else None
        previous_scene = story.scenes[scene.index - 1] if scene.index > 0 else None
        route_hint = scene_route_hints.get(scene.scene_id) if isinstance(scene_route_hints, dict) else None
        previous_route_hint = (
            scene_route_hints.get(previous_scene.scene_id)
            if isinstance(scene_route_hints, dict) and previous_scene is not None
            else None
        )

        for candidate_index in range(candidate_count):
            seed = _seed_for_candidate(base_seed, scene.index, candidate_index)
            route_decision = choose_scene_route(
                story=story,
                scene=scene,
                previous_scene=previous_scene,
                previous_selected_image_path=previous_selected_path,
                routing_config=config.get("generation", {}).get("routing", {}),
                route_hint=route_hint,
                previous_route_hint=previous_route_hint,
            )
            route_options = {
                "generation_mode": route_decision.generation_mode,
                "route_policy": route_decision.route_policy,
                "route_reason": route_decision.route_reason,
                "init_image_path": route_decision.init_image_path,
                "img2img_strength": route_decision.img2img_strength,
                "route_change_level": route_decision.route_change_level,
                "continuity_subject_ids": route_decision.continuity_subject_ids,
                "continuity_route_hint": route_decision.continuity_route_hint,
                "llm_route_change_level": route_decision.llm_route_change_level,
                "route_level_adjustment_reason": route_decision.route_level_adjustment_reason,
                "route_factors": route_decision.route_factors,
            }
            append_event(
                run_context,
                "generation_route_selected",
                stage="generation",
                scene_id=scene.scene_id,
                candidate_index=candidate_index,
                **route_options,
            )
            request = GenerationRequest(
                scene_id=scene.scene_id,
                candidate_index=candidate_index,
                seed=seed,
                prompt_spec=prompt_spec,
                width=int(config["model"]["width"]),
                height=int(config["model"]["height"]),
                guidance_scale=float(config["model"]["guidance_scale"]),
                num_inference_steps=int(config["model"]["num_inference_steps"]),
                reference_image_path=None,
                previous_selected_image_path=previous_selected_path,
                extra_options=route_options,
            )
            candidate = generator.generate_scene(request)
            candidate.image_path = save_candidate_image(candidate.image, run_context, scene.index, candidate_index, seed)
            candidate.image = None
            candidate_records.append(candidate)
            candidate_scores.append(
                scorer.score_candidate(
                    story=story,
                    scene=scene,
                    prompt_spec=prompt_spec,
                    candidate=candidate,
                    previous_results=previous_results,
                )
            )

        selection = scorer.select_best(
            scene_id=scene.scene_id,
            candidates=candidate_records,
            candidate_scores=candidate_scores,
        )
        selection.selected_image_path = save_selected_image(selection.selected_image_path, run_context, scene.index)

        scene_dir = scene_directory(run_context, scene.index)
        save_json(scene_dir / "prompt.json", prompt_spec)
        save_json(
            scene_dir / "scene_result.json",
            {
                "scene": scene,
                "selection": selection,
                "candidates": candidate_records,
            },
        )
        scene_results.append(selection)
        previous_results.append(selection)
        append_event(
            run_context,
            "scene_generation_completed",
            stage="generation",
            scene_id=scene.scene_id,
            scene_index=scene.index,
            selected_candidate_index=selection.selected_candidate_index,
            selected_seed=selection.selected_seed,
        )

    summary = RunSummary(
        run_name=run_context.run_name,
        runtime_profile=config["runtime"]["profile"],
        timestamp=get_timestamp_string(),
        pipeline_version=__version__,
        model_id=config["model"]["model_id"],
        prompt_pipeline=config["prompt"].get("pipeline", "rule_based"),
        generation_backend=config["model"]["backend"],
        generation_granularity=config["model"].get("granularity", "scene"),
        scorer_type=config["scoring"]["type"],
        scorer_config=config["scoring"],
        git_commit_id=_get_git_commit_id(repo_root),
        input_story_path=story.source_path,
        output_root=str(run_context.output_root),
        run_directory=str(run_context.run_directory),
        base_seed=base_seed,
        candidate_count=candidate_count,
        resolved_config=config,
        scene_results=scene_results,
    )

    save_json(run_context.run_directory / "run_summary.json", summary)
    save_json(run_context.run_directory / "manifest.json", build_manifest(summary))
    append_event(run_context, "run_completed", stage="run", scene_count=len(scene_results))
    return summary
