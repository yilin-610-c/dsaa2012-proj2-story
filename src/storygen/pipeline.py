from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from storygen import __version__
from storygen.anchor_bank import run_anchor_bank
from storygen.dual_face_refs import split_dual_face_refs
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
from storygen.dual_face_refs import split_dual_face_refs
from storygen.identity_conditioning import select_identity_anchor
from storygen.parser import parse_story_file
from storygen.prompt_pipelines import build_prompt_pipeline
from storygen.routing import choose_scene_route
from storygen.scoring import CLIPConsistencyScorer, HeuristicScorer
from storygen.types import (
    CandidateScore,
    GenerationRequest,
    PromptBundle,
    RunContext,
    RunSummary,
    SceneSelectionResult,
    Story,
    StoryGenerationRequest,
    StoryScenePlan,
)


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
    *,
    scene_plans: list[StoryScenePlan] | None = None,
    anchor_bank_summary: dict[str, Any] | None = None,
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
        scene_plans=scene_plans or [],
        anchor_bank_summary=anchor_bank_summary or {},
        character_specs=prompt_bundle.metadata.get("character_specs", {}),
        dual_face_refs=prompt_bundle.metadata.get("dual_face_refs", {}),
        previous_style_reference_path=prompt_bundle.metadata.get("previous_style_reference_path"),
        extra_options=config["model"].get("extra_options", {}),
    )


def _build_story_scene_plans(
    story: Story,
    prompt_bundle: PromptBundle,
    anchor_bank_summary: dict[str, Any],
    config: dict[str, Any],
    run_context: RunContext,
) -> list[StoryScenePlan]:
    prompt_specs = prompt_bundle.scene_prompts
    scene_route_hints = prompt_bundle.metadata.get("scene_route_hints", {})
    identity_config = config.get("generation", {}).get("identity_conditioning", {})
    plans: list[StoryScenePlan] = []
    dual_face_refs: dict[str, Any] = {}
    is_dual_story = len([entity for entity in story.all_entities if entity.strip()]) >= 2
    if is_dual_story and bool(config.get("generation", {}).get("dual_face_refs", {}).get("enabled", True)):
        for character_id, character_payload in prompt_bundle.metadata.get("character_specs", {}).items():
            anchor_spec = character_payload.get("anchor_spec", {}) if isinstance(character_payload, dict) else {}
            anchor_path = anchor_spec.get("half_body_path") or anchor_spec.get("portrait_path")
            if anchor_path:
                refs = split_dual_face_refs(
                    anchor_image_path=str(anchor_path),
                    output_dir=str(run_context.run_directory / "dual_face_refs" / str(character_id)),
                )
                dual_face_refs[str(character_id)] = {
                    "ref_a_path": refs.ref_a_path,
                    "ref_b_path": refs.ref_b_path,
                    "group_style_reference_path": refs.group_style_reference_path,
                    "metadata": refs.metadata or {},
                }

    for scene in story.scenes:
        prompt_spec = prompt_specs[scene.scene_id]
        route_hint = scene_route_hints.get(scene.scene_id) if isinstance(scene_route_hints, dict) else {}
        identity_plan: dict[str, Any]
        try:
            identity_plan = select_identity_anchor(
                scene=scene,
                route_hint=route_hint,
                generation_mode="text2img",
                anchor_bank_summary=anchor_bank_summary,
                identity_config=identity_config,
            )
        except ValueError as exc:
            identity_plan = {
                "identity_conditioning_enabled": False,
                "identity_conditioning_reason": str(exc),
                "identity_conditioning_error": True,
            }

        anchor_paths: dict[str, str] = {}
        selected_character_id = identity_plan.get("identity_anchor_character_id")
        selected_path = identity_plan.get("identity_anchor_path")
        if selected_character_id and selected_path:
            anchor_paths[str(selected_character_id)] = str(selected_path)

        primary_visible_character_ids = list((route_hint or {}).get("primary_visible_character_ids", []))
        if selected_character_id and str(selected_character_id) not in primary_visible_character_ids:
            primary_visible_character_ids.append(str(selected_character_id))

        plans.append(
            StoryScenePlan(
                scene_id=scene.scene_id,
                scene_index=scene.index,
                prompt_spec=prompt_spec,
                generation_prompt=prompt_spec.generation_prompt,
                scoring_prompt=prompt_spec.scoring_prompt,
                route_hint=dict(route_hint or {}),
                identity_plan=identity_plan,
                anchor_characters=primary_visible_character_ids,
                anchor_paths=anchor_paths,
                metadata={
                    "scene_entities": list(scene.entities),
                    "scene_text": scene.clean_text,
                },
            )
        )
    return plans


def _run_story_backend_placeholder(
    story: Story,
    prompt_bundle: PromptBundle,
    generator: BaseStoryGenerator,
    run_context: RunContext,
    config: dict[str, Any],
    *,
    scene_plans: list[StoryScenePlan],
    anchor_bank_summary: dict[str, Any],
) -> StoryGenerationResult:
    append_event(
        run_context,
        "placeholder_backend_attempted",
        stage="generation",
        backend=config["model"].get("backend"),
        granularity=config["model"].get("granularity"),
        scene_plan_count=len(scene_plans),
        anchor_bank_enabled=bool(anchor_bank_summary.get("enabled", False)),
    )
    request = _build_story_generation_request(
        story,
        prompt_bundle,
        config,
        scene_plans=scene_plans,
        anchor_bank_summary=anchor_bank_summary,
    )
    save_json(
        run_context.logs_directory / "story_backend_request.json",
        {
            "story_id": request.story_id,
            "scene_plans": request.scene_plans,
            "anchor_bank_summary": request.anchor_bank_summary,
            "character_specs": request.character_specs,
            "panel_prompts": request.panel_prompts,
        },
    )
    return generator.generate_story(request)


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
    dual_face_refs: dict[str, Any] = {}
    if len(story.all_entities) >= 2:
        first_anchor = next(iter(prompt_bundle.metadata.get("anchor_bank_summary", {}).get("characters", {}).values()), None)
        group_style_path = None
        if isinstance(first_anchor, dict):
            anchors = first_anchor.get("anchors", {})
            half_body = anchors.get("half_body", {}) if isinstance(anchors, dict) else {}
            group_style_path = str(half_body.get("canonical_image_path") or half_body.get("image_path") or "").strip()
        if group_style_path:
            dual_refs_root = run_context.run_directory / "dual_face_refs"
            dual_face_refs = {"group_style_reference_path": group_style_path}
            for character_id, payload in (prompt_bundle.metadata.get("anchor_bank_summary", {}).get("characters", {}) or {}).items():
                if not isinstance(payload, dict):
                    continue
                anchors = payload.get("anchors", {})
                half_body = anchors.get("half_body", {}) if isinstance(anchors, dict) else {}
                if isinstance(half_body, dict):
                    image_path = str(half_body.get("canonical_image_path") or half_body.get("image_path") or "").strip()
                    if image_path:
                        dual_face_refs[str(character_id)] = {"source_image_path": image_path}
    backend_metadata = build_backend_metadata(config["model"], config["runtime"])
    backend_metadata["img2img_enabled"] = bool(config.get("generation", {}).get("routing", {}).get("img2img_enabled", False))
    backend_metadata["route_policy"] = config.get("generation", {}).get("routing", {}).get("route_policy", "disabled")
    backend_metadata["identity_conditioning"] = {
        "enabled": bool(config.get("generation", {}).get("identity_conditioning", {}).get("enabled", False)),
        "adapter_type": config.get("generation", {}).get("identity_conditioning", {}).get("adapter_type"),
        "anchor_source": config.get("generation", {}).get("identity_conditioning", {}).get("anchor_source"),
        "anchor_type": config.get("generation", {}).get("identity_conditioning", {}).get("anchor_type"),
        "apply_to_modes": config.get("generation", {}).get("identity_conditioning", {}).get("apply_to_modes", []),
        "scale": config.get("generation", {}).get("identity_conditioning", {}).get("scale"),
        "adapter_model_id": config.get("generation", {}).get("identity_conditioning", {}).get("adapter_model_id"),
        "adapter_subfolder": config.get("generation", {}).get("identity_conditioning", {}).get("adapter_subfolder"),
        "adapter_weight_name": config.get("generation", {}).get("identity_conditioning", {}).get("adapter_weight_name"),
    }
    save_json(run_context.logs_directory / "generation_backend.json", backend_metadata)
    append_event(
        run_context,
        "backend_selected",
        stage="generation",
        backend=backend_metadata["backend"],
        granularity=backend_metadata["granularity"],
    )

    if isinstance(generator, BaseStoryGenerator):
        scene_stub_model_id = str(config["model"].get("anchor_bank_model_id") or config["model"].get("model_id") or "")
        if not scene_stub_model_id or scene_stub_model_id == "storydiffusion_direct_placeholder":
            scene_stub_model_id = "stabilityai/sdxl-turbo"
        scene_stub_generator = build_generation_backend(
            {
                **config["model"],
                "backend": "diffusers_text2img",
                "granularity": "scene",
                "model_id": scene_stub_model_id,
            },
            config["runtime"],
        )
        if not isinstance(scene_stub_generator, BaseSceneGenerator):
            raise TypeError(f"Expected a scene-level generator, got {type(scene_stub_generator).__name__}")
        anchor_bank_summary = run_anchor_bank(
            character_specs=prompt_bundle.metadata.get("character_specs", {}),
            anchor_config=config.get("generation", {}).get("anchor_bank", {}),
            run_context=run_context,
            prompt_config=config.get("prompt", {}),
            model_config=config.get("model", {}),
            generator=scene_stub_generator,
            event_logger=lambda event, **metadata: append_event(run_context, event, stage="anchor_bank", **metadata),
        )
        if anchor_bank_summary.get("enabled", False):
            save_json(run_context.logs_directory / "anchor_bank.json", anchor_bank_summary)
        scene_plans = _build_story_scene_plans(story, prompt_bundle, anchor_bank_summary, config, run_context)
        prompt_bundle.metadata["dual_face_refs"] = dual_face_refs
        if scene_plans:
            prompt_bundle.metadata["previous_style_reference_path"] = next(iter(dual_face_refs.values()), {}).get("group_style_reference_path") if dual_face_refs else None
        save_json(run_context.logs_directory / "story_scene_plans.json", scene_plans)
        story_result = _run_story_backend_placeholder(
            story,
            prompt_bundle,
            generator,
            run_context,
            config,
            scene_plans=scene_plans,
            anchor_bank_summary=anchor_bank_summary,
        )
        scene_results = []
        for panel_output, scene in zip(story_result.panel_outputs, story.scenes):
            scene_dir = scene_directory(run_context, scene.index)
            scene_dir.mkdir(parents=True, exist_ok=True)
            save_json(scene_dir / "prompt.json", prompt_specs[scene.scene_id])
            if panel_output.image_path:
                selected_path = save_selected_image(panel_output.image_path, run_context, scene.index)
            elif panel_output.image is not None:
                candidate_path = save_candidate_image(panel_output.image, run_context, scene.index, 0, int(config["generation"]["base_seed"]) + scene.index)
                selected_path = save_selected_image(candidate_path, run_context, scene.index)
            else:
                selected_path = ""
            panel_output.image = None
            panel_output.image_path = selected_path
            save_json(
                scene_dir / "scene_result.json",
                {
                    "scene": scene,
                    "selection": panel_output,
                    "candidates": [panel_output],
                },
            )
            scene_results.append(
                SceneSelectionResult(
                    scene_id=scene.scene_id,
                    selected_candidate_index=0,
                    selected_seed=int(config["generation"]["base_seed"]) + scene.index,
                    selected_image_path=selected_path,
                    selected_score=CandidateScore(
                        scene_id=scene.scene_id,
                        candidate_index=0,
                        seed=int(config["generation"]["base_seed"]) + scene.index,
                        score=0.0,
                        scorer_name="storydiffusion_direct",
                    ),
                    candidate_scores=[],
                    candidate_image_paths=[selected_path] if selected_path else [],
                )
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
            base_seed=int(config["generation"]["base_seed"]),
            candidate_count=int(config["generation"]["candidate_count"]),
            resolved_config=config,
            scene_results=scene_results,
        )
        save_json(run_context.run_directory / "run_summary.json", summary)
        save_json(run_context.run_directory / "manifest.json", build_manifest(summary))
        append_event(run_context, "run_completed", stage="run", run_name=summary.run_name)
        return summary

    if not isinstance(generator, BaseSceneGenerator):
        raise TypeError(f"Expected a scene-level generator, got {type(generator).__name__}")

    anchor_bank_summary = run_anchor_bank(
        character_specs=prompt_bundle.metadata.get("character_specs", {}),
        anchor_config=config.get("generation", {}).get("anchor_bank", {}),
        run_context=run_context,
        prompt_config=config.get("prompt", {}),
        model_config=config.get("model", {}),
        generator=generator,
        event_logger=lambda event, **metadata: append_event(run_context, event, stage="anchor_bank", **metadata),
    )
    if anchor_bank_summary.get("enabled", False):
        save_json(run_context.logs_directory / "anchor_bank.json", anchor_bank_summary)

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
                "identity_conditioning_subject_id": (route_hint or {}).get("identity_conditioning_subject_id"),
                "primary_visible_character_ids": list((route_hint or {}).get("primary_visible_character_ids", [])),
            }
            reference_image_path = None
            identity_config = config.get("generation", {}).get("identity_conditioning", {})
            if identity_config.get("enabled", False):
                try:
                    identity_options = select_identity_anchor(
                        scene=scene,
                        route_hint=route_hint,
                        generation_mode=route_decision.generation_mode,
                        anchor_bank_summary=anchor_bank_summary,
                        identity_config=identity_config,
                    )
                except ValueError as exc:
                    append_event(
                        run_context,
                        "identity_anchor_missing",
                        stage="identity_conditioning",
                        scene_id=scene.scene_id,
                        candidate_index=candidate_index,
                        error=str(exc),
                    )
                    raise
                route_options.update(identity_options)
                if identity_options.get("identity_conditioning_enabled"):
                    reference_image_path = identity_options.get("identity_anchor_path")
                    append_event(
                        run_context,
                        "identity_anchor_selected",
                        stage="identity_conditioning",
                        scene_id=scene.scene_id,
                        candidate_index=candidate_index,
                        character_id=identity_options.get("identity_anchor_character_id"),
                        anchor_type=identity_options.get("identity_anchor_type"),
                        anchor_path=reference_image_path,
                        generation_mode=route_decision.generation_mode,
                    )
                else:
                    append_event(
                        run_context,
                        "identity_anchor_missing",
                        stage="identity_conditioning",
                        scene_id=scene.scene_id,
                        candidate_index=candidate_index,
                        reason=identity_options.get("identity_conditioning_reason"),
                    )
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
                reference_image_path=reference_image_path,
                previous_selected_image_path=previous_selected_path,
                extra_options=route_options,
            )
            candidate = generator.generate_scene(request)
            if candidate.metadata.get("ip_adapter_loaded"):
                append_event(
                    run_context,
                    "ip_adapter_loaded",
                    stage="identity_conditioning",
                    scene_id=scene.scene_id,
                    candidate_index=candidate_index,
                    adapter_model_id=candidate.metadata.get("ip_adapter_model_id"),
                    adapter_subfolder=candidate.metadata.get("ip_adapter_subfolder"),
                    adapter_weight_name=candidate.metadata.get("ip_adapter_weight_name"),
                )
            if candidate.metadata.get("identity_conditioning_applied"):
                append_event(
                    run_context,
                    "ip_adapter_applied",
                    stage="identity_conditioning",
                    scene_id=scene.scene_id,
                    candidate_index=candidate_index,
                    character_id=candidate.metadata.get("identity_anchor_character_id"),
                    anchor_type=candidate.metadata.get("identity_anchor_type"),
                    anchor_path=candidate.metadata.get("identity_anchor_path"),
                    scale=candidate.metadata.get("ip_adapter_scale"),
                )
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
