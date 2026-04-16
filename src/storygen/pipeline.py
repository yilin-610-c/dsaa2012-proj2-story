from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from storygen import __version__
from storygen.generators import DiffusersTextToImageGenerator
from storygen.io.results import (
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
from storygen.prompt_builder import PromptBuilder
from storygen.scoring import CLIPConsistencyScorer, HeuristicScorer
from storygen.types import GenerationRequest, RunSummary


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


def _build_generator(config: dict[str, Any]) -> DiffusersTextToImageGenerator:
    backend_type = config["model"].get("backend")
    if backend_type != "diffusers_text2img":
        raise ValueError(f"Unsupported generator backend: {backend_type}")
    return DiffusersTextToImageGenerator(config["model"], config["runtime"])


def _build_scorer(config: dict[str, Any]):
    scorer_type = config["scoring"].get("type")
    if scorer_type == "heuristic":
        return HeuristicScorer()
    if scorer_type == "clip_consistency":
        return CLIPConsistencyScorer(config["scoring"], config["runtime"])
    raise ValueError(f"Unsupported scorer type: {scorer_type}")


def run_pipeline(config: dict[str, Any]) -> RunSummary:
    repo_root = Path(config["runtime"].get("repo_root", ".")).resolve()
    run_name = _resolve_run_name(config, config["runtime"].get("run_name"))
    run_context = create_run_context(config["runtime"]["output_root"], run_name)

    story = parse_story_file(config["runtime"]["input_path"])
    prompt_builder = PromptBuilder(config["prompt"])
    prompt_specs = prompt_builder.build_story_prompts(story)
    generator = _build_generator(config)
    scorer = _build_scorer(config)

    base_seed = int(config["generation"]["base_seed"])
    candidate_count = int(config["generation"]["candidate_count"])
    previous_results = []
    scene_results = []

    save_resolved_config(run_context.run_directory / "config_resolved.yaml", config)

    for scene in story.scenes:
        prompt_spec = prompt_specs[scene.scene_id]
        candidate_records = []
        candidate_scores = []
        previous_selected_path = previous_results[-1].selected_image_path if previous_results else None

        for candidate_index in range(candidate_count):
            seed = _seed_for_candidate(base_seed, scene.index, candidate_index)
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
            )
            candidate = generator.generate(request)
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

    summary = RunSummary(
        run_name=run_context.run_name,
        runtime_profile=config["runtime"]["profile"],
        timestamp=get_timestamp_string(),
        pipeline_version=__version__,
        model_id=config["model"]["model_id"],
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
    return summary
