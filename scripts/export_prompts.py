from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from storygen.cli import _parse_set_override
from storygen.config import resolve_config
from storygen.io.results import get_timestamp_string, save_json
from storygen.parser import parse_story_file
from storygen.prompt_pipelines import build_prompt_pipeline
from storygen.types import PromptBundle, PromptSpec, Story


PIPELINE_CHOICES = ("both", "rule_based", "llm_assisted")
ROUTE_HINT_KEYS = {
    "identity_conditioning_subject_id",
    "primary_visible_character_ids",
    "continuity_subject_ids",
    "continuity_route_hint",
    "llm_route_change_level",
    "route_change_level",
    "route_level_adjustment_reason",
    "route_hint_adjustment_reason",
    "route_factors",
    "route_reason",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export story prompt pipeline outputs without generating images")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["test_set/*.txt"],
        help="Story input paths or glob patterns. Quote globs to let the script expand them.",
    )
    parser.add_argument("--output-dir", default="outputs/prompt_audit", help="Directory for audit.json and audit.md")
    parser.add_argument("--pipelines", choices=PIPELINE_CHOICES, default="both", help="Prompt pipelines to export")
    parser.add_argument("--llm-profile", default="llm_prompt_text2img", help="Profile used for llm_assisted prompts")
    parser.add_argument("--rule-profile", default="smoke_test", help="Profile used for rule_based prompts")
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set a dotted config override. Can be repeated.",
    )
    return parser


def expand_input_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        candidates = matches if matches else [pattern]
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_file():
                raise FileNotFoundError(f"Prompt audit input does not exist or is not a file: {candidate}")
            key = str(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return sorted(paths, key=lambda item: str(item))


def parse_overrides(values: list[str]) -> dict[str, Any]:
    overrides = {}
    for override in values:
        key, value = _parse_set_override(override)
        overrides[key] = value
    return overrides


def selected_pipelines(value: str) -> list[str]:
    if value == "both":
        return ["rule_based", "llm_assisted"]
    return [value]


def _profile_for_pipeline(pipeline_name: str, *, rule_profile: str, llm_profile: str) -> str:
    if pipeline_name == "rule_based":
        return rule_profile
    if pipeline_name == "llm_assisted":
        return llm_profile
    raise ValueError(f"Unsupported prompt pipeline for audit: {pipeline_name}")


def _pipeline_config_overrides(pipeline_name: str) -> dict[str, Any]:
    if pipeline_name == "rule_based":
        return {"prompt.pipeline": "rule_based"}
    if pipeline_name == "llm_assisted":
        return {
            "prompt.pipeline": "llm_assisted",
            "prompt.llm.fallback_to_rule_based": False,
        }
    raise ValueError(f"Unsupported prompt pipeline for audit: {pipeline_name}")


def _scene_route_hints_from_plans(scene_plans: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hints = {}
    for scene_id, scene_plan in scene_plans.items():
        if not isinstance(scene_plan, dict):
            continue
        hints[scene_id] = {key: scene_plan.get(key) for key in ROUTE_HINT_KEYS if key in scene_plan}
    return hints


def _prompt_spec_payload(prompt_spec: PromptSpec) -> dict[str, Any]:
    return asdict(prompt_spec)


def _scene_payload(
    *,
    story: Story,
    scene_index: int,
    prompt_bundle: PromptBundle,
    scene_plans: dict[str, Any],
    scene_route_hints: dict[str, Any],
) -> dict[str, Any]:
    scene = story.scenes[scene_index]
    prompt_spec = prompt_bundle.scene_prompts[scene.scene_id]
    return {
        "source_path": story.source_path,
        "scene_id": scene.scene_id,
        "scene_index": scene.index,
        "raw_text": scene.raw_text,
        "clean_text": scene.clean_text,
        "entities": list(scene.entities),
        "story_all_entities": list(story.all_entities),
        "story_recurring_entities": list(story.recurring_entities),
        "prompt": _prompt_spec_payload(prompt_spec),
        "scene_plan": scene_plans.get(scene.scene_id),
        "scene_route_hint": scene_route_hints.get(scene.scene_id),
    }


def _build_pipeline_audit(
    *,
    story: Story,
    pipeline_name: str,
    config_path: str | Path,
    profile: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    pipeline_overrides = {
        **overrides,
        **_pipeline_config_overrides(pipeline_name),
    }
    config = resolve_config(config_path, profile, overrides=pipeline_overrides)
    events: list[dict[str, Any]] = []
    prompt_pipeline = build_prompt_pipeline(
        config["prompt"],
        event_logger=lambda event, **metadata: events.append({"event": event, **metadata}),
    )
    prompt_bundle = prompt_pipeline.build(story)
    metadata = prompt_pipeline.metadata()
    scene_plans = metadata.get("scene_plans", {})
    scene_route_hints = metadata.get("scene_route_hints", {})
    if not isinstance(scene_plans, dict):
        scene_plans = {}
    if not isinstance(scene_route_hints, dict) or not scene_route_hints:
        scene_route_hints = _scene_route_hints_from_plans(scene_plans)
    return {
        "status": "ok",
        "pipeline": pipeline_name,
        "profile": profile,
        "resolved_prompt_pipeline": config["prompt"].get("pipeline"),
        "metadata": metadata,
        "events": events,
        "scenes": [
            _scene_payload(
                story=story,
                scene_index=index,
                prompt_bundle=prompt_bundle,
                scene_plans=scene_plans,
                scene_route_hints=scene_route_hints,
            )
            for index in range(len(story.scenes))
        ],
    }


def build_story_audit(
    *,
    input_path: Path,
    pipeline_names: list[str],
    config_path: str | Path,
    rule_profile: str,
    llm_profile: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    story = parse_story_file(input_path)
    story_payload = {
        "source_path": story.source_path,
        "story_id": input_path.stem,
        "raw_text": story.raw_text,
        "all_entities": list(story.all_entities),
        "recurring_entities": list(story.recurring_entities),
        "entity_to_scene_ids": dict(story.entity_to_scene_ids),
        "pipelines": {},
    }
    for pipeline_name in pipeline_names:
        profile = _profile_for_pipeline(pipeline_name, rule_profile=rule_profile, llm_profile=llm_profile)
        try:
            story_payload["pipelines"][pipeline_name] = _build_pipeline_audit(
                story=story,
                pipeline_name=pipeline_name,
                config_path=config_path,
                profile=profile,
                overrides=overrides,
            )
        except Exception as exc:
            story_payload["pipelines"][pipeline_name] = {
                "status": "failed",
                "pipeline": pipeline_name,
                "profile": profile,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scenes": [],
            }
    return story_payload


def build_audit_payload(
    *,
    input_paths: list[Path],
    pipeline_names: list[str],
    config_path: str | Path,
    rule_profile: str,
    llm_profile: str,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "prompt_audit",
        "generated_at": get_timestamp_string(),
        "config_path": str(config_path),
        "pipeline_selection": list(pipeline_names),
        "rule_profile": rule_profile,
        "llm_profile": llm_profile,
        "overrides": dict(overrides),
        "stories": [
            build_story_audit(
                input_path=input_path,
                pipeline_names=pipeline_names,
                config_path=config_path,
                rule_profile=rule_profile,
                llm_profile=llm_profile,
                overrides=overrides,
            )
            for input_path in input_paths
        ],
    }


def _json_inline(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_markdown(audit_payload: dict[str, Any]) -> str:
    lines = [
        "# Prompt Audit",
        "",
        f"- Generated at: `{audit_payload.get('generated_at')}`",
        f"- Config: `{audit_payload.get('config_path')}`",
        f"- Pipelines: `{', '.join(audit_payload.get('pipeline_selection', []))}`",
        "",
    ]
    for story in audit_payload.get("stories", []):
        lines.extend(
            [
                f"## {story.get('source_path')}",
                "",
                f"- All entities: `{', '.join(story.get('all_entities', [])) or 'none'}`",
                f"- Recurring entities: `{', '.join(story.get('recurring_entities', [])) or 'none'}`",
                "",
            ]
        )
        for pipeline_name, pipeline_payload in story.get("pipelines", {}).items():
            lines.extend([f"### Pipeline: `{pipeline_name}`", ""])
            if pipeline_payload.get("status") != "ok":
                lines.extend(
                    [
                        f"Status: `failed`",
                        "",
                        f"Error: `{pipeline_payload.get('error_type')}: {pipeline_payload.get('error')}`",
                        "",
                    ]
                )
                continue
            for scene_payload in pipeline_payload.get("scenes", []):
                prompt = scene_payload.get("prompt", {})
                lines.extend(
                    [
                        f"#### {scene_payload.get('scene_id')}",
                        "",
                        f"- Text: {scene_payload.get('clean_text')}",
                        f"- Parsed entities: `{', '.join(scene_payload.get('entities', [])) or 'none'}`",
                        f"- Generation: `{prompt.get('generation_prompt', '')}`",
                        f"- Scoring: `{prompt.get('scoring_prompt', '')}`",
                        f"- Action: `{prompt.get('action_prompt', '')}`",
                        f"- Character: `{prompt.get('character_prompt', '')}`",
                        f"- Global context: `{prompt.get('global_context_prompt', '')}`",
                    ]
                )
                scene_plan = _json_inline(scene_payload.get("scene_plan"))
                route_hint = _json_inline(scene_payload.get("scene_route_hint"))
                if scene_plan:
                    lines.append(f"- Scene plan: `{scene_plan}`")
                if route_hint:
                    lines.append(f"- Route hint: `{route_hint}`")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_audit_outputs(audit_payload: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    save_json(output_path / "audit.json", audit_payload)
    (output_path / "audit.md").parent.mkdir(parents=True, exist_ok=True)
    (output_path / "audit.md").write_text(render_markdown(audit_payload), encoding="utf-8")
    stories_dir = output_path / "stories"
    for story in audit_payload.get("stories", []):
        story_id = story.get("story_id") or Path(str(story.get("source_path", "story"))).stem
        save_json(stories_dir / f"{story_id}.json", story)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        overrides = parse_overrides(args.set_overrides)
        input_paths = expand_input_paths(args.inputs)
        audit_payload = build_audit_payload(
            input_paths=input_paths,
            pipeline_names=selected_pipelines(args.pipelines),
            config_path=args.config,
            rule_profile=args.rule_profile,
            llm_profile=args.llm_profile,
            overrides=overrides,
        )
        write_audit_outputs(audit_payload, args.output_dir)
    except Exception as exc:
        parser.error(str(exc))
    print(f"Prompt audit written to: {args.output_dir}")


if __name__ == "__main__":
    main()
