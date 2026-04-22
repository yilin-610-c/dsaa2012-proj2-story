from __future__ import annotations

from typing import Any

from storygen.types import Scene, SceneRouteDecision, Story


def _has_large_change_keyword(scene: Scene, keywords: list[str]) -> str | None:
    scene_text = scene.clean_text.lower()
    for keyword in keywords:
        normalized = str(keyword).strip().lower()
        if normalized and normalized in scene_text:
            return normalized
    return None


def _shares_entities(previous_scene: Scene, current_scene: Scene) -> bool:
    previous_entities = set(previous_scene.entities)
    current_entities = set(current_scene.entities)
    if previous_entities and current_entities:
        return bool(previous_entities & current_entities)
    return not previous_entities and not current_entities


def _normalized_set(values: list[Any] | tuple[Any, ...] | None) -> set[str]:
    return {str(value).strip().lower() for value in values or [] if str(value).strip()}


def _has_continuity_subject_overlap(
    previous_scene: Scene,
    current_scene: Scene,
    route_hint: dict[str, Any] | None,
    previous_route_hint: dict[str, Any] | None,
) -> bool:
    current_subjects = _normalized_set((route_hint or {}).get("continuity_subject_ids"))
    previous_subjects = _normalized_set((previous_route_hint or {}).get("continuity_subject_ids"))
    current_entities = _normalized_set(current_scene.entities)
    previous_entities = _normalized_set(previous_scene.entities)

    if current_subjects and (previous_subjects | previous_entities):
        return bool(current_subjects & (previous_subjects | previous_entities))
    if current_entities and previous_subjects:
        return bool(current_entities & previous_subjects)
    return _shares_entities(previous_scene, current_scene)


def _valid_route_hint(route_hint: dict[str, Any] | None) -> bool:
    if not isinstance(route_hint, dict):
        return False
    return (
        route_hint.get("continuity_route_hint") in {"text2img", "img2img"}
        and route_hint.get("route_change_level") in {"small", "medium", "large"}
    )


def _route_hint_metadata(route_hint: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_change_level": route_hint.get("route_change_level"),
        "continuity_subject_ids": list(route_hint.get("continuity_subject_ids", [])),
        "continuity_route_hint": route_hint.get("continuity_route_hint"),
        "llm_route_change_level": route_hint.get("llm_route_change_level"),
        "route_level_adjustment_reason": route_hint.get("route_level_adjustment_reason"),
        "route_factors": dict(route_hint.get("route_factors", {})),
    }


def _strength_for_change_level(routing_config: dict[str, Any], change_level: str | None) -> float:
    strength_by_level = routing_config.get("strength_by_change_level", {})
    if isinstance(strength_by_level, dict) and change_level in strength_by_level:
        return float(strength_by_level[change_level])
    return float(routing_config.get("img2img_strength", 0.45))


def _execution_for_change_level(routing_config: dict[str, Any], change_level: str | None, route_hint: str | None) -> str:
    execution_by_level = routing_config.get("execution_by_change_level", {})
    if isinstance(execution_by_level, dict) and change_level in execution_by_level:
        execution = str(execution_by_level[change_level]).strip().lower()
        if execution in {"text2img", "img2img"}:
            return execution
    return route_hint or "text2img"


def _should_force_text2img_for_composition_change(routing_config: dict[str, Any], route_hint: dict[str, Any]) -> bool:
    if not routing_config.get("text2img_when_composition_change_needed", False):
        return False
    route_factors = route_hint.get("route_factors", {})
    return isinstance(route_factors, dict) and bool(route_factors.get("composition_change_needed"))


def choose_scene_route(
    *,
    story: Story,
    scene: Scene,
    previous_scene: Scene | None,
    previous_selected_image_path: str | None,
    routing_config: dict[str, Any],
    route_hint: dict[str, Any] | None = None,
    previous_route_hint: dict[str, Any] | None = None,
) -> SceneRouteDecision:
    route_policy = routing_config.get("route_policy", "disabled")
    strength = float(routing_config.get("img2img_strength", 0.45))

    if scene.index == 0:
        return SceneRouteDecision("text2img", route_policy, "first_scene_uses_text2img")
    if not routing_config.get("img2img_enabled", False):
        return SceneRouteDecision("text2img", route_policy, "img2img_disabled")
    if not routing_config.get("use_previous_selected_as_init", True):
        return SceneRouteDecision("text2img", route_policy, "previous_selected_init_disabled")
    if not previous_selected_image_path:
        return SceneRouteDecision("text2img", route_policy, "missing_previous_selected_image")
    if route_policy not in {"conservative", "llm_guided_conservative", "disabled"}:
        return SceneRouteDecision("text2img", route_policy, f"unsupported_route_policy:{route_policy}")
    if route_policy == "disabled":
        return SceneRouteDecision("text2img", route_policy, "route_policy_disabled")
    if previous_scene is None:
        return SceneRouteDecision("text2img", route_policy, "missing_previous_scene")

    keyword = _has_large_change_keyword(scene, routing_config.get("large_change_keywords", []))
    if keyword:
        return SceneRouteDecision("text2img", route_policy, f"large_change_keyword:{keyword}")

    if route_policy == "llm_guided_conservative" and _valid_route_hint(route_hint):
        change_level = str(route_hint["route_change_level"])
        hint_mode = str(route_hint["continuity_route_hint"])
        execution_mode = _execution_for_change_level(routing_config, change_level, hint_mode)
        hint_metadata = _route_hint_metadata(route_hint)
        route_reason = route_hint.get("route_reason") or "llm_guided_route_hint"
        if _should_force_text2img_for_composition_change(routing_config, route_hint):
            return SceneRouteDecision(
                "text2img",
                route_policy,
                f"llm_guided_composition_change_text2img:{route_reason}",
                **hint_metadata,
            )
        if execution_mode == "text2img":
            return SceneRouteDecision(
                "text2img",
                route_policy,
                f"llm_guided_{change_level}:{route_reason}",
                **hint_metadata,
            )
        if not _has_continuity_subject_overlap(previous_scene, scene, route_hint, previous_route_hint):
            return SceneRouteDecision(
                "text2img",
                route_policy,
                f"llm_guided_no_subject_overlap:{route_reason}",
                **hint_metadata,
            )
        strength = _strength_for_change_level(routing_config, change_level)
        return SceneRouteDecision(
            "img2img",
            route_policy,
            f"llm_guided_{change_level}:{route_reason}",
            init_image_path=previous_selected_image_path,
            img2img_strength=strength,
            **hint_metadata,
        )

    if not _shares_entities(previous_scene, scene):
        return SceneRouteDecision("text2img", route_policy, "no_shared_scene_entities")

    return SceneRouteDecision(
        "img2img",
        route_policy,
        "conservative_shared_entities",
        init_image_path=previous_selected_image_path,
        img2img_strength=strength,
    )
