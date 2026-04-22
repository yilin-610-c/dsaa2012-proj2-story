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


def choose_scene_route(
    *,
    story: Story,
    scene: Scene,
    previous_scene: Scene | None,
    previous_selected_image_path: str | None,
    routing_config: dict[str, Any],
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
    if route_policy not in {"conservative", "disabled"}:
        return SceneRouteDecision("text2img", route_policy, f"unsupported_route_policy:{route_policy}")
    if route_policy == "disabled":
        return SceneRouteDecision("text2img", route_policy, "route_policy_disabled")
    if previous_scene is None:
        return SceneRouteDecision("text2img", route_policy, "missing_previous_scene")

    keyword = _has_large_change_keyword(scene, routing_config.get("large_change_keywords", []))
    if keyword:
        return SceneRouteDecision("text2img", route_policy, f"large_change_keyword:{keyword}")

    if not _shares_entities(previous_scene, scene):
        return SceneRouteDecision("text2img", route_policy, "no_shared_scene_entities")

    return SceneRouteDecision(
        "img2img",
        route_policy,
        "conservative_shared_entities",
        init_image_path=previous_selected_image_path,
        img2img_strength=strength,
    )
