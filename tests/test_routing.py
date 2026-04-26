from storygen.routing import choose_scene_route
from storygen.types import Scene, Story


def _story() -> Story:
    scenes = [
        Scene("SCENE-1", 0, "<Lily> makes breakfast.", "Lily makes breakfast.", ["Lily"]),
        Scene("SCENE-2", 1, "<Lily> looks out the window.", "Lily looks out the window.", ["Lily"]),
        Scene("SCENE-3", 2, "<Lily> later walks outside.", "Lily later walks outside.", ["Lily"]),
    ]
    return Story("story.txt", "", scenes, ["Lily"], ["Lily"], {"Lily": ["SCENE-1", "SCENE-2", "SCENE-3"]})


def _routing(enabled: bool = True) -> dict:
    return {
        "img2img_enabled": enabled,
        "route_policy": "conservative",
        "use_previous_selected_as_init": True,
        "img2img_strength": 0.45,
        "large_change_keywords": ["later"],
    }


def _guided_routing() -> dict:
    return {
        "img2img_enabled": True,
        "route_policy": "llm_guided_conservative",
        "use_previous_selected_as_init": True,
        "text2img_when_composition_change_needed": False,
        "img2img_strength": 0.45,
        "strength_by_change_level": {"small": 0.35, "medium": 0.65, "large": 0.85},
        "execution_by_change_level": {"small": "img2img", "medium": "img2img", "large": "text2img"},
        "large_change_keywords": [],
    }


def _hint(
    level: str,
    mode: str = "img2img",
    subjects: list[str] | None = None,
    *,
    llm_level: str | None = None,
    adjustment_reason: str | None = None,
    route_factors: dict | None = None,
) -> dict:
    return {
        "continuity_subject_ids": subjects or ["Lily"],
        "continuity_route_hint": mode,
        "llm_route_change_level": llm_level or level,
        "route_change_level": level,
        "route_level_adjustment_reason": adjustment_reason,
        "route_factors": route_factors or {},
        "route_reason": f"{level} visual change",
    }


def test_scene_one_always_uses_text2img() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[0],
        previous_scene=None,
        previous_selected_image_path=None,
        routing_config=_routing(),
    )

    assert decision.generation_mode == "text2img"
    assert decision.route_reason == "Initial scene setup"


def test_routing_disabled_uses_text2img() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_routing(False),
    )

    assert decision.generation_mode == "text2img"
    assert decision.route_reason == "img2img_disabled"


def test_shared_entities_small_change_uses_img2img() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_routing(),
    )

    assert decision.generation_mode == "img2img"
    assert decision.init_image_path == "prev.png"
    assert decision.img2img_strength == 0.45


def test_large_change_keyword_uses_text2img() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[2],
        previous_scene=story.scenes[1],
        previous_selected_image_path="prev.png",
        routing_config=_routing(),
    )

    assert decision.generation_mode == "text2img"
    assert decision.route_reason == "large_change_keyword:later"


def test_missing_previous_selected_image_uses_text2img() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path=None,
        routing_config=_routing(),
    )

    assert decision.generation_mode == "text2img"
    assert decision.route_reason == "missing_previous_selected_image"


def test_llm_guided_small_change_uses_low_strength_img2img() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_guided_routing(),
        route_hint=_hint("small"),
        previous_route_hint=_hint("large", "text2img"),
    )

    assert decision.generation_mode == "img2img"
    assert decision.img2img_strength == 0.35
    assert decision.route_change_level == "small"
    assert decision.continuity_route_hint == "img2img"
    assert decision.continuity_subject_ids == ["Lily"]


def test_llm_guided_medium_change_uses_higher_strength_img2img() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_guided_routing(),
        route_hint=_hint("medium"),
        previous_route_hint=_hint("large", "text2img"),
    )

    assert decision.generation_mode == "img2img"
    assert decision.img2img_strength == 0.65
    assert decision.route_change_level == "medium"


def test_llm_guided_medium_composition_preserving_change_uses_img2img_when_enabled() -> None:
    story = _story()
    routing_config = _guided_routing()
    routing_config["text2img_when_composition_change_needed"] = True

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=routing_config,
        route_hint=_hint("medium", route_factors={"composition_change_needed": False}),
        previous_route_hint=_hint("large", "text2img"),
    )

    assert decision.generation_mode == "img2img"
    assert decision.img2img_strength == 0.65
    assert decision.route_change_level == "medium"


def test_llm_guided_composition_change_uses_text2img_when_config_enabled() -> None:
    story = _story()
    routing_config = _guided_routing()
    routing_config["text2img_when_composition_change_needed"] = True

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=routing_config,
        route_hint=_hint("medium", route_factors={"composition_change_needed": True}),
        previous_route_hint=_hint("large", "text2img"),
    )

    assert decision.generation_mode == "text2img"
    assert decision.img2img_strength is None
    assert decision.route_change_level == "medium"
    assert decision.route_factors == {"composition_change_needed": True}
    assert decision.route_reason.startswith("llm_guided_composition_change_text2img:")


def test_llm_guided_composition_change_config_disabled_preserves_old_medium_img2img_behavior() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_guided_routing(),
        route_hint=_hint("medium", route_factors={"composition_change_needed": True}),
        previous_route_hint=_hint("large", "text2img"),
    )

    assert decision.generation_mode == "img2img"
    assert decision.img2img_strength == 0.65
    assert decision.route_change_level == "medium"


def test_llm_guided_uses_adjusted_route_level_for_execution() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_guided_routing(),
        route_hint=_hint(
            "medium",
            llm_level="small",
            adjustment_reason="small_inconsistent_with_route_factors",
            route_factors={"primary_action_change": True},
        ),
        previous_route_hint=_hint("large", "text2img"),
    )

    assert decision.generation_mode == "img2img"
    assert decision.img2img_strength == 0.65
    assert decision.llm_route_change_level == "small"
    assert decision.route_change_level == "medium"
    assert decision.route_level_adjustment_reason == "small_inconsistent_with_route_factors"
    assert decision.route_factors == {"primary_action_change": True}


def test_llm_guided_large_change_uses_text2img() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_guided_routing(),
        route_hint=_hint("large", "img2img"),
        previous_route_hint=_hint("large", "text2img"),
    )

    assert decision.generation_mode == "text2img"
    assert decision.img2img_strength is None
    assert decision.route_change_level == "large"


def test_llm_guided_subjects_can_resolve_missing_parser_entities() -> None:
    story = Story(
        "story.txt",
        "",
        [
            Scene("SCENE-1", 0, "<Lily> makes breakfast.", "Lily makes breakfast.", ["Lily"]),
            Scene("SCENE-2", 1, "She looks out the window.", "She looks out the window.", []),
        ],
        ["Lily"],
        ["Lily"],
        {"Lily": ["SCENE-1"]},
    )

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_guided_routing(),
        route_hint=_hint("small", subjects=["Lily"]),
        previous_route_hint=_hint("large", "text2img", subjects=["Lily"]),
    )

    assert decision.generation_mode == "img2img"
    assert decision.route_reason.startswith("llm_guided_small:")


def test_llm_guided_invalid_hint_falls_back_to_conservative() -> None:
    story = _story()

    decision = choose_scene_route(
        story=story,
        scene=story.scenes[1],
        previous_scene=story.scenes[0],
        previous_selected_image_path="prev.png",
        routing_config=_guided_routing(),
        route_hint={"continuity_route_hint": "bad", "route_change_level": "small"},
    )

    assert decision.generation_mode == "img2img"
    assert decision.route_reason == "conservative_shared_entities"
