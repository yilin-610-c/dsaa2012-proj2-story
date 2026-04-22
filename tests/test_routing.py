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
    assert decision.route_reason == "first_scene_uses_text2img"


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
