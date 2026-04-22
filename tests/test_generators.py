import pytest

from storygen.generators import (
    DiffusersTextToImageGenerator,
    StoryDiffusionDirectGenerator,
    build_backend_metadata,
    build_generation_backend,
)


def test_build_generation_backend_supports_diffusers_scene_backend() -> None:
    backend = build_generation_backend(
        {"backend": "diffusers_text2img", "granularity": "scene", "model_id": "fake"},
        {"device": "cpu", "torch_dtype": "float32"},
    )

    assert isinstance(backend, DiffusersTextToImageGenerator)


def test_build_generation_backend_supports_storydiffusion_story_placeholder() -> None:
    backend = build_generation_backend(
        {"backend": "storydiffusion_direct", "granularity": "story", "model_id": "fake"},
        {"device": "cpu", "torch_dtype": "float32"},
    )

    assert isinstance(backend, StoryDiffusionDirectGenerator)


@pytest.mark.parametrize(
    "model_config, message",
    [
        ({"backend": "diffusers_text2img", "granularity": "story"}, "diffusers_text2img only supports"),
        ({"backend": "storydiffusion_direct", "granularity": "scene"}, "storydiffusion_direct only supports"),
        ({"backend": "missing", "granularity": "scene"}, "Unsupported generator backend"),
    ],
)
def test_build_generation_backend_rejects_invalid_combinations(model_config: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_generation_backend(model_config, {"device": "cpu"})


def test_build_backend_metadata_marks_placeholders_unimplemented() -> None:
    metadata = build_backend_metadata(
        {"backend": "storydiffusion_direct", "granularity": "story", "model_id": "fake"},
        {"device": "cuda", "torch_dtype": "float16"},
    )

    assert metadata["backend"] == "storydiffusion_direct"
    assert metadata["granularity"] == "story"
    assert metadata["implemented"] is False
