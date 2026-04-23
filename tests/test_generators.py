import pytest

from storygen.generators import (
    DiffusersTextToImageGenerator,
    StoryDiffusionDirectGenerator,
    build_backend_metadata,
    build_generation_backend,
)
from storygen.types import GenerationRequest, PromptSpec


class FakePipelineResult:
    def __init__(self) -> None:
        self.images = ["fake-image"]


class FakePipeline:
    def __init__(self) -> None:
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return FakePipelineResult()


class FakeIPAdapterPipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.loaded_adapters = []
        self.ip_adapter_scale = None

    def load_ip_adapter(self, model_id, *, subfolder=None, weight_name=None):
        self.loaded_adapters.append(
            {
                "model_id": model_id,
                "subfolder": subfolder,
                "weight_name": weight_name,
            }
        )

    def set_ip_adapter_scale(self, scale):
        self.ip_adapter_scale = scale


def _prompt_spec() -> PromptSpec:
    return PromptSpec(
        scene_id="SCENE-1",
        style_prompt="style",
        character_prompt="character",
        global_context_prompt="global",
        local_prompt="local",
        action_prompt="action",
        generation_prompt="generation prompt",
        scoring_prompt="scoring prompt",
        full_prompt="full prompt",
        negative_prompt="negative",
    )


def _request(
    extra_options: dict | None = None,
    previous_selected_image_path: str | None = None,
    reference_image_path: str | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        scene_id="SCENE-1",
        candidate_index=0,
        seed=1,
        prompt_spec=_prompt_spec(),
        width=64,
        height=64,
        guidance_scale=0.0,
        num_inference_steps=1,
        reference_image_path=reference_image_path,
        previous_selected_image_path=previous_selected_image_path,
        extra_options=extra_options or {},
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


def test_diffusers_generator_text2img_path_records_route_metadata() -> None:
    backend = DiffusersTextToImageGenerator(
        {"backend": "diffusers_text2img", "model_id": "fake"},
        {"device": "cpu", "torch_dtype": "float32"},
    )
    fake_pipeline = FakePipeline()
    backend.pipeline = fake_pipeline

    candidate = backend.generate_scene(_request({"generation_mode": "text2img", "route_reason": "test"}))

    assert candidate.image == "fake-image"
    assert candidate.metadata["generation_mode"] == "text2img"
    assert candidate.metadata["route_reason"] == "test"
    assert fake_pipeline.kwargs["prompt"] == "generation prompt"
    assert "ip_adapter_image" not in fake_pipeline.kwargs


def test_diffusers_generator_img2img_path_uses_previous_image(tmp_path) -> None:
    from PIL import Image

    init_path = tmp_path / "prev.png"
    Image.new("RGB", (32, 32), color="white").save(init_path)
    backend = DiffusersTextToImageGenerator(
        {"backend": "diffusers_text2img", "model_id": "fake"},
        {"device": "cpu", "torch_dtype": "float32"},
    )
    fake_pipeline = FakePipeline()
    backend.img2img_pipeline = fake_pipeline

    candidate = backend.generate_scene(
        _request(
            {
                "generation_mode": "img2img",
                "route_reason": "test_img2img",
                "init_image_path": str(init_path),
                "img2img_strength": 0.4,
            },
            previous_selected_image_path=str(init_path),
        )
    )

    assert candidate.image == "fake-image"
    assert candidate.metadata["generation_mode"] == "img2img"
    assert candidate.metadata["init_image_path"] == str(init_path)
    assert candidate.metadata["img2img_strength"] == 0.4
    assert fake_pipeline.kwargs["strength"] == 0.4
    assert fake_pipeline.kwargs["image"].size == (64, 64)


def test_diffusers_generator_img2img_requires_init_image() -> None:
    backend = DiffusersTextToImageGenerator(
        {"backend": "diffusers_text2img", "model_id": "fake"},
        {"device": "cpu", "torch_dtype": "float32"},
    )

    with pytest.raises(ValueError, match="img2img generation requires"):
        backend.generate_scene(_request({"generation_mode": "img2img"}))


def test_diffusers_generator_applies_ip_adapter_when_reference_image_present(tmp_path) -> None:
    from PIL import Image

    reference_path = tmp_path / "anchor.png"
    Image.new("RGB", (32, 32), color="white").save(reference_path)
    backend = DiffusersTextToImageGenerator(
        {"backend": "diffusers_text2img", "model_id": "fake"},
        {"device": "cpu", "torch_dtype": "float32"},
    )
    fake_pipeline = FakeIPAdapterPipeline()
    backend.pipeline = fake_pipeline

    candidate = backend.generate_scene(
        _request(
            {
                "generation_mode": "text2img",
                "identity_conditioning_enabled": True,
                "identity_anchor_character_id": "Hero",
                "identity_anchor_type": "half_body",
                "identity_anchor_path": str(reference_path),
                "identity_conditioning_reason": "test",
                "identity_apply_to_modes": ["text2img"],
                "ip_adapter_scale": 0.6,
                "ip_adapter_model_id": "h94/IP-Adapter",
                "ip_adapter_subfolder": "sdxl_models",
                "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
            },
            reference_image_path=str(reference_path),
        )
    )

    assert candidate.metadata["identity_conditioning_applied"] is True
    assert candidate.metadata["identity_anchor_character_id"] == "Hero"
    assert candidate.metadata["ip_adapter_loaded"] is True
    assert fake_pipeline.loaded_adapters == [
        {
            "model_id": "h94/IP-Adapter",
            "subfolder": "sdxl_models",
            "weight_name": "ip-adapter_sdxl.bin",
        }
    ]
    assert fake_pipeline.ip_adapter_scale == 0.6
    assert fake_pipeline.kwargs["ip_adapter_image"].size == (32, 32)


def test_diffusers_generator_rejects_unsupported_ip_adapter_pipeline(tmp_path) -> None:
    from PIL import Image

    reference_path = tmp_path / "anchor.png"
    Image.new("RGB", (32, 32), color="white").save(reference_path)
    backend = DiffusersTextToImageGenerator(
        {"backend": "diffusers_text2img", "model_id": "fake"},
        {"device": "cpu", "torch_dtype": "float32"},
    )
    backend.pipeline = FakePipeline()

    with pytest.raises(ValueError, match="does not support IP-Adapter"):
        backend.generate_scene(
            _request(
                {
                    "generation_mode": "text2img",
                    "identity_conditioning_enabled": True,
                    "identity_apply_to_modes": ["text2img"],
                    "ip_adapter_scale": 0.6,
                    "ip_adapter_model_id": "h94/IP-Adapter",
                    "ip_adapter_subfolder": "sdxl_models",
                    "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
                },
                reference_image_path=str(reference_path),
            )
        )


def test_diffusers_generator_does_not_apply_ip_adapter_to_img2img_by_default(tmp_path) -> None:
    from PIL import Image

    init_path = tmp_path / "prev.png"
    reference_path = tmp_path / "anchor.png"
    Image.new("RGB", (32, 32), color="white").save(init_path)
    Image.new("RGB", (32, 32), color="blue").save(reference_path)
    backend = DiffusersTextToImageGenerator(
        {"backend": "diffusers_text2img", "model_id": "fake"},
        {"device": "cpu", "torch_dtype": "float32"},
    )
    fake_pipeline = FakeIPAdapterPipeline()
    backend.img2img_pipeline = fake_pipeline

    candidate = backend.generate_scene(
        _request(
            {
                "generation_mode": "img2img",
                "init_image_path": str(init_path),
                "identity_conditioning_enabled": True,
                "identity_apply_to_modes": ["text2img"],
                "ip_adapter_model_id": "h94/IP-Adapter",
                "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
            },
            previous_selected_image_path=str(init_path),
            reference_image_path=str(reference_path),
        )
    )

    assert candidate.metadata["identity_conditioning_applied"] is False
    assert "ip_adapter_image" not in fake_pipeline.kwargs
