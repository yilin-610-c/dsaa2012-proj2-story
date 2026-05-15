#!/usr/bin/env python3
"""Generate DIVERSE character reference images for PixArt DreamBooth LoRA.

Two-phase:
  Phase 1: anchor_bank → 1 canonical character image (same backbone as LoRA target).
  Phase 2: diverse prompts → 12+ training images (same backbone).

Backends (--ref-backend):
  pixart (default): PixArt-α (dit_text2img) for both phases — matches PixArt LoRA train/infer,
                    avoids SDXL→PixArt domain gap. No IP-Adapter (PixArt has none); diversity
                    comes from prompt variety + different seeds. Canonical still anchors identity
                    via anchor_bank half-body selection.
  sdxl: Original path — SDXL-Turbo anchor_bank + IP-Adapter Phase 2 (legacy).

Usage:
    PYTHONPATH=src python scripts/gen_lora_ref_images.py \
        --input test_set/01.txt --output-dir training_data/my_character

    PYTHONPATH=src python scripts/gen_lora_ref_images.py \
        --input test_set/01.txt --output-dir training_data/x --ref-backend sdxl
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from storygen.types import GenerationRequest, PromptSpec


DIVERSE_PROMPTS = [
    # Full body, outdoor
    "sks {name}, full body shot, standing in a sunlit park, relaxed pose, natural daylight, photorealistic",
    "sks {name}, walking on a quiet street, full body, candid shot, afternoon light",
    "sks {name}, sitting on a park bench, full body, reading, soft shade, garden background",
    # Half body, indoor
    "sks {name}, standing by a window, half body, side profile, soft indoor light, thoughtful",
    "sks {name}, sitting at a wooden desk, half body, writing, warm lamp light, bookshelves behind",
    "sks {name}, in a cozy living room, half body, relaxing on a chair, warm evening light",
    # Close-up / portrait (different angles)
    "sks {name}, close-up portrait, looking ahead, soft natural light, neutral background",
    "sks {name}, medium close-up, three-quarter turn, gentle expression, soft background blur",
    # Action poses
    "sks {name}, reaching for a book on a shelf, full body, library, soft light",
    "sks {name}, holding a cup, half body, cafe background, warm morning light",
    "sks {name}, looking out a window, three-quarter back view, contemplative, soft daylight",
    # Outdoor different lighting
    "sks {name}, golden hour, half body, warm sunset backlight, outdoor",
    "sks {name}, overcast day, full body walking on a path, soft diffused light, park",
    "sks {name}, dappled sunlight through trees, medium shot, candid, outdoor",
    "sks {name}, evening outdoor, half body, soft twilight, city lights in background",
]


def _build_sdxl_anchor_config(base_seed: int = 42) -> dict:
    return {
        "runtime": {
            "input_path": "",
            "output_root": "",
            "run_name_prefix": "lora_ref",
            "device": "cuda",
            "torch_dtype": "float16",
            "repo_root": ".",
        },
        "model": {
            "backend": "diffusers_text2img",
            "granularity": "scene",
            "model_id": "stabilityai/sdxl-turbo",
            "width": 768,
            "height": 768,
            "guidance_scale": 0.0,
            "num_inference_steps": 4,
            "enable_attention_slicing": False,
            "device": "cuda",
        },
        "prompt": {
            "pipeline": "rule_based",
            "builder": "legacy",
            "style_prompt": "clean character reference, simple background, photorealistic",
            "negative_prompt": (
                "blurry, distorted, low quality, bad anatomy, extra limbs, duplicate, mutation, "
                "deformed, ugly, extra heads, merged bodies"
            ),
        },
        "generation": {
            "candidate_count": 1,
            "base_seed": base_seed,
            "routing": {"img2img_enabled": False, "route_policy": "disabled"},
            "anchor_bank": {
                "enabled": True,
                "generate": True,
                "anchor_types": ["half_body"],
                "output_dir_name": "anchors",
                "base_seed_offset": 900000,
                "half_body_candidate_count": 3,
                "half_body_selector": {
                    "method": "clip_text_alignment",
                    "clip_model_id": "openai/clip-vit-base-patch32",
                    "clip_max_text_length": 77,
                },
                "prompt_suffix": "clean identity reference image, simple background, consistent character design",
            },
            "identity_conditioning": {
                "enabled": True,
                "adapter_type": "ip_adapter",
                "anchor_source": "character_anchor_bank",
                "anchor_type": "half_body",
                "apply_to_modes": ["text2img"],
                "scale": 0.6,
                "adapter_model_id": "h94/IP-Adapter",
                "adapter_subfolder": "sdxl_models",
                "adapter_weight_name": "ip-adapter_sdxl.bin",
                "fail_on_missing_anchor": False,
            },
        },
        "scoring": {
            "type": "clip_consistency",
            "clip_model_id": "openai/clip-vit-base-patch32",
            "clip_max_text_length": 77,
        },
    }


def _build_pixart_anchor_config(base_seed: int = 42) -> dict:
    """PixArt-α for reference generation (512² to match default LoRA training resolution)."""
    return {
        "runtime": {
            "input_path": "",
            "output_root": "",
            "run_name_prefix": "lora_ref_pixart",
            "device": "cuda",
            "torch_dtype": "float16",
            "repo_root": ".",
        },
        "model": {
            "backend": "dit_text2img",
            "granularity": "scene",
            "model_id": "PixArt-alpha/PixArt-XL-2-1024-MS",
            "width": 512,
            "height": 512,
            "guidance_scale": 2.5,
            "num_inference_steps": 12,
            "enable_attention_slicing": True,
            "enable_model_cpu_offload": True,
            "max_sequence_length": 120,
            "lora_path": None,
            "lora_trigger": None,
            "device": "cuda",
        },
        "prompt": {
            "pipeline": "rule_based",
            "builder": "legacy",
            "style_prompt": (
                "photorealistic portrait, natural skin texture, soft natural light, "
                "sharp facial detail, consistent identity"
            ),
            "negative_prompt": (
                "blurry, distorted, low quality, bad anatomy, extra limbs, duplicate, mutation, "
                "deformed, ugly, extra heads, merged bodies"
            ),
        },
        "generation": {
            "candidate_count": 1,
            "base_seed": base_seed,
            "routing": {"img2img_enabled": False, "route_policy": "disabled"},
            "anchor_bank": {
                "enabled": True,
                "generate": True,
                "anchor_types": ["half_body"],
                "output_dir_name": "anchors",
                "base_seed_offset": 900000,
                "half_body_candidate_count": 3,
                "half_body_selector": {
                    "method": "clip_text_alignment",
                    "clip_model_id": "openai/clip-vit-base-patch32",
                    "clip_max_text_length": 77,
                },
                "prompt_suffix": "clean identity reference image, simple background, consistent character design",
            },
            "identity_conditioning": {"enabled": False},
        },
        "scoring": {
            "type": "clip_consistency",
            "clip_model_id": "openai/clip-vit-base-patch32",
            "clip_max_text_length": 77,
        },
    }


def _generate_with_ip_adapter(gen, prompt: str, neg: str, ref_path: str, seed: int, width: int = 768, height: int = 768):
    """SDXL Phase 2: IP-Adapter identity conditioning."""
    fake_spec = PromptSpec(
        scene_id="ref",
        style_prompt="",
        character_prompt="",
        global_context_prompt="",
        scene_consistency_prompt="",
        local_prompt="",
        action_prompt="",
        generation_prompt=prompt,
        scoring_prompt=prompt,
        full_prompt=prompt,
        negative_prompt=neg,
    )
    request = GenerationRequest(
        scene_id="ref",
        candidate_index=0,
        seed=seed,
        prompt_spec=fake_spec,
        width=width,
        height=height,
        guidance_scale=0.0,
        num_inference_steps=4,
        reference_image_path=ref_path,
        extra_options={
            "generation_mode": "text2img",
            "identity_conditioning_enabled": True,
            "identity_apply_to_modes": ["text2img"],
            "ip_adapter_scale": 0.55,
            "ip_adapter_model_id": "h94/IP-Adapter",
            "ip_adapter_subfolder": "sdxl_models",
            "ip_adapter_weight_name": "ip-adapter_sdxl.bin",
        },
    )
    return gen.generate_scene(request)


def _generate_pixart_text2img(gen, prompt: str, neg: str, seed: int, model_config: dict):
    """PixArt Phase 2: plain text2img (same backbone as LoRA training target)."""
    fake_spec = PromptSpec(
        scene_id="ref",
        style_prompt="",
        character_prompt="",
        global_context_prompt="",
        scene_consistency_prompt="",
        local_prompt="",
        action_prompt="",
        generation_prompt=prompt,
        scoring_prompt=prompt,
        full_prompt=prompt,
        negative_prompt=neg,
    )
    request = GenerationRequest(
        scene_id="ref",
        candidate_index=0,
        seed=seed,
        prompt_spec=fake_spec,
        width=int(model_config["width"]),
        height=int(model_config["height"]),
        guidance_scale=float(model_config["guidance_scale"]),
        num_inference_steps=int(model_config["num_inference_steps"]),
        reference_image_path=None,
        extra_options={
            "generation_mode": "text2img",
            "lora_ref_pixart_phase2": True,
        },
    )
    return gen.generate_scene(request)


def generate_lora_refs(
    input_story: str,
    output_dir: str,
    base_seed: int = 42,
    *,
    ref_backend: str = "pixart",
) -> list[dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ref_backend = str(ref_backend or "pixart").strip().lower()
    if ref_backend not in {"pixart", "sdxl"}:
        raise ValueError(f"ref_backend must be 'pixart' or 'sdxl', got {ref_backend!r}")

    from storygen.anchor_bank import run_anchor_bank
    from storygen.generators import build_generation_backend
    from storygen.generators.base import BaseSceneGenerator
    from storygen.io.results import create_run_context
    from storygen.parser import parse_story_file
    from storygen.prompt_pipelines import build_prompt_pipeline

    if ref_backend == "pixart":
        config = _build_pixart_anchor_config(base_seed)
    else:
        config = _build_sdxl_anchor_config(base_seed)

    story = parse_story_file(input_story)

    prompt_pipeline = build_prompt_pipeline(config["prompt"], event_logger=lambda **kw: None)
    prompt_bundle = prompt_pipeline.build(story)
    character_specs = prompt_bundle.metadata.get("character_specs", {})
    if not character_specs:
        from storygen.character_specs import build_rule_based_character_specs

        character_specs = build_rule_based_character_specs(story)

    char_name = (
        list(character_specs.keys())[0]
        if character_specs
        else (story.all_entities[0] if story.all_entities else "character")
    )

    print(f"── ref-backend={ref_backend} (canonical + diverse use the same generator family) ──")

    # ── Phase 1: anchor_bank → canonical character image ──
    print(f"── Phase 1: Anchor Bank → canonical {char_name} ──")
    gen = build_generation_backend(config["model"], config["runtime"])
    if not isinstance(gen, BaseSceneGenerator):
        raise TypeError("Expected scene-level generator")

    captions_for_dataset: list[dict] = []
    ref_path: str = ""

    with tempfile.TemporaryDirectory() as tmpdir:
        run_context = create_run_context(tmpdir, "lora_refs")
        anchor_summary = run_anchor_bank(
            character_specs=character_specs,
            anchor_config=config["generation"]["anchor_bank"],
            run_context=run_context,
            prompt_config=config["prompt"],
            model_config=config["model"],
            generator=gen,
            event_logger=lambda event, **m: print(f"  [{event}]"),
        )

        if not anchor_summary.get("characters"):
            print("ERROR: anchor_bank failed")
            return []

        first_char = list(anchor_summary["characters"].values())[0]
        half_body_info = first_char.get("anchors", {}).get("half_body", {})

        canonical = (
            half_body_info.get("canonical_anchor", {})
            if isinstance(half_body_info.get("canonical_anchor"), dict)
            else {}
        )
        ref_path = canonical.get("selected_image_path") or canonical.get("image_path")

        if not ref_path:
            candidates = half_body_info.get("candidates", [])
            if candidates:
                ref_path = candidates[0].get("image_path") or candidates[0].get("selected_image_path")

        if not ref_path or not Path(ref_path).exists():
            print(
                f"ERROR: no anchor image found (canonical={canonical}, "
                f"candidates={len(half_body_info.get('candidates', []))})"
            )
            return []

        ref_name = char_name.replace(" ", "_").replace("<", "").replace(">", "").replace("'", "")
        canonical_dest = output_path / f"{ref_name}_canonical.png"
        try:
            if Path(ref_path).resolve() != canonical_dest.resolve():
                shutil.copy2(ref_path, canonical_dest)
        except shutil.SameFileError:
            pass
        ref_path = str(canonical_dest)
        print(f"  Canonical anchor: {canonical_dest}")

        captions_for_dataset.append(
            {
                "file_name": canonical_dest.name,
                "text": (
                    f"sks {ref_name}, half body shot, clean identity reference, "
                    f"simple background, photorealistic"
                ),
            }
        )

    # ── Phase 2: diverse training images ──
    if ref_backend == "sdxl":
        print("\n── Phase 2: Diverse Training Images (SDXL + IP-Adapter ref + varied prompts) ──")
    else:
        print("\n── Phase 2: Diverse Training Images (PixArt text2img + varied prompts) ──")

    neg = (
        "blurry, distorted, low quality, bad anatomy, extra limbs, duplicate, mutation, "
        "deformed, ugly, extra heads, merged bodies, asymmetric face"
    )

    for i, template in enumerate(DIVERSE_PROMPTS):
        prompt = template.format(name=ref_name)
        seed_i = base_seed + 500 + i * 37
        print(f"  [{i + 1}/{len(DIVERSE_PROMPTS)}] {prompt[:85]}...")

        if ref_backend == "sdxl":
            candidate = _generate_with_ip_adapter(
                gen,
                prompt=prompt,
                neg=neg,
                ref_path=ref_path,
                seed=seed_i,
                width=int(config["model"]["width"]),
                height=int(config["model"]["height"]),
            )
        else:
            candidate = _generate_pixart_text2img(gen, prompt=prompt, neg=neg, seed=seed_i, model_config=config["model"])

        if candidate.image is None:
            print("    SKIP (no image)")
            continue

        fname = f"{ref_name}_diverse_{i:03d}.png"
        img_path = output_path / fname
        candidate.image.save(img_path)

        captions_for_dataset.append({"file_name": fname, "text": prompt})

    meta_path = output_path / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for entry in captions_for_dataset:
            f.write(json.dumps(entry) + "\n")

    total = len(captions_for_dataset)
    print(f"\nDone! {total} images → {output_dir}")
    print(f"  1 canonical (anchor_bank) + {total - 1} diverse")
    return []


def main():
    p = argparse.ArgumentParser(description="Generate diverse LoRA training images")
    p.add_argument("--input", type=str, required=True, help="Story text file")
    p.add_argument("--output-dir", type=str, default="training_data/character")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ref-backend",
        type=str,
        default="pixart",
        choices=("pixart", "sdxl"),
        help="pixart: PixArt-α for all ref images (matches LoRA backbone). sdxl: legacy SDXL+IP-Adapter Phase 2.",
    )
    args = p.parse_args()
    generate_lora_refs(
        input_story=args.input,
        output_dir=args.output_dir,
        base_seed=args.seed,
        ref_backend=args.ref_backend,
    )


if __name__ == "__main__":
    main()
