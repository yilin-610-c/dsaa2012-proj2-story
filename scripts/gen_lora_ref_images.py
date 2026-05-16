#!/usr/bin/env python3
"""Generate character reference images for PixArt DreamBooth LoRA.

Bootstrap:
  pixart: canonical + 3 augmentations (PixArt-native).
  sdxl / sdxl_ipadapter: SDXL anchor canonical + IP-Adapter images from story scenes (photoreal).

Backends:
  pixart (default): PixArt canonical + minimal augmentations.
  sdxl_ipadapter: character from story → SDXL canonical → 3–4 IP-Adapter scene images → train PixArt LoRA.

Usage:
    PYTHONPATH=src python scripts/gen_lora_ref_images.py \\
        --input test_set/01.txt --output-dir training_data/my_character --phase bootstrap

    PYTHONPATH=src python scripts/gen_lora_ref_images.py \\
        --input test_set/01.txt --output-dir training_data/my_character --phase expand \\
        --lora-path outputs/.../lora_checkpoints --lora-trigger "sks "
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from storygen.types import GenerationRequest, PromptSpec


_PHOTOREAL_STYLE = (
    "professional photograph, Canon EOS 5D, 85mm portrait lens, photorealistic skin texture, "
    "sharp focus on face, natural skin imperfections, realistic human, 8K detail"
)
_PHOTOREAL_NEGATIVE = (
    "illustration, cartoon, anime, painting, sketch, 3d render, cgi, "
    "blurry, distorted, low quality, bad anatomy, extra limbs, duplicate, mutation, "
    "deformed, ugly, extra heads, merged bodies, smooth plastic skin, doll face"
)


def _build_sdxl_anchor_config(base_seed: int = 42, resolution: int = 768) -> dict:
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
            "width": resolution,
            "height": resolution,
            "guidance_scale": 0.0,
            "num_inference_steps": 4,
            "enable_attention_slicing": False,
            "device": "cuda",
        },
        "prompt": {
            "pipeline": "rule_based",
            "builder": "legacy",
            "style_prompt": _PHOTOREAL_STYLE,
            "negative_prompt": _PHOTOREAL_NEGATIVE,
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
                "prompt_suffix": (
                    "professional headshot, studio portrait, neutral gray background, "
                    "photorealistic, consistent identity, front-facing, natural expression"
                ),
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


def _is_sdxl_ipadapter_backend(ref_backend: str) -> bool:
    return ref_backend.strip().lower() in {"sdxl", "sdxl_ipadapter"}


def _story_diverse_prompts(story, ref_name: str, gender_label: str, count: int) -> list[str]:
    """Build photoreal IP-Adapter prompts from story scenes (same person, varied context)."""
    prompts: list[str] = []
    for scene in story.scenes:
        action = scene.clean_text.strip().rstrip(".")
        if not action:
            continue
        prompts.append(f"sks {ref_name}, {gender_label}, {action}, {_PHOTOREAL_STYLE}")
        if len(prompts) >= count:
            break
    fallbacks = [
        f"sks {ref_name}, {gender_label}, half body portrait, neutral background, looking at camera, {_PHOTOREAL_STYLE}",
        f"sks {ref_name}, {gender_label}, three-quarter view, walking outdoors, daylight, {_PHOTOREAL_STYLE}",
        f"sks {ref_name}, {gender_label}, sitting indoors by a window, soft natural light, {_PHOTOREAL_STYLE}",
        f"sks {ref_name}, {gender_label}, close-up portrait, looking ahead, {_PHOTOREAL_STYLE}",
    ]
    for fb in fallbacks:
        if len(prompts) >= count:
            break
        prompts.append(fb)
    return prompts[:count]


def _build_pixart_anchor_config(base_seed: int = 42, resolution: int = 768) -> dict:
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
            "width": resolution,
            "height": resolution,
            "guidance_scale": 4.5,
            "num_inference_steps": 20,
            "enable_attention_slicing": True,
            "enable_model_cpu_offload": True,
            "max_sequence_length": 120,
            "lora_path": None,
            "lora_trigger": None,
            "prompt_mode": "full",
            "device": "cuda",
        },
        "prompt": {
            "pipeline": "rule_based",
            "builder": "legacy",
            "style_prompt": (
                "professional photograph, Canon EOS 5D, 85mm portrait lens, studio lighting, "
                "photorealistic skin texture, sharp focus on face, natural skin imperfections, "
                "high fidelity, 8K detail, realistic human proportions"
            ),
            "negative_prompt": (
                "illustration, cartoon, anime, painting, sketch, 3d render, cgi, "
                "blurry, distorted, low quality, bad anatomy, extra limbs, duplicate, mutation, "
                "deformed, ugly, extra heads, merged bodies, smooth plastic skin"
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
                "prompt_suffix": (
                    "professional headshot, studio portrait, neutral gray background, "
                    "consistent identity, front-facing, natural expression"
                ),
            },
            "identity_conditioning": {"enabled": False},
        },
        "scoring": {
            "type": "clip_consistency",
            "clip_model_id": "openai/clip-vit-base-patch32",
            "clip_max_text_length": 77,
        },
    }


def _build_pixart_gen_config(
    base_seed: int,
    resolution: int,
    *,
    lora_path: str | None = None,
    lora_trigger: str | None = None,
) -> dict:
    cfg = _build_pixart_anchor_config(base_seed, resolution)
    cfg["model"]["lora_path"] = lora_path
    cfg["model"]["lora_trigger"] = lora_trigger
    cfg["generation"]["anchor_bank"]["enabled"] = False
    return cfg


# Minimal augmentations: same person as canonical (flip + mild crop + brightness).
DEFAULT_AUG_SUFFIXES = ("flip_h", "crop_center", "bright_up")

# Optional expand (IP-Adapter locks face to canonical; pure LoRA text2img drifts).
EXPAND_PROMPTS_IPADAPTER = [
    "sks {name}, {gender}, side profile portrait, looking left, studio lighting, neutral background, photorealistic",
    "sks {name}, {gender}, three-quarter view, looking straight ahead, soft natural light, photorealistic portrait",
    "sks {name}, {gender}, sitting on a bus by the window, half body, looking forward, interior lighting",
    "sks {name}, {gender}, standing at a bus stop, half body, waiting, overcast daylight, photorealistic",
]

DIVERSE_PROMPTS_SDXL = [
    "sks {name}, full body shot, standing in a sunlit park, relaxed pose, natural daylight, photorealistic",
    "sks {name}, walking on a quiet street, full body, candid shot, afternoon light",
    "sks {name}, sitting on a park bench, full body, reading, soft shade, garden background",
    "sks {name}, standing by a window, half body, side profile, soft indoor light, thoughtful",
    "sks {name}, sitting at a wooden desk, half body, writing, warm lamp light, bookshelves behind",
    "sks {name}, in a cozy living room, half body, relaxing on a chair, warm evening light",
    "sks {name}, close-up portrait, looking ahead, soft natural light, neutral background",
    "sks {name}, medium close-up, three-quarter turn, gentle expression, soft background blur",
    "sks {name}, reaching for a book on a shelf, full body, library, soft light",
    "sks {name}, holding a cup, half body, cafe background, warm morning light",
    "sks {name}, looking out a window, three-quarter back view, contemplative, soft daylight",
    "sks {name}, golden hour, half body, warm sunset backlight, outdoor",
    "sks {name}, overcast day, full body walking on a path, soft diffused light, park",
    "sks {name}, dappled sunlight through trees, medium shot, candid, outdoor",
    "sks {name}, evening outdoor, half body, soft twilight, city lights in background",
]


def _generate_with_ip_adapter(
    gen, prompt: str, neg: str, ref_path: str, seed: int, width: int = 768, height: int = 768
):
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


def _generate_pixart_text2img(
    gen,
    prompt: str,
    neg: str,
    seed: int,
    width: int,
    height: int,
    *,
    guidance_scale: float = 4.5,
    num_steps: int = 20,
):
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
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        extra_options={"generation_mode": "text2img"},
    )
    return gen.generate_scene(request)


def _gender_tokens(character_specs: dict[str, dict]) -> tuple[str, str]:
    """Caption gender hint and extra negative tokens to reduce cross-gender drift."""
    if not character_specs:
        return "", ""
    spec = next(iter(character_specs.values()))
    gender = str(spec.get("gender_presentation") or "").strip().lower()
    if gender == "male":
        return "adult male man", "woman, female, girl, feminine face"
    if gender == "female":
        return "adult female woman", "man, male, boy, masculine face"
    return "", ""


def _make_augmentations(
    image_path: str,
    output_dir: Path,
    base_name: str,
    count: int = 3,
    *,
    gender_hint: str = "",
) -> list[dict]:
    from PIL import Image, ImageEnhance

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    entries: list[dict] = []

    aug_by_suffix: dict[str, callable] = {
        "flip_h": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        "crop_center": lambda im: im.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92))).resize((w, h)),
        "bright_up": lambda im: ImageEnhance.Brightness(im).enhance(1.15),
        "crop_tl": lambda im: im.crop((0, 0, int(w * 0.85), int(h * 0.85))).resize((w, h)),
        "bright_down": lambda im: ImageEnhance.Brightness(im).enhance(0.85),
    }
    suffixes = list(DEFAULT_AUG_SUFFIXES)
    if count > len(suffixes):
        suffixes.extend([s for s in aug_by_suffix if s not in suffixes])
    suffixes = suffixes[: max(0, count)]

    caption_gender = f", {gender_hint}" if gender_hint else ""
    for i, suffix in enumerate(suffixes):
        aug_fn = aug_by_suffix[suffix]
        fname = f"{base_name}_aug_{i:02d}_{suffix}.png"
        out_path = output_dir / fname
        aug_fn(img).save(out_path)
        entries.append(
            {
                "file_name": fname,
                "text": (
                    f"sks {base_name}{caption_gender}, photorealistic portrait, "
                    f"consistent identity, natural skin texture"
                ),
            }
        )

    return entries


def _write_metadata(meta_path: Path, entries: list[dict]) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _append_metadata(meta_path: Path, entries: list[dict]) -> None:
    existing: list[dict] = []
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            existing = [json.loads(line) for line in f if line.strip()]
    _write_metadata(meta_path, existing + entries)


def _load_story_context(input_story: str) -> tuple[object, dict[str, dict]]:
    from storygen.character_specs import build_rule_based_character_specs
    from storygen.parser import parse_story_file
    from storygen.prompt_pipelines import build_prompt_pipeline

    story = parse_story_file(input_story)
    config = _build_pixart_anchor_config()
    prompt_pipeline = build_prompt_pipeline(config["prompt"], event_logger=lambda **kw: None)
    prompt_bundle = prompt_pipeline.build(story)
    character_specs = prompt_bundle.metadata.get("character_specs", {})
    if not character_specs:
        character_specs = build_rule_based_character_specs(story)
    return story, character_specs


def _resolve_character_name(input_story: str) -> str:
    story, character_specs = _load_story_context(input_story)
    if character_specs:
        return list(character_specs.keys())[0]
    if story.all_entities:
        return story.all_entities[0]
    return "character"


def _sanitize_name(char_name: str) -> str:
    return char_name.replace(" ", "_").replace("<", "").replace(">", "").replace("'", "")


def generate_bootstrap_refs(
    input_story: str,
    output_dir: str,
    base_seed: int = 42,
    *,
    ref_backend: str = "pixart",
    resolution: int = 768,
    aug_count: int = 3,
    diverse_count: int = 4,
) -> list[dict]:
    """Phase A: pixart = canonical + augs; sdxl_ipadapter = canonical + IP-Adapter scene images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ref_backend = str(ref_backend or "pixart").strip().lower()

    from storygen.anchor_bank import run_anchor_bank
    from storygen.character_specs import build_rule_based_character_specs
    from storygen.generators import build_generation_backend
    from storygen.generators.base import BaseSceneGenerator
    from storygen.io.results import create_run_context
    from storygen.parser import parse_story_file
    from storygen.prompt_pipelines import build_prompt_pipeline

    story = parse_story_file(input_story)
    char_name = _resolve_character_name(input_story)
    ref_name = _sanitize_name(char_name)

    if ref_backend == "pixart":
        config = _build_pixart_anchor_config(base_seed, resolution)
    elif _is_sdxl_ipadapter_backend(ref_backend):
        config = _build_sdxl_anchor_config(base_seed, resolution)
    else:
        raise ValueError(f"ref_backend must be pixart|sdxl|sdxl_ipadapter, got {ref_backend!r}")

    prompt_pipeline = build_prompt_pipeline(config["prompt"], event_logger=lambda **kw: None)
    prompt_bundle = prompt_pipeline.build(story)
    character_specs = prompt_bundle.metadata.get("character_specs", {})
    if not character_specs:
        character_specs = build_rule_based_character_specs(story)
    gender_hint, gender_neg = _gender_tokens(character_specs)
    if gender_neg:
        config["prompt"]["negative_prompt"] = ", ".join(
            p for p in [config["prompt"]["negative_prompt"], gender_neg] if p
        )

    if _is_sdxl_ipadapter_backend(ref_backend):
        print(
            f"── bootstrap ref-backend={ref_backend} resolution={resolution}² "
            f"diverse_count={diverse_count} (SDXL canonical + IP-Adapter) ──"
        )
    else:
        print(f"── bootstrap ref-backend={ref_backend} resolution={resolution}² aug_count={aug_count} ──")
    print(f"── Phase 1: Anchor Bank → canonical {char_name} ──")

    gen = build_generation_backend(config["model"], config["runtime"])
    if not isinstance(gen, BaseSceneGenerator):
        raise TypeError("Expected scene-level generator")

    captions: list[dict] = []
    ref_path = ""

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
            print("ERROR: no anchor image found")
            return []

        canonical_dest = output_path / f"{ref_name}_canonical.png"
        if Path(ref_path).resolve() != canonical_dest.resolve():
            shutil.copy2(ref_path, canonical_dest)
        ref_path = str(canonical_dest)
        print(f"  Canonical anchor: {canonical_dest}")

        gender_cap = f", {gender_hint}" if gender_hint else ""
        captions.append(
            {
                "file_name": canonical_dest.name,
                "text": (
                    f"sks {ref_name}{gender_cap}, half body shot, clean identity reference, "
                    f"simple background, photorealistic, consistent identity"
                ),
            }
        )

    if _is_sdxl_ipadapter_backend(ref_backend):
        gender_label = gender_hint or "person"
        scene_prompts = _story_diverse_prompts(story, ref_name, gender_label, diverse_count)
        print(f"\n── Phase 2: SDXL + IP-Adapter ({len(scene_prompts)} story scenes, photoreal) ──")
        neg = config["prompt"]["negative_prompt"]
        for i, prompt in enumerate(scene_prompts):
            seed_i = base_seed + 500 + i * 37
            print(f"  [{i + 1}/{len(scene_prompts)}] {prompt[:95]}...")
            candidate = _generate_with_ip_adapter(
                gen,
                prompt=prompt,
                neg=neg,
                ref_path=ref_path,
                seed=seed_i,
                width=resolution,
                height=resolution,
            )
            if candidate.image is None:
                print("    SKIP (no image)")
                continue
            fname = f"{ref_name}_scene_{i:02d}.png"
            candidate.image.save(output_path / fname)
            captions.append({"file_name": fname, "text": prompt})
    else:
        print("\n── Phase 2: Augmentations only (same person as canonical) ──")
        aug_entries = _make_augmentations(
            ref_path, output_path, ref_name, count=aug_count, gender_hint=gender_hint
        )
        captions.extend(aug_entries)
        for entry in aug_entries:
            print(f"  {entry['file_name']}")

    meta_path = output_path / "metadata.jsonl"
    _write_metadata(meta_path, captions)
    print(f"\nBootstrap done: {len(captions)} images → {output_dir}")
    return captions


def generate_expand_refs(
    input_story: str,
    output_dir: str,
    base_seed: int = 42,
    *,
    expand_mode: str = "ipadapter",
    lora_path: str | None = None,
    resolution: int = 768,
) -> list[dict]:
    """Optional phase B: a few diverse training images anchored to canonical.

    expand_mode=ipadapter (recommended): SDXL + IP-Adapter from canonical — same face.
    expand_mode=lora: PixArt + phase-1 LoRA text2img — often drifts (gender/pose); not recommended.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    expand_mode = (expand_mode or "ipadapter").strip().lower()

    _, character_specs = _load_story_context(input_story)
    ref_name = _sanitize_name(_resolve_character_name(input_story))
    gender_hint, gender_neg = _gender_tokens(character_specs)
    gender_label = gender_hint or "person"

    canonical = output_path / f"{ref_name}_canonical.png"
    if not canonical.exists():
        matches = list(output_path.glob("*_canonical.png"))
        if not matches:
            print(f"ERROR: no canonical in {output_dir}; run --phase bootstrap first")
            return []
        canonical = matches[0]
        ref_name = canonical.stem.replace("_canonical", "")

    ref_path = str(canonical)
    from storygen.generators import build_generation_backend
    from storygen.generators.base import BaseSceneGenerator

    new_entries: list[dict] = []

    if expand_mode == "ipadapter":
        config = _build_sdxl_anchor_config(base_seed, resolution)
        neg = config["prompt"]["negative_prompt"]
        if gender_neg:
            neg = ", ".join(p for p in [neg, gender_neg] if p)
        gen = build_generation_backend(config["model"], config["runtime"])
        if not isinstance(gen, BaseSceneGenerator):
            raise TypeError("Expected scene-level generator")
        print(f"── expand: IP-Adapter (canonical ref) resolution={resolution}² ──")
        templates = EXPAND_PROMPTS_IPADAPTER
        for i, template in enumerate(templates):
            prompt = template.format(name=ref_name, gender=gender_label)
            seed_i = base_seed + 700 + i * 53
            print(f"  [{i + 1}/{len(templates)}] {prompt[:90]}...")
            candidate = _generate_with_ip_adapter(
                gen, prompt=prompt, neg=neg, ref_path=ref_path, seed=seed_i,
                width=resolution, height=resolution,
            )
            if candidate.image is None:
                print("    SKIP (no image)")
                continue
            fname = f"{ref_name}_expand_{i:02d}.png"
            candidate.image.save(output_path / fname)
            new_entries.append({"file_name": fname, "text": prompt})
    elif expand_mode == "lora":
        if not lora_path:
            raise ValueError("expand_mode=lora requires --lora-path")
        config = _build_pixart_gen_config(base_seed, resolution, lora_path=lora_path, lora_trigger=None)
        neg = config["prompt"]["negative_prompt"]
        if gender_neg:
            neg = ", ".join(p for p in [neg, gender_neg] if p)
        gen = build_generation_backend(config["model"], config["runtime"])
        if not isinstance(gen, BaseSceneGenerator):
            raise TypeError("Expected scene-level generator")
        print(f"── expand: LoRA text2img (may drift) LoRA={lora_path} ──")
        for i, template in enumerate(EXPAND_PROMPTS_IPADAPTER):
            prompt = template.format(name=ref_name, gender=gender_label)
            seed_i = base_seed + 700 + i * 53
            candidate = _generate_pixart_text2img(
                gen, prompt=prompt, neg=neg, seed=seed_i,
                width=resolution, height=resolution,
            )
            if candidate.image is None:
                continue
            fname = f"{ref_name}_lora_{i:02d}.png"
            candidate.image.save(output_path / fname)
            new_entries.append({"file_name": fname, "text": prompt})
    else:
        raise ValueError(f"expand_mode must be ipadapter or lora, got {expand_mode!r}")

    meta_path = output_path / "metadata.jsonl"
    _append_metadata(meta_path, new_entries)
    print(f"\nExpand done: +{len(new_entries)} images → {output_dir}")
    return new_entries


def generate_lora_refs(
    input_story: str,
    output_dir: str,
    base_seed: int = 42,
    *,
    ref_backend: str = "pixart",
    phase: str = "bootstrap",
    resolution: int = 768,
    lora_path: str | None = None,
    lora_trigger: str = "sks ",
    aug_count: int = 3,
    diverse_count: int = 4,
    expand_mode: str = "ipadapter",
) -> list[dict]:
    phase = (phase or "bootstrap").strip().lower()
    if phase == "bootstrap":
        return generate_bootstrap_refs(
            input_story,
            output_dir,
            base_seed,
            ref_backend=ref_backend,
            resolution=resolution,
            aug_count=aug_count,
            diverse_count=diverse_count,
        )
    if phase == "expand":
        return generate_expand_refs(
            input_story,
            output_dir,
            base_seed,
            expand_mode=expand_mode,
            lora_path=lora_path,
            resolution=resolution,
        )
    if phase == "all":
        bootstrap = generate_bootstrap_refs(
            input_story,
            output_dir,
            base_seed,
            ref_backend=ref_backend,
            resolution=resolution,
            aug_count=aug_count,
            diverse_count=diverse_count,
        )
        generate_expand_refs(
            input_story, output_dir, base_seed, expand_mode=expand_mode, lora_path=lora_path, resolution=resolution
        )
        return bootstrap
    raise ValueError(f"phase must be bootstrap|expand|all, got {phase!r}")


def main():
    p = argparse.ArgumentParser(description="Generate LoRA training images (bootstrap / expand)")
    p.add_argument("--input", type=str, required=True, help="Story text file")
    p.add_argument("--output-dir", type=str, default="training_data/character")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=768, help="Square resolution for ref gen (default 768)")
    p.add_argument(
        "--phase",
        type=str,
        default="bootstrap",
        choices=("bootstrap", "expand", "all"),
        help="bootstrap: canonical + few augs; expand: optional diverse refs; all: both",
    )
    p.add_argument(
        "--ref-backend",
        type=str,
        default="pixart",
        choices=("pixart", "sdxl", "sdxl_ipadapter"),
        help=(
            "pixart: canonical+augs. sdxl / sdxl_ipadapter: SDXL canonical + IP-Adapter "
            "scene images from story (for PixArt LoRA train; cross-model experiment)."
        ),
    )
    p.add_argument(
        "--diverse-count",
        type=int,
        default=4,
        help="With sdxl_ipadapter: IP-Adapter images from story scenes (default 4)",
    )
    p.add_argument("--lora-path", type=str, default=None, help="Only for --expand-mode lora")
    p.add_argument(
        "--expand-mode",
        type=str,
        default="ipadapter",
        choices=("ipadapter", "lora"),
        help="expand: ipadapter=canonical-locked (recommended); lora=PixArt text2img (drifts)",
    )
    p.add_argument("--aug-count", type=int, default=3, help="Augmentations from canonical (default 3)")
    args = p.parse_args()
    generate_lora_refs(
        input_story=args.input,
        output_dir=args.output_dir,
        base_seed=args.seed,
        ref_backend=args.ref_backend,
        phase=args.phase,
        resolution=args.resolution,
        lora_path=args.lora_path,
        aug_count=args.aug_count,
        diverse_count=args.diverse_count,
        expand_mode=args.expand_mode,
    )


if __name__ == "__main__":
    main()
