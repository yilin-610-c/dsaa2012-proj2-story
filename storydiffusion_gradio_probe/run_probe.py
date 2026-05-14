from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORYDIFFUSION_ROOT = REPO_ROOT.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "storydiffusion_gradio_probe"


DEFAULT_NEGATIVE_PROMPT = (
    "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, "
    "bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, "
    "cloned face, ugly fingers, cartoon, cg, 3d, unreal, amputation, disconnected limbs, "
    "character sheet, turnaround, multiple views, duplicate person, repeated person, triptych"
)


@dataclass(frozen=True)
class ProbeConfig:
    storydiffusion_root: Path
    output_dir: Path
    sd_type: str
    use_reference_images: bool
    reference_images: list[Path]
    general_prompt: str
    prompt_array: str
    negative_prompt: str
    style: str
    seed: int
    num_steps: int
    guidance_scale: float
    sa32: float
    sa64: float
    id_length: int
    height: int
    width: int
    style_strength_ratio: float
    ip_adapter_strength: float
    comic_type: str
    font_choice: str
    character_files: str
    save_image_start_index: int
    save_identity_images: bool


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        import yaml

        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _as_lines(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _path_list(value: Any, *, base_dir: Path, repo_root: Path) -> list[Path]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            # Most configs reference assets like "outputs/..." which are repo-root relative,
            # not relative to the config file directory.
            path = repo_root / path
        paths.append(path.resolve())
    return paths


def load_probe_config(path: Path, overrides: argparse.Namespace) -> ProbeConfig:
    payload = _read_mapping(path)
    base_dir = path.resolve().parent

    storydiffusion_root = Path(
        overrides.storydiffusion_root
        or payload.get("storydiffusion_root")
        or DEFAULT_STORYDIFFUSION_ROOT
    ).expanduser()
    if not storydiffusion_root.is_absolute():
        storydiffusion_root = (base_dir / storydiffusion_root).resolve()
    else:
        storydiffusion_root = storydiffusion_root.resolve()

    output_dir = Path(overrides.output_dir or payload.get("output_dir") or DEFAULT_OUTPUT_DIR).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()

    generation = dict(payload.get("generation") or {})
    prompts = dict(payload.get("prompts") or {})
    reference_images = _path_list(payload.get("reference_images"), base_dir=base_dir, repo_root=REPO_ROOT)
    use_reference_images = bool(payload.get("use_reference_images", bool(reference_images)))

    prompt_array = _as_lines(overrides.prompt or prompts.get("prompt_array"))
    general_prompt = _as_lines(overrides.general_prompt or prompts.get("general_prompt"))
    negative_prompt = _as_lines(prompts.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT)

    return ProbeConfig(
        storydiffusion_root=storydiffusion_root,
        output_dir=output_dir,
        sd_type=str(generation.get("sd_type", "Unstable")),
        use_reference_images=use_reference_images,
        reference_images=reference_images,
        general_prompt=general_prompt,
        prompt_array=prompt_array,
        negative_prompt=negative_prompt,
        style=str(generation.get("style", "Japanese Anime")),
        seed=int(generation.get("seed", 0)),
        num_steps=int(generation.get("num_steps", 35)),
        guidance_scale=float(generation.get("guidance_scale", 5.0)),
        sa32=float(generation.get("sa32", 0.5)),
        sa64=float(generation.get("sa64", 0.5)),
        id_length=int(generation.get("storydiffusion_internal_id_length", generation.get("id_length", 3))),
        height=int(generation.get("height", 768)),
        width=int(generation.get("width", 768)),
        style_strength_ratio=float(generation.get("style_strength_ratio", 20)),
        ip_adapter_strength=float(generation.get("ip_adapter_strength", 0.5)),
        comic_type=str(generation.get("comic_type", "No typesetting (default)")),
        font_choice=str(generation.get("font_choice", "Inkfree.ttf")),
        character_files=str(payload.get("character_files") or ""),
        save_image_start_index=int(payload.get("save_image_start_index", 0)),
        save_identity_images=bool(overrides.save_identity_images or payload.get("save_identity_images", False)),
    )


def validate_config(config: ProbeConfig) -> None:
    if not config.storydiffusion_root.exists():
        raise FileNotFoundError(f"StoryDiffusion root does not exist: {config.storydiffusion_root}")
    app_path = config.storydiffusion_root / "gradio_app_sdxl_specific_id_low_vram.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Cannot find original Gradio app: {app_path}")
    if not config.general_prompt.strip():
        raise ValueError("prompts.general_prompt is required")
    if not config.prompt_array.strip():
        raise ValueError("prompts.prompt_array is required")
    if config.use_reference_images:
        if not config.reference_images:
            raise ValueError("use_reference_images=true requires at least one reference image")
        missing = [str(path) for path in config.reference_images if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing reference image(s): " + ", ".join(missing))
        if " img" not in config.general_prompt:
            raise ValueError('Reference-image mode needs the PhotoMaker trigger word " img" in general_prompt')


def _patch_gradio_launch_methods() -> list[tuple[Any, str, Any]]:
    patched: list[tuple[Any, str, Any]] = []

    def skipped_launch(*args: Any, **kwargs: Any) -> None:
        print("[storydiffusion_probe] skipped Gradio launch during import")
        return None

    try:
        import gradio
    except Exception:
        return patched

    blocks = getattr(getattr(gradio, "blocks", None), "Blocks", None)
    if blocks is not None and hasattr(blocks, "launch"):
        patched.append((blocks, "launch", blocks.launch))
        blocks.launch = skipped_launch

    interface = getattr(gradio, "Interface", None)
    if interface is not None and hasattr(interface, "launch"):
        patched.append((interface, "launch", interface.launch))
        interface.launch = skipped_launch

    return patched


def _restore_gradio_launch_methods(patched: list[tuple[Any, str, Any]]) -> None:
    for owner, attr_name, original in reversed(patched):
        setattr(owner, attr_name, original)


def import_gradio_app(storydiffusion_root: Path):
    os.environ["STORYDIFFUSION_DISABLE_GRADIO_LAUNCH"] = "1"
    patched_launch_methods = _patch_gradio_launch_methods()
    sys.path.insert(0, str(storydiffusion_root))
    old_cwd = Path.cwd()
    os.chdir(storydiffusion_root)
    try:
        import gradio_app_sdxl_specific_id_low_vram as gradio_app
    finally:
        os.chdir(old_cwd)
        _restore_gradio_launch_methods(patched_launch_methods)
    return gradio_app


def run_generation(config: ProbeConfig) -> list[Path]:
    validate_config(config)
    gradio_app = import_gradio_app(config.storydiffusion_root)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_type = "Using Ref Images" if config.use_reference_images else "Only Using Textual Description"
    upload_images = [str(path) for path in config.reference_images] if config.use_reference_images else None

    old_cwd = Path.cwd()
    os.chdir(config.storydiffusion_root)
    try:
        generator = gradio_app.process_generation(
            config.sd_type,
            model_type,
            upload_images,
            config.num_steps,
            config.style,
            config.ip_adapter_strength,
            config.style_strength_ratio,
            config.guidance_scale,
            config.seed,
            config.sa32,
            config.sa64,
            config.id_length,
            config.general_prompt,
            config.negative_prompt,
            config.prompt_array,
            config.height,
            config.width,
            config.comic_type,
            config.font_choice,
            config.character_files,
        )
        final_images = None
        for step_index, images in enumerate(generator, start=1):
            final_images = images
            print(f"[storydiffusion] generation update {step_index}: {len(images)} image(s)")
    finally:
        os.chdir(old_cwd)

    if final_images is None:
        raise RuntimeError("StoryDiffusion returned no images")

    prompt_lines = config.prompt_array.splitlines()
    identity_paths: list[Path] = []
    identity_image_prompt_map: dict[str, dict[str, Any]] = {}
    if config.save_identity_images and config.save_image_start_index > 0:
        identity_dir = config.output_dir / "identity_refs"
        identity_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(final_images[: config.save_image_start_index]):
            output_path = identity_dir / f"identity_{index:03d}.png"
            image.save(output_path)
            identity_paths.append(output_path)
            identity_image_prompt_map[output_path.name] = {
                "path": str(output_path),
                "prompt_array_index": index,
                "prompt": prompt_lines[index] if index < len(prompt_lines) else "",
            }

    images_to_save = final_images[config.save_image_start_index :]
    saved_paths = []
    for index, image in enumerate(images_to_save):
        output_path = config.output_dir / f"image_{index:03d}.png"
        image.save(output_path)
        saved_paths.append(output_path)

    manifest_path = config.output_dir / "manifest.json"
    manifest = {
        "storydiffusion_root": str(config.storydiffusion_root),
        "sd_type": config.sd_type,
        "model_type": model_type,
        "reference_images": [str(path) for path in config.reference_images],
        "general_prompt": config.general_prompt,
        "prompt_array": config.prompt_array,
        "negative_prompt": config.negative_prompt,
        "generation": {
            "style": config.style,
            "seed": config.seed,
            "num_steps": config.num_steps,
            "guidance_scale": config.guidance_scale,
            "sa32": config.sa32,
            "sa64": config.sa64,
            "id_length": config.id_length,
            "height": config.height,
            "width": config.width,
            "style_strength_ratio": config.style_strength_ratio,
            "comic_type": config.comic_type,
        },
        "images": [str(path) for path in saved_paths],
        "identity_images": [str(path) for path in identity_paths],
        "identity_image_prompt_map": identity_image_prompt_map,
        "raw_image_count": len(final_images),
        "save_image_start_index": config.save_image_start_index,
        "save_identity_images": config.save_identity_images,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(manifest_path, config.output_dir / "last_manifest.json")
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original StoryDiffusion Gradio generation function from a simple config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("example_config.yaml"),
        help="YAML or JSON config with prompts, reference images, and generation settings.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument("--storydiffusion-root", type=Path, help="Override original StoryDiffusion repo path.")
    parser.add_argument("--general-prompt", help="Override prompts.general_prompt.")
    parser.add_argument(
        "--prompt",
        help="Override prompts.prompt_array. Use shell $'line1\\nline2' quoting for multiple lines.",
    )
    parser.add_argument("--save-identity-images", action="store_true", help="Save skipped identity reference images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_probe_config(args.config, args)
    saved_paths = run_generation(config)
    print("Saved images:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
