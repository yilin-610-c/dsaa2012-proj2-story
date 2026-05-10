from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from storygen.prompt_builder import PromptBuilder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def merge_prompt_config_with_pack(
    prompt_config: dict[str, Any],
    pack_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Shallow-copy prompt_config, apply optional YAML pack merges, return (merged, storydiffusion_postprocess_rules).
    """
    merged: dict[str, Any] = dict(prompt_config)
    post: dict[str, Any] = {}
    name = (pack_name or "").strip()
    if not name or name.lower() == "none":
        return merged, post
    path = _repo_root() / "configs" / "prompt_templates" / f"{name}.yaml"
    if not path.is_file():
        return merged, post
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    extra = data.get("merge_into_prompt_config") or {}
    if isinstance(extra, dict):
        merged.update(extra)
    post_block = data.get("storydiffusion_postprocess")
    if isinstance(post_block, dict):
        post = post_block
    return merged, post


def build_rule_prompt_builder(prompt_config: dict[str, Any]) -> Any:
    """
    Return either legacy PromptBuilder or ModularPromptBuilder (same public surface:
    build_story_prompts, build_prompt_for_scene, build_story_context when present).
    """
    kind = str(prompt_config.get("builder", "legacy")).strip().lower()
    if kind == "modular":
        from storygen.prompt_stack.facade import ModularPromptBuilder

        modular_cfg = prompt_config.get("modular")
        modular_dict = modular_cfg if isinstance(modular_cfg, dict) else {}
        backend = str(modular_dict.get("backend", "sdxl")).strip().lower()
        pack = modular_dict.get("template_pack", "default")
        return ModularPromptBuilder(prompt_config, backend=backend, template_pack=str(pack))
    return PromptBuilder(prompt_config)
