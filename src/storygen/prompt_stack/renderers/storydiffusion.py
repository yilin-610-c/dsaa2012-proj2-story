from __future__ import annotations

from copy import deepcopy
from typing import Any

from storygen.types import PromptSpec


def _split_comma_clauses(text: str) -> list[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def _filter_comma_clauses(text: str, bans: list[str]) -> str:
    if not text or not bans:
        return text
    bans_l = [b.lower() for b in bans if b]
    kept: list[str] = []
    for clause in _split_comma_clauses(text):
        low = clause.lower()
        if any(b in low for b in bans_l):
            continue
        kept.append(clause)
    return ", ".join(kept)


def _strip_leading_prefix_chunks(text: str, prefixes: list[str]) -> str:
    """Remove leading comma-separated chunks that match any prefix (case-insensitive)."""
    if not text or not prefixes:
        return text
    prefixes_l = [p.strip().lower() for p in prefixes if p and p.strip()]
    remainder = text.strip()
    changed = True
    while changed and remainder:
        changed = False
        clauses = _split_comma_clauses(remainder)
        if not clauses:
            break
        first = clauses[0]
        if first.lower() in prefixes_l:
            clauses = clauses[1:]
            remainder = ", ".join(clauses)
            changed = True
            continue
        for pref in prefixes_l:
            if first.lower().startswith(pref + " ") or first.lower() == pref:
                clauses = clauses[1:]
                remainder = ", ".join(clauses)
                changed = True
                break
    return remainder.strip().strip(",").strip()


def sanitize_prompt_specs_for_storydiffusion(
    specs: dict[str, PromptSpec],
    rules: dict[str, Any],
) -> dict[str, PromptSpec]:
    """
    Drop SDXL-oriented spatial / repetition clauses that StoryDiffusion handles via attention.
    """
    bans = list(rules.get("segment_bans") or [])
    fields = list(rules.get("comma_filter_fields") or [])
    gen_prefixes = list(rules.get("generation_prompt_strip_prefix_chunks") or [])

    out: dict[str, PromptSpec] = {}
    for scene_id, spec in specs.items():
        new_spec = deepcopy(spec)
        for field in fields:
            cur = getattr(new_spec, field, "") or ""
            if isinstance(cur, str):
                setattr(new_spec, field, _filter_comma_clauses(cur, bans))
        gp = new_spec.generation_prompt or ""
        if gen_prefixes:
            new_spec.generation_prompt = _strip_leading_prefix_chunks(gp, gen_prefixes)
        out[scene_id] = new_spec
    return out
