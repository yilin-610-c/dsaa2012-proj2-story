from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from storygen.types import PromptSpec, Story


@dataclass
class PromptState:
    """Structured prompt assembly state (extensible; used by the modular stack)."""

    story: Story
    story_context: dict[str, Any] = field(default_factory=dict)
    scene_specs: dict[str, PromptSpec] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
