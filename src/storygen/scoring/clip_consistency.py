from __future__ import annotations

from pathlib import Path
from typing import Any

from storygen.types import (
    CandidateScore,
    GenerationCandidate,
    PromptSpec,
    Scene,
    SceneSelectionResult,
    Story,
)

from .base import BaseScorer


class CLIPConsistencyScorer(BaseScorer):
    def __init__(self, scoring_config: dict[str, Any], runtime_config: dict[str, Any]) -> None:
        self.scoring_config = scoring_config
        self.runtime_config = runtime_config
        self.scorer_name = scoring_config.get("type", "clip_consistency")
        self.model_id = scoring_config.get("clip_model_id", "openai/clip-vit-base-patch32")
        self.device = scoring_config.get("device") or runtime_config.get("device", "cpu")
        self.model = None
        self.processor = None
        self._image_feature_cache: dict[str, Any] = {}
        self._text_feature_cache: dict[str, Any] = {}

    def score_candidate(
        self,
        *,
        story: Story,
        scene: Scene,
        prompt_spec: PromptSpec,
        candidate: GenerationCandidate,
        previous_results: list[SceneSelectionResult],
    ) -> CandidateScore:
        del story, scene

        previous_image_path = previous_results[-1].selected_image_path if previous_results else None
        text_alignment = self._text_alignment_score(candidate.image_path, self._build_full_alignment_text(prompt_spec))
        action_alignment = self._text_alignment_score(candidate.image_path, prompt_spec.local_prompt)
        previous_image_consistency = self._image_consistency_score(candidate.image_path, previous_image_path)

        configured_weights = {
            "text_alignment": float(self.scoring_config.get("text_image_weight", 0.45)),
            "action_alignment": float(self.scoring_config.get("action_weight", 0.2)),
            "previous_image_consistency": float(self.scoring_config.get("consistency_weight", 0.35)),
        }

        component_values = {
            "text_alignment": text_alignment,
            "action_alignment": action_alignment,
            "previous_image_consistency": previous_image_consistency,
        }
        active_weights = {
            name: weight
            for name, weight in configured_weights.items()
            if weight > 0 and (name != "previous_image_consistency" or previous_image_path is not None)
        }
        total_weight = sum(active_weights.values()) or 1.0
        weighted_score = sum(component_values[name] * weight for name, weight in active_weights.items()) / total_weight

        return CandidateScore(
            scene_id=candidate.scene_id,
            candidate_index=candidate.candidate_index,
            seed=candidate.seed,
            score=round(float(weighted_score), 6),
            scorer_name=self.scorer_name,
            components={key: round(float(value), 6) for key, value in component_values.items()},
            metadata={
                "clip_model_id": self.model_id,
                "device": self._resolved_device(),
                "weights": configured_weights,
                "previous_image_available": previous_image_path is not None,
                "previous_image_path": previous_image_path,
            },
        )

    def select_best(
        self,
        *,
        scene_id: str,
        candidates: list[GenerationCandidate],
        candidate_scores: list[CandidateScore],
    ) -> SceneSelectionResult:
        if not candidates or not candidate_scores:
            raise ValueError(f"No candidates to select for scene {scene_id}")

        candidate_by_index = {candidate.candidate_index: candidate for candidate in candidates}
        selected_score = sorted(
            candidate_scores,
            key=lambda item: (-item.score, item.candidate_index, item.seed),
        )[0]
        selected_candidate = candidate_by_index[selected_score.candidate_index]
        if not selected_candidate.image_path:
            raise ValueError(f"Selected candidate for {scene_id} is missing an image path")

        return SceneSelectionResult(
            scene_id=scene_id,
            selected_candidate_index=selected_candidate.candidate_index,
            selected_seed=selected_candidate.seed,
            selected_image_path=selected_candidate.image_path,
            selected_score=selected_score,
            candidate_scores=sorted(candidate_scores, key=lambda item: item.candidate_index),
            candidate_image_paths=[
                candidate.image_path or "" for candidate in sorted(candidates, key=lambda item: item.candidate_index)
            ],
        )

    def _text_alignment_score(self, image_path: str | None, text: str) -> float:
        if not image_path or not text.strip():
            return 0.0
        image_features = self._get_image_features(image_path)
        text_features = self._get_text_features(text)
        return float((image_features @ text_features.T).item())

    def _image_consistency_score(self, image_path: str | None, previous_image_path: str | None) -> float:
        if not image_path or not previous_image_path:
            return 0.0
        image_features = self._get_image_features(image_path)
        previous_features = self._get_image_features(previous_image_path)
        return float((image_features @ previous_features.T).item())

    def _get_image_features(self, image_path: str):
        normalized_path = str(Path(image_path))
        cached = self._image_feature_cache.get(normalized_path)
        if cached is not None:
            return cached

        self._load()

        import torch
        from PIL import Image

        with Image.open(normalized_path) as image_handle:
            image = image_handle.convert("RGB")
            image_inputs = self.processor(images=image, return_tensors="pt")
        image_inputs = {key: value.to(self._resolved_device()) for key, value in image_inputs.items()}

        with torch.no_grad():
            image_features = self._normalize_features(self.model.get_image_features(**image_inputs))

        self._image_feature_cache[normalized_path] = image_features
        return image_features

    def _get_text_features(self, text: str):
        cached = self._text_feature_cache.get(text)
        if cached is not None:
            return cached

        self._load()

        import torch

        text_inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_text_length(),
        )
        text_inputs = {key: value.to(self._resolved_device()) for key, value in text_inputs.items()}

        with torch.no_grad():
            text_features = self._normalize_features(self.model.get_text_features(**text_inputs))

        self._text_feature_cache[text] = text_features
        return text_features

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import AutoProcessor, CLIPModel

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.model = self.model.to(self._resolved_device())
        self.model.eval()

    def _build_full_alignment_text(self, prompt_spec: PromptSpec) -> str:
        fragments = [
            prompt_spec.local_prompt,
            prompt_spec.character_prompt,
            prompt_spec.global_context_prompt,
            prompt_spec.style_prompt,
        ]
        return ", ".join(fragment.strip() for fragment in fragments if fragment.strip())

    def _resolved_device(self) -> str:
        if self.device != "cuda":
            return self.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _max_text_length(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", None)
        model_max_length = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max_length, int) and model_max_length > 0:
            return model_max_length
        return 77

    @staticmethod
    def _normalize_features(features):
        import torch

        if isinstance(features, torch.Tensor):
            tensor_features = features
        elif hasattr(features, "pooler_output") and features.pooler_output is not None:
            tensor_features = features.pooler_output
        elif hasattr(features, "image_embeds") and features.image_embeds is not None:
            tensor_features = features.image_embeds
        elif hasattr(features, "text_embeds") and features.text_embeds is not None:
            tensor_features = features.text_embeds
        elif hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            tensor_features = features.last_hidden_state[:, 0]
        else:
            raise TypeError(f"Unsupported CLIP feature output type: {type(features)!r}")

        return tensor_features / tensor_features.norm(dim=-1, keepdim=True)
