from __future__ import annotations

from storygen.types import (
    CandidateScore,
    GenerationCandidate,
    PromptSpec,
    Scene,
    SceneSelectionResult,
    Story,
)

from .base import BaseScorer


class HeuristicScorer(BaseScorer):
    """Deterministic placeholder scorer based on candidate metadata only."""

    def __init__(self, scorer_name: str = "heuristic_v1") -> None:
        self.scorer_name = scorer_name

    def score_candidate(
        self,
        *,
        story: Story,
        scene: Scene,
        prompt_spec: PromptSpec,
        candidate: GenerationCandidate,
        previous_results: list[SceneSelectionResult],
    ) -> CandidateScore:
        del story, scene, prompt_spec, previous_results

        seed_component = 1.0 / (1 + (candidate.seed % 997))
        index_component = 1.0 / (1 + candidate.candidate_index)
        score = round(seed_component + index_component, 6)
        return CandidateScore(
            scene_id=candidate.scene_id,
            candidate_index=candidate.candidate_index,
            seed=candidate.seed,
            score=score,
            scorer_name=self.scorer_name,
            components={
                "seed_component": round(seed_component, 6),
                "index_component": round(index_component, 6),
            },
            metadata={"selection_mode": "metadata_only_placeholder"},
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
