from __future__ import annotations

from abc import ABC, abstractmethod

from storygen.types import (
    CandidateScore,
    GenerationCandidate,
    PromptSpec,
    Scene,
    SceneSelectionResult,
    Story,
)


class BaseScorer(ABC):
    @abstractmethod
    def score_candidate(
        self,
        *,
        story: Story,
        scene: Scene,
        prompt_spec: PromptSpec,
        candidate: GenerationCandidate,
        previous_results: list[SceneSelectionResult],
    ) -> CandidateScore:
        raise NotImplementedError

    @abstractmethod
    def select_best(
        self,
        *,
        scene_id: str,
        candidates: list[GenerationCandidate],
        candidate_scores: list[CandidateScore],
    ) -> SceneSelectionResult:
        raise NotImplementedError
