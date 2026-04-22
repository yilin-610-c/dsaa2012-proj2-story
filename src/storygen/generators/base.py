from __future__ import annotations

from abc import ABC, abstractmethod

from storygen.types import GenerationCandidate, GenerationRequest, StoryGenerationRequest, StoryGenerationResult


class BaseSceneGenerator(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_scene(self, request: GenerationRequest) -> GenerationCandidate:
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> GenerationCandidate:
        return self.generate_scene(request)


class BaseStoryGenerator(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate_story(self, request: StoryGenerationRequest) -> StoryGenerationResult:
        raise NotImplementedError


class BaseImageGenerator(BaseSceneGenerator):
    """Backward-compatible name for scene-level image generators."""
