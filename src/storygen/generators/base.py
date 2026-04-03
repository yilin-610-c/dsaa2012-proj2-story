from __future__ import annotations

from abc import ABC, abstractmethod

from storygen.types import GenerationCandidate, GenerationRequest


class BaseImageGenerator(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationCandidate:
        raise NotImplementedError
