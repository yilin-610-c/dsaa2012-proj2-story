import json
from pathlib import Path

from storygen.config import resolve_config
from storygen.generators import BaseSceneGenerator
from storygen.pipeline import run_pipeline
from storygen.types import GenerationCandidate, GenerationRequest


class FakeImage:
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"fake image")


class FakeSceneGenerator(BaseSceneGenerator):
    def load(self) -> None:
        return None

    def generate_scene(self, request: GenerationRequest) -> GenerationCandidate:
        return GenerationCandidate(
            scene_id=request.scene_id,
            candidate_index=request.candidate_index,
            seed=request.seed,
            prompt_spec=request.prompt_spec,
            image=FakeImage(),
            metadata={"backend": "fake"},
        )


def test_run_pipeline_writes_minimal_logs_without_changing_summary_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("storygen.pipeline.build_generation_backend", lambda model, runtime: FakeSceneGenerator())
    config = resolve_config(
        "configs/base.yaml",
        "smoke_test",
        overrides={
            "runtime.output_root": str(tmp_path),
            "runtime.run_name": "log_test",
            "scoring.type": "heuristic",
            "generation.candidate_count": 1,
        },
    )

    summary = run_pipeline(config)
    run_dir = Path(summary.run_directory)

    prompt_log = run_dir / "logs" / "prompt_pipeline.json"
    backend_log = run_dir / "logs" / "generation_backend.json"
    events_log = run_dir / "logs" / "events.jsonl"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert prompt_log.exists()
    assert backend_log.exists()
    assert events_log.exists()
    assert manifest["prompt_pipeline"] == "rule_based"
    assert manifest["generation_backend"] == "diffusers_text2img"
    assert manifest["generation_granularity"] == "scene"
    assert len(manifest["selected_outputs"]) == len(summary.scene_results)

    events = [json.loads(line)["event"] for line in events_log.read_text(encoding="utf-8").splitlines()]
    assert "run_started" in events
    assert "prompt_pipeline_selected" in events
    assert "backend_selected" in events
    assert "scene_generation_started" in events
    assert "scene_generation_completed" in events
    assert "run_completed" in events
