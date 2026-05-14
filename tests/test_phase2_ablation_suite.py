from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType


class Completed:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def _load_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_phase2_ablation_suite.py"
    spec = importlib.util.spec_from_file_location("run_phase2_ablation_suite", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_story(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _capture_subprocess(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command, *args, **kwargs):
        calls.append(list(command))
        return Completed(0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _manifest_entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_smoke_suite_builds_expected_commands_without_heavy_generation(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    calls = _capture_subprocess(monkeypatch)
    single = _write_story(tmp_path, "single.txt", "[SCENE-1] <Nina> walks home.")
    double = _write_story(tmp_path, "double.txt", "[SCENE-1] <Nina> meets <Leo>.")
    output_root = tmp_path / "outputs_ablation"

    result = module.main(
        [
            "--suite",
            "smoke",
            "--single-stories",
            str(single),
            "--double-stories",
            str(double),
            "--output-root",
            str(output_root),
            "--storydiffusion-root",
            "/opt/StoryDiffusion",
            "--native-width",
            "512",
            "--native-height",
            "512",
            "--native-num-steps",
            "20",
            "--native-seed",
            "3",
            "--native-id-length",
            "2",
            "--storydiffusion-prompt-mode",
            "clean_v2",
            "--dry-run",
        ]
    )

    assert result == 0
    assert len(calls) == 3
    joined = [" ".join(call) for call in calls]
    assert "--run-name single_storygen_default_single" in joined[0]
    assert "--single-route native_storydiffusion" not in joined[0]
    assert "--run-name single_native_storydiffusion_single" in joined[1]
    assert "--single-route native_storydiffusion" in joined[1]
    assert "--storydiffusion-root /opt/StoryDiffusion" in joined[1]
    assert "--native-width 512" in joined[1]
    assert "--native-height 512" in joined[1]
    assert "--native-num-steps 20" in joined[1]
    assert "--native-seed 3" in joined[1]
    assert "--native-id-length 2" in joined[1]
    assert "--storydiffusion-prompt-mode clean_v2" in joined[1]
    assert "--storydiffusion-prompt-mode clean_v2" not in joined[0]
    assert "--run-name double_native_storydiffusion_double" in joined[2]
    assert "--storydiffusion-root /opt/StoryDiffusion" in joined[2]
    assert "--storydiffusion-prompt-mode clean_v2" in joined[2]
    assert "--native-id-length 2" in joined[2]
    manifest = output_root / "smoke" / "suite_manifest.jsonl"
    entries = _manifest_entries(manifest)
    assert [entry["status"] for entry in entries] == ["passed", "passed", "passed"]


def test_smoke_suite_forwards_natural_prompt_mode_to_native_routes(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    calls = _capture_subprocess(monkeypatch)
    single = _write_story(tmp_path, "single.txt", "[SCENE-1] <Nina> walks home.")
    double = _write_story(tmp_path, "double.txt", "[SCENE-1] <Nina> meets <Leo>.")

    result = module.main(
        [
            "--suite",
            "smoke",
            "--single-stories",
            str(single),
            "--double-stories",
            str(double),
            "--output-root",
            str(tmp_path / "outputs_ablation"),
            "--storydiffusion-prompt-mode",
            "natural",
            "--save-identity-images",
            "--dry-run",
        ]
    )

    assert result == 0
    joined = [" ".join(call) for call in calls]
    assert "--storydiffusion-prompt-mode natural" not in joined[0]
    assert "--storydiffusion-prompt-mode natural" in joined[1]
    assert "--storydiffusion-prompt-mode natural" in joined[2]
    assert "--save-identity-images" in joined[1]
    assert "--save-identity-images" in joined[2]


def test_full_suite_builds_expected_command_count_without_heavy_generation(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    calls = _capture_subprocess(monkeypatch)
    single_a = _write_story(tmp_path, "single_a.txt", "[SCENE-1] <Nina> walks home.")
    single_b = _write_story(tmp_path, "single_b.txt", "[SCENE-1] <Girl> runs outside.")
    double = _write_story(tmp_path, "double.txt", "[SCENE-1] <Nina> meets <Leo>.")

    result = module.main(
        [
            "--suite",
            "full",
            "--single-stories",
            f"{single_a},{single_b}",
            "--double-stories",
            str(double),
            "--output-root",
            str(tmp_path / "outputs_ablation"),
            "--dry-run",
        ]
    )

    assert result == 0
    assert len(calls) == 5
    joined = [" ".join(call) for call in calls]
    assert sum("--single-route native_storydiffusion" in call for call in joined) == 2
    assert sum("double_native_storydiffusion" in call for call in joined) == 1


def test_suite_resume_skips_completed_output(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    calls = _capture_subprocess(monkeypatch)
    single = _write_story(tmp_path, "single.txt", "[SCENE-1] <Nina> walks home.")
    output_root = tmp_path / "outputs_ablation"
    completed_dir = output_root / "smoke" / "single_storygen_default_single"
    completed_dir.mkdir(parents=True)
    (completed_dir / "run_summary.json").write_text("{}", encoding="utf-8")

    result = module.main(
        [
            "--suite",
            "smoke",
            "--single-stories",
            str(single),
            "--double-stories",
            "",
            "--output-root",
            str(output_root),
            "--resume",
            "--dry-run",
        ]
    )

    assert result == 0
    assert len(calls) == 1
    manifest = output_root / "smoke" / "suite_manifest.jsonl"
    entries = _manifest_entries(manifest)
    assert entries[0]["status"] == "skipped"
    assert entries[1]["status"] == "passed"
