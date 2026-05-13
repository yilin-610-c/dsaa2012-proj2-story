from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_auto_story_pipeline_modular.py"
    spec = importlib.util.spec_from_file_location("run_auto_story_pipeline_modular", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_story(tmp_path: Path, text: str, name: str = "story.txt") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _forbid_subprocess(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise AssertionError("dry-run should not execute subprocess.run")

    monkeypatch.setattr(subprocess, "run", fail_run)


def test_single_story_default_route_builds_storygen_command(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    _forbid_subprocess(monkeypatch)
    story = _write_story(tmp_path, "[SCENE-1] <Nina> walks home.")

    result = module.main(["--input", str(story), "--run-name", "dry_single_default", "--dry-run"])

    assert result == 0
    output = capsys.readouterr().out
    assert "route=single entities=1" in output
    assert "storygen.cli" in output
    assert "prompt.builder=modular" in output
    assert "prompt.modular.backend=sdxl" in output
    assert "storydiffusion_gradio_probe/run_test_set.py" not in output


def test_single_story_native_route_builds_storydiffusion_probe_command(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    _forbid_subprocess(monkeypatch)
    story = _write_story(tmp_path, "[SCENE-1] <Nina> walks home.")

    result = module.main(
        [
            "--input",
            str(story),
            "--run-name",
            "dry_single_native",
            "--single-route",
            "native_storydiffusion",
            "--dry-run",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "route=single entities=1" in output
    assert "storydiffusion_gradio_probe/run_test_set.py" in output
    assert "--prompt-builder modular" in output
    assert "--prompt-modular-backend storydiffusion" in output
    assert "--run" in output
    assert "storygen.cli" not in output


def test_double_story_still_builds_storydiffusion_probe_command(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    _forbid_subprocess(monkeypatch)
    story = _write_story(tmp_path, "[SCENE-1] <Nina> meets <Leo> in the snow.")

    result = module.main(["--input", str(story), "--run-name", "dry_double_default", "--dry-run"])

    assert result == 0
    output = capsys.readouterr().out
    assert "route=double entities=2" in output
    assert "storydiffusion_gradio_probe/run_test_set.py" in output
    assert "--prompt-builder modular" in output
    assert "--prompt-modular-backend storydiffusion" in output
    assert "--run" in output
    assert "storygen.cli" not in output


def test_single_route_storygen_matches_default_route(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    _forbid_subprocess(monkeypatch)
    story = _write_story(tmp_path, "[SCENE-1] <Nina> walks home.")

    result = module.main(
        [
            "--input",
            str(story),
            "--run-name",
            "dry_single_explicit_storygen",
            "--single-route",
            "storygen",
            "--dry-run",
        ]
    )

    assert result == 0
