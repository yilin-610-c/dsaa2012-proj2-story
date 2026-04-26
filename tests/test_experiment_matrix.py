from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_experiment_matrix", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def _args(tmp_path: Path, **overrides):
    parser = matrix.build_parser()
    args = parser.parse_args(
        [
            "--experiment-id",
            "exp1",
            "--profiles",
            "llm_prompt_text2img",
            "--stories",
            str(tmp_path / "*.txt"),
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_parse_profiles_rejects_empty_profiles() -> None:
    assert matrix.parse_profiles("a,b") == ["a", "b"]
    with pytest.raises(Exception):
        matrix.parse_profiles(" , ")


def test_resolve_story_paths_accepts_glob_and_comma_list(tmp_path: Path) -> None:
    story_b = tmp_path / "b.txt"
    story_a = tmp_path / "a.txt"
    story_b.write_text("b", encoding="utf-8")
    story_a.write_text("a", encoding="utf-8")

    assert matrix.resolve_story_paths(str(tmp_path / "*.txt")) == [story_a, story_b]
    assert matrix.resolve_story_paths(f"{story_b},{story_a}") == [story_a, story_b]


def test_build_jobs_and_child_argv_forward_extra_set(tmp_path: Path) -> None:
    story = tmp_path / "01.txt"
    story.write_text("story", encoding="utf-8")
    output_root = Path("outputs_remote") / "exp1"
    job = matrix.build_jobs("exp1", ["llm_prompt_text2img"], [story], output_root)[0]
    args = _args(
        tmp_path,
        config="configs/base.yaml",
        candidate_count=1,
        width=512,
        height=512,
        base_seed=2026,
        extra_set=["model.num_inference_steps=4", "model.enable_attention_slicing=true"],
    )

    argv = matrix.build_child_argv(job, args)

    assert job.run_name == "exp1__llm_prompt_text2img__01"
    assert argv[:3] == ["python3", "-m", "storygen.cli"]
    assert ["--run-name", "exp1__llm_prompt_text2img__01"][0] in argv
    assert argv.count("--set") == 5
    assert "runtime.output_root=outputs_remote/exp1" in argv
    assert "model.width=512" in argv
    assert "model.height=512" in argv
    assert "model.num_inference_steps=4" in argv
    assert "model.enable_attention_slicing=true" in argv


def test_dry_run_writes_records_and_does_not_call_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = tmp_path / "01.txt"
    story.write_text("story", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "git_commit_hash", lambda repo_root: "abc123")
    monkeypatch.setattr(matrix, "repo_is_dirty", lambda repo_root: False)

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess should not run during dry-run")

    monkeypatch.setattr(matrix.subprocess, "run", fail_run)

    result = matrix.main(
        [
            "--experiment-id",
            "dry",
            "--profiles",
            "llm_prompt_text2img",
            "--stories",
            str(story),
            "--candidate-count",
            "1",
            "--width",
            "512",
            "--height",
            "512",
            "--base-seed",
            "2026",
            "--extra-set",
            "model.num_inference_steps=4",
            "--dry-run",
        ]
    )

    assert result == 0
    output_root = tmp_path / "outputs_remote" / "dry"
    manifest = output_root / "manifest.jsonl"
    summary = output_root / "summary.csv"
    readme = output_root / "README.md"
    log_out = output_root / "batch_logs" / "dry__llm_prompt_text2img__01.out"
    log_err = output_root / "batch_logs" / "dry__llm_prompt_text2img__01.err"
    record = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])

    assert record["status"] == "dry_run"
    assert record["argv"]
    assert "model.num_inference_steps=4" in record["argv"]
    assert "PYTHONPATH=src" in record["command"]
    assert log_out.exists()
    assert log_err.exists()
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    assert rows[0]["status"] == "dry_run"
    assert "manifest.jsonl` is append-only" in readme.read_text(encoding="utf-8")


def test_output_root_override_writes_records_under_custom_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = tmp_path / "01.txt"
    story.write_text("story", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "git_commit_hash", lambda repo_root: "abc123")
    monkeypatch.setattr(matrix, "repo_is_dirty", lambda repo_root: False)

    result = matrix.main(
        [
            "--experiment-id",
            "local_exp",
            "--profiles",
            "llm_prompt_text2img",
            "--stories",
            str(story),
            "--output-root",
            "outputs_local/local_exp",
            "--dry-run",
        ]
    )

    assert result == 0
    output_root = tmp_path / "outputs_local" / "local_exp"
    record = json.loads((output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["output_directory"] == "outputs_local/local_exp/local_exp__llm_prompt_text2img__01"
    assert "runtime.output_root=outputs_local/local_exp" in record["argv"]


def test_resume_records_skipped_when_run_summary_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = tmp_path / "01.txt"
    story.write_text("story", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "git_commit_hash", lambda repo_root: "abc123")
    monkeypatch.setattr(matrix, "repo_is_dirty", lambda repo_root: False)
    run_dir = tmp_path / "outputs_remote" / "exp" / "exp__llm_prompt_text2img__01"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess should not run for resume skip")

    monkeypatch.setattr(matrix.subprocess, "run", fail_run)

    result = matrix.main(
        [
            "--experiment-id",
            "exp",
            "--profiles",
            "llm_prompt_text2img",
            "--stories",
            str(story),
            "--resume",
        ]
    )

    assert result == 0
    record = json.loads((tmp_path / "outputs_remote" / "exp" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "skipped"


def test_continue_on_error_records_failure_and_runs_next_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story1 = tmp_path / "01.txt"
    story2 = tmp_path / "02.txt"
    story1.write_text("story", encoding="utf-8")
    story2.write_text("story", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "git_commit_hash", lambda repo_root: "abc123")
    monkeypatch.setattr(matrix, "repo_is_dirty", lambda repo_root: True)
    calls = []

    def fake_run(argv, stdout, stderr, env):
        calls.append((argv, dict(env)))
        stdout.write("out")
        stderr.write("err")
        return subprocess.CompletedProcess(argv, returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)

    result = matrix.main(
        [
            "--experiment-id",
            "exp",
            "--profiles",
            "llm_prompt_text2img",
            "--stories",
            f"{story1},{story2}",
            "--continue-on-error",
        ]
    )

    assert result == 1
    assert len(calls) == 2
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "1"
    records = [
        json.loads(line)
        for line in (tmp_path / "outputs_remote" / "exp" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["status"] for record in records] == ["failed", "success"]


def test_explicit_cuda_visible_devices_is_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = tmp_path / "01.txt"
    story.write_text("story", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "git_commit_hash", lambda repo_root: "abc123")
    monkeypatch.setattr(matrix, "repo_is_dirty", lambda repo_root: False)
    captured_env = {}

    def fake_run(argv, stdout, stderr, env):
        captured_env.update(env)
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)

    result = matrix.main(
        [
            "--experiment-id",
            "exp",
            "--profiles",
            "llm_prompt_text2img",
            "--stories",
            str(story),
            "--cuda-visible-devices",
            "0",
        ]
    )

    assert result == 0
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "0"
    record = json.loads((tmp_path / "outputs_remote" / "exp" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["cuda_visible_devices"] == "0"
    assert record["command"].startswith("CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src")


def test_stops_after_first_failure_without_continue_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story1 = tmp_path / "01.txt"
    story2 = tmp_path / "02.txt"
    story1.write_text("story", encoding="utf-8")
    story2.write_text("story", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(matrix, "git_commit_hash", lambda repo_root: "abc123")
    monkeypatch.setattr(matrix, "repo_is_dirty", lambda repo_root: False)
    calls = []

    def fake_run(argv, stdout, stderr, env):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=1)

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)

    result = matrix.main(
        [
            "--experiment-id",
            "exp",
            "--profiles",
            "llm_prompt_text2img",
            "--stories",
            f"{story1},{story2}",
        ]
    )

    assert result == 1
    assert len(calls) == 1
