from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_local_ablation.py"
SPEC = importlib.util.spec_from_file_location("run_local_ablation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
local = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT_PATH.parent))
sys.modules[SPEC.name] = local
SPEC.loader.exec_module(local)
sys.path.pop(0)


def test_latest_suite_expands_to_two_character_profile() -> None:
    parser = local.build_parser()
    args = parser.parse_args(["--suite", "latest", "--experiment-id", "local_latest"])

    argv = local.matrix_argv_from_args(args)

    assert argv[argv.index("--profiles") + 1] == "llm_prompt_two_character_text2img"
    assert argv[argv.index("--output-root") + 1] == "outputs_local/local_latest"
    assert "--resume" in argv
    assert "--continue-on-error" in argv


def test_main_suite_excludes_rule_based_profiles() -> None:
    parser = local.build_parser()
    args = parser.parse_args(["--suite", "main", "--experiment-id", "local_main", "--preset", "main"])

    argv = local.matrix_argv_from_args(args)
    profiles = argv[argv.index("--profiles") + 1].split(",")

    assert profiles == [
        "llm_prompt_text2img",
        "llm_prompt_ip_adapter_text2img",
        "llm_prompt_two_character_text2img",
    ]
    assert all("rule" not in profile and profile != "smoke_test" for profile in profiles)
    assert argv[argv.index("--candidate-count") + 1] == "3"
    assert argv[argv.index("--width") + 1] == "768"
    assert argv[argv.index("--height") + 1] == "768"


def test_cli_overrides_profiles_stories_and_preset_values() -> None:
    parser = local.build_parser()
    args = parser.parse_args(
        [
            "--suite",
            "main",
            "--profiles",
            "llm_prompt_text2img",
            "--stories",
            "test_set/01.txt,test_set/07.txt",
            "--experiment-id",
            "custom",
            "--candidate-count",
            "2",
            "--width",
            "640",
            "--height",
            "640",
            "--base-seed",
            "7",
            "--extra-set",
            "prompt.cache.enabled=false",
            "--dry-run",
            "--no-resume",
            "--stop-on-error",
        ]
    )

    argv = local.matrix_argv_from_args(args)

    assert argv[argv.index("--profiles") + 1] == "llm_prompt_text2img"
    assert argv[argv.index("--stories") + 1] == "test_set/01.txt,test_set/07.txt"
    assert argv[argv.index("--candidate-count") + 1] == "2"
    assert argv[argv.index("--width") + 1] == "640"
    assert argv[argv.index("--height") + 1] == "640"
    assert argv[argv.index("--base-seed") + 1] == "7"
    assert "prompt.cache.enabled=false" in argv
    assert "--dry-run" in argv
    assert "--resume" not in argv
    assert "--continue-on-error" not in argv


def test_local_ablation_dry_run_writes_outputs_local_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    story = tmp_path / "01.txt"
    story.write_text("story", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local.run_experiment_matrix, "git_commit_hash", lambda repo_root: "abc123")
    monkeypatch.setattr(local.run_experiment_matrix, "repo_is_dirty", lambda repo_root: False)

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess should not run during dry-run")

    monkeypatch.setattr(local.run_experiment_matrix.subprocess, "run", fail_run)

    result = local.main(
        [
            "--suite",
            "latest",
            "--experiment-id",
            "local_dry",
            "--stories",
            str(story),
            "--dry-run",
        ]
    )

    assert result == 0
    output_root = tmp_path / "outputs_local" / "local_dry"
    record = json.loads((output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["profile"] == "llm_prompt_two_character_text2img"
    assert record["status"] == "dry_run"
    assert record["candidate_count"] == 1
    assert record["width"] == 512
    assert record["height"] == 512
    assert record["base_seed"] == 2026
    assert "model.num_inference_steps=4" in record["argv"]
    rows = list(csv.DictReader((output_root / "summary.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "dry_run"
