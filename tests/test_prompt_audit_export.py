from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest

from storygen.types import PromptBundle, PromptSpec


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_prompts.py"
SPEC = importlib.util.spec_from_file_location("export_prompts", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
export_prompts = importlib.util.module_from_spec(SPEC)
sys.modules["export_prompts"] = export_prompts
SPEC.loader.exec_module(export_prompts)


def _write_story(path: Path) -> None:
    path.write_text(
        "[SCENE-1] <Ryan> walks quickly toward a bus.\n\n"
        "[SEP]\n\n"
        "[SCENE-2] He pauses at the door and looks ahead.\n",
        encoding="utf-8",
    )


def test_expand_input_paths_supports_globs(tmp_path: Path) -> None:
    first = tmp_path / "01.txt"
    second = tmp_path / "02.txt"
    _write_story(first)
    _write_story(second)

    paths = export_prompts.expand_input_paths([str(tmp_path / "*.txt")])

    assert paths == [first, second]


def test_rule_based_audit_payload_contains_scene_prompt_fields() -> None:
    input_path = Path("test_set/02.txt")

    payload = export_prompts.build_audit_payload(
        input_paths=[input_path],
        pipeline_names=["rule_based"],
        config_path="configs/base.yaml",
        rule_profile="smoke_test",
        llm_profile="llm_prompt_text2img",
        overrides={},
    )

    story = payload["stories"][0]
    pipeline = story["pipelines"]["rule_based"]
    scene_two = pipeline["scenes"][1]

    assert pipeline["status"] == "ok"
    assert story["source_path"] == "test_set/02.txt"
    assert scene_two["scene_id"] == "SCENE-2"
    assert scene_two["entities"] == []
    assert scene_two["story_all_entities"] == ["Ryan"]
    assert scene_two["prompt"]["generation_prompt"] == (
        "Ryan, pauses, cinematic story illustration, new scene setting: at the door and looks ahead, same vehicle context:"
    )
    assert scene_two["prompt"]["scoring_prompt"] == "Ryan, pauses, at the door and looks ahead"


def test_write_audit_outputs_writes_json_markdown_and_story_files(tmp_path: Path) -> None:
    payload = export_prompts.build_audit_payload(
        input_paths=[Path("test_set/02.txt")],
        pipeline_names=["rule_based"],
        config_path="configs/base.yaml",
        rule_profile="smoke_test",
        llm_profile="llm_prompt_text2img",
        overrides={},
    )

    export_prompts.write_audit_outputs(payload, tmp_path)

    audit_json = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    audit_markdown = (tmp_path / "audit.md").read_text(encoding="utf-8")
    story_json = json.loads((tmp_path / "stories" / "02.json").read_text(encoding="utf-8"))

    assert audit_json["artifact_type"] == "prompt_audit"
    assert "test_set/02.txt" in audit_markdown
    assert "SCENE-2" in audit_markdown
    assert "Ryan, pauses, at the door and looks ahead" in audit_markdown
    assert story_json["story_id"] == "02"


def test_pipeline_failure_is_recorded_and_other_pipelines_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    original_builder = export_prompts.build_prompt_pipeline

    def fake_build_prompt_pipeline(prompt_config, *, event_logger=None):
        if prompt_config.get("pipeline") == "llm_assisted":
            raise RuntimeError("missing test api key")
        return original_builder(prompt_config, event_logger=event_logger)

    monkeypatch.setattr(export_prompts, "build_prompt_pipeline", fake_build_prompt_pipeline)

    payload = export_prompts.build_audit_payload(
        input_paths=[Path("test_set/02.txt")],
        pipeline_names=["rule_based", "llm_assisted"],
        config_path="configs/base.yaml",
        rule_profile="smoke_test",
        llm_profile="llm_prompt_text2img",
        overrides={},
    )

    pipelines = payload["stories"][0]["pipelines"]

    assert pipelines["rule_based"]["status"] == "ok"
    assert pipelines["llm_assisted"]["status"] == "failed"
    assert pipelines["llm_assisted"]["error_type"] == "RuntimeError"
    assert "missing test api key" in pipelines["llm_assisted"]["error"]


def test_cli_overrides_take_precedence_for_llm_direct_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_targets: list[str] = []

    class FakePipeline:
        def build(self, story):
            return PromptBundle(
                scene_prompts={
                    scene.scene_id: PromptSpec(
                        scene_id=scene.scene_id,
                        style_prompt="",
                        character_prompt="",
                        global_context_prompt="",
                        generation_prompt=f"prompt for {scene.scene_id}",
                        scoring_prompt=f"score for {scene.scene_id}",
                    )
                    for scene in story.scenes
                },
                metadata={},
            )

        def metadata(self):
            return {"target_backends": list(captured_targets), "validation_status": "passed"}

    def fake_build_prompt_pipeline(prompt_config, *, event_logger=None):
        del event_logger
        captured_targets[:] = list(prompt_config["llm_direct"]["targets"])
        return FakePipeline()

    monkeypatch.setattr(export_prompts, "build_prompt_pipeline", fake_build_prompt_pipeline)

    payload = export_prompts.build_audit_payload(
        input_paths=[Path("test_set/02.txt")],
        pipeline_names=["llm_direct"],
        config_path="configs/base.yaml",
        rule_profile="smoke_test",
        llm_profile="llm_prompt_anchor_bank",
        overrides={"prompt.llm_direct.targets": ["anchor"]},
    )

    pipeline = payload["stories"][0]["pipelines"]["llm_direct"]
    assert pipeline["status"] == "ok"
    assert captured_targets == ["anchor"]
