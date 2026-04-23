import argparse

import pytest

from storygen.cli import _parse_set_override, build_parser


def test_parse_set_override_parses_yaml_scalar_values() -> None:
    assert _parse_set_override("prompt.artifact.export_enabled=true") == (
        "prompt.artifact.export_enabled",
        True,
    )
    assert _parse_set_override("generation.candidate_count=4") == ("generation.candidate_count", 4)
    assert _parse_set_override("prompt.artifact.path=null") == ("prompt.artifact.path", None)


def test_parse_set_override_requires_key_value_shape() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_set_override("missing_equals")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_set_override("=value")


def test_cli_parser_accepts_repeated_set_overrides() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "--profile",
            "llm_prompt_text2img",
            "--set",
            "prompt.artifact.export_enabled=true",
            "--set",
            "prompt.artifact.path=null",
        ]
    )

    assert args.set_overrides == [
        "prompt.artifact.export_enabled=true",
        "prompt.artifact.path=null",
    ]
