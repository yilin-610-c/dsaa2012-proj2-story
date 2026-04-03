# Minimal Story Image Baseline

This repository contains a small, deterministic baseline for multi-scene story image generation. It parses the custom scene text format, builds rule-based prompts, generates multiple text-to-image candidates per scene, selects one candidate deterministically, and saves images plus metadata for inspection.

## Structure

```text
configs/        YAML config and runtime profiles
scripts/        demo entrypoint
src/storygen/   package code
test_set/       sample scene files
outputs/        generated runs
```

## Setup

Use Python 3.10+ inside WSL on the machine that can access your NVIDIA GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pick a model id that works on your local setup. The default in `configs/base.yaml` is `stabilityai/sdxl-turbo`, but you can override it from the CLI.

## Run

Cheap smoke test:

```bash
PYTHONPATH=src python3 -m storygen.cli --profile smoke_test --input test_set/01.txt
```

Fuller local demo:

```bash
PYTHONPATH=src python3 -m storygen.cli --profile demo_run --input test_set/01.txt
```

Demo script:

```bash
PYTHONPATH=src python3 scripts/run_demo.py --profile demo_run --input test_set/01.txt
```

## Outputs

Each run writes to `outputs/<run_name>/` and includes:

- `config_resolved.yaml`
- `manifest.json`
- `run_summary.json`
- `scenes/scene_XXX/prompt.json`
- `scenes/scene_XXX/candidates/*.png`
- `scenes/scene_XXX/scene_result.json`
- `scenes/scene_XXX/selected.png`

The run metadata includes the resolved config, runtime profile, model id, timestamp, git commit id when available, pipeline version, seeds, and selected outputs.

## Notes

- Prompt building is strictly rule-based and template-based in v1.
- The backend is pure text-to-image only in v1.
- `reference_image_path` and `previous_selected_image_path` are reserved in the request contract but unused by the generator backend.
- The scoring layer is a placeholder interface designed to be replaced later with image-aware or cross-scene consistency scoring.
