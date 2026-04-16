# Minimal Story Image Baseline

This repository contains a small, deterministic baseline for multi-scene story image generation. It parses the custom scene text format, builds rule-based prompts, generates multiple text-to-image candidates per scene, reranks them with CLIP-based text/image consistency scoring, and saves images plus metadata for inspection.

The prompt builder is rule-based and continuity-oriented: it keeps a stable main subject anchor, rewrites leading pronouns back to the primary entity when appropriate, adds explicit continuity text for subject identity, setting, lighting, and palette, and injects action-emphasis phrases so pose-critical scenes are easier to score and inspect.

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

The default scorer is `clip_consistency`. It compares each candidate image against the current scene prompt and, for scenes after the first, also compares it with the previous selected frame. This is the first real cross-scene consistency mechanism in the pipeline; prompt wording alone remains a soft bias.

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

`demo_run` now defaults to 4 candidates per scene because reranking needs real choice to matter. Expect CLIP reranking to add extra inference time on top of image generation, but it is still practical on a local RTX 4060.

## Notes

- Prompt building is strictly rule-based and template-based in v1.
- The backend is pure text-to-image only in v1.
- `reference_image_path` and `previous_selected_image_path` are reserved in the request contract but unused by the generator backend.
- The scoring layer now includes a CLIP-based reranker for prompt adherence and previous-frame consistency, while remaining replaceable for future scorers.
