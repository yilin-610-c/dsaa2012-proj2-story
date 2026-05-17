# Ablation Plan

## Phase 1 Goal

Compare the current single-character default path with an opt-in native StoryDiffusion path, without changing default routing behavior.

This phase is intentionally narrow: it only adds a switch for single-character ablation runs and documents how to compare outputs. It does not change prompt cleanup, generation backends, configs, or the existing two-character route.

## Experiment Routes

| Experiment name | Meaning | Main components | Output path |
| --- | --- | --- | --- |
| `single_storygen_default_<id>` | Existing single-character baseline | `storygen.cli`, `cloud_anchor_ipadapter_story`, modular SDXL prompts, Anchor Bank, IP-Adapter | `outputs/single_storygen_default_<id>/` |
| `single_native_storydiffusion_<id>` | Opt-in single-character native StoryDiffusion ablation | `storydiffusion_gradio_probe/run_test_set.py`, modular StoryDiffusion prompts, native Gradio probe | `outputs/single_native_storydiffusion_<id>/` |
| `double_native_storydiffusion_<id>` | Existing two-character default | `storydiffusion_gradio_probe/run_test_set.py`, modular StoryDiffusion prompts, native Gradio probe | `outputs/double_native_storydiffusion_<id>/` |

## Expected Commands

Default single-character storygen route:

```bash
python3 scripts/run_auto_story_pipeline_modular.py \
  --input test_set/14.txt \
  --run-name single_storygen_default_14
```

Opt-in single-character native StoryDiffusion route:

```bash
python3 scripts/run_auto_story_pipeline_modular.py \
  --input test_set/14.txt \
  --run-name single_native_storydiffusion_14 \
  --single-route native_storydiffusion
```

Default two-character native StoryDiffusion route:

```bash
python3 scripts/run_auto_story_pipeline_modular.py \
  --input test_set/07.txt \
  --run-name double_native_storydiffusion_07
```

Use `--dry-run` first to confirm the selected command without running model generation.

## Component Switches

The default single-character route keeps the current storygen behavior:

- `cloud_anchor_ipadapter_story`
- `prompt.builder=modular`
- `prompt.modular.backend=sdxl`
- Anchor Bank enabled
- IP-Adapter identity conditioning enabled

The opt-in native StoryDiffusion route reuses the existing probe path:

- `storydiffusion_gradio_probe/run_test_set.py`
- `--prompt-builder modular`
- `--prompt-modular-backend storydiffusion`
- native StoryDiffusion Gradio `process_generation(...)`

## Manual Evaluation Criteria

Compare each run using the same story and seed where possible.

- Identity consistency: face, hair, outfit, and body remain stable across scenes.
- Scene and action correctness: each panel depicts the intended scene action.
- Setting continuity: background and visual style remain coherent when the story setting does not change.
- Artifacts: check extra faces, duplicated people, bad anatomy, and severe distortions.
