# Phase 2 Ablation Runbook

## External StoryDiffusion Dependency

Native StoryDiffusion routes call the official external StoryDiffusion repository. This project does not vendor the upstream repo and does not use a git submodule.

Expected external file:

```text
<storydiffusion-root>/gradio_app_sdxl_specific_id_low_vram.py
```

Recommended layout:

```text
spring25/
  dsaa2012-proj2-story/
  StoryDiffusion/
```

If the upstream repo is not in the default location, pass `--storydiffusion-root /path/to/StoryDiffusion`.

## Environment Checks

```bash
conda env list

test -f /path/to/StoryDiffusion/gradio_app_sdxl_specific_id_low_vram.py

conda run -n storydiffusion python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

conda run -n storydiffusion python -c "import gradio, diffusers, transformers; print(gradio.__version__); print(diffusers.__version__); print(transformers.__version__)"
```

## A. Teammate High-VRAM Full Ablation

Use this when the `storydiffusion` conda environment and official StoryDiffusion repo already exist and the machine has enough GPU memory.

Default external repo location:

```bash
python3 scripts/run_phase2_ablation_suite.py \
  --suite full \
  --output-root outputs_ablation \
  --continue-on-error \
  --resume
```

Explicit external repo location:

```bash
python3 scripts/run_phase2_ablation_suite.py \
  --suite full \
  --output-root outputs_ablation \
  --storydiffusion-root /path/to/StoryDiffusion \
  --continue-on-error \
  --resume
```

Expected suite manifest:

```text
outputs_ablation/full/suite_manifest.jsonl
```

Expected experiment folders include:

```text
outputs_ablation/full/single_storygen_default_<id>/
outputs_ablation/full/single_native_storydiffusion_<id>/
outputs_ablation/full/double_native_storydiffusion_<id>/
```

## B. Local Low-VRAM Smoke

First validate command construction without running generation:

```bash
python3 scripts/run_phase2_ablation_suite.py \
  --suite smoke \
  --output-root outputs_ablation \
  --storydiffusion-root /path/to/StoryDiffusion \
  --native-width 512 \
  --native-height 512 \
  --native-num-steps 20 \
  --native-seed 0 \
  --dry-run
```

Then run a small smoke only if the local machine can access CUDA and has enough memory:

```bash
python3 scripts/run_phase2_ablation_suite.py \
  --suite smoke \
  --output-root outputs_ablation \
  --storydiffusion-root /path/to/StoryDiffusion \
  --native-width 512 \
  --native-height 512 \
  --native-num-steps 20 \
  --native-seed 0 \
  --continue-on-error
```

This smoke is only for path validation. It is not the final quality run.

## Files To Send Back

For each completed experiment, send:

- `outputs_ablation/<suite>/suite_manifest.jsonl`
- native StoryDiffusion `manifest.json`
- storygen `run_summary.json`
- generated scene images
- generated native config YAML files, if present

## OOM Fallbacks

If native StoryDiffusion fails with CUDA OOM:

- reduce `--native-width` and `--native-height` to `512`
- reduce `--native-num-steps` to `10` or `20`
- run `--suite smoke` before `--suite full`
- run with fewer stories via `--single-stories 14 --double-stories 07`
- use `--continue-on-error` so one failed run does not stop the whole suite
