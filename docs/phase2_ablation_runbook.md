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

## C. Representative Three-Story Ablation

Use this before a broader ablation. It compares the current storygen baseline against native StoryDiffusion natural prompts on three representative cases:

- `test_setA/19.txt`: single human identity, Student
- `test_setA/03.txt`: dual animal identity, Cat/Dog
- `test_set/06.txt`: dual human identity, Jack/Sara

Methods:

- `storygen`: `storygen.cli` with `cloud_anchor_ipadapter_story`, modular SDXL prompts, Anchor Bank, and IP-Adapter
- `native`: native StoryDiffusion probe with `--storydiffusion-prompt-mode natural` and saved identity reference images

First dry-run the matrix:

```bash
conda run -n storygen env PYTHONPATH=src python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_setA/19.txt,test_setA/03.txt,test_set/06.txt \
  --methods storygen,native \
  --output-root outputs_ablation/representative_v1 \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /home/lyl610/spring25/StoryDiffusion \
  --storydiffusion-prompt-mode natural \
  --native-width 512 \
  --native-height 512 \
  --native-num-steps 30 \
  --native-id-lengths 1 \
  --native-id-lengths-override test_setA/19.txt=1,2 \
  --native-id-lengths-override test_set/06.txt=1,2 \
  --save-identity-images \
  --dry-run
```

Then run the same command without `--dry-run`:

```bash
conda run -n storygen env PYTHONPATH=src python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_setA/19.txt,test_setA/03.txt,test_set/06.txt \
  --methods storygen,native \
  --output-root outputs_ablation/representative_v1 \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /home/lyl610/spring25/StoryDiffusion \
  --storydiffusion-prompt-mode natural \
  --native-width 512 \
  --native-height 512 \
  --native-num-steps 30 \
  --native-id-lengths 1 \
  --native-id-lengths-override test_setA/19.txt=1,2 \
  --native-id-lengths-override test_set/06.txt=1,2 \
  --save-identity-images \
  --continue-on-error
```

This produces eight runs:

```text
storygen_default_test_setA_19/
native_natural_id1_test_setA_19/
native_natural_id2_test_setA_19/
storygen_default_test_setA_03/
native_natural_id1_test_setA_03/
storygen_default_test_set_06/
native_natural_id1_test_set_06/
native_natural_id2_test_set_06/
```

`--native-id-lengths` is the default list of native identity prompt counts per character. `--native-id-lengths-override STORY=1,2` replaces that list for one story. In this matrix, Cat/Dog only runs `id_length=1`, while Student and Jack/Sara run both `id_length=1` and `id_length=2` so their `identity_refs/` can be compared before choosing the cleaner setting.

Inspect these files before judging image quality:

- `outputs_ablation/representative_v1/custom/suite_manifest.jsonl`
- native `storydiffusion_prompt_debug.json`
- native `identity_refs/identity_*.png`
- native generated story images
- storygen `logs/prompt_bundle.json`
- storygen `logs/story_scene_plans.json`
- storygen generated scene images

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
