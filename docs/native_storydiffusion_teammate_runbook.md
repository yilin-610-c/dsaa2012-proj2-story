# Native StoryDiffusion Teammate Runbook

This runbook is for high-VRAM native StoryDiffusion ablation runs delegated to a teammate. It focuses on the native StoryDiffusion path:

- `scripts/run_phase2_ablation_suite.py`
- `scripts/run_auto_story_pipeline_modular.py`
- `storydiffusion_gradio_probe/run_test_set.py`
- external official StoryDiffusion repo through `--storydiffusion-root`

It does not run the default `cloud_anchor_ipadapter_story` storygen + Anchor Bank + IP-Adapter route unless `--methods storygen` is added explicitly.

## Environment

There are two separate repositories involved:

1. This project repo: `dsaa2012-proj2-story`
2. The official external StoryDiffusion repo: `StoryDiffusion`

The official StoryDiffusion repo is **not vendored** into this project and is **not a git submodule**. Native StoryDiffusion runs import the official Gradio app from that external folder, so every real native run must know where that folder is.

If the external repo is not already installed, clone it outside this project repo, for example:

```bash
mkdir -p /path/to/external/repos
cd /path/to/external/repos
git clone https://github.com/HVision-NKU/StoryDiffusion.git
```

Then use that folder path in every native command:

```bash
--storydiffusion-root /path/to/external/repos/StoryDiffusion
```

For example, if the repo was cloned to `/home/teammate/spring25/StoryDiffusion`, every command below should use:

```bash
--storydiffusion-root /home/teammate/spring25/StoryDiffusion
```

Required checks:

```bash
cd /path/to/dsaa2012-proj2-story
git switch phase1/native-single-storydiffusion-ablation
git pull

conda activate storygen
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

conda run -n storydiffusion python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"

test -f /path/to/external/repos/StoryDiffusion/gradio_app_sdxl_specific_id_low_vram.py
```

Replace `/path/to/external/repos/StoryDiffusion` in the commands below with the real external StoryDiffusion folder on the teammate's machine.

For WSL machines that need the CUDA driver library path:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

For `--storydiffusion-prompt-mode natural`, the prompt builder uses the LLM-assisted prompt path. Set:

```bash
export OPENAI_API_KEY="..."
```

## Model Choice

The native probe forwards `--native-sd-type` to StoryDiffusion's `--sd-type`. Valid names depend on the external StoryDiffusion repo config; common choices are:

- `Unstable`
- `RealVision`
- `Juggernaut`
- `SDXL`

Do not test all models on all stories first. If identity reference images still look like character sheets or three-view turnarounds, test model choice on one or two stories only, then use the best-looking setting for the broader run.

Recommended first model sanity check:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_setA/19.txt \
  --methods native \
  --output-root outputs_ablation/model_sanity_unstable \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /path/to/StoryDiffusion \
  --storydiffusion-prompt-mode natural \
  --native-sd-type Unstable \
  --native-width 768 \
  --native-height 768 \
  --native-num-steps 35 \
  --native-id-lengths 1 \
  --save-identity-images \
  --continue-on-error
```

Optional one-story comparison:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_setA/19.txt \
  --methods native \
  --output-root outputs_ablation/model_sanity_realvision \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /path/to/StoryDiffusion \
  --storydiffusion-prompt-mode natural \
  --native-sd-type RealVision \
  --native-width 768 \
  --native-height 768 \
  --native-num-steps 35 \
  --native-id-lengths 1 \
  --save-identity-images \
  --continue-on-error
```

Compare:

- `outputs_ablation/model_sanity_*/custom/native_natural_id1_test_setA_19/identity_refs/`
- `outputs_ablation/model_sanity_*/custom/native_natural_id1_test_setA_19/*.storydiffusion_prompt_debug.json`
- saved story frames in the same run folder

## Stage 1: Three Representative Stories

Run this first. It covers one single human identity, one animal pair, and one two-human story.

Stories:

- `test_setA/19.txt`: Student, single human identity
- `test_setA/03.txt`: Cat/Dog, animal dual subject
- `test_set/06.txt`: Jack/Sara, two-human story

Dry run:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_setA/19.txt,test_setA/03.txt,test_set/06.txt \
  --methods native \
  --output-root outputs_ablation/native_stage1_representative \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /path/to/StoryDiffusion \
  --storydiffusion-prompt-mode natural \
  --native-sd-type Unstable \
  --native-width 768 \
  --native-height 768 \
  --native-num-steps 35 \
  --native-id-lengths 1 \
  --save-identity-images \
  --dry-run
```

Real run:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_setA/19.txt,test_setA/03.txt,test_set/06.txt \
  --methods native \
  --output-root outputs_ablation/native_stage1_representative \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /path/to/StoryDiffusion \
  --storydiffusion-prompt-mode natural \
  --native-sd-type Unstable \
  --native-width 768 \
  --native-height 768 \
  --native-num-steps 35 \
  --native-id-lengths 1 \
  --save-identity-images \
  --continue-on-error
```

Check before expanding:

- `outputs_ablation/native_stage1_representative/custom/suite_manifest.jsonl`
- each run's `identity_refs/`
- each run's `*.storydiffusion_prompt_debug.json`
- whether identity refs are single-subject images, not three-view character sheets
- whether story frames preserve identity across panels

## Stage 2: Half-Coverage Suite

Run this if Stage 1 looks usable. This list is intentionally mixed across human, animal, robot, child, occupation/action, travel/vehicle, and two-character stories.

Stories:

```text
test_setA/19.txt,test_setA/03.txt,test_set/06.txt,test_setA/13.txt,
test_set/14.txt,test_set/16.txt,test_set/17.txt,test_setA/10.txt,
test_setA/15.txt,test_setA/18.txt,test_setA/20.txt,test_set/07.txt,
test_set/extra_03.txt,test_set/extra_06.txt,test_setA/extra_10.txt,test_setA/extra_11.txt
```

Command:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_setA/19.txt,test_setA/03.txt,test_set/06.txt,test_setA/13.txt,test_set/14.txt,test_set/16.txt,test_set/17.txt,test_setA/10.txt,test_setA/15.txt,test_setA/18.txt,test_setA/20.txt,test_set/07.txt,test_set/extra_03.txt,test_set/extra_06.txt,test_setA/extra_10.txt,test_setA/extra_11.txt \
  --methods native \
  --output-root outputs_ablation/native_stage2_half \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /path/to/StoryDiffusion \
  --storydiffusion-prompt-mode natural \
  --native-sd-type Unstable \
  --native-width 768 \
  --native-height 768 \
  --native-num-steps 35 \
  --native-id-lengths 1 \
  --save-identity-images \
  --continue-on-error \
  --resume
```

If this is too slow or hits OOM, reduce only one dimension at a time:

- first try `--native-width 640 --native-height 640`
- then try `--native-num-steps 30`
- keep `--native-id-lengths 1` unless identity refs are clearly too weak

## Stage 3: Full Native Suite

Run this after Stage 1 and Stage 2 are acceptable. The command uses all currently tracked story files from both test folders.

Command:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_set/01.txt,test_set/02.txt,test_set/04.txt,test_set/05.txt,test_set/06.txt,test_set/07.txt,test_set/08.txt,test_set/09.txt,test_set/11.txt,test_set/14.txt,test_set/16.txt,test_set/17.txt,test_set/extra_03.txt,test_set/extra_06.txt,test_set/extra_08.txt,test_set/extra_09.txt,test_setA/03.txt,test_setA/10.txt,test_setA/12.txt,test_setA/13.txt,test_setA/15.txt,test_setA/18.txt,test_setA/19.txt,test_setA/20.txt,test_setA/extra_01.txt,test_setA/extra_02.txt,test_setA/extra_04.txt,test_setA/extra_05.txt,test_setA/extra_07.txt,test_setA/extra_10.txt,test_setA/extra_11.txt,test_setA/extra_12.txt \
  --methods native \
  --output-root outputs_ablation/native_stage3_full \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /path/to/StoryDiffusion \
  --storydiffusion-prompt-mode natural \
  --native-sd-type Unstable \
  --native-width 768 \
  --native-height 768 \
  --native-num-steps 35 \
  --native-id-lengths 1 \
  --save-identity-images \
  --continue-on-error \
  --resume
```

## What To Send Back

Please send back:

- `outputs_ablation/<run>/custom/suite_manifest.jsonl`
- for each failed run, the terminal traceback and the run folder
- for each representative run, the full run folder including:
  - `identity_refs/`
  - generated story images
  - `manifest.json`
  - `*.yaml`
  - `*.storydiffusion_prompt_debug.json`

For quick visual review, prioritize:

- `native_natural_id1_test_setA_19`
- `native_natural_id1_test_setA_03`
- `native_natural_id1_test_set_06`
- one robot run, e.g. `native_natural_id1_test_setA_13`
- one animal single run, e.g. `native_natural_id1_test_set_extra_03`

## Notes

- `--native-id-lengths 1` means one front-loaded identity reference prompt per character. This usually avoids multiplying bad identity refs.
- `--save-identity-images` saves those front-loaded identity frames under `identity_refs/` so we can inspect whether the identity bank itself is clean.
- `--resume` skips completed run folders when `manifest.json` or `run_summary.json` exists.
- `--continue-on-error` keeps the suite moving if one story fails.
- Use separate `--output-root` folders when comparing `--native-sd-type`; otherwise run names can collide because model type is not part of the experiment name.
