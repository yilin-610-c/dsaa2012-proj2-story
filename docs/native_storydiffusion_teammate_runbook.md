# Teammate Image Generation Runbook

This runbook asks a teammate to run two image-generation batches:

1. Native StoryDiffusion on six selected stories.
2. Anchor Bank + IP-Adapter on four selected stories, using both a distilled SDXL model and a non-distilled SDXL model.

The goal is to get real image outputs for qualitative comparison. Do not reduce image size, steps, or candidate settings unless the run fails from OOM.

Working convention:

- Final image-generation experiments should be run on the remote host.
- Local runs are for prompt-only audit, unit tests, and low-budget sanity checks unless a task explicitly says otherwise.

## Setup

Use this project repo plus the external official StoryDiffusion repo.

```bash
cd /path/to/dsaa2012-phase1-ablation
git switch phase1/native-single-storydiffusion-ablation
git pull

export OPENAI_API_KEY="..."
export STORYDIFFUSION_ROOT=/path/to/StoryDiffusion
```

Sanity checks:

```bash
test -f "$STORYDIFFUSION_ROOT/gradio_app_sdxl_specific_id_low_vram.py"

conda run -n storygen python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
conda run -n storydiffusion python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For WSL, also set:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## Batch A: Native StoryDiffusion

Stories:

```text
test_set/02.txt
test_set/04.txt
test_set/05.txt
test_set/06.txt
test_set/07.txt
test_set/17.txt
```

Output root:

```text
outputs_teammate/native_storydiffusion_llm_direct_selected
```

Dry run:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" \
  python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_set/02.txt,test_set/04.txt,test_set/05.txt,test_set/06.txt,test_set/07.txt,test_set/17.txt \
  --methods native \
  --output-root outputs_teammate/native_storydiffusion_llm_direct_selected \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root "$STORYDIFFUSION_ROOT" \
  --storydiffusion-prompt-mode llm_direct \
  --native-sd-type Unstable \
  --native-width 768 \
  --native-height 768 \
  --native-num-steps 35 \
  --native-guidance-scale 5.0 \
  --native-id-lengths 1 \
  --save-identity-images \
  --dry-run
```

Real run:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" \
  python scripts/run_phase2_ablation_suite.py \
  --suite custom \
  --stories test_set/02.txt,test_set/04.txt,test_set/05.txt,test_set/06.txt,test_set/07.txt,test_set/17.txt \
  --methods native \
  --output-root outputs_teammate/native_storydiffusion_llm_direct_selected \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root "$STORYDIFFUSION_ROOT" \
  --storydiffusion-prompt-mode llm_direct \
  --native-sd-type Unstable \
  --native-width 768 \
  --native-height 768 \
  --native-num-steps 35 \
  --native-guidance-scale 5.0 \
  --native-id-lengths 1 \
  --save-identity-images \
  --continue-on-error \
  --resume
```

Expected run folders:

```text
outputs_teammate/native_storydiffusion_llm_direct_selected/custom/native_llm_direct_id1_test_set_02
outputs_teammate/native_storydiffusion_llm_direct_selected/custom/native_llm_direct_id1_test_set_04
outputs_teammate/native_storydiffusion_llm_direct_selected/custom/native_llm_direct_id1_test_set_05
outputs_teammate/native_storydiffusion_llm_direct_selected/custom/native_llm_direct_id1_test_set_06
outputs_teammate/native_storydiffusion_llm_direct_selected/custom/native_llm_direct_id1_test_set_07
outputs_teammate/native_storydiffusion_llm_direct_selected/custom/native_llm_direct_id1_test_set_17
```

Check in each folder:

- generated story images
- `identity_refs/`
- `manifest.json`
- `*.yaml`
- `*.storydiffusion_prompt_debug.json`

## Batch B: Anchor Bank + IP-Adapter, Distilled SDXL

Stories:

```text
test_set/02.txt
test_set/04.txt
test_set/05.txt
test_set/17.txt
```

This uses the distilled SDXL-Turbo scene model.

Output root:

```text
outputs_anchor_fix/anchor_ipadapter_distilled_scene_llm_direct_selected
```

Dry run:

```bash
for story in 02 04 05 17; do
  conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" \
    python scripts/run_auto_story_pipeline_modular.py \
    --input "test_set/${story}.txt" \
    --run-name "anchor_ipadapter_distilled_test_set_${story}" \
    --output-root outputs_anchor_fix/anchor_ipadapter_distilled_scene_llm_direct_selected \
    --single-env ipadapter \
    --single-profile cloud_anchor_ipadapter_scene \
    --double-env storygen \
    --single-route storygen \
    --double-route storygen \
    --set prompt.pipeline=llm_direct \
    --set 'prompt.llm_direct.targets=["anchor"]' \
    --set prompt.llm.max_output_tokens=6000 \
    --set model.model_id=stabilityai/sdxl-turbo \
    --set model.anchor_bank_model_id=stabilityai/sdxl-turbo \
    --set model.width=768 \
    --set model.height=768 \
    --set model.num_inference_steps=4 \
    --set model.guidance_scale=0.0 \
    --set generation.candidate_count=3 \
    --set generation.identity_conditioning.scale=0.55 \
    --dry-run
done
```

Real run:

```bash
for story in 02 04 05 17; do
  conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" \
    python scripts/run_auto_story_pipeline_modular.py \
    --input "test_set/${story}.txt" \
    --run-name "anchor_ipadapter_distilled_test_set_${story}" \
    --output-root outputs_anchor_fix/anchor_ipadapter_distilled_scene_llm_direct_selected \
    --single-env ipadapter \
    --single-profile cloud_anchor_ipadapter_scene \
    --double-env storygen \
    --single-route storygen \
    --double-route storygen \
    --set prompt.pipeline=llm_direct \
    --set 'prompt.llm_direct.targets=["anchor"]' \
    --set prompt.llm.max_output_tokens=6000 \
    --set model.model_id=stabilityai/sdxl-turbo \
    --set model.anchor_bank_model_id=stabilityai/sdxl-turbo \
    --set model.width=768 \
    --set model.height=768 \
    --set model.num_inference_steps=4 \
    --set model.guidance_scale=0.0 \
    --set generation.candidate_count=3 \
    --set generation.identity_conditioning.scale=0.55
done
```

Expected run folders:

```text
outputs_anchor_fix/anchor_ipadapter_distilled_scene_llm_direct_selected/anchor_ipadapter_distilled_test_set_02
outputs_anchor_fix/anchor_ipadapter_distilled_scene_llm_direct_selected/anchor_ipadapter_distilled_test_set_04
outputs_anchor_fix/anchor_ipadapter_distilled_scene_llm_direct_selected/anchor_ipadapter_distilled_test_set_05
outputs_anchor_fix/anchor_ipadapter_distilled_scene_llm_direct_selected/anchor_ipadapter_distilled_test_set_17
```

## Batch C: Anchor Bank + IP-Adapter, Non-Distilled SDXL

This uses the non-distilled SDXL base model with higher step count and normal CFG.

Output root:

```text
outputs_anchor_fix/anchor_ipadapter_sdxl_scene_llm_direct_selected
```

Dry run:

```bash
for story in 02 04 05 17; do
  conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" \
    python scripts/run_auto_story_pipeline_modular.py \
    --input "test_set/${story}.txt" \
    --run-name "anchor_ipadapter_sdxl_base_test_set_${story}" \
    --output-root outputs_anchor_fix/anchor_ipadapter_sdxl_scene_llm_direct_selected \
    --single-env ipadapter \
    --single-profile cloud_anchor_ipadapter_scene \
    --double-env storygen \
    --single-route storygen \
    --double-route storygen \
    --set prompt.pipeline=llm_direct \
    --set 'prompt.llm_direct.targets=["anchor"]' \
    --set prompt.llm.max_output_tokens=6000 \
    --set model.model_id=stabilityai/stable-diffusion-xl-base-1.0 \
    --set model.anchor_bank_model_id=stabilityai/stable-diffusion-xl-base-1.0 \
    --set model.width=768 \
    --set model.height=768 \
    --set model.num_inference_steps=35 \
    --set model.guidance_scale=5.0 \
    --set generation.candidate_count=3 \
    --set generation.identity_conditioning.scale=0.55 \
    --dry-run
done
```

Real run:

```bash
for story in 02 04 05 17; do
  conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" \
    python scripts/run_auto_story_pipeline_modular.py \
    --input "test_set/${story}.txt" \
    --run-name "anchor_ipadapter_sdxl_base_test_set_${story}" \
    --output-root outputs_anchor_fix/anchor_ipadapter_sdxl_scene_llm_direct_selected \
    --single-env ipadapter \
    --single-profile cloud_anchor_ipadapter_scene \
    --double-env storygen \
    --single-route storygen \
    --double-route storygen \
    --set prompt.pipeline=llm_direct \
    --set 'prompt.llm_direct.targets=["anchor"]' \
    --set prompt.llm.max_output_tokens=6000 \
    --set model.model_id=stabilityai/stable-diffusion-xl-base-1.0 \
    --set model.anchor_bank_model_id=stabilityai/stable-diffusion-xl-base-1.0 \
    --set model.width=768 \
    --set model.height=768 \
    --set model.num_inference_steps=35 \
    --set model.guidance_scale=5.0 \
    --set generation.candidate_count=3 \
    --set generation.identity_conditioning.scale=0.55
done
```

Expected run folders:

```text
outputs_anchor_fix/anchor_ipadapter_sdxl_scene_llm_direct_selected/anchor_ipadapter_sdxl_base_test_set_02
outputs_anchor_fix/anchor_ipadapter_sdxl_scene_llm_direct_selected/anchor_ipadapter_sdxl_base_test_set_04
outputs_anchor_fix/anchor_ipadapter_sdxl_scene_llm_direct_selected/anchor_ipadapter_sdxl_base_test_set_05
outputs_anchor_fix/anchor_ipadapter_sdxl_scene_llm_direct_selected/anchor_ipadapter_sdxl_base_test_set_17
```

## What To Send Back

Please send these artifacts back.

Native StoryDiffusion:

- `outputs_teammate/native_storydiffusion_llm_direct_selected/custom/suite_manifest.jsonl`
- each selected run folder
- especially `identity_refs/`, generated story images, `manifest.json`, `*.yaml`, and `*.storydiffusion_prompt_debug.json`

Anchor + IP-Adapter:

- all four run folders under `outputs_anchor_fix/anchor_ipadapter_distilled_scene_llm_direct_selected/`
- all four run folders under `outputs_anchor_fix/anchor_ipadapter_sdxl_scene_llm_direct_selected/`
- for each run: `run_summary.json`, `manifest.json` if present, `logs/prompt_bundle.json`, `logs/story_scene_plans.json`, `anchors/`, and generated scene images

If any run fails, send:

- terminal traceback
- exact command
- output folder if it was created
- `logs/` folder if present

## Notes

- Native StoryDiffusion uses `--storydiffusion-prompt-mode llm_direct`, so prompt generation uses the new best-effort `llm_direct` pipeline.
- Anchor + IP-Adapter commands use `prompt.pipeline=llm_direct` with `prompt.llm_direct.targets=["anchor"]`; this generates only anchor-compatible prompts.
- For manual prompt search, use `prompt.pipeline=manual_payload` with `prompt.manual_payload.path=<json>`. This bypasses the LLM and copies final executable prompts from the JSON payload into the same Anchor Bank + IP-Adapter scene pipeline.
- Batch B and Batch C use the scene-level `cloud_anchor_ipadapter_scene` profile. Do not use the older `cloud_anchor_ipadapter_story` profile for final Anchor + IP-Adapter runs; that legacy story wrapper bypasses scene multi-candidate CLIP selection.
- Batch B and Batch C intentionally use separate output roots so distilled and non-distilled outputs never collide.
- `generation.candidate_count=3` is intentional and should not be lowered unless the run fails from memory.
- For non-distilled SDXL, use `35` steps and `guidance_scale=5.0`; for SDXL-Turbo, use `4` steps and `guidance_scale=0.0`.

Manual bird-flight prompt search example:

```bash
python scripts/run_auto_story_pipeline_modular.py \
  --input test_set/extra_06.txt \
  --run-name anchor_ipadapter_turbo_manual_extra06_flight_v1 \
  --output-root outputs_anchor_fix/anchor_ipadapter_manual_selected \
  --single-env ipadapter \
  --single-profile cloud_anchor_ipadapter_scene \
  --double-env storydiffusion \
  --single-route storygen \
  --double-route storygen \
  --set prompt.pipeline=manual_payload \
  --set prompt.manual_payload.path=manual_payloads/extra06_bird_flight_anchor.json \
  --set model.model_id=stabilityai/sdxl-turbo \
  --set model.anchor_bank_model_id=stabilityai/sdxl-turbo \
  --set model.width=768 \
  --set model.height=768 \
  --set model.num_inference_steps=4 \
  --set model.guidance_scale=0.0 \
  --set generation.candidate_count=6 \
  --set generation.anchor_bank.half_body_candidate_count=3 \
  --set generation.identity_conditioning.scale=0.15
```
