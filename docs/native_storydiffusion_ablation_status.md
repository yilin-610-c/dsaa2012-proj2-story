# Native StoryDiffusion Ablation Status

## Current Supported Routes

This branch keeps the original default routing and adds opt-in native StoryDiffusion ablation support.

| Story type | Default route | Command switch | Generation path | Prompt path |
| --- | --- | --- | --- | --- |
| Single-character | `storygen` | default `--single-route storygen` | `python -m storygen.cli --profile cloud_storydiffusion_debug` | `llm_assisted_v9` PromptSpec through storygen, Anchor Bank/IP-Adapter path available |
| Single-character native ablation | native StoryDiffusion probe | `--single-route native_storydiffusion` | `storydiffusion_gradio_probe/run_test_set.py` -> external `StoryDiffusion/gradio_app_sdxl_specific_id_low_vram.py` | `--prompt-builder modular --prompt-modular-backend storydiffusion`; optional `--storydiffusion-prompt-mode clean` |
| Two-character | native StoryDiffusion probe | default for two-character stories | `storydiffusion_gradio_probe/run_test_set.py` -> external StoryDiffusion Gradio app | modular StoryDiffusion prompt backend |

The native probe path does not vendor the official StoryDiffusion repository. A real native run requires:

```bash
--storydiffusion-root /path/to/StoryDiffusion
```

and the file:

```text
/path/to/StoryDiffusion/gradio_app_sdxl_specific_id_low_vram.py
```

## What Changed In This Branch

- Added `--single-route {storygen,native_storydiffusion}` to `scripts/run_auto_story_pipeline_modular.py`.
- Added native runtime forwarding for `--storydiffusion-root`, width, height, step count, seed, and guidance scale.
- Added native preflight validation for the external StoryDiffusion Gradio app file on real runs.
- Added `scripts/run_phase2_ablation_suite.py` for smoke/full ablation orchestration.
- Added `--storydiffusion-prompt-mode {current,clean}` forwarding for native routes.
- Added Gradio launch monkeypatching in `storydiffusion_gradio_probe/run_probe.py` so importing the official app does not start a web server.
- Added clean native prompt rendering that uses the repo `llm_assisted_v9` PromptSpec bundle and writes `storydiffusion_prompt_debug.json`.
- Updated clean native rendering so scene prompts consume `PromptSpec.generation_prompt`, not only raw scene text.
- Clean native rendering now disables rule-based fallback for prompt building. If LLM-assisted prompt planning fails, the run should fail instead of silently producing rule-based prompts.
- Prompt cache is disabled by default. This prevents prompt experiments from reusing stale `.cache/prompt_builder` outputs when prompt logic changes. It can still be explicitly enabled with `--set prompt.cache.enabled=true`.
- Output directories such as `outputs_ablation/`, `outputs_native_llm/`, and `outputs_prompt_verify/` are ignored via `outputs_*/`.

## Important Observed Problem

The run:

```text
outputs_native_llm/native_clean_llm_single_14_generation_prompt_v2
```

was a real native StoryDiffusion probe run. Evidence:

- `storydiffusion_root` points to `/home/lyl610/spring25/StoryDiffusion`.
- `manifest.json` reports `model_type: Only Using Textual Description`.
- The probe calls the official app's `process_generation(...)` through `gradio_app_sdxl_specific_id_low_vram.py`.
- Images were written as `image_000.png`, `image_001.png`, and `image_002.png`.

However, its prompt quality was weak because the LLM-assisted output for `test_set/14.txt` produced only a minimal character identity:

```json
{
  "character_id": "girl",
  "age_band": "child",
  "gender_presentation": "female",
  "hair_color": "",
  "hairstyle": "",
  "signature_outfit": ""
}
```

As a result, the native StoryDiffusion identity prompt became essentially:

```text
[Girl] human girl
```

This is not enough to stabilize a single character across scenes, especially because the native text-only ablation does not use Anchor Bank or IP-Adapter reference images.

## Likely Root Cause

There are two separate prompt issues to separate during debugging:

1. **Stale cache risk**: older runs reused `.cache/prompt_builder` entries. Those cached LLM outputs for story 14 already lacked `hair_color`, `hairstyle`, and `signature_outfit`. With cache enabled, prompt code changes would not necessarily produce new LLM outputs.
2. **Prompt schema behavior**: even with a fresh API call, the current LLM prompt may not force enough visual identity detail for under-specified characters such as `Girl` in `test_set/14.txt`.

The first issue is addressed by disabling prompt cache by default. The second still needs prompt-builder/schema instruction improvement.

## Recommended Next Steps

1. Re-run prompt-only export for `test_set/14.txt` with cache disabled and fallback disabled.
2. Inspect whether fresh LLM output contains stable identity fields:
   - `hair_color`
   - `hairstyle`
   - `skin_tone`
   - `signature_outfit`
   - `signature_accessory`
3. If those fields are still empty, update `src/storygen/llm_assisted_prompt_builder.py` instructions/schema validation so the LLM must infer visually useful, non-generic identity details when the story text is under-specified.
4. Add tests that fail when a named human character receives only generic identity such as `human girl` and empty visual fields.
5. Only after prompt identity fields are stable, re-run native StoryDiffusion clean mode and compare with the default storygen single-character path.

## Commands

Default single-character storygen path:

```bash
conda run -n storygen env PYTHONPATH=src python scripts/run_auto_story_pipeline_modular.py \
  --input test_set/14.txt \
  --run-name single_storygen_default_14 \
  --output-root outputs_ablation/phase2_smoke/smoke \
  --single-env storygen \
  --double-env storydiffusion
```

Single-character native StoryDiffusion clean prompt ablation:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
conda run -n storygen env PYTHONPATH=src python scripts/run_auto_story_pipeline_modular.py \
  --input test_set/14.txt \
  --run-name native_clean_llm_single_14 \
  --output-root outputs_native_llm \
  --single-env storygen \
  --double-env storydiffusion \
  --single-route native_storydiffusion \
  --storydiffusion-root /home/lyl610/spring25/StoryDiffusion \
  --storydiffusion-prompt-mode clean \
  --native-width 512 \
  --native-height 512 \
  --native-num-steps 20
```

Suite smoke with clean native prompt mode:

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
conda run -n storygen env PYTHONPATH=src python scripts/run_phase2_ablation_suite.py \
  --suite smoke \
  --output-root outputs_ablation/phase2_smoke \
  --single-env storygen \
  --double-env storydiffusion \
  --storydiffusion-root /home/lyl610/spring25/StoryDiffusion \
  --storydiffusion-prompt-mode clean \
  --native-width 512 \
  --native-height 512 \
  --native-num-steps 20
```

After a native clean run, inspect:

```bash
cat outputs_native_llm/<run-name>/storydiffusion_prompt_debug.json
cat outputs_native_llm/<run-name>/manifest.json
```

The prompt debug should show `metadata.source: llm_assisted` and non-empty visual identity fields before the image quality comparison is meaningful.
