# Prompt Pipeline Robustness Notes

This branch adds subject-type-aware prompt normalization for the `llm_assisted_v9` path and the native StoryDiffusion clean prompt renderer. The goal is to keep the generation backends unchanged while preventing animal and robot subjects from falling back to human identity wording.

## Supported Subject Types

`llm_assisted_v9` now normalizes character specs with:

- `subject_type`: `human`, `animal`, `robot`, `object`, `vehicle`, or `unknown`
- human fields: `hair_color`, `hairstyle`, `skin_tone`, `body_build`, `signature_outfit`, `signature_accessory`
- animal fields: `species`, `fur_color`, `fur_pattern`, `markings`, `body_size`
- robot/object fields: `material`, `color_scheme`, `shape_features`, `signature_parts`

Deterministic fallback handles common tags:

- `Cat`, `Dog`, `Bird` become animal specs with stable species/body defaults.
- `Robot` becomes a robot spec with metal/body defaults.
- human names and roles still get stable visual defaults when under-specified.
- pronoun `it` is not enough to force a human identity.

## Native StoryDiffusion Clean Prompts

The native clean renderer now builds type-aware prompts:

- animal general prompt: `[Cat] small gray cat, short fur`
- animal identity prompt: `full body animal character reference ..., single animal only`
- robot general prompt: `[Robot] silver metal robot, boxy body`
- robot identity prompt: `full body robot character reference ..., single robot only`
- human prompts keep face/outfit identity wording.

Dual animal scenes use `both animals visible` and `two-animal composition`. Broken empty identity fragments such as `Cat is. Dog is.` are stripped.

## LLM Audit Logging

Real LLM calls expose a response record containing request metadata, response metadata, raw response, parsed response, validated output, cache key, and builder version. In `storygen.cli` runs this is written separately:

```text
outputs/<run-name>/logs/llm_prompt_response.json
```

The API key is not recorded. `prompt_bundle.json` keeps only summary metadata and a pointer to the audit log.

For native clean StoryDiffusion prompt-only config generation, the debug JSON also includes the same `llm_response_record` when the prompt builder made or loaded an LLM response.

## Prompt-Only Audit

To regenerate clean native StoryDiffusion prompt debug files without image generation, run `run_test_set.py` without `--run`. This requires `OPENAI_API_KEY` because `clean` mode uses `llm_assisted_v9`.

Example for one animal story:

```bash
conda run -n storygen env PYTHONPATH=src python storydiffusion_gradio_probe/run_test_set.py \
  --input-dir test_setA \
  --config-dir outputs_prompt_verify/a03_clean \
  --output-root outputs_prompt_verify/a03_clean \
  --unwrap-output-dir \
  --glob 03.txt \
  --limit 1 \
  --prompt-builder modular \
  --prompt-modular-backend storydiffusion \
  --storydiffusion-prompt-mode clean
```

Expected checks for `outputs_prompt_verify/a03_clean/03.storydiffusion_prompt_debug.json`:

- no `[Cat] human person` or `[Dog] human person`
- no animal `hairstyle`, `signature_outfit`, or `complete outfit visible`
- no `Cat is. Dog is.`
- animal identity prompt says `single animal only`
- dual animal scene says `both animals visible` and `two-animal composition`

For a full prompt-only audit, run the same command once for `test_set` and once for `test_setA` with a broader glob such as `*.txt`. This covers the 32 local stories without invoking image generation.
