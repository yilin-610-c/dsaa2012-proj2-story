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

Use Python 3.10+ on a machine that can access the target compute device. For local WSL/GPU work, make sure the NVIDIA driver, CUDA-compatible PyTorch build, and GPU visibility are already working before running image generation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pick a model id that works on your local setup. The default in `configs/base.yaml` is `stabilityai/sdxl-turbo`, but you can override it from the CLI.

The default scorer is `clip_consistency`. It compares each candidate image against the current scene prompt and, for scenes after the first, also compares it with the previous selected frame. This is the first real cross-scene consistency mechanism in the pipeline; prompt wording alone remains a soft bias.

## Run

Profiles are read dynamically from `configs/base.yaml`; the CLI no longer hard-codes the available profile names.

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

Run unit tests after installing requirements:

```bash
PYTHONPATH=src pytest -q
```

Available profiles in the base config:

- `smoke_test`: cheap local baseline run.
- `demo_run`: fuller local baseline run.
- `cloud_strong_backbone`: scene-level `diffusers_text2img` profile reserved for stronger cloud models; override `--model-id` or edit the profile for the target machine.
- `cloud_storydiffusion`: story-level `storydiffusion_direct` placeholder for future StoryDiffusion integration. It is intentionally not implemented yet.
- `llm_prompt_smoke`: optional LLM-assisted prompt planning profile. It calls the configured LLM only when there is no prompt artifact or local cache hit.
- `llm_prompt_text2img`: LLM-assisted prompts with text-to-image generation only.
- `rule_prompt_img2img`: rule-based prompts with conservative img2img continuity routing.
- `llm_prompt_img2img`: LLM-assisted prompts with conservative img2img continuity routing.

## LLM-Assisted Prompts

The default prompt pipeline remains `rule_based`. To use OpenAI-assisted structured prompt planning, select:

```yaml
prompt:
  pipeline: llm_assisted
```

The LLM-assisted path asks the model for strict JSON containing shared identity, setting, and short per-scene prompts. It does not generate images, score candidates, or write long freeform artistic prompts. The final downstream contract is still `PromptSpec`, so generators use `generation_prompt` and scorers use `scoring_prompt` / `action_prompt` exactly as before.

For generation, `negative_prompt` is passed directly into both text2img and img2img diffusers calls. In the LLM-assisted path, human-character prompts may add animal/pet suppression terms to reduce failures where a human character is replaced by a pet or animal.

API keys are read only from environment variables:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_smoke --input test_set/01.txt
```

Prompt reuse has two layers:

- `.cache/prompt_builder/`: local disk cache for avoiding repeated API calls on the same machine. This directory is ignored by git.
- `prompt_artifacts/`: shareable, reviewed prompt JSON files for teammates. These can be committed or sent directly.

To reuse a shared artifact without calling the API:

```yaml
prompt:
  pipeline: llm_assisted
  artifact:
    path: prompt_artifacts/llm_assisted_v3/example.json
```

## Ablation Runs

Use profiles to switch major components without code edits:

```bash
# Rule-based prompt + text2img baseline
PYTHONPATH=src python3 -m storygen.cli --profile smoke_test --input test_set/01.txt

# LLM-assisted prompt + text2img only
PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_text2img --input test_set/01.txt

# Rule-based prompt + conservative img2img routing
PYTHONPATH=src python3 -m storygen.cli --profile rule_prompt_img2img --input test_set/01.txt

# LLM-assisted prompt + conservative img2img routing
PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_img2img --input test_set/01.txt
```

Img2img routing is disabled by default. When enabled, scene 1 always uses text2img. Later scenes use img2img only when the conservative route policy finds a small continuity-preserving change and a previous selected image is available. Route decisions are logged in `logs/events.jsonl` as `generation_route_selected`.

## Outputs

Each run writes to `outputs/<run_name>/` and includes:

- `config_resolved.yaml`
- `manifest.json`
- `run_summary.json`
- `logs/prompt_pipeline.json`
- `logs/generation_backend.json`
- `logs/events.jsonl`
- `scenes/scene_XXX/prompt.json`
- `scenes/scene_XXX/candidates/*.png`
- `scenes/scene_XXX/scene_result.json`
- `scenes/scene_XXX/selected.png`

The run metadata includes the resolved config, runtime profile, model id, timestamp, git commit id when available, pipeline version, seeds, and selected outputs.

`demo_run` now defaults to 4 candidates per scene because reranking needs real choice to matter. Expect CLIP reranking to add extra inference time on top of image generation, but it is still practical on a local RTX 4060.

## Notes

- Prompt building is strictly rule-based and template-based in v1.
- The default backend is pure text-to-image.
- `prompt.pipeline=rule_based` is implemented and preserves the baseline prompt behavior.
- `prompt.pipeline=llm_assisted` uses structured OpenAI prompt planning with validation, cache, artifacts, and rule-based fallback.
- `prompt.pipeline=api` is kept as a compatibility alias for `llm_assisted`.
- `model.backend=storydiffusion_direct` is a story-level placeholder and raises `NotImplementedError` until the StoryDiffusion backend is implemented.
- `previous_selected_image_path` is used by the optional img2img continuity route when enabled.
- `reference_image_path` is reserved for future anchor/IP-Adapter conditioning.
- The scoring layer now includes a CLIP-based reranker for prompt adherence and previous-frame consistency, while remaining replaceable for future scorers.
