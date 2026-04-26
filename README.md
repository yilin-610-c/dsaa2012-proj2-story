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
- `llm_prompt_img2img_guided`: LLM-assisted prompts with LLM-guided small/medium/large route execution.
- `llm_prompt_anchor_bank`: LLM-assisted prompts plus run-local anchor generation; anchors are not consumed by scene generation yet.
- `llm_prompt_ip_adapter_text2img`: LLM-assisted prompts, run-local anchors, and IP-Adapter identity conditioning for text2img scenes.
- `llm_prompt_two_character_text2img`: conservative two-character text2img policy layer with run-local anchors and IP-Adapter allowed only for single-primary scenes.
- `llm_prompt_hybrid_identity`: guided routing plus IP-Adapter identity conditioning for text2img scenes.

## LLM-Assisted Prompts

The default prompt pipeline remains `rule_based`. To use OpenAI-assisted structured prompt planning, select:

```yaml
prompt:
  pipeline: llm_assisted
```

The LLM-assisted path asks the model for strict JSON containing shared identity, setting, and short per-scene prompts. It does not generate images, score candidates, or write long freeform artistic prompts. The final downstream contract is still `PromptSpec`, so generators use `generation_prompt` and scorers use `scoring_prompt` / `action_prompt` exactly as before.

The prompt pipeline also writes `character_specs` into `logs/prompt_bundle.json`. The contract is:

```text
PromptBundle.metadata["character_specs"] = dict[character_id, JSON-serialized CharacterSpec]
```

These specs are metadata-only identity plans for future anchor/IP-Adapter conditioning; they are not consumed by generation, scoring, routing, or selection yet. Rule-based runs write minimal specs for all recurring characters, or all parsed entities when no recurring characters are tagged. LLM-assisted runs can write stable visual identity fields such as hair, outfit, body build, and accessories.

The canonical per-scene planning metadata is:

```text
PromptBundle.metadata["scene_plans"] = dict[scene_id, scene_plan]
```

`scene_plan` carries visible-character planning, interaction/composition hints, and derived local policy fields such as `policy.visible_character_count` and `policy.scene_focus_mode`. `scene_route_hints` remains available as a route-only derived view for routing.

For generation, `negative_prompt` is passed directly into both text2img and img2img diffusers calls. In the LLM-assisted path, human-character prompts may add narrow pet-substitution suppression terms to reduce failures where a human character is replaced by a pet. Non-human characters should not receive a generic non-human suppression term.

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
    path: prompt_artifacts/llm_assisted_v7/example.json
```

## Prompt-Only Audit

Use the prompt-only audit script to inspect prompt generation across test stories without loading image models, generating candidates, scoring, or writing normal run artifacts:

```bash
PYTHONPATH=src python3 scripts/export_prompts.py --inputs 'test_set/*.txt' --pipelines both
```

By default this writes:

```text
outputs/prompt_audit/audit.json
outputs/prompt_audit/audit.md
outputs/prompt_audit/stories/<story_id>.json
```

`audit.json` keeps the full structured prompt fields, parsed entities, pipeline metadata, LLM scene plans, and route hints. `audit.md` is a compact human-readable summary for reviewing scene-by-scene prompt quality.

`--pipelines both` runs `rule_based` and `llm_assisted`. The LLM-assisted path can call the configured API when no cache or prompt artifact is available. For audit runs, LLM fallback to rule-based is disabled so failures are recorded as `status: failed` instead of being silently mixed with rule-based output.

Checkpoint note for this prompt-audit change: implementation was started from `abe2863`, then moved to branch `feat/prompt-audit-optimization` with checkpoint commit `055a7b6` (`checkpoint: before prompt-audit-export`).

## Anchor Bank v1

Anchor Bank v1 is an optional run-local identity asset step. It consumes `PromptBundle.metadata["character_specs"]`, builds portrait and half-body anchor prompts, and writes generated identity anchors under:

```text
outputs/<run_name>/anchors/<character_id>/
```

Portrait anchors still generate a single inspect-only image. Half-body anchors now generate multiple candidates per character, run a lightweight canonical selector, and save:

```text
outputs/<run_name>/anchors/<character_id>/half_body_cand_0.png
outputs/<run_name>/anchors/<character_id>/half_body_cand_1.png
outputs/<run_name>/anchors/<character_id>/half_body_cand_2.png
outputs/<run_name>/anchors/<character_id>/canonical_half_body.png
outputs/<run_name>/anchors/<character_id>/canonical_anchor.json
```

The selector is intentionally minimal in this version: it tries CLIP text-image alignment against the half-body anchor prompt, adds a simple file/openability quality check, and then writes the chosen canonical half-body anchor. Portrait remains inspect-only.

Run-local anchor generation:

```bash
PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_anchor_bank --input test_set/06.txt
```

## IP-Adapter Identity Conditioning

IP-Adapter identity conditioning is opt-in. It consumes the run-local Anchor Bank output and passes the selected anchor image through `GenerationRequest.reference_image_path` only when `generation.identity_conditioning.enabled=true`.

The v1 default uses the selected `canonical_half_body.png` and applies IP-Adapter only to `text2img` scenes. This keeps previous-frame img2img behavior separate from identity conditioning. Portrait anchors are generated for inspection only and are not selected by the default identity-conditioning path.

For multi-character stories, scene planning now derives local policy from `primary_visible_character_ids`. Ordinary IP-Adapter is only allowed for `policy.scene_focus_mode="single_primary"` scenes. Dual-primary scenes are forced to plain text2img with stronger prompt composition cues, and identity conditioning is skipped with an explicit policy reason instead of guessing a single anchor target.

For example, `test_set/06.txt` contains Jack and Sara in every scene. In the conservative two-character policy layer, those panels should resolve to `policy.scene_focus_mode: "dual_primary"` with `identity_conditioning_subject_id: null`, and the identity-conditioning path should record `policy_skip:dual_primary_scene` instead of forcing one character anchor onto a two-character panel.

The IP-Adapter profiles disable `model.enable_attention_slicing` by default. Attention slicing is a memory-saving diffusers option, but in the current SDXL + IP-Adapter setup it can conflict with IP-Adapter attention processor loading. Non-IP-Adapter profiles keep the base setting unchanged.

```bash
# LLM prompt + text2img + anchor bank + IP-Adapter identity conditioning
PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_ip_adapter_text2img --input test_set/01.txt

# Conservative two-character text2img experiment
PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_two_character_text2img --input test_set/06.txt

# LLM guided routing + anchor bank + IP-Adapter on text2img scenes
PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_hybrid_identity --input test_set/01.txt
```

Multi-character safety validation:

```bash
PYTHONPATH=src python3 -m storygen.cli \
  --profile llm_prompt_ip_adapter_text2img \
  --input test_set/06.txt \
  --run-name ablation_06_ip_adapter_text2img_skip_ambiguous \
  --set prompt.artifact.path=$ARTIFACT_06 \
  --set generation.identity_conditioning.fail_on_missing_anchor=false
```

The configured default adapter is SDXL-oriented:

```yaml
adapter_model_id: h94/IP-Adapter
adapter_subfolder: sdxl_models
adapter_weight_name: ip-adapter_sdxl.bin
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

# LLM-assisted prompt + LLM-guided multi-level img2img routing
PYTHONPATH=src python3 -m storygen.cli --profile llm_prompt_img2img_guided --input test_set/01.txt
```

Img2img routing is disabled by default. When enabled, scene 1 always uses text2img. Later scenes use img2img only when the conservative route policy finds a small continuity-preserving change and a previous selected image is available. Route decisions are logged in `logs/events.jsonl` as `generation_route_selected`.

The guided route policy uses LLM metadata as a planning signal, not as a backend replacement. For each scene, the LLM-assisted prompt pipeline can provide `continuity_subject_ids`, `continuity_route_hint`, `route_change_level`, `route_reason`, and structured `route_factors`. Local routing still makes the final execution decision. By default, `small` uses low-strength img2img, composition-preserving `medium` uses higher-strength img2img, and `large` falls back to text2img. In the guided profile, `route_factors.composition_change_needed=true` also routes to text2img because previous-frame img2img tends to lock the old layout.

The guided profile also enables route-aware scoring. For medium/large changes, previous-image consistency is downweighted and excessive similarity can be penalized, so reranking does not automatically prefer the image that is most similar to the previous panel.

This composition-aware route improves action/layout changes but does not solve character identity by itself. For composition-changing text2img scenes, identity consistency currently comes from prompt text and reranking; future anchor/IP-Adapter work should handle stronger identity conditioning.

## Outputs

Each run writes to `outputs/<run_name>/` and includes:

- `config_resolved.yaml`
- `manifest.json`
- `run_summary.json`
- `logs/prompt_pipeline.json`
- `logs/prompt_bundle.json`
- `logs/anchor_bank.json` when anchor bank is enabled
- `logs/generation_backend.json`
- `logs/events.jsonl`
- `scenes/scene_XXX/prompt.json`
- `scenes/scene_XXX/candidates/*.png`
- `scenes/scene_XXX/scene_result.json`
- `scenes/scene_XXX/selected.png`
- `anchors/<character_id>/anchor_spec.json` and generated anchor images when anchor bank is enabled

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
- `character_specs` are written as prompt-bundle metadata for future anchor/IP-Adapter conditioning.
- `generation.anchor_bank` can generate run-local identity anchors from `character_specs`, but v1 does not feed them back into scene generation.
- `generation.identity_conditioning.enabled=true` uses run-local anchors as IP-Adapter reference images for configured generation modes.
- `reference_image_path` carries the selected anchor path when identity conditioning is enabled.
- The scoring layer now includes a CLIP-based reranker for prompt adherence and previous-frame consistency, while remaining replaceable for future scorers.
