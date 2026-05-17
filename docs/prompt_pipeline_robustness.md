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

`clean_v2` is an opt-in native StoryDiffusion ablation mode that keeps the default storygen/IP-Adapter path unchanged. It moves stable identity into `general_prompt` and front-loaded identity reference prompts, while story scene prompts stay lightweight and only describe action, setting, spatial relation, and framing. In `clean_v2`, `--native-id-length` means identity reference prompts per character; the generated YAML records `generation.id_length` as the total front-loaded identity prompt count and `generation.storydiffusion_internal_id_length` as the per-character value used by the official StoryDiffusion runner.

`natural` is a second opt-in native StoryDiffusion ablation mode. It keeps the same code-generated `general_prompt`, identity reference prompts, `id_length`, and `save_image_start_index` as `clean_v2`, but rewrites only the story scene prompts into shorter storyboard-style text. The goal is to avoid mechanical clauses such as `both animals visible`, `two-animal composition`, and `action readable` unless a future warning mode explicitly needs them.

## Native StoryDiffusion Debug Findings

The first `natural` image experiments surfaced two native-only debugging gaps:

- Character specs can be keyed differently from display tags. For example, the story tag `[Student]` may receive an LLM spec under `character_specs["student"]`. Native renderers now resolve character specs case-insensitively while preserving the original prompt tag casing, so `[Student]` no longer falls back to `[Student] human person`.
- The official StoryDiffusion flow generates identity reference images before story frames, but the probe used to skip saving them. Native runs can now opt in to `--save-identity-images`, which writes skipped identity frames to `identity_refs/identity_000.png`, `identity_refs/identity_001.png`, and records prompt mappings in the debug JSON and manifest.

For native quality experiments, prefer:

```bash
--storydiffusion-prompt-mode natural \
--native-id-length 2 \
--save-identity-images
```

The saved identity refs are the fastest way to diagnose whether failures come from the identity bank or from later story-frame generation.

The first saved identity refs showed that wording such as `full body character reference`, `full-body portrait`, and `side view character reference` can encourage StoryDiffusion to generate character-sheet or turnaround-style identity images. The native `clean_v2` / `natural` identity prompts now use single-subject wording instead:

- human: `a single human ..., standing alone, centered, plain background, one person in the image`
- animal: `a single ... dog/cat, standing alone, centered, plain background, one animal in the image`
- robot/object: `a single ... robot/object, standing alone, centered, plain background`

The native negative prompt also includes `character sheet`, `turnaround`, `multiple views`, `duplicate person`, `repeated person`, and `triptych`. This is intentionally limited to the native StoryDiffusion identity/debug path; story scene prompts and the default storygen/IP-Adapter pipeline are unchanged.

Native clean/natural prompt generation also raises the LLM structured-output token budget above the base default. A double-character story can exceed the previous 800-token budget and return truncated JSON such as `Unterminated string`; the native probe now uses a larger `prompt.llm.max_output_tokens` override for config/debug generation.

## Anchor Bank Human Age Labels

Anchor Bank prompts now treat `young adult` as an adult human label instead of a child label. This prevents under-specified Student-style characters from producing anchor prompts such as `one human boy child` when the validated character spec says `age_band: young adult` and `gender_presentation: male`. Generic `girl` / `boy` ids and explicit `child` / `kid` / `toddler` / `baby` ages still generate child labels.

## Storygen Anchor/IP-Adapter Profile Name

The storygen route formerly documented as `cloud_storydiffusion_debug` is now named `cloud_anchor_ipadapter_story`. This name is more precise: the route builds story-level `StoryScenePlan` records, generates Anchor Bank references, and uses IP-Adapter over scene-level diffusers generation. It is not the native StoryDiffusion Gradio path. The old `cloud_storydiffusion_debug` profile remains as a backward-compatible alias for existing commands.

## LLM Audit Logging

Real LLM calls expose a response record containing request metadata, response metadata, raw response, parsed response, validated output, cache key, and builder version. In `storygen.cli` runs this is written separately:

```text
outputs/<run-name>/logs/llm_prompt_response.json
```

The API key is not recorded. `prompt_bundle.json` keeps only summary metadata and a pointer to the audit log.

For native clean StoryDiffusion prompt-only config generation, the debug JSON also includes the same `llm_response_record` when the prompt builder made or loaded an LLM response.

## LLM-Direct Prompt Validation

`llm_direct` is a general LLM-owned prompt pipeline for Anchor/IP-Adapter and native StoryDiffusion prompt payloads. It uses target-conditional schemas:

- `targets: ["anchor"]` requires anchor reference prompts, anchor scene prompts, scoring prompts, action prompts, visible character ids, identity-conditioning ids, and scene planning metadata.
- `targets: ["storydiffusion"]` requires StoryDiffusion scene prompts plus `general_prompt`, identity reference prompts, identity prompt count, action prompts, scoring prompts, and scene planning metadata. Anchor scene fields are not required.
- `targets: ["anchor", "storydiffusion"]` requires both sets. The two scene prompt fields stay separate; adapters must not convert one into the other.

Validation now records structured issues with severity `hard_error`, `repair_error`, or `warning`. Hard errors cover malformed interfaces such as bad tags, scene count mismatch, missing target-required fields, and invalid subject types. Repair errors cover prompt boundary violations such as scene leakage in identity/reference fields. Warnings cover non-blocking risks such as token overlap, mood-heavy scene prompts, or unsafe negative prompt wording.

The builder defaults to one repair attempt and best-effort continuation: hard or repair errors trigger repair once, warning-only payloads continue without repair, and unresolved non-fatal validator issues are logged without blocking generation when the payload still contains the minimum fields needed by the selected backend. Local code never rewrites semantic prompt content. API/no-payload failures and payloads that cannot construct backend prompts remain fatal and are recorded as `failed_no_payload` or `failed_unparseable_payload`.

## Stateful Visual Prompt Planning

`llm_direct` now uses Stateful Visual Prompt Planning. The LLM must emit self-contained panel prompts plus planning metadata:

- `visual_continuity_anchors` for persistent or evolving visual state, such as a repeated setting/object/task or lighting state.
- `scene_visual_plan` with `visual_action`, `action_visibility_cue`, and an allowed camera framing.
- `scene_change_level` (`small`, `medium`, or `large`) and `action_critical`.
- final `action_prompt` and `scoring_prompt` with concrete visible evidence for candidate selection.

These fields are not local prompt ingredients. The LLM-authored `anchor_generation_prompt` and `storydiffusion_prompt` must already contain the relevant continuity anchors, action evidence, and framing. Validators check that the plan is reflected in the final prompt text and may request LLM repair, but adapters do not synthesize or rewrite visual content.

For the Anchor/IP-Adapter scene profile, `scene_change_level` is copied into `metadata.scene_route_hints` as `route_change_level`, and `cloud_anchor_ipadapter_scene` enables the existing route-aware scorer. The scorer now treats `action_critical` as a direct weighting signal for this profile: `small/medium/large` changes use lower continuity weights, and `action_critical=true` lowers previous-image consistency further so CLIP selection can prioritize pose/action evidence over near-miss continuity. The score metadata records both the base weights and the effective weights after route-aware/action-critical adjustment.

The current instruction also distinguishes prompt detail levels explicitly:

- `anchor_generation_prompt` is the richest executable scene prompt and should include stable identity, current action, pose or object interaction, relevant setting/continuity anchor, spatial relation, visible near-miss-disambiguating evidence, and camera framing.
- `action_prompt` stays short and focuses on the decisive visible action cue.
- `scoring_prompt` is a compact CLIP selection query rather than a scene summary. It stays short, focuses on the visible discriminator that separates the correct candidate from a near-miss, and avoids long identity descriptions or cinematic wording.

Minimal sanity-check command for the Anchor scene profile:

```bash
conda run -n storygen env PYTHONPATH=src OPENAI_API_KEY="$OPENAI_API_KEY" \
  python scripts/export_prompts.py \
  --inputs test_set/04.txt test_set/extra_06.txt \
  --output-dir outputs_anchor_fix/prompt_audit_stateful_scoring_check \
  --pipelines llm_direct \
  --llm-profile llm_prompt_anchor_bank \
  --set 'prompt.llm_direct.targets=["anchor"]' \
  --set prompt.llm.max_output_tokens=6000
```

## Relaxed Validation

`llm_direct` validation is now intentionally lighter. Repair is reserved for structural or backend-usability problems such as malformed payloads, missing target-required fields, bad ids/tags, invalid scene metadata enums, or anchor identity prompts that leak story scene content.

Prompt-quality issues no longer trigger repair by default. Warnings are still logged in audit/debug outputs for:

- scene plan fields not strongly reflected in the final prompt text
- action or scoring prompts that are still usable but too generic
- continuity anchors that are weakly reflected
- reference-only wording or similar prompt-quality risks

This reduces unnecessary extra LLM repair calls while preserving audit visibility into prompt quality.

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
  --storydiffusion-prompt-mode clean_v2 \
  --id-length 2
```

Expected checks for `outputs_prompt_verify/a03_clean/03.storydiffusion_prompt_debug.json`:

- no `[Cat] human person` or `[Dog] human person`
- no animal `hairstyle`, `signature_outfit`, or `complete outfit visible`
- no `Cat is. Dog is.`
- animal identity prompt says `single animal only`
- dual animal scene says `both animals visible` and `two-animal composition`

For a full prompt-only audit, run the same command once for `test_set` and once for `test_setA` with a broader glob such as `*.txt`. This covers the 32 local stories without invoking image generation.
