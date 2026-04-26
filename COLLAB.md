# Collaboration Guide

This repo is built on top of the current `storygen` baseline. The goal is to evolve the existing pipeline into a stronger, modular, reproducible system for multi-panel story image generation, not to replace it with a separate codebase.

## Current Objective

The current project goal is to improve the baseline along:
- per-panel correctness
- cross-panel consistency
- narrative continuity
- style consistency
- overall image quality

Engineering goals:
- keep the pipeline fixed and automatic
- keep runs reproducible and inspectable
- preserve modular boundaries so collaborators can work in parallel
- support quick controlled experiments on dev/Test-A cases
- preserve a clean baseline path for comparison and reporting

Non-goals for the current repo stage:
- full repo rewrite
- large custom training stack first
- paper-faithful reimplementation at the cost of engineering stability
- large academic ablation infrastructure

## Team Split

### Collaborator A: orchestration / prompting / evaluation
- pipeline orchestration
- parsing and prompt pipeline upgrades
- future `SceneSpec` design
- scoring / reranking / evaluation
- dev subset workflow and controlled comparisons
- run scripts, output organization, reproducibility docs

### Collaborator B: generation / consistency mechanisms
- generation-module improvements
- reference-conditioned generation
- identity and style consistency mechanisms
- anchor and reference integration
- img2img and continuity-oriented generation routes
- StoryDiffusion-like generation experiments when they fit the repo

### Shared ownership
- config shape
- output naming and logging conventions
- shared dataclass and interface fields
- experiment naming and comparison setup
- final integration and submission path

Rules:
- Do not silently rename shared fields, config keys, CLI flags, or output files.
- Do not remove an existing runnable path without replacing and documenting it.
- Any shared-interface change must be reflected in code and in this document.

## Current Code Map

Current working layout:
- `src/storygen/parser.py`: scene text parsing
- `src/storygen/types.py`: active internal contracts
- `src/storygen/prompt_builder.py`: rule-based prompt construction
- `src/storygen/prompt_pipelines.py`: prompt pipeline selector and placeholders
- `src/storygen/generators/`: generation backends and wrappers
- `src/storygen/scoring/`: scoring and reranking backends
- `src/storygen/pipeline.py`: end-to-end orchestration
- `src/storygen/cli.py`: stable CLI entrypoint
- `configs/base.yaml`: main config and runtime profiles
- `scripts/run_demo.py`: demo entrypoint
- `outputs/<run_name>/...`: run artifacts

Current active internal contracts are still `Scene`, `Story`, and `PromptSpec`. The repo now also has minimal extensibility contracts for `PromptBundle`, `StoryPromptSpec`, `StoryGenerationRequest`, and `StoryGenerationResult`, but the live baseline still runs through scene-level `PromptSpec` generation.

## Active Interfaces

### `Scene` and `Story`

Current source of truth: `src/storygen/types.py`

Current `Scene` fields:
- `scene_id`
- `index`
- `raw_text`
- `clean_text`
- `entities`

Current `Story` fields:
- `source_path`
- `raw_text`
- `scenes`
- `all_entities`
- `recurring_entities`
- `entity_to_scene_ids`

These are already shared pipeline interfaces. Do not rename or remove fields casually.

### `PromptSpec`

Current source of truth: `src/storygen/types.py`

Current `PromptSpec` fields:
- `style_prompt`
- `character_prompt`
- `global_context_prompt`
- `local_prompt`
- `action_prompt`
- `generation_prompt`
- `scoring_prompt`
- `full_prompt`
- `negative_prompt`

Intended meaning:
- `full_prompt`: rich human-inspectable prompt for debugging and metadata
- `generation_prompt`: shorter prompt actually passed into the text-to-image generator
- `scoring_prompt`: short prompt used for CLIP text-image scoring
- `action_prompt`: short action-only prompt used for action-sensitive scoring
- `negative_prompt`: passed to both text2img and img2img generation calls as the negative conditioning text

Any change to these semantics must be coordinated because prompting, generation, scoring, and output inspection all depend on them.

### Generator and scorer contracts

Current source of truth:
- `src/storygen/generators/base.py`
- `src/storygen/scoring/base.py`

Scene-level generator contract:
- consumes `GenerationRequest`
- returns `GenerationCandidate`

Story-level generator contract:
- consumes `StoryGenerationRequest`
- returns `StoryGenerationResult`

Implemented generation backends:
- `diffusers_text2img`: scene-level baseline backend
- `storydiffusion_direct`: story-level placeholder only; it raises `NotImplementedError`

`diffusers_text2img` now supports two generation routes:
- `text2img`: default route and baseline behavior
- `img2img`: optional continuity route using the previous selected scene image as the init image

Img2img is a route inside the scene-level backend, not a replacement backend. Scene 1 must remain text2img. Portrait or anchor images must not be used as scene 1 init images.

Do not implement StoryDiffusion by changing `diffusers_text2img` or by treating StoryDiffusion as only another `model_id`. StoryDiffusion should be implemented behind `storydiffusion_direct` because its execution unit is the whole story, not one scene candidate.

Scorer contract:
- consumes `Story`, `Scene`, `PromptSpec`, `GenerationCandidate`, and previous selected results
- returns structured `CandidateScore`
- selects one `SceneSelectionResult`

This contract should stay stable while generation/scoring implementations change underneath.

### Prompt pipeline contracts

Current source of truth: `src/storygen/prompt_pipelines.py`

Implemented prompt pipelines:
- `rule_based`: wraps the existing `PromptBuilder` and preserves baseline behavior
- `llm_assisted`: OpenAI-backed structured prompt planning with validation, local cache, shareable artifacts, and rule-based fallback
- `api`: compatibility alias for `llm_assisted`

`prompt.pipeline` is the config switch. LLM-assisted prompting must keep the downstream `PromptSpec` contract unchanged and must not replace the rule-based baseline.

## Planned Stable Interfaces

These are collaboration targets, not fully implemented repo-wide contracts yet.

### `SceneSpec` (planned)

Purpose:
- structured scene representation before generation

Intended minimum fields:
- `scene_id`
- `raw_text`
- `characters`
- `objects`
- `setting`
- `action`
- `style`
- `continuity_tags`

Possible extensions:
- `camera` or `shot_type`
- `mood`
- `scene_transition_flags`
- `recommended_generation_mode`

Current mapping:
- parts of this are currently spread across `Scene`, `Story`, and prompt-building logic

### `AnchorSpec` (planned)

Purpose:
- cross-panel consistency anchors for character/object/style/background identity

Intended minimum fields:
- `anchor_id`
- `type`
- `reference_paths`
- `must_keep`
- `optional_keep`

Possible extensions:
- `adapter_scale`
- `scope`
- `linked_scene_ids`

Current mapping:
- active placeholder dataclass exists for future anchor bank paths
- no anchor generation or IP-Adapter conditioning is implemented yet
- future generation-side work should use it without breaking the baseline interfaces

### `CharacterSpec` (planned)

Purpose:
- stable identity attributes for recurring characters only

Intended fields:
- age band
- gender presentation
- hair color and hairstyle
- skin tone
- body build
- signature outfit
- signature accessory
- persistent profession marker when visually relevant

Rule:
- do not put scene-specific action, pose, emotion, object state, or transient lighting into `CharacterSpec`

Active metadata contract:
- `PromptBundle.metadata["character_specs"]` is a `dict`
- each key is a `character_id`
- each value is the JSON-serialized `CharacterSpec`
- this metadata is not consumed by generator, scorer, routing, or selection yet
- future anchor bank and IP-Adapter work should consume this contract instead of parsing identity back out of prompt strings

### `ScoringSpec` (planned)

Purpose:
- declarative scoring and reranking definition

Intended minimum fields:
- `scene_id`
- scoring weights
- must-have checks
- consistency targets
- selected metrics

Possible dimensions:
- text alignment
- character consistency
- object continuity
- background or style consistency
- image quality
- narrative continuity

Current mapping:
- parts of this currently live in `configs/base.yaml`, `PromptSpec`, and scorer implementations

## Run Expectations

Stable entrypoints already in the repo:

```bash
PYTHONPATH=src python3 -m storygen --config configs/base.yaml --profile smoke_test --input test_set/01.txt
PYTHONPATH=src python3 -m storygen --config configs/base.yaml --profile demo_run --input test_set/01.txt
PYTHONPATH=src python3 scripts/run_demo.py --profile demo_run --input test_set/01.txt
```

Profiles are dynamic and validated from `configs/base.yaml`; the CLI does not hard-code the profile list.

Current base profiles:
- `smoke_test`: cheap scene-level baseline
- `demo_run`: fuller scene-level baseline
- `cloud_strong_backbone`: scene-level diffusers profile reserved for stronger cloud backbones
- `cloud_storydiffusion`: story-level StoryDiffusion placeholder; currently not runnable for real generation
- `llm_prompt_smoke`: optional LLM-assisted prompt planning profile
- `llm_prompt_text2img`: LLM prompts with text2img only
- `rule_prompt_img2img`: rule prompts with conservative img2img routing
- `llm_prompt_img2img`: LLM prompts with conservative img2img routing
- `llm_prompt_img2img_guided`: LLM prompts with guided small/medium/large route execution

Expected comparison workflow:
- baseline path
- prompt-improved path
- anchor or reference-conditioned path
- stronger combined path
- quick dev subset path

Rule:
- major experiment choices should be controlled through config or CLI, not by editing code for each run

Current config pattern already supports this style. Prefer adding switches under existing YAML sections, for example:
- `prompt.rewriter.type`
- `generation.routing.img2img_enabled`
- future `generation.identity_conditioning.enabled`
- future `scoring.type`

## Output and Logging Conventions

Current source of truth: `src/storygen/pipeline.py` and `src/storygen/io/results.py`

Current run layout:

```text
outputs/<run_name>/
  config_resolved.yaml
  manifest.json
  run_summary.json
  logs/
    prompt_pipeline.json
    generation_backend.json
    events.jsonl
  scenes/
    scene_001/
      prompt.json
      scene_result.json
      selected.png
      candidates/
        cand_000_seed_....png
```

Required collaboration rule:
- do not silently change output naming or placement
- if output structure must change, keep backward compatibility where feasible and update this document

Each run should continue to preserve:
- resolved config
- prompt pipeline metadata
- generation backend metadata
- event log
- prompt artifacts
- candidate images
- selected images
- scores and selection metadata
- summary metadata for reproducibility

## Current Extensibility Status

Implemented:
- dynamic runtime profiles from YAML
- `prompt.pipeline=rule_based`
- `prompt.pipeline=llm_assisted`
- `prompt.pipeline=api` alias for `llm_assisted`
- `model.backend=diffusers_text2img` with `model.granularity=scene`
- `model.backend=storydiffusion_direct` placeholder with `model.granularity=story`
- minimal run logs under `outputs/<run_name>/logs/`
- local prompt cache under `.cache/prompt_builder/`
- shareable prompt artifacts under `prompt_artifacts/`
- `character_specs` prompt-bundle metadata for future identity conditioning
- optional run-local anchor bank generation under `outputs/<run_name>/anchors/`
- optional IP-Adapter identity conditioning for scene-level diffusers text2img routes
- optional conservative img2img routing through `generation.routing`
- optional LLM-guided multi-level routing through `route_policy=llm_guided_conservative`

Not implemented:
- real StoryDiffusion generation
- training-free attention/reference/latent components
- shared anchor artifacts / anchor reuse
- multi-anchor or multi-character IP-Adapter conditioning

The current baseline behavior should remain unchanged for `smoke_test`, `demo_run`, and any profile using `diffusers_text2img + scene`.

### LLM Prompt Sharing Rules

- `.cache/prompt_builder/` is local acceleration state and should not be committed.
- `prompt_artifacts/` is the collaboration surface for reviewed LLM prompt outputs.
- To share generated prompts with teammates, export an artifact and commit or send that JSON.
- To reuse teammate prompts, set `prompt.artifact.path`; this bypasses API calls.
- API keys must come from environment variables and must never be written into configs, cache records, artifacts, or docs.
- LLM-assisted prompting is for structured extraction and short prompt rewriting only; it does not own image generation or scoring.
- Human-character LLM prompts may augment `negative_prompt` with animal/pet suppression terms to avoid human-to-animal substitutions.
- LLM route metadata is a planning signal only. It may provide `continuity_subject_ids`, `continuity_route_hint`, `route_change_level`, and `route_reason`, but local routing still owns the final generation mode and strength.

### Img2Img Routing Rules

- `generation.routing.img2img_enabled=false` preserves text2img-only behavior.
- `route_policy=conservative` may use img2img for scene 2+ when a previous selected image exists and the scene appears to be a small continuity-preserving change.
- `route_policy=llm_guided_conservative` may use LLM-resolved continuity subjects to recover pronoun-only scenes that parser entities miss.
- guided routing uses `route_change_level`: `small` uses low-strength img2img, `medium` uses higher-strength img2img, and `large` routes back to text2img by default.
- route decisions are logged as `generation_route_selected`.
- candidate metadata must preserve `generation_mode`, `route_reason`, `route_change_level`, `continuity_subject_ids`, `continuity_route_hint`, `init_image_path`, and `img2img_strength`.
- current img2img init source is only the previous selected scene image.
- img2img is a route execution choice inside the scene backend, not a replacement for `diffusers_text2img`.
- future anchors and IP-Adapter conditioning should remain separately switchable for ablations.

## Controlled Experiment Expectations

We do not need a large formal ablation framework right now.

We do need a few clean, repeatable comparisons such as:
- baseline
- baseline + prompt improvements
- baseline + anchor or reference conditioning
- baseline + anchor/reference + rerank

That means:
- baseline behavior must remain runnable
- improved variants should be toggled through config where possible
- collaborators should avoid hard-coded one-off experiment branches inside shared modules

## Recent Implementation Notes

### 2026-04-22: LLM-Guided Multi-Level Img2Img Routing

Implemented:
- added LLM route metadata fields: `continuity_subject_ids`, `continuity_route_hint`, `route_change_level`, and `route_reason`
- added `PromptBundle.metadata["scene_route_hints"]` as the carrier for route hints
- added `route_policy=llm_guided_conservative`
- added small/medium/large route execution levels through `strength_by_change_level` and `execution_by_change_level`
- added `llm_prompt_img2img_guided` profile for LLM prompts with guided multi-level routing
- bumped the LLM prompt builder/cache/artifact namespace to `llm_assisted_v4`
- extended route event logs and candidate metadata with route hint fields

Behavior preserved:
- `smoke_test`, `demo_run`, and text2img-only profiles remain baseline-compatible
- `PromptSpec` is unchanged
- `diffusers_text2img` remains the scene backend; img2img is a route inside that backend
- `large` guided changes default to text2img instead of high-strength img2img retry

Validation:
- `conda run -n storygen env PYTHONPATH=src pytest -q`
- result: `57 passed`

Known next-step considerations:
- old `llm_assisted_v3` cache/artifacts do not contain route metadata; use v4 artifacts for guided routing
- if medium-strength img2img still under-changes a scene, the next likely fix is semantic failure detection or scorer weight adjustment, not IP-Adapter yet
- anchor bank and IP-Adapter remain future, separately switchable components

### 2026-04-22: LLM Routing Rubric And Route-Aware Scoring

Implemented:
- bumped the LLM prompt builder/cache/artifact namespace to `llm_assisted_v5`
- added structured `route_factors` to LLM-assisted route metadata
- tightened the LLM route rubric so `small` requires nearly unchanged subject, setting, framing, body state, and action
- added generic self-consistency correction from route factors, without story-specific keyword route overrides
- added route-aware CLIP consistency scoring for guided routing
- fixed prompt metadata logging so route hints are visible after prompt build

Defaults:
- guided route-aware scoring is enabled only in `llm_prompt_img2img_guided`
- `small` keeps continuity-heavy scoring and has no over-similarity penalty by default
- `medium` uses previous-image consistency weight `0.10` and penalty threshold `0.90`
- `large` uses previous-image consistency weight `0.0` and penalty threshold `0.85`

Behavior preserved:
- baseline and text2img-only profiles keep the old scoring behavior unless route-aware scoring is explicitly enabled
- `PromptSpec` remains unchanged
- anchor bank, IP-Adapter, ControlNet, and automatic regeneration loops remain out of scope

Validation:
- `conda run -n storygen env PYTHONPATH=src pytest -q`
- result: `64 passed`

Known next-step considerations:
- old `llm_assisted_v4` cache/artifacts do not include route factors; use v5 artifacts for route-aware guided experiments
- route-aware scoring reduces over-similar selections but does not regenerate failed candidates
- no per-story hardcoded route keywords should be added for released unseen examples

### 2026-04-22: Guided Img2Img V5 Failure Observation

Observed run:
- `outputs/llm_img2img_guided_2026-04-22T13-12-58+00-00`

What worked:
- `llm_assisted_v5` was active
- route factors were logged in `prompt_pipeline.json` and `prompt_bundle.json`
- scene 2 and scene 3 were no longer classified as `small`; both were classified as `medium`
- `img2img_strength=0.65` was used for medium scenes
- route-aware scoring was active and applied over-similarity penalties

Failure:
- scene 2 and scene 3 still visually stayed close to the cooking/breakfast composition
- the generated candidates did not create enough composition change for looking out the window or sitting with a book
- scorer could only rerank available candidates; it could not fix the fact that all candidates were under-changed

Conclusion:
- previous-frame img2img is still too restrictive when `route_factors.composition_change_needed=true`
- medium-strength img2img is not sufficient for composition-changing panels with the current SDXL Turbo setup
- identity continuity should not depend on previous-frame img2img for composition-changing scenes; use text2img now, and later use anchors/IP-Adapter for identity conditioning

Recommended next change:
- keep `small -> img2img`
- route `composition_change_needed=true -> text2img`
- keep `large -> text2img`
- optionally allow `medium -> img2img` only when `composition_change_needed=false`

### 2026-04-22: Composition-Aware Guided Routing

Implemented:
- added `generation.routing.text2img_when_composition_change_needed`
- enabled it in `llm_prompt_img2img_guided`
- guided routing now sends `route_factors.composition_change_needed=true` scenes to text2img
- composition-preserving `medium` scenes may still use img2img with medium strength
- route metadata and event logs still preserve `route_factors`, final mode, and route reason

Behavior preserved:
- baseline, text2img-only, rule-based img2img, and conservative LLM img2img profiles keep existing behavior
- no story-specific keyword route overrides were added
- `PromptSpec` remains unchanged

Validation:
- `conda run -n storygen env PYTHONPATH=src pytest -q`
- result: `67 passed`

Observed run after this change:
- `outputs/llm_img2img_guided_2026-04-22T13-53-57+00-00`

What changed in the run:
- scene 2 and scene 3 had `route_factors.composition_change_needed=true`
- both scenes were routed to `text2img` with route reason `llm_guided_composition_change_text2img`
- previous-frame img2img no longer locked scene 2 and scene 3 into the breakfast/cooking composition

Remaining limitation:
- character identity consistency became weaker across text2img composition-changing scenes
- scene 3 captured the book/reading composition better than the eating action
- this supports the next phase: identity anchors or IP-Adapter for text2img scenes, rather than using previous-frame img2img as the identity mechanism

### 2026-04-23: CharacterSpec Extraction Metadata

Implemented:
- added `character_specs` to `PromptBundle.metadata`
- rule-based prompting now emits a minimal `CharacterSpec` with `metadata.source=rule_based`
- LLM-assisted prompting now asks for structured stable character identity blocks under `global.characters`
- bumped the LLM prompt builder/cache/artifact namespace to `llm_assisted_v6`
- added `src/storygen/character_specs.py` as the extraction boundary for future anchor bank and IP-Adapter work

Behavior preserved:
- `PromptSpec` is unchanged
- generator, scorer, routing, and image selection do not read `CharacterSpec` yet
- baseline profiles remain runnable and should produce the same generation behavior apart from extra metadata logs

Rules:
- `CharacterSpec` must contain stable visible identity only
- do not include action, pose, emotion, temporary objects, object state, lighting, camera framing, or scene-specific context
- future anchor bank and IP-Adapter work should consume `PromptBundle.metadata["character_specs"]` through config-gated paths

Validation:
- targeted tests: `conda run -n storygen env PYTHONPATH=src pytest -q tests/test_llm_assisted_prompt_builder.py tests/test_prompt_pipelines.py tests/test_pipeline_logging.py tests/test_config.py`
- result: `32 passed`

Traceability note:
- attempted to create `feat/character-spec-extraction` and `character-spec-extraction` branches before editing, but `.git/refs` was read-only in this environment
- baseline recorded before edits: `main@34d0d17`

### 2026-04-23: CharacterSpec Multi-Character Hardening

Implemented:
- rule-based `character_specs` now emits all recurring characters in deterministic order
- when no recurring characters are tagged, rule-based extraction falls back to all parsed entities instead of only the first entity
- stories with no extractable entities now produce an empty `character_specs` dict
- LLM-assisted tests cover multi-character specs and main-character alignment validation
- human negative prompt augmentation no longer adds a generic `non-human subject` suppression term

Behavior preserved:
- `character_specs` remains metadata-only
- generator, scorer, routing, and selection inputs are unchanged by the presence of `character_specs`
- non-human characters should not be forced into `human person` identity prompts or excluded by generic non-human negative prompts

Validation:
- targeted tests include rule-based sparse/multi-character stories, non-human LLM prompt behavior, and a downstream request regression proving `character_specs` is non-intrusive
- targeted command: `conda run -n storygen env PYTHONPATH=src pytest -q tests/test_llm_assisted_prompt_builder.py tests/test_prompt_pipelines.py tests/test_pipeline_logging.py tests/test_prompt_builder.py`
- targeted result: `43 passed`
- full regression command: `conda run -n storygen env PYTHONPATH=src pytest -q`
- full regression result: `77 passed`

### 2026-04-23: Anchor Bank v1

Implemented:
- added config-gated run-local anchor bank generation
- anchor bank consumes `PromptBundle.metadata["character_specs"]`
- generated anchors are written under `outputs/<run_name>/anchors/<character_id>/`
- `logs/anchor_bank.json` records anchor prompts, seeds, paths, and source character specs
- added profile `llm_prompt_anchor_bank`

Behavior preserved:
- v1 does not implement IP-Adapter or reference conditioning
- anchors are not passed into scene `GenerationRequest.reference_image_path`
- anchors are not used by routing, scoring, scene generation, or selection
- portrait anchors must not be used as scene 1 init images

Scope:
- v1 is run-local only
- no `anchor_artifacts/` sharing/export/reuse yet
- future IP-Adapter work should consume anchor paths through a separate config-gated identity-conditioning path

Traceability note:
- attempted checkpoint was skipped because `.git/refs/heads` was not writable in this environment
- baseline recorded before edits: `main@1e4feca`

Validation:
- targeted command: `conda run -n storygen env PYTHONPATH=src pytest -q tests/test_anchor_bank.py tests/test_pipeline_logging.py tests/test_generators.py tests/test_prompt_pipelines.py`
- targeted result: `25 passed`
- full regression command: `conda run -n storygen env PYTHONPATH=src pytest -q`
- full regression result: `82 passed`

### 2026-04-23: IP-Adapter Identity Conditioning v1

Implemented:
- added config-gated IP-Adapter identity conditioning
- added `llm_prompt_ip_adapter_text2img` and `llm_prompt_hybrid_identity` profiles
- scene requests can receive `reference_image_path` from run-local Anchor Bank outputs
- default anchor type is `half_body`
- default apply mode is `text2img` only
- candidate metadata records anchor character, anchor type, anchor path, adapter model, adapter weight, and scale

Behavior preserved:
- identity conditioning is disabled by default
- baseline, prompt-only, routing-only, and anchor-bank-only profiles preserve existing scene request behavior
- anchor generation requests never receive `reference_image_path`
- img2img scenes do not use IP-Adapter unless `apply_to_modes` explicitly includes `img2img`
- IP-Adapter is a route-conditioned scene generation option inside `diffusers_text2img`, not a new backend

Runtime note:
- first real `llm_prompt_ip_adapter_text2img` run downloaded the SDXL IP-Adapter weights and then failed while loading adapter attention processors
- root cause: base `model.enable_attention_slicing=true` installs sliced attention processors before `load_ip_adapter(...)`; this is incompatible with the current diffusers SDXL IP-Adapter loading path and raised `SlicedAttnProcessor.__init__() missing 1 required positional argument: 'slice_size'`
- fix: IP-Adapter profiles now override `model.enable_attention_slicing: false`
- attention slicing is a memory optimization, not a method component; disabling it is expected to mainly affect VRAM/speed, not the intended ablation semantics

Profile semantics:
- `llm_prompt_ip_adapter_text2img`: LLM prompt + run-local anchor bank + text2img scenes with IP-Adapter identity conditioning
- `llm_prompt_hybrid_identity`: the same identity-conditioning path plus LLM-guided routing; composition-changing scenes use text2img + IP-Adapter, while composition-preserving small changes may use img2img without IP-Adapter by default
- hybrid is therefore an ablation of route execution plus identity conditioning, not a stronger standalone IP-Adapter setting

Traceability note:
- checkpoint was skipped because `.git/refs/heads` was not writable in this environment
- baseline recorded before edits: `main@f2c8654`

Validation:
- targeted command: `conda run -n storygen env PYTHONPATH=src pytest -q tests/test_generators.py tests/test_pipeline_logging.py tests/test_anchor_bank.py tests/test_identity_conditioning.py tests/test_config.py`
- targeted result: `32 passed`
- full regression command: `conda run -n storygen env PYTHONPATH=src pytest -q`
- full regression result: `93 passed`

### 2026-04-23: Multi-Character Identity Conditioning Hardening

Implemented:
- bumped the LLM prompt builder/cache/artifact namespace to `llm_assisted_v7`
- added scene-level `identity_conditioning_subject_id` and `primary_visible_character_ids` metadata
- upgraded anchor selection order to explicit identity subject, continuity subjects, scene entities, then single-character fallback
- made multi-character ambiguous scenes skip or raise instead of silently using single-character fallback

Policy:
- `half_body` is the canonical IP-Adapter anchor type for now
- `portrait` anchors remain inspect-only and are not selected by default identity conditioning
- `llm_prompt_ip_adapter_text2img` remains the preferred current route for identity consistency
- `llm_prompt_hybrid_identity` remains exploratory because previous-frame img2img can propagate wrong objects or composition

Traceability note:
- feature branch/checkpoint creation failed because `.git/refs/heads` could not create nested refs in this environment
- baseline recorded before edits: `main@01dc0b8`

Validation:
- targeted command: `conda run -n storygen env PYTHONPATH=src pytest -q tests/test_llm_assisted_prompt_builder.py tests/test_identity_conditioning.py tests/test_pipeline_logging.py tests/test_anchor_bank.py tests/test_config.py`
- targeted result: `52 passed`
- full regression command: `conda run -n storygen env PYTHONPATH=src pytest -q`
- full regression result: `103 passed`

Follow-up validation note:
- first `test_set/06.txt` v7 artifact export attempt fell back to rule-based because the LLM returned `global.main_character: "human man"` instead of a character id from `global.characters`
- prompt instruction was tightened so `global.main_character` must be an exact `character_id`, or an empty string when there is no single main character

Real-run validation on `test_set/06.txt`:
- `outputs/ablation_06_anchor_bank` loaded `prompt_artifacts/llm_assisted_v7/06_gpt-4o-2024-08-06_v1_c5f38192.json` without fallback
- Anchor Bank generated both `jack` and `sara` anchors, with `portrait.png`, `half_body.png`, and `anchor_spec.json` under each character directory
- LLM v7 metadata set `primary_visible_character_ids: ["jack", "sara"]` and `identity_conditioning_subject_id: null` for all scenes, correctly representing the story as a two-character ambiguous identity-conditioning case
- `outputs/ablation_06_ip_adapter_text2img_skip_ambiguous` used `generation.identity_conditioning.fail_on_missing_anchor=false`; every candidate logged `identity_anchor_missing` with `ambiguous_or_missing_scene_character`
- no `identity_anchor_selected` or `ip_adapter_applied` events were emitted, and scene `reference_image_path` stayed `null`
- interpretation: the multi-character safety path works, but this run is not an IP-Adapter-applied two-character result; it is a safe skip case that avoids incorrectly forcing Jack or Sara as the sole identity anchor
- next optional debug step would be a config-gated forced-subject override for ablation only, not a default method

## Shared Code Rules

## 2026-04-26 Prompt Audit Integration

Integrated the prompt-audit optimization path into `0425_luo` without replacing the story backend path.

Preserved story-level contracts:
- `StoryScenePlan`
- `StoryGenerationRequest.scene_plans`
- `anchor_bank_summary`
- `dual_face_refs`
- `previous_style_reference_path`
- `storydiffusion_direct`
- `diffusers_text2img_consistent`

Prompt updates:
- LLM-assisted builder namespace is now `llm_assisted_v9`
- optimized `generation_prompt` remains inside the existing `PromptSpec`, including `scene_consistency_prompt`
- `PromptBundle.metadata["scene_route_hints"]` carries normalized route hints, including `route_hint_adjustment_reason`
- `PromptBundle.metadata["scene_plans"]` carries audit-oriented scene-plan details for prompt review and export
- `_build_story_scene_plans` consumes optimized `PromptSpec.generation_prompt` and `scene_route_hints` when constructing story backend scene plans

Experiment/audit commands:
- prompt audit export: `PYTHONPATH=src python3 scripts/export_prompts.py --inputs 'test_set/*.txt' --pipelines both --output-dir outputs/prompt_audit`
- remote matrix dry run: `PYTHONPATH=src python3 scripts/run_experiment_matrix.py --experiment-id prompt_audit_check --profiles llm_prompt_text2img --stories 'test_set/*.txt' --dry-run`
- storydiffusion smoke profile: `PYTHONPATH=src python3 -m storygen.cli --profile cloud_storydiffusion_debug --input test_set/01.txt --run-name smoke_storydiffusion_prompt_audit`

When modifying shared code:
- prefer additive and modular changes
- preserve working baseline behavior unless intentionally upgraded
- avoid broad repo refactors unless they remove a concrete blocker
- keep interface contracts explicit
- update docs whenever shared behavior changes
- preserve the current config-driven execution model

When changing shared interfaces:
- document the change here
- keep backward compatibility when practical
- make downstream modules still work before merging related changes

## Coding-Agent Rules

When using coding agents on this repo:
- prioritize modular, documented changes
- prefer config-driven toggles over hard-coded branches
- preserve runnable baseline paths
- avoid silent interface drift
- avoid introducing external API dependencies
- avoid agent-based runtime logic
- keep the pipeline automatic and reproducible

If changing a shared contract:
- update code
- update docs
- verify affected entrypoints still run

## Resource Tracking

This project may use public pretrained models, datasets, and other permissible resources, but they must be trackable for later reporting.

At minimum, collaborators should keep track of:
- pretrained model IDs used in configs
- scoring models used in configs
- any future reference datasets or curated dev resources
- any new external resources that affect training, generation, or evaluation

## Immediate Next Collaboration Goals

Near-term repo goals:
1. keep `COLLAB.md` current as the shared contract
2. preserve stable run entrypoints
3. stabilize the future `SceneSpec` / `AnchorSpec` / `ScoringSpec` direction
4. keep major modules switchable through config
5. maintain a quick dev subset workflow
6. keep outputs and metadata easy to inspect
7. preserve a clean baseline comparison path
