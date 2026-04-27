# Presentation Pipeline Deep Dive

Working baseline for this note:
- Branch: `integrate/prompt-audit-into-storydiffusion`
- Commit inspected: `c517128`
- Working tree status when this note was written: existing uncommitted changes in `README.md`, `COLLAB.md`, `configs/base.yaml`, `src/storygen/llm_assisted_prompt_builder.py`, `tests/test_config.py`, and `tests/test_llm_assisted_prompt_builder.py`; this document was added without modifying those files.
- Purpose: build a clear 5-minute presentation story around the full pipeline, the LLM decision points, the generation/identity changes, the StoryDiffusion-style integration, and known failures.

## Short Answer

Yes, it is worth showing failures. For this project, the strongest presentation is not "we built one final model"; it is "we identified concrete failure modes, converted them into explicit pipeline contracts, and improved reproducibility and consistency step by step."

Recommended 5-minute arc:

1. Start with the task difficulty: each panel must match the text, while characters and story state must stay consistent across panels.
2. Show one baseline failure: independent text2img gives weak character consistency, and weak prompts can even change a human into a pet-like subject.
3. Introduce the modular pipeline: parser -> prompt planning -> route/identity planning -> generation -> scoring/logging.
4. Explain the main design: LLM plans structured metadata; local code validates and makes final execution decisions.
5. Show the generation evolution: text2img baseline -> LLM prompts -> img2img routing -> Anchor Bank -> IP-Adapter -> story-level `storydiffusion_direct`.
6. End with rigor: ablations, logs, failure analysis, and the honest limitation that single-anchor IP-Adapter does not solve multi-character scenes.

Do not spend the whole talk walking linearly through every component. Use the failure to motivate why each component exists.

## What The Current Pipeline Does

The codebase has two runnable generation shapes.

The scene-level pipeline uses `model.backend=diffusers_text2img` and `model.granularity=scene`. It iterates over scenes, generates multiple candidates per scene, scores them with CLIP-based text/action/previous-image consistency, and saves the selected frame. This is the older main experimental route used by profiles such as `llm_prompt_text2img`, `llm_prompt_img2img_guided`, `llm_prompt_ip_adapter_text2img`, and `llm_prompt_hybrid_identity`.

The story-level pipeline uses `model.backend=storydiffusion_direct` and `model.granularity=story`. It builds a full list of `StoryScenePlan` objects first, writes `story_scene_plans.json` and `story_backend_request.json`, then delegates each panel to the scene-level diffusers generator. This is the current `cloud_storydiffusion_debug` route. It is "StoryDiffusion-style" at the orchestration/interface level, but it is not yet a full paper-faithful StoryDiffusion implementation because consistent attention is disabled by default and the backend still delegates to ordinary scene generation.

Key source files:
- `src/storygen/pipeline.py`: end-to-end orchestration.
- `src/storygen/llm_assisted_prompt_builder.py`: LLM structured prompt planning, validation, route hints, identity subject planning.
- `src/storygen/routing.py`: final local route decision for text2img/img2img.
- `src/storygen/anchor_bank.py`: run-local identity anchor generation and canonical half-body selection.
- `src/storygen/identity_conditioning.py`: decides whether a scene can receive an IP-Adapter reference image.
- `src/storygen/generators/diffusers_text2img.py`: text2img, img2img, and IP-Adapter execution.
- `src/storygen/generators/storydiffusion_direct.py`: story-level backend that consumes `StoryScenePlan` and delegates panel rendering.
- `src/storygen/generators/consistent_attention.py`: experimental StoryDiffusion-style consistent attention processor, not the default active path.

## What The LLM Decides

The LLM does not generate images and does not directly decide the final backend call. It returns strict JSON for local code to validate and normalize.

Global LLM outputs:
- `global.main_character`: single main character when applicable.
- `global.identity_cues`: reusable visual identity cues.
- `global.shared_setting`: recurring setting information.
- `global.style_cues`: extra style details not already in local config.
- `global.characters`: stable `CharacterSpec` blocks with age, gender presentation, hair, skin tone, body build, outfit, accessories, and profession marker.

Per-scene LLM outputs:
- `generation_prompt`: short phrase used for image generation after local assembly.
- `scoring_prompt`: short phrase for CLIP text-image scoring.
- `action_prompt`: action-only phrase for action-sensitive scoring.
- `continuity_subject_ids`: which recurring subjects should stay continuous.
- `continuity_route_hint`: `text2img` or `img2img`, but only as a hint.
- `route_change_level`: `small`, `medium`, or `large`.
- `route_factors`: booleans for same subject, same setting, body state change, primary action change, new key objects, and whether composition change is needed.
- `primary_visible_character_ids`: foreground recurring characters in the panel.
- `identity_conditioning_subject_id`: one character whose anchor may condition the scene, or `null` when no single clear target exists.
- `interaction_summary`, `spatial_relation`, `framing`, `setting_focus`: especially important for two-character scenes.

Local validation then:
- Rejects wrong scene ids or missing fields.
- Trims overlong prompts.
- Removes command-style prompt openings like "show" or "generate".
- Removes question framing from scoring prompts.
- Adds human labels and pet-suppression negative terms when needed.
- Adjusts route level when the LLM's own route factors contradict the label.
- Forces dual-primary scenes to use `identity_conditioning_subject_id=null` instead of guessing one person.

Presentation phrasing:

"The LLM is a structured scene planner, not an image generator and not the final authority. It predicts identity, scene focus, and route metadata; our deterministic code validates the JSON and decides what execution path is safe."

## What The Pipeline Decides From LLM Results

Routing:
- Scene 1 always uses text2img.
- If img2img is disabled, all scenes use text2img.
- In conservative routing, a later scene can use previous selected image as img2img init only when continuity is safe.
- In LLM-guided routing, route hints and route factors decide whether a scene is small/medium/large.
- If `composition_change_needed=true`, the guided route can force text2img, because previous-frame img2img tends to preserve the old layout too much.

Identity conditioning:
- Anchor Bank generates portrait and half-body identity references from `CharacterSpec`.
- The canonical half-body anchor is the default IP-Adapter reference.
- IP-Adapter is currently applied only to modes listed in `generation.identity_conditioning.apply_to_modes`, normally `text2img`.
- The selector first tries `identity_conditioning_subject_id`, then continuity subjects, then scene entities, then single-character fallback.
- If multiple characters are equally important and no single identity target exists, it skips IP-Adapter when `fail_on_missing_anchor=false`.

Scoring and selection in scene-level runs:
- Generates `candidate_count` images per scene.
- Scores text alignment, action alignment, and previous-image consistency.
- Route-aware scoring can reduce previous-image consistency for medium/large changes and penalize over-similarity.

Story-level `cloud_storydiffusion_debug`:
- Builds `story_scene_plans.json` first.
- Generates Anchor Bank.
- Builds `StoryGenerationRequest`.
- Calls `storydiffusion_direct`.
- Current backend generates one panel per scene, so it does not use the old multi-candidate CLIP reranking loop.

## Old Pipeline vs Current `cloud_storydiffusion_debug`

Older identity-focused command:

```bash
PYTHONPATH=src python3 -m storygen.cli \
  --profile llm_prompt_ip_adapter_text2img \
  --input test_set/01.txt
```

What it isolates:
- Scene-level backend: `diffusers_text2img`.
- LLM-assisted prompt planning.
- Anchor Bank enabled.
- IP-Adapter identity conditioning enabled.
- Routing disabled, so every scene is text2img.
- `candidate_count=3`, followed by CLIP-based reranking.
- For a single clear character, every scene can use the same half-body anchor.

Hybrid command:

```bash
PYTHONPATH=src python3 -m storygen.cli \
  --profile llm_prompt_hybrid_identity \
  --input test_set/01.txt
```

What it changes:
- Adds LLM-guided img2img routing.
- IP-Adapter still applies only to text2img scenes.
- If a later scene is routed to img2img, the log shows `generation_mode_not_enabled:img2img` for identity conditioning. This is why hybrid can be less stable than pure `text2img + IP-Adapter`.

Current StoryDiffusion-style command:

```bash
PYTHONPATH=src python -m storygen.cli \
  --profile cloud_storydiffusion_debug \
  --input test_set/14.txt \
  --run-name compare_14_with_scene_consistency_v4 \
  --set generation.identity_conditioning.scale=0.3 \
  --set model.consistent_attention.enabled=false
```

What it means:
- Story-level backend: `storydiffusion_direct`.
- LLM-assisted prompt planning.
- Anchor Bank enabled.
- Identity conditioning enabled.
- IP-Adapter scale lowered to `0.3`, which is a tradeoff: less identity lock, more composition freedom.
- Consistent attention explicitly disabled.
- It preserves story-level plans and identity plans, but rendering is still delegated to scene-level diffusers.

Important distinction for the presentation:

Do not claim that the current command fully implements StoryDiffusion's consistent self-attention. A technically sound phrasing is:

"We integrated a story-level StoryDiffusion-style backend interface. It consumes the whole story plan and identity anchors, and it gives us the right place to add consistent attention. In the current verified route, consistent attention is off, so the main consistency mechanism is still Anchor Bank plus IP-Adapter."

## Git Iteration Story

The commit history supports a strong engineering narrative:

- `30b75f3` added the LLM-based prompt pipeline with OpenAI structured planning, cache/artifacts, tests, and docs.
- `4482961` added initial img2img routing, route decisions, and generation metadata.
- `104a910` added LLM-guided route factors and route-aware scoring.
- `34d0d17` checkpointed composition-aware routing, motivated by img2img under-change.
- `1e4feca` added `CharacterSpec` extraction metadata.
- `f2c8654` added run-local Anchor Bank generation.
- `6e4c756` added IP-Adapter identity conditioning.
- `01dc0b8` disabled attention slicing for IP-Adapter profiles because sliced attention processors conflicted with adapter loading.
- `93d3178` hardened multi-character anchor selection.
- `b6a8722` added canonical half-body selection for IP-Adapter.
- `d0701e3` added conservative two-character scene planning policy.
- `ec4384a` integrated prompt audit with the StoryDiffusion path.
- `6a7c600` / `e47fd07` cleaned up anchor prompts to use single-subject identity prompts.
- `c517128` is the inspected checkpoint before changing prompt-cache defaults.

This is a good "design evolution" story:

1. Prompt failure showed text alone was too weak.
2. Img2img improved continuity but could over-preserve composition.
3. Route metadata made the choice explicit and inspectable.
4. Text2img was needed for composition-changing scenes.
5. Anchor/IP-Adapter added visual identity conditioning for text2img.
6. Story-level backend unified prompt plans, anchors, and future consistent attention.

## Failure Cases Worth Showing

Show 2 failures maximum in a 5-minute talk.

Failure 1: independent text2img / weak prompt identity
- Symptom: same character changes across panels, or weak human prompt causes pet/animal substitution.
- Why it matters: baseline text prompts are not enough for identity.
- Fix: LLM prompt planner adds stable identity cues; local prompt assembler merges character identity into `generation_prompt`; negative prompt suppresses pet substitution for human characters.

Failure 2: img2img under-change
- Symptom: later scenes preserve previous composition too strongly, so a new action or layout is not shown.
- Why it matters: visual continuity and story correctness conflict.
- Fix: LLM route factors detect body/action/composition changes; route-aware scoring reduces over-similarity; composition-changing scenes are routed back to text2img.

Failure 3, optional if discussing future work: multi-character identity
- Example: `outputs/ablation_06_ip_adapter_text2img_skip_ambiguous`.
- Jack and Sara both appear in every scene.
- The LLM metadata sets `identity_conditioning_subject_id=null` and `primary_visible_character_ids=["jack", "sara"]`.
- The identity selector logs `ambiguous_or_missing_scene_character`, so IP-Adapter is skipped.
- This is a safe limitation, not a hidden success: single-anchor IP-Adapter v1 should not force Jack's anchor or Sara's anchor onto a two-person panel.

For your stated concern, yes: include one bad baseline example of poor character consistency if you have a visually clear image. It makes the need for Anchor Bank/IP-Adapter obvious and helps the audience understand the design pressure quickly.

## Single-Character vs Multi-Character Behavior

Single-character scene:
- The LLM can safely set `identity_conditioning_subject_id` to that character.
- Anchor Bank can generate one canonical half-body anchor.
- IP-Adapter can condition each text2img scene on that anchor.
- This is currently the strongest identity path.

Multi-character scene:
- If both people are equally important, the LLM should set `identity_conditioning_subject_id=null`.
- The current selector skips IP-Adapter rather than guessing.
- Text2img then relies on prompt text and, in story-level runs, any available story-plan/style/reference metadata.
- Expected failure: weaker identity consistency across panels.
- Future work: multi-anchor conditioning, regional conditioning, or verified consistent attention.

Presentation phrasing:

"Our system chooses safety over a false improvement: for two-character scenes, single-anchor IP-Adapter is skipped unless one character is clearly the identity target. That prevents contaminating both faces with one anchor, but it leaves multi-character consistency as future work."

## StoryDiffusion Positioning

External reference: StoryDiffusion proposes consistent self-attention for long-range image/video generation and reports improved consistency across story/comic panels. Project page: https://storydiffusion.github.io/ and paper page: https://huggingface.co/papers/2405.01434.

In this repo:
- `storydiffusion_direct` is a story-level backend interface.
- It consumes full-story scene plans and anchor metadata.
- It delegates each panel to `DiffusersTextToImageGenerator`.
- `consistent_attention.py` contains an experimental StoryDiffusion-style attention processor with an identity bank, but it is not the verified active path when `model.consistent_attention.enabled=false`.

So the honest technical comparison is:
- StoryDiffusion paper idea: share attention/identity information across generated panels.
- Current project implementation: build the story-level interface and identity-anchor path first; consistent attention remains planned/experimental.
- Main currently demonstrated consistency mechanism: Anchor Bank + IP-Adapter.

External reference: IP-Adapter is a lightweight image-prompt adapter for text-to-image diffusion, using image conditioning compatible with text prompts. Project page: https://ip-adapter.github.io/ and model card: https://huggingface.co/h94/IP-Adapter.

## Suggested 5-Minute Slide Plan

Slide 1, 30 seconds: task and baseline failure
- Show input story -> baseline panels.
- State the core problem: panel correctness vs cross-panel identity consistency.
- Point to one obvious failure.

Slide 2, 50 seconds: pipeline overview
- Diagram: text -> parser -> LLM structured planner -> local validation -> route/identity planner -> generation -> scoring/logs.
- Emphasize that LLM is not the final authority.

Slide 3, 60 seconds: prompt and route planning
- Show the fields the LLM predicts: character specs, route factors, visible characters, identity subject.
- Explain how local routing chooses text2img vs img2img.
- Mention composition-change forcing text2img.

Slide 4, 70 seconds: identity consistency design
- Show Anchor Bank -> canonical half-body -> IP-Adapter.
- Compare text2img only vs text2img + IP-Adapter if you have images.
- Mention scale tradeoff: high scale locks identity but can narrow composition; lower scale preserves scene freedom.

Slide 5, 50 seconds: StoryDiffusion-style integration
- Explain `cloud_storydiffusion_debug`: story-level plans, anchor metadata, one backend entrypoint.
- Be precise: current verified run has consistent attention off; this is an integration point, not a full StoryDiffusion claim.

Slide 6, 40 seconds: rigor, limitations, future work
- Ablations: rule vs LLM prompts, text2img vs img2img, IP-Adapter on/off, single-character vs multi-character skip.
- Limitation: multi-character identity remains unsolved with single-anchor IP-Adapter.
- Future: consistent attention or multi-anchor/regional conditioning.

## Mapping To Rubric

Technical Soundness:
- Structured contracts: `PromptSpec`, `CharacterSpec`, `StoryScenePlan`, `GenerationRequest`.
- Local validation prevents malformed LLM output from silently controlling execution.
- Routing and identity decisions are logged in `events.jsonl`.
- Story-level and scene-level backends are separated.

Novelty and Performance:
- LLM is used as a planner for route/identity metadata, not just prompt rewriting.
- Anchor Bank plus canonical half-body selection provides run-local identity assets.
- IP-Adapter gives visual conditioning for single-character text2img scenes.
- Story-level backend prepares for consistent attention integration.

Completeness and Rigor:
- Multiple ablation profiles exist in `configs/base.yaml`.
- Existing outputs record run summaries, resolved configs, prompt bundles, story scene plans, anchors, and events.
- Known failures are documented instead of hidden.
- Tests cover config, LLM prompt builder, routing, scoring, anchor bank, identity conditioning, and pipeline logging.

Presentation Clarity:
- Tell the story through one failure and one design diagram.
- Use one visual comparison for identity improvement.
- Use one limitation slide to show honesty and technical maturity.

## Best Final Narrative

"We started from a deterministic multi-panel text-to-image baseline. The main failure was that each panel could be locally plausible but globally inconsistent. We first improved text understanding with an LLM, but constrained it to structured JSON so the system stayed reproducible. Then we added route planning because previous-frame img2img helps small continuity changes but hurts large composition changes. Finally, we added Anchor Bank and IP-Adapter so composition-changing text2img scenes could still preserve a single character's identity. The current StoryDiffusion-style backend integrates these plans at the whole-story level and creates a clean interface for future consistent attention. The current limitation is multi-character identity: we intentionally skip single-anchor IP-Adapter when two characters are equally important, because guessing would be technically unsound."

