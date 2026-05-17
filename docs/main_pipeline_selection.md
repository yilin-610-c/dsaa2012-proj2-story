# Main Pipeline Selection

The intended final routing policy is simple:

- Single recurring visual entity: use the scene-level Anchor Bank + IP-Adapter path, normally `cloud_anchor_ipadapter_scene`.
- Two or more recurring visual entities: use the native StoryDiffusion path with `llm_direct` prompts and front-loaded identity reference rows.
- DiT/PixArt: optional comparison backend, not the default production path.

## Anchor/IP-Adapter

Use `prompt.pipeline=llm_direct` with `prompt.llm_direct.targets=["anchor","storydiffusion"]`. The LLM writes final executable prompts; the adapter only copies fields into `PromptSpec`, metadata, and Anchor Bank inputs.

Non-human subjects use config-driven identity-conditioning scale overrides:

```yaml
generation:
  identity_conditioning:
    scale: 0.6
    scale_by_subject_type:
      animal: 0.1
      robot: 0.1
      object: 0.1
      vehicle: 0.1
```

This keeps IP-Adapter from over-constraining animal/object/robot pose while preserving broad identity cues.

## Native StoryDiffusion

Use `--storydiffusion-prompt-mode llm_direct` for prompt config generation. StoryDiffusion consumes the LLM payload as:

```text
prompt_array = identity_reference_prompts + storydiffusion_prompt list
```

The native runner owns the image-generation behavior; the prompt pipeline only prepares validated prompt payloads and debug metadata.

## DiT/PixArt

`dit_text2img` and `dit_story_joint` are available for optional experiments. They are explicit profiles and should not replace the main Anchor/IP-Adapter or native StoryDiffusion routes without a separate comparison.
