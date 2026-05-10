# Original StoryDiffusion Gradio Probe

This folder is a small adapter for testing the original StoryDiffusion repo at:

```text
/home/xyz/Desktop/xluo/StoryDiffusion
```

It imports `gradio_app_sdxl_specific_id_low_vram.py`, disables `demo.launch(...)`, calls the same `process_generation(...)` function used by the Gradio button, and saves the final yielded images plus a manifest under `outputs/storydiffusion_gradio_probe/`.

## Run

Use your existing `story diffusion` conda environment from this project root:

```bash
conda activate "story diffusion"
python storydiffusion_gradio_probe/run_probe.py \
  --config storydiffusion_gradio_probe/example_config.yaml
```

The original Gradio app loads models during import, so the first run may take a while and will use GPU memory immediately.

## Edit Prompts

For quick experiments, edit `example_config.yaml`:

- `use_reference_images`: set to `true` after filling `reference_images`.
- `reference_images`: one image path per character in the low-VRAM Gradio script.
- `prompts.general_prompt`: character description. In reference-image mode, the original PhotoMaker path requires the trigger word `img`, for example `a woman img`.
- `prompts.prompt_array`: one line per generated frame. The first `generation.id_length` lines are identity-reference prompts used by StoryDiffusion before later frames are generated.
- `generation.id_length`: number of identity prompt lines per character.

You can also override the main prompt from the command line:

```bash
python storydiffusion_gradio_probe/run_probe.py \
  --config storydiffusion_gradio_probe/example_config.yaml \
  --prompt $'[Taylor] wakes up in bed\n[Taylor] eats breakfast\n[Taylor] walks on the road\n[Taylor] works in an office'
```

## Notes

The original Gradio code supports at most two characters in this low-VRAM script and temporarily does not support multi-character panels in reference-image mode. For a no-character scene, prefix a line with `[NC]`, but do not put `[NC]` in the first `id_length` identity lines.

## Batch Test Set

To convert `test_set/*.txt` into Gradio-probe configs without running generation:

```bash
python storydiffusion_gradio_probe/run_test_set.py
```

To use your pipeline's Anchor Bank images as StoryDiffusion ref images (PhotoMaker mode), pass `--anchor-run-dir` pointing at a storygen run folder that contains `logs/anchor_bank.json` (for example `outputs/compare_aesthetic_16_low_ip/`). The probe will look up `canonical_half_body.png` paths from the Anchor Bank summary and use them as ref images.

```bash
python storydiffusion_gradio_probe/run_test_set.py \
  --anchor-run-dir outputs/compare_aesthetic_16_low_ip
```

If the run directory does not have `logs/anchor_bank.json` or the referenced `canonical_half_body.png` files are missing, you can ask the probe to generate them first:

```bash
conda run -n storydiffusion python storydiffusion_gradio_probe/run_test_set.py \
  --anchor-run-dir outputs/compare_aesthetic_16_low_ip \
  --ensure-anchors
```

To generate all matching stories:

```bash
conda activate "story diffusion"
python storydiffusion_gradio_probe/run_test_set.py --run
```

The generated configs are written to `outputs/storydiffusion_gradio_probe/configs/`, and images are saved per story under `outputs/storydiffusion_gradio_probe/test_set/<story_id>/`. For multi-character stories, the script prepends one identity prompt per character for the original StoryDiffusion attention bank and saves only the original story-scene images.
