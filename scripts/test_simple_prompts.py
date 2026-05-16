#!/usr/bin/env python3
"""Quick test: simple prompts vs structural prompts for character consistency."""
import sys, time, json
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from storygen.parser import parse_story_file
from storygen.generators.dit_text2img import DitTextToImageGenerator


def gen_simple(story_file: str, lora_path: str, lora_trigger: str, output_dir: str):
    story = parse_story_file(str(REPO / story_file))
    char = story.all_entities[0] if story.all_entities else "character"
    story_code = Path(story_file).stem

    gen = DitTextToImageGenerator(
        model_config={
            "model_id": "PixArt-alpha/PixArt-XL-2-1024-MS",
            "width": 512, "height": 512,
            "guidance_scale": 2.5, "num_inference_steps": 12,
            "enable_model_cpu_offload": True,
            "lora_path": lora_path,
            "lora_trigger": lora_trigger,
        },
        runtime_config={"device": "cuda", "torch_dtype": "float16"},
    )

    out = Path(output_dir) / f"simple_{story_code}"
    out.mkdir(parents=True, exist_ok=True)

    from storygen.types import PromptSpec, GenerationRequest
    results = {}

    for i, scene in enumerate(story.scenes):
        # Simple: just sks + character + scene text
        simple_prompt = f"{lora_trigger} {char}, {scene.clean_text}"

        spec = PromptSpec(
            scene_id=scene.scene_id, style_prompt="", character_prompt="",
            global_context_prompt="", scene_consistency_prompt="",
            local_prompt="", action_prompt="",
            generation_prompt=simple_prompt,
            scoring_prompt=simple_prompt, full_prompt=simple_prompt,
            negative_prompt="blurry, distorted, low quality, bad anatomy, extra limbs, duplicate subjects",
        )
        req = GenerationRequest(
            scene_id=scene.scene_id, candidate_index=0,
            seed=42 + scene.index * 100,
            prompt_spec=spec, width=512, height=512,
            guidance_scale=2.5, num_inference_steps=12,
        )

        print(f"  [{i+1}/{len(story.scenes)}] {simple_prompt[:90]}...")
        t0 = time.time()
        cand = gen.generate_scene(req)
        elapsed = time.time() - t0

        img_path = out / f"scene_{i+1:03d}.png"
        if cand.image:
            cand.image.save(img_path)
            print(f"    saved {img_path} ({elapsed:.1f}s)")
        results[scene.scene_id] = {"prompt": simple_prompt, "image": str(img_path)}

    (out / "prompts.json").write_text(json.dumps(results, indent=2))
    print(f"  Done → {out}")


if __name__ == "__main__":
    lora_path = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "lora_checkpoints/character/checkpoint-800")
    lora_trigger = sys.argv[2] if len(sys.argv) > 2 else "sks "

    print(f"LoRA: {lora_path}")
    print(f"Trigger: {repr(lora_trigger)}")
    print()

    for story in ["test_set/01.txt", "test_set/02.txt"]:
        print(f"── {story} ──")
        gen_simple(story, lora_path, lora_trigger, str(REPO / "outputs/simple_prompt_test"))
        print()

    print("All done! Compare: outputs/simple_prompt_test/simple_01/ vs simple_02/")
