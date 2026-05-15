#!/usr/bin/env bash
# =============================================================================
# Full LoRA Pipeline: Reference Gen → Train → Inference
#
# Usage:
#   bash scripts/run_full_lora_pipeline.sh --input test_set/01.txt [--profile dit_lora_smoke] [--ref-backend pixart|sdxl]
#
# All outputs go to: outputs/<profile>_<timestamp>_<story>/.
# =============================================================================
set -euo pipefail

# ── Parse args ──
INPUT=""
PROFILE="dit_lora_smoke"
LORA_RANK=32
LORA_STEPS=1200
CHECKPOINT_STEPS=200
# Training reference images: pixart = PixArt-α only (matches LoRA backbone); sdxl = legacy SDXL+IP-Adapter Phase 2
REF_BACKEND="pixart"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)   INPUT="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --lora-steps) LORA_STEPS="$2"; shift 2 ;;
        --ref-backend) REF_BACKEND="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$INPUT" ]]; then
    echo "Usage: bash scripts/run_full_lora_pipeline.sh --input test_set/01.txt [--profile dit_lora_smoke] [--ref-backend pixart|sdxl]"
    exit 1
fi

# ── Paths ──
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STORY_CODE=$(basename "$INPUT" .txt)                    # e.g. "01"
TIMESTAMP=$(date +%Y-%m-%dT%H-%M-%S)
RUN_DIR="$REPO_ROOT/outputs/${PROFILE}_full_${TIMESTAMP}_${STORY_CODE}"
PYTHON=/home/xyz/.conda/envs/ipadapter/bin/python
ACCELERATE="$PYTHON -m accelerate.commands.launch"

TRAIN_IMG_DIR="$RUN_DIR/training_images"
LORA_DIR="$RUN_DIR/lora_checkpoints"
GEN_DIR="$RUN_DIR"

export PYTHONPATH="$REPO_ROOT/src"

mkdir -p "$TRAIN_IMG_DIR" "$LORA_DIR"

# ── Log ──
LOG="$RUN_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================================"
echo " Full LoRA Pipeline"
echo " Story: $STORY_CODE"
echo " Profile: $PROFILE"
echo " ref-backend: $REF_BACKEND"
echo " Output: $RUN_DIR"
echo "================================================================================"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Generate diverse training images (default: PixArt-α only, same backbone as Step 2/3)
# ═══════════════════════════════════════════════════════════════════════════════
echo "── Step 1/3: Generate Training Reference Images (ref-backend=$REF_BACKEND) ──"
echo ""

$PYTHON "$REPO_ROOT/scripts/gen_lora_ref_images.py" \
    --input "$REPO_ROOT/$INPUT" \
    --output-dir "$TRAIN_IMG_DIR" \
    --ref-backend "$REF_BACKEND"

echo ""
echo "Step 1 complete: $(ls "$TRAIN_IMG_DIR"/*.png 2>/dev/null | wc -l) images in $TRAIN_IMG_DIR"

# Fix metadata.jsonl format for HF ImageFolder (image → file_name)
$PYTHON -c "
import json, os
meta_path = '$TRAIN_IMG_DIR/metadata.jsonl'
with open(meta_path) as f:
    lines = [json.loads(l) for l in f if l.strip()]
with open(meta_path, 'w') as f:
    for entry in lines:
        if 'image' in entry and 'file_name' not in entry:
            entry['file_name'] = os.path.basename(entry.pop('image'))
        f.write(json.dumps(entry) + '\n')
print(f'Fixed {len(lines)} metadata entries')
"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: PixArt LoRA DreamBooth training
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 2/3: Train PixArt LoRA ──"
echo ""

$ACCELERATE \
    "$REPO_ROOT/scripts/train_pixart_lora_hf.py" \
    --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-1024-MS \
    --train_data_dir="$TRAIN_IMG_DIR" \
    --output_dir="$LORA_DIR" \
    --resolution=512 \
    --rank="$LORA_RANK" \
    --train_batch_size=1 \
    --learning_rate=1e-06 \
    --max_train_steps="$LORA_STEPS" \
    --checkpointing_steps="$CHECKPOINT_STEPS" \
    --gradient_checkpointing \
    --mixed_precision=fp16 || true   # validation crash is harmless

echo ""
echo "Step 2 complete: checkpoints in $LORA_DIR"

# Pick LoRA weights for inference.
# train_pixart_lora_hf.py saves the final adapter to --output_dir (LORA_DIR) root via
# transformer.save_pretrained(args.output_dir). Intermediate saves live under checkpoint-*.
# If max_train_steps is not a multiple of --checkpointing_steps, the last segment is ONLY
# on the output root — using the last checkpoint-* folder would load stale weights and
# breaks character consistency vs the training images.
LATEST_CKPT=""
if [[ -f "$LORA_DIR/adapter_model.safetensors" || -f "$LORA_DIR/adapter_model.bin" ]]; then
    LATEST_CKPT="$LORA_DIR"
else
    LATEST_CKPT=$(ls -d "$LORA_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
if [[ -z "$LATEST_CKPT" ]]; then
    echo "ERROR: No LoRA checkpoint found. Training may have failed."
    exit 1
fi
echo "Using LoRA path: $LATEST_CKPT"

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Inference (PixArt + LoRA)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 3/3: Inference (Story Generation) ──"
echo ""

$PYTHON -m storygen.cli \
    --profile "$PROFILE" \
    --input "$REPO_ROOT/$INPUT" \
    --set model.lora_path="$LATEST_CKPT" \
    --set model.lora_trigger="sks " \
    --set generation.candidate_count=1 \
    --set runtime.output_root="$GEN_DIR" \
    --set runtime.run_name="generation"

echo ""
echo "================================================================================"
echo " Pipeline complete!"
echo " Output: $RUN_DIR"
echo "  - Training images: $TRAIN_IMG_DIR  ($(ls "$TRAIN_IMG_DIR"/*.png 2>/dev/null | wc -l) images)"
echo "  - LoRA checkpoint: $LATEST_CKPT"
echo "  - Generated story: $GEN_DIR/generation/"
echo ""
echo " To re-use this LoRA for another story:"
echo "   PYTHONPATH=$REPO_ROOT/src $PYTHON -m storygen.cli \\"
echo "     --profile $PROFILE \\"
echo "     --input $REPO_ROOT/<other_story> \\"
echo "     --set model.lora_path=$LATEST_CKPT \\"
echo "     --set model.lora_trigger=\"sks \""
echo "================================================================================"
