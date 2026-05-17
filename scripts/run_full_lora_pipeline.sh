#!/usr/bin/env bash
# =============================================================================
# Full LoRA Pipeline
#
# Default (pixart):
#   canonical + 3 augmentations → PixArt LoRA train → inference
#
# Experiment (sdxl_ipadapter): SDXL canonical + IP-Adapter story scenes → PixArt LoRA train
#   bash scripts/run_full_lora_pipeline.sh --input test_set/02.txt --ref-backend sdxl_ipadapter
#
# Usage:
#   bash scripts/run_full_lora_pipeline.sh --input test_set/02.txt
# =============================================================================
set -euo pipefail

INPUT=""
PROFILE="dit_lora_smoke"
LORA_RANK=32
T5_LORA_RANK=4
LORA_STEPS_PHASE1=500
LORA_STEPS_TOTAL=800
CHECKPOINT_STEPS=200
LEARNING_RATE="1e-6"
RESOLUTION=768
REF_BACKEND="pixart"
PROMPT_MODE="simple"
BASE_SEED=42
AUG_COUNT=3
DIVERSE_COUNT=4
ENABLE_EXPAND=0
EXPAND_MODE="ipadapter"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)              INPUT="$2"; shift 2 ;;
        --profile)            PROFILE="$2"; shift 2 ;;
        --lora-rank)          LORA_RANK="$2"; shift 2 ;;
        --t5-lora-rank)       T5_LORA_RANK="$2"; shift 2 ;;
        --lora-steps-phase1)  LORA_STEPS_PHASE1="$2"; shift 2 ;;
        --lora-steps-total)   LORA_STEPS_TOTAL="$2"; shift 2 ;;
        --lora-steps)         LORA_STEPS_TOTAL="$2"; shift 2 ;;
        --learning-rate)      LEARNING_RATE="$2"; shift 2 ;;
        --resolution)         RESOLUTION="$2"; shift 2 ;;
        --ref-backend)        REF_BACKEND="$2"; shift 2 ;;
        --prompt-mode)        PROMPT_MODE="$2"; shift 2 ;;
        --base-seed)          BASE_SEED="$2"; shift 2 ;;
        --aug-count)          AUG_COUNT="$2"; shift 2 ;;
        --diverse-count)      DIVERSE_COUNT="$2"; shift 2 ;;
        --expand)             ENABLE_EXPAND=1; shift ;;
        --expand-mode)        EXPAND_MODE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$INPUT" ]]; then
    echo "Usage: bash scripts/run_full_lora_pipeline.sh --input test_set/01.txt [--expand] [--aug-count 3]"
    exit 1
fi

if [[ "$REF_BACKEND" != "pixart" && "$REF_BACKEND" != "sdxl" && "$REF_BACKEND" != "sdxl_ipadapter" ]]; then
    echo "ERROR: --ref-backend must be pixart, sdxl, or sdxl_ipadapter (got: $REF_BACKEND)"
    exit 1
fi
if [[ "$PROMPT_MODE" != "simple" && "$PROMPT_MODE" != "full" ]]; then
    echo "ERROR: --prompt-mode must be simple or full (got: $PROMPT_MODE)"
    exit 1
fi
if [[ "$EXPAND_MODE" != "ipadapter" && "$EXPAND_MODE" != "lora" ]]; then
    echo "ERROR: --expand-mode must be ipadapter or lora (got: $EXPAND_MODE)"
    exit 1
fi
if [[ "$ENABLE_EXPAND" == "1" && "$REF_BACKEND" == "pixart" ]]; then
    echo "ERROR: --expand with --ref-backend pixart is not supported."
    echo "  SDXL/IP-Adapter training images do not match PixArt LoRA (domain gap)."
    echo "  Use default bootstrap only (canonical + augmentations), or provide your own"
    echo "  reference photos under training_images/ before training."
    echo "  For SDXL+IP-Adapter diverse refs, use: --ref-backend sdxl (legacy; inference still PixArt unless you change profile)."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STORY_CODE=$(basename "$INPUT" .txt)
TIMESTAMP=$(date +%Y-%m-%dT%H-%M-%S)
RUN_DIR="$REPO_ROOT/outputs/${PROFILE}_full_${TIMESTAMP}_${STORY_CODE}"
PYTHON=/home/xyz/.conda/envs/ipadapter/bin/python
ACCELERATE="$PYTHON -m accelerate.commands.launch"

TRAIN_IMG_DIR="$RUN_DIR/training_images"
LORA_DIR="$RUN_DIR/lora_checkpoints"
GEN_DIR="$RUN_DIR"

export PYTHONPATH="$REPO_ROOT/src"

mkdir -p "$TRAIN_IMG_DIR" "$LORA_DIR"

LOG="$RUN_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================================"
echo " Full LoRA Pipeline"
echo " Story: $STORY_CODE"
echo " Profile: $PROFILE  ref-backend: $REF_BACKEND  prompt-mode: $PROMPT_MODE"
if [[ "$REF_BACKEND" == "sdxl" || "$REF_BACKEND" == "sdxl_ipadapter" ]]; then
    echo " NOTE: SDXL training refs → PixArt LoRA (cross-model experiment; identity may differ at infer)."
fi
echo " resolution: ${RESOLUTION}²  aug_count: $AUG_COUNT  diverse_count: $DIVERSE_COUNT  expand: $ENABLE_EXPAND"
if [[ "$ENABLE_EXPAND" == "1" ]]; then
    echo " expand_mode: $EXPAND_MODE  train steps: phase1=$LORA_STEPS_PHASE1 total=$LORA_STEPS_TOTAL"
else
    echo " train steps: $LORA_STEPS_TOTAL (single phase)"
fi
echo " base_seed: $BASE_SEED  shared_scene_seed: true"
echo " Output: $RUN_DIR"
echo "================================================================================"
echo ""

fix_metadata() {
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
print(f'Metadata: {len(lines)} entries')
"
}

pick_lora_path() {
    if [[ -f "$LORA_DIR/adapter_model.safetensors" || -f "$LORA_DIR/adapter_model.bin" ]]; then
        echo "$LORA_DIR"
    else
        ls -d "$LORA_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1
    fi
}

train_lora() {
    local extra_args=("$@")
    $ACCELERATE \
        "$REPO_ROOT/scripts/train_pixart_lora_hf.py" \
        --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-1024-MS \
        --train_data_dir="$TRAIN_IMG_DIR" \
        --output_dir="$LORA_DIR" \
        --resolution="$RESOLUTION" \
        --rank="$LORA_RANK" \
        --t5_lora_rank="$T5_LORA_RANK" \
        --train_batch_size=1 \
        --learning_rate="$LEARNING_RATE" \
        --checkpointing_steps="$CHECKPOINT_STEPS" \
        --gradient_checkpointing \
        --mixed_precision=fp16 \
        --train_text_encoder \
        "${extra_args[@]}" || true
}

# ── Step 1: Training reference images ──
if [[ "$REF_BACKEND" == "sdxl" || "$REF_BACKEND" == "sdxl_ipadapter" ]]; then
    echo "── Step 1: SDXL canonical + IP-Adapter (${DIVERSE_COUNT} story scenes, photoreal) ──"
else
    echo "── Step 1: PixArt bootstrap (canonical + ${AUG_COUNT} augmentations) ──"
fi
$PYTHON "$REPO_ROOT/scripts/gen_lora_ref_images.py" \
    --input "$REPO_ROOT/$INPUT" \
    --output-dir "$TRAIN_IMG_DIR" \
    --ref-backend "$REF_BACKEND" \
    --phase bootstrap \
    --resolution "$RESOLUTION" \
    --seed "$BASE_SEED" \
    --aug-count "$AUG_COUNT" \
    --diverse-count "$DIVERSE_COUNT"
fix_metadata
echo "Step 1 complete: $(ls "$TRAIN_IMG_DIR"/*.png 2>/dev/null | wc -l) images"
echo ""

if [[ "$ENABLE_EXPAND" == "1" && "$REF_BACKEND" == "pixart" ]]; then
    echo "── Step 2a: LoRA train phase 1 ($LORA_STEPS_PHASE1 steps) ──"
    train_lora --max_train_steps="$LORA_STEPS_PHASE1"
    PHASE1_LORA=$(pick_lora_path)
    if [[ -z "$PHASE1_LORA" ]]; then
        echo "ERROR: No LoRA checkpoint after phase 1"
        exit 1
    fi
    echo "Phase 1 LoRA: $PHASE1_LORA"
    echo ""

    echo "── Step 2b: Expand training set (mode=$EXPAND_MODE) ──"
    EXPAND_ARGS=(
        --input "$REPO_ROOT/$INPUT"
        --output-dir "$TRAIN_IMG_DIR"
        --phase expand
        --resolution "$RESOLUTION"
        --seed "$BASE_SEED"
        --expand-mode "$EXPAND_MODE"
    )
    if [[ "$EXPAND_MODE" == "lora" ]]; then
        EXPAND_ARGS+=(--lora-path "$PHASE1_LORA")
    fi
    $PYTHON "$REPO_ROOT/scripts/gen_lora_ref_images.py" "${EXPAND_ARGS[@]}"
    fix_metadata
    echo ""

    echo "── Step 3: LoRA train phase 2 (resume → $LORA_STEPS_TOTAL steps) ──"
    train_lora --resume_from_checkpoint=latest --max_train_steps="$LORA_STEPS_TOTAL"
else
    echo "── Step 2: LoRA train ($LORA_STEPS_TOTAL steps, bootstrap set only) ──"
    train_lora --max_train_steps="$LORA_STEPS_TOTAL"
fi

LATEST_CKPT=$(pick_lora_path)
if [[ -z "$LATEST_CKPT" ]]; then
    echo "ERROR: No LoRA checkpoint found"
    exit 1
fi
echo "LoRA checkpoint: $LATEST_CKPT"
echo ""

echo "── Final: Story inference (shared seed=$BASE_SEED) ──"
$PYTHON -m storygen.cli \
    --profile "$PROFILE" \
    --input "$REPO_ROOT/$INPUT" \
    --set model.lora_path="$LATEST_CKPT" \
    --set model.lora_trigger="sks " \
    --set model.prompt_mode="$PROMPT_MODE" \
    --set generation.candidate_count=1 \
    --set generation.base_seed="$BASE_SEED" \
    --set generation.shared_scene_seed=true \
    --set runtime.output_root="$GEN_DIR" \
    --set runtime.run_name="generation"

echo ""
echo "================================================================================"
echo " Pipeline complete!  Output: $RUN_DIR"
echo "================================================================================"
