#!/usr/bin/env bash
set -euo pipefail

GEOEDIT_ROOT="${GEOEDIT_ROOT:-/host/space0/guo-z/GeoEdit}"
GEOEDIT_PYTHON="${GEOEDIT_PYTHON:-/host/space0/guo-z/envs/geoedit/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/host/space0/guo-z/models}"
CASE_DIR="${CASE_DIR:-/host/space0/guo-z/tf-ufi/outputs/foodmanip_bench/image_edit_benchmark_20260807_v1/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop/case_736x592}"
REMOTE_BASE="${REMOTE_BASE:-/host/space0/guo-z/tf-ufi/outputs/foodmanip_bench/image_edit_benchmark_20260807_v1/foodstateedit_spoon_scoop_v0_3_staged/clear_broth_spoon_scoop}"
GPU="${GPU:-6}"
VARIANT="${VARIANT:-staged_rigid12_material8}"
RIGID_TSTRONG="${RIGID_TSTRONG:-12}"
MATERIAL_TSTRONG="${MATERIAL_TSTRONG:-8}"
OUTPUT_DIR="$REMOTE_BASE/$VARIANT"
OUTPUT="$OUTPUT_DIR/result.mp4"
STATUS="$OUTPUT_DIR/status.txt"
RUN_LOG="$OUTPUT_DIR/run.log"

if [[ ! "$VARIANT" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "invalid VARIANT: $VARIANT" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR" "$MODEL_ROOT/cache/huggingface" "$MODEL_ROOT/cache/transformers"
if [[ -s "$OUTPUT" && -s "$OUTPUT_DIR/result.png" ]]; then
    echo "COMPLETE existing_result $(date --iso-8601=seconds)" > "$STATUS"
    exit 0
fi

required=(
  "$CASE_DIR/first_frame.png"
  "$CASE_DIR/motion_signal.png"
  "$CASE_DIR/depth.png"
  "$CASE_DIR/mask_utensil.png"
  "$CASE_DIR/mask_material.png"
  "$CASE_DIR/mask_old.png"
  "$CASE_DIR/prompt.txt"
  "$MODEL_ROOT/PAI/Wan2.2-VACE-Fun-A14B/high_noise_model/diffusion_pytorch_model.safetensors"
  "$MODEL_ROOT/PAI/Wan2.2-VACE-Fun-A14B/low_noise_model/diffusion_pytorch_model.safetensors"
  "$MODEL_ROOT/PAI/Wan2.2-VACE-Fun-A14B/models_t5_umt5-xxl-enc-bf16.pth"
  "$MODEL_ROOT/PAI/Wan2.2-VACE-Fun-A14B/Wan2.1_VAE.pth"
  "$MODEL_ROOT/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json"
  "$MODEL_ROOT/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model"
)
for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then
        echo "FAILED missing=$path $(date --iso-8601=seconds)" > "$STATUS"
        exit 3
    fi
done

export PYTHONPATH="$GEOEDIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HF_HOME="$MODEL_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$MODEL_ROOT/cache/transformers"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "RUNNING $(date --iso-8601=seconds) gpu=$GPU frames=21 steps=20 rigid_tstrong=$RIGID_TSTRONG material_tstrong=$MATERIAL_TSTRONG warm_start=false" > "$STATUS"
cd "$GEOEDIT_ROOT"
if "$GEOEDIT_PYTHON" -m geoedit.inference \
    --input-dir "$CASE_DIR" \
    --motion-signal-mask "$CASE_DIR/mask_utensil.png" \
    --material-mask "$CASE_DIR/mask_material.png" \
    --material-tstrong-index "$MATERIAL_TSTRONG" \
    --output "$OUTPUT" \
    --num-frames 21 \
    --num-inference-steps 20 \
    --tweak-index 3 \
    --tstrong-index "$RIGID_TSTRONG" \
    --replace-mode mask_new \
    --no-warm-start \
    --vram-limit 46 \
    --seed 1 > "$RUN_LOG" 2>&1; then
    if [[ -s "$OUTPUT" && -s "$OUTPUT_DIR/result.png" ]]; then
        echo "COMPLETE $(date --iso-8601=seconds) gpu=$GPU rigid_tstrong=$RIGID_TSTRONG material_tstrong=$MATERIAL_TSTRONG" > "$STATUS"
        exit 0
    fi
    echo "FAILED missing_result $(date --iso-8601=seconds)" > "$STATUS"
    exit 4
else
    rc=$?
    echo "FAILED exit=$rc $(date --iso-8601=seconds)" > "$STATUS"
    exit "$rc"
fi
