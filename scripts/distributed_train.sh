#!/bin/bash
# Distributed training launcher for multi-GPU and multi-node setups
# Usage: ./distributed_train.sh [config.yaml]

set -e

CONFIG_FILE="${1:-config.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/dataset/dataset-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
RUN_NAME="${RUN_NAME:-distributed_$(date +%Y%m%d_%H%M%S)}"

# Detect number of GPUs
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPU(s)"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected!"
    exit 1
fi

# Build device string (0,1,2,3 for 4 GPUs)
DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
echo "Using devices: $DEVICES"

# Calculate batch size (scale with GPU count)
BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
TOTAL_BATCH_SIZE=$((BASE_BATCH_SIZE * NUM_GPUS))
echo "Total batch size: $TOTAL_BATCH_SIZE ($BASE_BATCH_SIZE per GPU)"

# Training parameters
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
MODEL="${MODEL:-yolov8n.pt}"

echo ""
echo "=========================================="
echo "Starting Distributed Training"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL"
echo "Data: $DATA_ROOT"
echo "Epochs: $EPOCHS"
echo "Image Size: $IMGSZ"
echo "Batch Size: $TOTAL_BATCH_SIZE"
echo "Output: $OUTPUT_DIR/$RUN_NAME"
echo "=========================================="
echo ""

# Run training with multi-GPU support
python3 train.py \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$TOTAL_BATCH_SIZE" \
    --imgsz "$IMGSZ" \
    --device "$DEVICES" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --workers $((NUM_GPUS * 4)) \
    --amp

echo ""
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR/$RUN_NAME"
