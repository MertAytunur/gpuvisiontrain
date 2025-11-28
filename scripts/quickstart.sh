#!/bin/bash
# Quick Start Script - Run this on any GPU instance
# This script automates the full setup and training process

set -e

echo "=========================================="
echo "YOLO Training Quick Start"
echo "=========================================="

# Configuration - SET THESE BEFORE RUNNING
export GCS_BUCKET="${GCS_BUCKET:-gs://my-vision-datasets}"
export DATASET_NAME="${DATASET_NAME:-dataset-v1}"
export PROJECT_ID="${PROJECT_ID:-your-project-id}"
export KEY_FILE="${KEY_FILE:-./gcs-sa-key.json}"

# Paths
export LOCAL_DATA="/data/dataset"
export WORK_DIR="$(pwd)"

# Training settings
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MODEL="${MODEL:-yolov8n.pt}"
export IMGSZ="${IMGSZ:-640}"

# ============================================
# Step 1: Check prerequisites
# ============================================
echo ">>> Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

# Check service account key
if [ ! -f "$KEY_FILE" ]; then
    echo "WARNING: GCS service account key not found at $KEY_FILE"
    echo "If you're using GCS, please provide the key file"
fi

# ============================================
# Step 2: Install dependencies
# ============================================
echo ">>> Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# ============================================
# Step 3: Verify GPU
# ============================================
echo ">>> Checking GPU status..."
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"         Memory: {mem:.1f} GB")
EOF

# ============================================
# Step 4: Download dataset (if using GCS)
# ============================================
if [ -f "$KEY_FILE" ]; then
    echo ">>> Downloading dataset from GCS..."

    # Authenticate
    export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE"

    python3 sync_data.py download \
        --provider gcs \
        --remote "$GCS_BUCKET/$DATASET_NAME" \
        --local "$LOCAL_DATA/$DATASET_NAME" \
        --key-file "$KEY_FILE" \
        --project "$PROJECT_ID"

    DATA_ROOT="$LOCAL_DATA/$DATASET_NAME"
else
    echo ">>> Skipping GCS download (no key file)"
    DATA_ROOT="${DATA_ROOT:-$LOCAL_DATA/$DATASET_NAME}"
fi

# ============================================
# Step 5: Validate dataset
# ============================================
echo ">>> Validating dataset..."
if [ -f "$DATA_ROOT/data.yaml" ]; then
    echo "Dataset found at: $DATA_ROOT"
    echo "Classes:"
    python3 -c "import yaml; d=yaml.safe_load(open('$DATA_ROOT/data.yaml')); print(f'  Count: {d[\"nc\"]}'); print(f'  Names: {d[\"names\"]}')"
else
    echo "ERROR: data.yaml not found at $DATA_ROOT"
    echo "Please ensure your dataset is in YOLO format"
    exit 1
fi

# ============================================
# Step 6: Start training
# ============================================
echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Model: $MODEL"
echo "Data: $DATA_ROOT"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Image Size: $IMGSZ"
echo "=========================================="
echo ""

# Detect GPUs for device string
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$NUM_GPUS" -gt 1 ]; then
    DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    TOTAL_BATCH=$((BATCH_SIZE * NUM_GPUS))
    echo "Multi-GPU detected: using devices $DEVICES with batch size $TOTAL_BATCH"
else
    DEVICES="0"
    TOTAL_BATCH="$BATCH_SIZE"
fi

RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"

python3 train.py \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$TOTAL_BATCH" \
    --imgsz "$IMGSZ" \
    --device "$DEVICES" \
    --run_name "$RUN_NAME" \
    --amp

# ============================================
# Step 7: Upload results (if using GCS)
# ============================================
if [ -f "$KEY_FILE" ]; then
    echo ">>> Uploading results to GCS..."
    python3 sync_data.py upload \
        --provider gcs \
        --local "./outputs/$RUN_NAME" \
        --remote "$GCS_BUCKET/outputs/$RUN_NAME" \
        --key-file "$KEY_FILE" \
        --project "$PROJECT_ID"
fi

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results: ./outputs/$RUN_NAME"
echo "Best weights: ./outputs/$RUN_NAME/weights/best.pt"
