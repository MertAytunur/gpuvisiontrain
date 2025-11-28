#!/bin/bash
# Setup script for Vast.ai GPU instances
# Run this after SSHing into a Vast.ai instance

set -e

echo "=========================================="
echo "Vast.ai Setup Script"
echo "=========================================="

# Configuration - modify these
GCS_BUCKET="${GCS_BUCKET:-gs://my-vision-datasets}"
DATASET_NAME="${DATASET_NAME:-dataset-v1}"
PROJECT_ID="${PROJECT_ID:-your-project-id}"
KEY_FILE_PATH="${KEY_FILE_PATH:-/root/gcs-sa-key.json}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/workspace/data}"
REPO_DIR="${REPO_DIR:-/workspace/yolo-training}"

# Vast.ai instances typically use /workspace as the main directory
cd /workspace || cd ~

# Check if service account key exists
if [ ! -f "$KEY_FILE_PATH" ]; then
    echo "ERROR: Service account key not found at $KEY_FILE_PATH"
    echo "Please copy your GCS service account key to the instance first:"
    echo "  scp gcs-sa-key.json root@vast-instance:$KEY_FILE_PATH"
    exit 1
fi

# 1. Update system packages
echo ">>> Updating system packages..."
apt-get update -qq

# 2. Install dependencies
echo ">>> Installing system dependencies..."
apt-get install -y curl git python3-pip python3-venv

# 3. Install gcloud CLI if not present
if ! command -v gcloud &> /dev/null; then
    echo ">>> Installing Google Cloud SDK..."
    curl -sO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-linux-x86_64.tar.gz
    ./google-cloud-sdk/install.sh -q --path-update true
    export PATH="$PATH:/workspace/google-cloud-sdk/bin"
    rm google-cloud-cli-linux-x86_64.tar.gz
fi

# 4. Authenticate with GCS
echo ">>> Authenticating with Google Cloud..."
gcloud auth activate-service-account --key-file="$KEY_FILE_PATH"
gcloud config set project "$PROJECT_ID"
export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE_PATH"

# 5. Setup Python environment
echo ">>> Setting up Python environment..."
python3 -m pip install --upgrade pip
pip install ultralytics torch torchvision pyyaml

# 6. Clone or update training repo (if using git)
if [ -n "$GITHUB_REPO" ]; then
    echo ">>> Cloning training repository..."
    git clone "$GITHUB_REPO" "$REPO_DIR" || (cd "$REPO_DIR" && git pull)
fi

# 7. Download dataset from GCS to local storage
echo ">>> Downloading dataset from GCS..."
mkdir -p "$LOCAL_DATA_DIR"
gcloud storage cp -r "$GCS_BUCKET/$DATASET_NAME" "$LOCAL_DATA_DIR/"

# 8. Verify GPU availability
echo ">>> Checking GPU status..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 9. Verify dataset
echo ">>> Verifying dataset structure..."
if [ -f "$LOCAL_DATA_DIR/$DATASET_NAME/data.yaml" ]; then
    echo "Dataset verified: data.yaml found"
    cat "$LOCAL_DATA_DIR/$DATASET_NAME/data.yaml"
else
    echo "WARNING: data.yaml not found in dataset"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Dataset location: $LOCAL_DATA_DIR/$DATASET_NAME"
echo ""
echo "To start training, run:"
echo "  python train.py --data_root $LOCAL_DATA_DIR/$DATASET_NAME --epochs 100 --batch_size 16"
echo ""
echo "For multi-GPU training:"
echo "  python train.py --data_root $LOCAL_DATA_DIR/$DATASET_NAME --device 0,1,2,3 --batch_size 64"
echo ""
echo "After training, upload results:"
echo "  python sync_data.py upload --provider gcs --local ./outputs/run_name --remote $GCS_BUCKET/outputs/run_name"
