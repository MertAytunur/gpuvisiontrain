#!/bin/bash
# Setup script for fal Compute GPU instances
# Run this after SSHing into a fal instance

set -e

echo "=========================================="
echo "fal Compute Setup Script"
echo "=========================================="

# Configuration - modify these
GCS_BUCKET="${GCS_BUCKET:-gs://my-vision-datasets}"
DATASET_NAME="${DATASET_NAME:-dataset-v1}"
PROJECT_ID="${PROJECT_ID:-your-project-id}"
KEY_FILE_PATH="${KEY_FILE_PATH:-/home/ubuntu/gcs-sa-key.json}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/data/dataset}"
REPO_DIR="${REPO_DIR:-/home/ubuntu/yolo-training}"

# Check if service account key exists
if [ ! -f "$KEY_FILE_PATH" ]; then
    echo "ERROR: Service account key not found at $KEY_FILE_PATH"
    echo "Please copy your GCS service account key to the instance first:"
    echo "  scp gcs-sa-key.json user@fal-instance:$KEY_FILE_PATH"
    exit 1
fi

# 1. Update system packages
echo ">>> Updating system packages..."
sudo apt-get update -qq

# 2. Install gcloud CLI if not present
if ! command -v gcloud &> /dev/null; then
    echo ">>> Installing Google Cloud SDK..."
    curl -sO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-linux-x86_64.tar.gz
    ./google-cloud-sdk/install.sh -q --path-update true
    source ~/.bashrc
    rm google-cloud-cli-linux-x86_64.tar.gz
fi

# 3. Authenticate with GCS
echo ">>> Authenticating with Google Cloud..."
gcloud auth activate-service-account --key-file="$KEY_FILE_PATH"
gcloud config set project "$PROJECT_ID"
export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE_PATH"

# 4. Create Python environment
echo ">>> Setting up Python environment..."
if command -v conda &> /dev/null; then
    conda create -n yolo python=3.10 -y || true
    conda activate yolo
else
    python3 -m venv venv
    source venv/bin/activate
fi

# 5. Install dependencies
echo ">>> Installing Python dependencies..."
pip install --upgrade pip
pip install ultralytics torch torchvision pyyaml

# 6. Download dataset from GCS to local SSD
echo ">>> Downloading dataset from GCS..."
mkdir -p "$LOCAL_DATA_DIR"
gcloud storage cp -r "$GCS_BUCKET/$DATASET_NAME" "$LOCAL_DATA_DIR/"

# 7. Verify dataset
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
echo "After training, upload results:"
echo "  python sync_data.py upload --provider gcs --local ./outputs/run_name --remote $GCS_BUCKET/outputs/run_name"
