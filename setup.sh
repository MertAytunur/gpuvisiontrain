#!/bin/bash
# One-command setup script for YOLO training
# Run: chmod +x setup.sh && ./setup.sh

set -e

echo "=========================================="
echo "GPU Vision Train - Setup Script"
echo "=========================================="

# Configuration
GCS_BUCKET="${GCS_BUCKET:-gs://trainingdatamert}"
DATASET_NAME="${DATASET_NAME:-BumperTraining}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$HOME/data}"

# Install system dependencies
echo ">>> Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-full python3-venv curl
fi

# Setup Python virtual environment first (needed for dotenv)
echo ">>> Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install Python dependencies
echo ">>> Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Use Python to load .env and create credentials file
echo ">>> Creating GCS credentials from .env..."
python3 << 'PYEOF'
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Build credentials dict
credentials = {
    "type": os.environ.get('GCS_TYPE', 'service_account'),
    "project_id": os.environ.get('GCS_PROJECT_ID'),
    "private_key_id": os.environ.get('GCS_PRIVATE_KEY_ID', ''),
    "private_key": os.environ.get('GCS_PRIVATE_KEY', '').replace('\\n', '\n'),
    "client_email": os.environ.get('GCS_CLIENT_EMAIL'),
    "client_id": os.environ.get('GCS_CLIENT_ID', ''),
    "auth_uri": os.environ.get('GCS_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
    "token_uri": os.environ.get('GCS_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
    "auth_provider_x509_cert_url": os.environ.get('GCS_AUTH_PROVIDER_CERT_URL', 'https://www.googleapis.com/oauth2/v1/certs'),
    "client_x509_cert_url": os.environ.get('GCS_CLIENT_CERT_URL', ''),
    "universe_domain": "googleapis.com"
}

if not credentials['project_id'] or not credentials['client_email']:
    print("ERROR: Missing GCS credentials in .env file")
    exit(1)

# Write to temp file
key_path = '/tmp/gcs-credentials.json'
with open(key_path, 'w') as f:
    json.dump(credentials, f, indent=2)

print(f"Credentials saved to {key_path}")
print(f"Project: {credentials['project_id']}")
PYEOF

# Install gcloud CLI if not present
GCLOUD_CMD=""
if command -v gcloud &> /dev/null; then
    GCLOUD_CMD="gcloud"
elif [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD_CMD="$HOME/google-cloud-sdk/bin/gcloud"
else
    echo ">>> Installing Google Cloud SDK..."
    cd ~
    curl -sO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
    tar -xf google-cloud-cli-linux-x86_64.tar.gz
    rm google-cloud-cli-linux-x86_64.tar.gz
    ./google-cloud-sdk/install.sh -q --path-update false
    GCLOUD_CMD="$HOME/google-cloud-sdk/bin/gcloud"
    cd - > /dev/null
fi

# Get project ID from credentials
GCS_PROJECT_ID=$(python3 -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ.get('GCS_PROJECT_ID', ''))")

# Authenticate with gcloud
echo ">>> Authenticating with Google Cloud..."
$GCLOUD_CMD auth activate-service-account --key-file="/tmp/gcs-credentials.json"
$GCLOUD_CMD config set project "$GCS_PROJECT_ID"
export GOOGLE_APPLICATION_CREDENTIALS="/tmp/gcs-credentials.json"

# Create data directory
mkdir -p "$LOCAL_DATA_DIR"

# Download dataset
echo ">>> Downloading dataset from GCS..."
REMOTE_PATH="${GCS_BUCKET}/${DATASET_NAME}"
LOCAL_PATH="${LOCAL_DATA_DIR}/${DATASET_NAME}"

echo "    From: $REMOTE_PATH"
echo "    To:   $LOCAL_PATH"

if [ -d "$LOCAL_PATH" ]; then
    echo "    Dataset already exists, skipping download"
else
    $GCLOUD_CMD storage cp -r "$REMOTE_PATH" "$LOCAL_DATA_DIR/"
fi

# Verify dataset
if [ -f "$LOCAL_PATH/data.yaml" ]; then
    echo ">>> Dataset verified: data.yaml found"
else
    echo ">>> WARNING: data.yaml not found in dataset"
fi

# Check GPU
echo ">>> Checking GPU status..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "    PyTorch not fully installed or no GPU detected"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Dataset location: $LOCAL_PATH"
echo ""
echo "To start training, run:"
echo "  source venv/bin/activate"
echo "  python3 train.py --data_root $LOCAL_PATH"
echo ""
