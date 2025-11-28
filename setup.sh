#!/bin/bash
# One-command setup script for YOLO training
# Run: chmod +x setup.sh && ./setup.sh

set -e

echo "=========================================="
echo "GPU Vision Train - Setup Script"
echo "=========================================="

# Load .env if exists
if [ -f .env ]; then
    echo ">>> Loading .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check required env vars
if [ -z "$GCS_PROJECT_ID" ] || [ -z "$GCS_CLIENT_EMAIL" ] || [ -z "$GCS_PRIVATE_KEY" ]; then
    echo "ERROR: Missing GCS credentials in .env file"
    echo "Please create .env from .env.example and fill in your credentials"
    exit 1
fi

# Install system dependencies
echo ">>> Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-full python3-venv curl
fi

# Install gcloud CLI if not present
if ! command -v gcloud &> /dev/null; then
    echo ">>> Installing Google Cloud SDK..."
    cd ~
    if [ ! -d "google-cloud-sdk" ]; then
        curl -sO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
        tar -xf google-cloud-cli-linux-x86_64.tar.gz
        rm google-cloud-cli-linux-x86_64.tar.gz
        ./google-cloud-sdk/install.sh -q --path-update true
    fi
    export PATH="$HOME/google-cloud-sdk/bin:$PATH"
    cd - > /dev/null
fi

# Create service account key from env vars
echo ">>> Setting up GCS authentication..."
TEMP_KEY_FILE=$(mktemp /tmp/gcs-key-XXXXXX.json)

cat > "$TEMP_KEY_FILE" << EOF
{
  "type": "${GCS_TYPE:-service_account}",
  "project_id": "${GCS_PROJECT_ID}",
  "private_key_id": "${GCS_PRIVATE_KEY_ID}",
  "private_key": "${GCS_PRIVATE_KEY}",
  "client_email": "${GCS_CLIENT_EMAIL}",
  "client_id": "${GCS_CLIENT_ID}",
  "auth_uri": "${GCS_AUTH_URI:-https://accounts.google.com/o/oauth2/auth}",
  "token_uri": "${GCS_TOKEN_URI:-https://oauth2.googleapis.com/token}",
  "auth_provider_x509_cert_url": "${GCS_AUTH_PROVIDER_CERT_URL:-https://www.googleapis.com/oauth2/v1/certs}",
  "client_x509_cert_url": "${GCS_CLIENT_CERT_URL}",
  "universe_domain": "googleapis.com"
}
EOF

# Fix newlines in private key
sed -i 's/\\n/\n/g' "$TEMP_KEY_FILE"

# Authenticate with gcloud
echo ">>> Authenticating with Google Cloud..."
gcloud auth activate-service-account --key-file="$TEMP_KEY_FILE" 2>/dev/null || \
    ~/google-cloud-sdk/bin/gcloud auth activate-service-account --key-file="$TEMP_KEY_FILE"

gcloud config set project "$GCS_PROJECT_ID" 2>/dev/null || \
    ~/google-cloud-sdk/bin/gcloud config set project "$GCS_PROJECT_ID"

export GOOGLE_APPLICATION_CREDENTIALS="$TEMP_KEY_FILE"

# Setup Python virtual environment
echo ">>> Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install Python dependencies
echo ">>> Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# Create data directory
DATA_DIR="${LOCAL_DATA_DIR:-$HOME/data}"
mkdir -p "$DATA_DIR"

# Download dataset
echo ">>> Downloading dataset from GCS..."
REMOTE_PATH="${GCS_BUCKET}/${DATASET_NAME}"
LOCAL_PATH="${DATA_DIR}/${DATASET_NAME}"

echo "    From: $REMOTE_PATH"
echo "    To:   $LOCAL_PATH"

gcloud storage cp -r "$REMOTE_PATH" "$DATA_DIR/" 2>/dev/null || \
    ~/google-cloud-sdk/bin/gcloud storage cp -r "$REMOTE_PATH" "$DATA_DIR/"

# Verify dataset
if [ -f "$LOCAL_PATH/data.yaml" ]; then
    echo ">>> Dataset verified: data.yaml found"
else
    echo ">>> WARNING: data.yaml not found in dataset"
fi

# Check GPU
echo ">>> Checking GPU status..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not fully installed or no GPU"

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
