# GPU Vision Train

Train YOLO models on rented GPU clusters (fal Compute, Vast.ai, Azure ML) with Google Cloud Storage integration.

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/MertAytunur/gpuvisiontrain.git
cd gpuvisiontrain

# 2. Copy and fill .env
cp .env.example .env
# Edit .env with your credentials

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training
python train.py --data_root /path/to/dataset
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Setup GCS Credentials](#setup-gcs-credentials)
4. [Configure Environment](#configure-environment)
5. [Prepare Dataset](#prepare-dataset)
6. [Upload Dataset to GCS](#upload-dataset-to-gcs)
7. [Training on GPU Providers](#training-on-gpu-providers)
8. [Multi-GPU Training](#multi-gpu-training)
9. [Commands Reference](#commands-reference)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA
- Google Cloud account
- GPU provider account (fal, Vast.ai, etc.)

---

## Project Structure

```
gpuvisiontrain/
├── train.py              # Main training script
├── sync_data.py          # GCS/Azure sync utilities
├── run_training.py       # Pipeline orchestrator
├── config.yaml           # Config template
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
└── scripts/
    ├── setup_fal.sh      # fal Compute setup
    ├── setup_vastai.sh   # Vast.ai setup
    ├── distributed_train.sh
    └── quickstart.sh
```

---

## Setup GCS Credentials

### Step 1: Create Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** > **Service Accounts**
3. Click **Create Service Account**
4. Name: `vision-training-sa`
5. Grant role: **Storage Object Admin**
6. Click **Done**

### Step 2: Generate JSON Key

1. Click on your service account
2. Go to **Keys** tab
3. Click **Add Key** > **Create new key**
4. Select **JSON**
5. Download the file (e.g., `storage-479605-xxxxx.json`)

### Step 3: Create GCS Bucket

```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud storage buckets create gs://your-bucket-name --location=EUROPE-WEST1
```

---

## Configure Environment

### Step 1: Copy Template

```bash
cp .env.example .env
```

### Step 2: Fill in Your Credentials

Open `.env` and fill in values from your JSON key file:

```bash
# From your JSON key file
GCS_TYPE=service_account
GCS_PROJECT_ID=storage-479605
GCS_PRIVATE_KEY_ID=2606526bf2a9369fb63d62bd911115d7e0981d90
GCS_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nMIIEvQI...(your key)...\n-----END PRIVATE KEY-----\n"
GCS_CLIENT_EMAIL=mertaytunur@storage-479605.iam.gserviceaccount.com
GCS_CLIENT_ID=113737909009229371293
GCS_AUTH_URI=https://accounts.google.com/o/oauth2/auth
GCS_TOKEN_URI=https://oauth2.googleapis.com/token
GCS_AUTH_PROVIDER_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
GCS_CLIENT_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/your-service-account

# Your bucket
GCS_BUCKET=gs://your-bucket-name

# Dataset
DATASET_NAME=dataset-v1
LOCAL_DATA_DIR=/data/dataset

# Training
MODEL=yolov8n.pt
EPOCHS=100
BATCH_SIZE=16
IMGSZ=640
DEVICE=0
```

> **Security**: Never commit `.env` to git. Only `.env.example` should be committed.

---

## Prepare Dataset

### YOLO Format Structure

```
dataset-v1/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── ...
│   └── val/
│       └── ...
└── data.yaml
```

### Label Format

Each `.txt` file (one per image):

```
class_id x_center y_center width height
```

- `class_id`: 0-indexed class number
- Coordinates: normalized [0, 1]

Example `img001.txt`:
```
0 0.5 0.5 0.2 0.3
1 0.25 0.75 0.1 0.15
```

### data.yaml

```yaml
path: dataset-v1
train: images/train
val: images/val

nc: 3
names:
  - cat
  - dog
  - bird
```

---

## Upload Dataset to GCS

```bash
# Upload
gcloud storage cp -r /path/to/dataset-v1 gs://your-bucket/dataset-v1

# Verify
gcloud storage ls gs://your-bucket/dataset-v1/
```

---

## Training on GPU Providers

### fal Compute

```bash
# 1. Create instance at fal.ai dashboard

# 2. SSH into instance
ssh user@fal-instance

# 3. Clone repo
git clone https://github.com/MertAytunur/gpuvisiontrain.git
cd gpuvisiontrain

# 4. Create .env with your credentials
nano .env  # paste your credentials

# 5. Install and run
pip install -r requirements.txt
python sync_data.py download
python train.py
```

### Vast.ai

```bash
# 1. Rent GPU at vast.ai

# 2. SSH into instance
ssh root@vast-instance

# 3. Setup
git clone https://github.com/MertAytunur/gpuvisiontrain.git
cd gpuvisiontrain
nano .env  # paste your credentials

# 4. Run
pip install -r requirements.txt
./scripts/quickstart.sh
```

### Local Machine

```bash
pip install -r requirements.txt
python train.py --data_root /path/to/local/dataset
```

---

## Multi-GPU Training

### Automatic Detection

```bash
# Uses all available GPUs
./scripts/distributed_train.sh
```

### Manual Selection

```bash
# Single GPU
python train.py --device 0

# Multiple GPUs
python train.py --device 0,1,2,3 --batch_size 64
```

### Environment Variables

```bash
export DEVICE=0,1,2,3
export BATCH_SIZE=64
python train.py
```

---

## Commands Reference

### Data Sync

```bash
# Download dataset from GCS
python sync_data.py download

# With explicit paths
python sync_data.py download --remote gs://bucket/dataset --local /data/dataset

# Upload results
python sync_data.py upload --local ./outputs/run_001 --remote gs://bucket/outputs/run_001
```

### Training

```bash
# Basic
python train.py --data_root /data/dataset/dataset-v1

# Full options
python train.py \
    --data_root /data/dataset/dataset-v1 \
    --model yolov8m.pt \
    --epochs 100 \
    --batch_size 32 \
    --imgsz 640 \
    --device 0,1 \
    --amp
```

### Full Pipeline

```bash
# Config-based
python run_training.py --config config.yaml

# One-command setup
./scripts/quickstart.sh
```

### Model Variants

| Model | Params | Speed | Accuracy |
|-------|--------|-------|----------|
| yolov8n.pt | 3.2M | Fastest | Base |
| yolov8s.pt | 11.2M | Fast | Better |
| yolov8m.pt | 25.9M | Medium | Good |
| yolov8l.pt | 43.7M | Slow | High |
| yolov8x.pt | 68.2M | Slowest | Best |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 8

# Use smaller model
python train.py --model yolov8n.pt
```

### GCS Auth Failed

```bash
# Verify .env is loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ.get('GCS_PROJECT_ID'))"

# Check credentials
python sync_data.py download --remote gs://your-bucket/test
```

### Dataset Not Found

```bash
# Verify structure
ls -la /data/dataset/dataset-v1/
cat /data/dataset/dataset-v1/data.yaml
```

### Resume Training

```bash
python train.py --resume ./outputs/run_name/weights/last.pt
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GCS_PROJECT_ID` | GCP project ID | `storage-479605` |
| `GCS_PRIVATE_KEY` | Service account private key | `-----BEGIN...` |
| `GCS_CLIENT_EMAIL` | Service account email | `sa@project.iam...` |
| `GCS_BUCKET` | GCS bucket URL | `gs://my-bucket` |
| `DATASET_NAME` | Dataset folder name | `dataset-v1` |
| `LOCAL_DATA_DIR` | Local data directory | `/data/dataset` |
| `MODEL` | YOLO model variant | `yolov8n.pt` |
| `EPOCHS` | Training epochs | `100` |
| `BATCH_SIZE` | Batch size per GPU | `16` |
| `IMGSZ` | Image size | `640` |
| `DEVICE` | GPU device(s) | `0` or `0,1,2,3` |

---

## License

MIT
