# Vision Training Infrastructure Plan

## Overview

This project trains custom vision models (YOLO format) using cloud GPUs and object storage.

Current stack:

- Code & issues: GitHub
- Dataset storage (~50 GB): Google Cloud Storage (GCS)
- Compute for training: fal Compute and/or Vast.ai (GPU servers via SSH)
- Optional later: Azure ML GPU clusters (using startup credits)

Goal: training code is provider‑agnostic; only setup scripts differ per platform.

---

## 1. Storage Strategy

### 1.1 GitHub

- Use GitHub only for code, configs, and small sample images.
- Do **not** commit full datasets or large binaries to the repo (GitHub has repo and file size limits and is not designed for 50 GB+ datasets). [web:31][web:33]

### 1.2 Google Cloud Storage (GCS)

- Create one bucket for all datasets and outputs, e.g. `gs://my-vision-datasets`.
- Store in GCS:
  - Raw and preprocessed datasets (images, labels)
  - Model checkpoints and final weights
  - Logs/metrics exports

Example one‑time setup (local machine):

Create bucket
gcloud storage buckets create gs://my-vision-datasets
--location=EUROPE-WEST1

Upload dataset
gcloud storage cp -r /path/to/local_dataset
gs://my-vision-datasets/dataset-v1

text

GCS is built for large datasets (tens–hundreds of TB) and charges per GB per month, which fits the ~50 GB dataset size well. [web:23][web:26]

---

## 2. Authentication (Headless)

All training environments are headless (SSH only). Authentication to GCS uses a **service account**.

### 2.1 Service account setup (local, once)

1. In Google Cloud Console, create a **service account** in the project.
2. Grant roles for GCS access (for example `Storage Object Admin` on the dataset bucket). [web:78][web:73]
3. Create a **JSON key** and download it (e.g. `gcs-sa-key.json`). [web:76]

### 2.2 Using the service account on GPU servers

Copy the JSON key to the server, then:

gcloud auth activate-service-account
--key-file=/home/ubuntu/gcs-sa-key.json

gcloud config set project YOUR_PROJECT_ID
export GOOGLE_APPLICATION_CREDENTIALS=/home/ubuntu/gcs-sa-key.json

text

After this, `gcloud storage` and GCS client libraries can access the bucket headlessly. [web:72][web:78]

---

## 3. Data Format (YOLO)

The dataset uses standard YOLO format:

### 3.1 Directory layout

dataset-v1/
images/
train/
val/
test/ # optional
labels/
train/
val/
test/ # optional
data.yaml

text

- Images live in `images/{train,val,test}/` (e.g. `.jpg`, `.png`).
- Labels live in `labels/{train,val,test}/`.
- Filenames match: `images/train/000123.jpg` ↔ `labels/train/000123.txt`.

### 3.2 Label files

- Each `.txt` label file: one line per object:
class_id x_center y_center width height

text
- Coordinates are normalized to \([0, 1]\) relative to image width/height.
- `class_id` is zero‑based (0 … nc‑1).

### 3.3 data.yaml

Example:

path: dataset-v1
train: images/train
val: images/val
test: images/test # optional

nc: 12 # number of classes
names:

class0

class1

class2

...
text

When synced to GCS, keep this structure:

gs://my-vision-datasets/dataset-v1/...

text

---

## 4. Training Environments

### 4.1 Local (RTX 3060)

- Use for debugging data loading, augmentation, and configs.
- Use a **small subset** of the dataset locally.
- Example:

python train.py --data_root /datasets/sample-dataset-v1

text

### 4.2 fal Compute

Purpose: higher‑end GPUs (e.g. H100/A100) for faster runs.

Steps:

1. In fal dashboard, create a **Compute instance** with 1× GPU.
2. SSH into the instance.
3. Install `gcloud` CLI and authenticate with the service account.
4. **Copy** dataset from GCS to local SSD once at the start (do not stream every batch from GCS).
5. Run training pointing to the local path.
6. Copy outputs (checkpoints, logs) back to GCS.

Example setup script:

1) Install gcloud CLI (simplified)
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh -q

2) Auth (service account key already copied to /home/ubuntu)
./google-cloud-sdk/bin/gcloud auth activate-service-account
--key-file=/home/ubuntu/gcs-sa-key.json
./google-cloud-sdk/bin/gcloud config set project YOUR_PROJECT_ID

3) Download dataset to local SSD
mkdir -p /data/dataset
./google-cloud-sdk/bin/gcloud storage cp -r
gs://my-vision-datasets/dataset-v1
/data/dataset

4) Create/activate environment and train
cd /home/ubuntu/your_repo

conda activate vision_env # or source venv, etc.
python train.py --data_root /data/dataset/dataset-v1

5) Sync outputs back to GCS
./google-cloud-sdk/bin/gcloud storage cp -r
/data/outputs/run-001
gs://my-vision-datasets/outputs/run-001

text

Copying once to SSD avoids slow, repeated network reads; training then runs at local disk speed. [web:23][web:78]

### 4.3 Vast.ai (optional)

- Same GCS usage as fal: authenticate, copy dataset once, train from local disk.
- Vast‑specific features like Cloud Sync can be used later if needed. [web:49][web:59]

---

## 5. Azure (Future Option with Startup Credits)

Optional path using Azure credits:

- Store a copy of the dataset in **Azure Blob Storage**.
- Use **Azure ML compute clusters** with GPU VMs (e.g. A10/T4/H100) as training targets. [web:81][web:85]
- Reuse the same training code; only data paths and setup scripts change.

---

## 6. Code & Config Conventions

- Training script always accepts a generic data root:

python train.py --data_root /data/dataset/dataset-v1

text

- Provider differences (local, fal, Vast, Azure) are isolated in:
  - Environment setup scripts
  - Data sync scripts (GCS/Blob)

- Checkpoint often and always sync important artifacts to GCS or Blob before shutting down GPU instances.

---

## 7. Initial Milestones

1. **M0 – Dataset on GCS**
   - `dataset-v1` uploaded to `gs://my-vision-datasets`.
   - YOLO structure validated.

2. **M1 – Local test**
   - Subset training run completes on RTX 3060.

3. **M2 – Single‑GPU fal run**
   - End‑to‑end: GCS → fal SSD → train → results back to GCS.

4. **M3 – Optional Vast/Azure run**
   - Same repo and config run successfully on another provider.