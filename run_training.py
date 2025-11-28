#!/usr/bin/env python3
"""
Orchestration Script for YOLO Training on Rented GPUs
Handles: data sync, training, checkpoint management, and result upload.
"""

import argparse
import os
import sys
import subprocess
import yaml
import logging
import signal
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates the full training pipeline."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run_name = config.get('run_name') or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.interrupted = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.warning("Received interrupt signal, finishing current epoch and saving...")
        self.interrupted = True

    def setup_environment(self):
        """Verify environment and dependencies."""
        logger.info("Checking environment...")

        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                logger.info(f"Found {gpu_count} GPU(s): {gpu_names}")
            else:
                logger.warning("CUDA not available, training will be slow!")
        except ImportError:
            logger.error("PyTorch not installed!")
            return False

        # Check ultralytics
        try:
            from ultralytics import YOLO
            logger.info("Ultralytics YOLO available")
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return False

        return True

    def sync_dataset(self) -> Optional[str]:
        """Download dataset from cloud storage."""
        storage_config = self.config.get('storage', {})
        provider = storage_config.get('provider', 'gcs')
        remote_dataset = storage_config.get('remote_dataset')
        local_dataset = storage_config.get('local_dataset', '/data/dataset')

        if not remote_dataset:
            # Assume dataset is already local
            local_path = self.config.get('data_root')
            if local_path and Path(local_path).exists():
                logger.info(f"Using local dataset: {local_path}")
                return local_path
            logger.error("No dataset configured!")
            return None

        logger.info(f"Syncing dataset from {remote_dataset}...")

        sync_cmd = [
            sys.executable, 'sync_data.py',
            'download',
            '--provider', provider,
            '--remote', remote_dataset,
            '--local', local_dataset
        ]

        # Add auth options for GCS
        if provider == 'gcs':
            key_file = storage_config.get('key_file')
            project_id = storage_config.get('project_id')
            if key_file:
                sync_cmd.extend(['--key-file', key_file])
            if project_id:
                sync_cmd.extend(['--project', project_id])

        try:
            subprocess.run(sync_cmd, check=True)
            return local_dataset
        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset sync failed: {e}")
            return None

    def run_training(self, data_root: str) -> bool:
        """Execute the training script."""
        train_config = self.config.get('training', {})

        train_cmd = [
            sys.executable, 'train.py',
            '--data_root', data_root,
            '--run_name', self.run_name,
            '--epochs', str(train_config.get('epochs', 100)),
            '--batch_size', str(train_config.get('batch_size', 16)),
            '--imgsz', str(train_config.get('imgsz', 640)),
            '--device', train_config.get('device', '0'),
            '--workers', str(train_config.get('workers', 8)),
            '--output_dir', train_config.get('output_dir', './outputs'),
        ]

        # Optional arguments
        if train_config.get('model'):
            train_cmd.extend(['--model', train_config['model']])
        if train_config.get('resume'):
            train_cmd.extend(['--resume', train_config['resume']])
        if train_config.get('cache'):
            train_cmd.append('--cache')

        logger.info(f"Starting training: {' '.join(train_cmd)}")

        try:
            subprocess.run(train_cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
        except KeyboardInterrupt:
            logger.warning("Training interrupted")
            return False

    def upload_results(self) -> bool:
        """Upload training results to cloud storage."""
        storage_config = self.config.get('storage', {})
        provider = storage_config.get('provider', 'gcs')
        remote_outputs = storage_config.get('remote_outputs')
        local_outputs = Path(self.config.get('training', {}).get('output_dir', './outputs')) / self.run_name

        if not remote_outputs:
            logger.info("No remote output path configured, skipping upload")
            return True

        if not local_outputs.exists():
            logger.warning(f"Output directory not found: {local_outputs}")
            return False

        logger.info(f"Uploading results to {remote_outputs}/{self.run_name}...")

        sync_cmd = [
            sys.executable, 'sync_data.py',
            'upload',
            '--provider', provider,
            '--local', str(local_outputs),
            '--remote', f"{remote_outputs}/{self.run_name}"
        ]

        # Add auth options for GCS
        if provider == 'gcs':
            key_file = storage_config.get('key_file')
            project_id = storage_config.get('project_id')
            if key_file:
                sync_cmd.extend(['--key-file', key_file])
            if project_id:
                sync_cmd.extend(['--project', project_id])

        try:
            subprocess.run(sync_cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Upload failed: {e}")
            return False

    def run(self) -> bool:
        """Execute the full training pipeline."""
        logger.info("=" * 60)
        logger.info(f"Starting training run: {self.run_name}")
        logger.info("=" * 60)

        # 1. Setup environment
        if not self.setup_environment():
            return False

        # 2. Sync dataset
        data_root = self.sync_dataset()
        if not data_root:
            return False

        # 3. Run training
        success = self.run_training(data_root)

        # 4. Upload results (even if training was interrupted)
        self.upload_results()

        if success:
            logger.info("=" * 60)
            logger.info("Training completed successfully!")
            logger.info("=" * 60)
        else:
            logger.warning("Training did not complete successfully")

        return success


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Orchestrate YOLO training on rented GPUs')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Override run name'
    )

    args = parser.parse_args()

    # Load config
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        logger.info("Create a config.yaml file or use train.py directly with CLI arguments")
        sys.exit(1)

    config = load_config(args.config)

    if args.run_name:
        config['run_name'] = args.run_name

    # Run orchestrator
    orchestrator = TrainingOrchestrator(config)
    success = orchestrator.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
