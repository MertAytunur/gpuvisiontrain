#!/usr/bin/env python3
"""
YOLO Training Script - Provider Agnostic
Supports: Local, fal Compute, Vast.ai, Azure ML
"""

import argparse
import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLO Training Script for GPU Clusters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to dataset root directory (containing data.yaml)'
    )
    parser.add_argument(
        '--data_yaml',
        type=str,
        default='data.yaml',
        help='Name of the data config file within data_root'
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Model to use (yolov8n/s/m/l/x.pt or path to custom weights)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for training'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Name for this training run (auto-generated if not provided)'
    )

    # Distributed training arguments
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (e.g., 0 or 0,1,2,3 for multi-GPU)'
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed training across multiple nodes'
    )

    # Advanced arguments
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (epochs without improvement)'
    )
    parser.add_argument(
        '--save_period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=True,
        help='Use Automatic Mixed Precision training'
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache images in RAM for faster training'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['SGD', 'Adam', 'AdamW', 'auto'],
        help='Optimizer to use'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='Final learning rate (lr0 * lrf)'
    )

    # Config file (overrides command line args)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file'
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_config_with_args(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """Merge config file settings with command line arguments."""
    for key, value in config.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
        elif hasattr(args, key):
            # Config file values override defaults, but CLI args take precedence
            pass
    return args


def validate_dataset(data_root: str, data_yaml: str) -> Path:
    """Validate dataset structure and return path to data.yaml."""
    data_root_path = Path(data_root)
    data_yaml_path = data_root_path / data_yaml

    if not data_root_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    # Validate YOLO structure
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_name in required_dirs:
        dir_path = data_root_path / dir_name
        if not dir_path.exists():
            logger.warning(f"Expected directory not found: {dir_path}")

    # Load and validate data.yaml
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    if 'nc' not in data_config:
        raise ValueError("data.yaml must contain 'nc' (number of classes)")
    if 'names' not in data_config:
        raise ValueError("data.yaml must contain 'names' (class names)")

    logger.info(f"Dataset validated: {data_config['nc']} classes")
    logger.info(f"Classes: {data_config['names']}")

    return data_yaml_path


def setup_distributed():
    """Setup distributed training environment."""
    import torch.distributed as dist

    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        logger.info(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")

        return rank, world_size, local_rank

    return 0, 1, 0


def train(args: argparse.Namespace):
    """Main training function using Ultralytics YOLO."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Validate dataset
    data_yaml_path = validate_dataset(args.data_root, args.data_yaml)

    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.run_name = f"run_{timestamp}"

    # Setup output directory
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training run: {args.run_name}")
    logger.info(f"Output directory: {output_dir}")

    # Setup distributed if requested
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        args.device = local_rank

    # Load model
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        logger.info(f"Loading model: {args.model}")
        model = YOLO(args.model)

    # Update data.yaml path to be absolute
    data_yaml_abs = str(data_yaml_path.absolute())

    # Training configuration
    train_args = {
        'data': data_yaml_abs,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'patience': args.patience,
        'save_period': args.save_period,
        'project': str(args.output_dir),
        'name': args.run_name,
        'exist_ok': True,
        'amp': args.amp,
        'cache': args.cache,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'verbose': True,
    }

    # Log training configuration
    logger.info("Training configuration:")
    for key, value in train_args.items():
        logger.info(f"  {key}: {value}")

    # Start training
    try:
        results = model.train(**train_args)
        logger.info("Training completed successfully!")

        # Save final model info
        info_path = output_dir / 'training_info.yaml'
        with open(info_path, 'w', encoding='utf-8') as f:
            yaml.dump({
                'run_name': args.run_name,
                'model': args.model,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'imgsz': args.imgsz,
                'data_root': args.data_root,
                'completed': True,
            }, f)

        return results

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main():
    """Main entry point."""
    args = parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(args, config)

    logger.info("=" * 60)
    logger.info("YOLO Training Script")
    logger.info("=" * 60)

    # Run training
    train(args)


if __name__ == '__main__':
    main()
