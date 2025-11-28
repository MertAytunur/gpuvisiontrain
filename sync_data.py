#!/usr/bin/env python3
"""
Data Sync Utilities for GCS/Azure Blob Storage
Handles downloading datasets and uploading training outputs.
"""

import argparse
import os
import sys
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StorageProvider:
    """Base class for storage providers."""

    def download(self, remote_path: str, local_path: str, parallel: bool = True) -> bool:
        raise NotImplementedError

    def upload(self, local_path: str, remote_path: str, parallel: bool = True) -> bool:
        raise NotImplementedError

    def list_files(self, remote_path: str) -> List[str]:
        raise NotImplementedError


class GCSProvider(StorageProvider):
    """Google Cloud Storage provider."""

    def __init__(self, project_id: Optional[str] = None, key_file: Optional[str] = None):
        self.project_id = project_id
        self.key_file = key_file
        self._setup_auth()

    def _setup_auth(self):
        """Setup GCS authentication."""
        if self.key_file:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.key_file
            logger.info(f"Using service account key: {self.key_file}")

            # Activate service account if gcloud is available
            try:
                subprocess.run(
                    ['gcloud', 'auth', 'activate-service-account',
                     f'--key-file={self.key_file}'],
                    check=True,
                    capture_output=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("gcloud CLI not available or auth failed, using SDK auth")

        if self.project_id:
            os.environ['GCLOUD_PROJECT'] = self.project_id

    def _run_gsutil(self, args: List[str], parallel: bool = True) -> subprocess.CompletedProcess:
        """Run gsutil command with optional parallel transfer."""
        cmd = ['gcloud', 'storage']
        if parallel:
            # Use parallel composite uploads/downloads
            cmd.extend(['--verbosity', 'warning'])
        cmd.extend(args)

        logger.info(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd, check=True, capture_output=True, text=True)

    def download(self, remote_path: str, local_path: str, parallel: bool = True) -> bool:
        """Download from GCS to local path."""
        try:
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)

            # Use recursive copy
            self._run_gsutil(['cp', '-r', remote_path, local_path], parallel)
            logger.info(f"Downloaded {remote_path} to {local_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e.stderr}")
            return False

    def upload(self, local_path: str, remote_path: str, parallel: bool = True) -> bool:
        """Upload from local path to GCS."""
        try:
            self._run_gsutil(['cp', '-r', local_path, remote_path], parallel)
            logger.info(f"Uploaded {local_path} to {remote_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Upload failed: {e.stderr}")
            return False

    def list_files(self, remote_path: str) -> List[str]:
        """List files in GCS path."""
        try:
            result = self._run_gsutil(['ls', '-r', remote_path], parallel=False)
            return result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            return []


class AzureBlobProvider(StorageProvider):
    """Azure Blob Storage provider."""

    def __init__(self, connection_string: Optional[str] = None, account_name: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        self.account_name = account_name

    def _run_azcopy(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run azcopy command."""
        cmd = ['azcopy'] + args
        logger.info(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd, check=True, capture_output=True, text=True)

    def download(self, remote_path: str, local_path: str, parallel: bool = True) -> bool:
        """Download from Azure Blob to local path."""
        try:
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)

            self._run_azcopy(['copy', remote_path, local_path, '--recursive'])
            logger.info(f"Downloaded {remote_path} to {local_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e.stderr}")
            return False

    def upload(self, local_path: str, remote_path: str, parallel: bool = True) -> bool:
        """Upload from local path to Azure Blob."""
        try:
            self._run_azcopy(['copy', local_path, remote_path, '--recursive'])
            logger.info(f"Uploaded {local_path} to {remote_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Upload failed: {e.stderr}")
            return False

    def list_files(self, remote_path: str) -> List[str]:
        """List files in Azure Blob path."""
        try:
            result = self._run_azcopy(['list', remote_path])
            return result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            return []


def get_provider(provider_type: str, **kwargs) -> StorageProvider:
    """Factory function to get storage provider."""
    providers = {
        'gcs': GCSProvider,
        'azure': AzureBlobProvider,
    }

    if provider_type not in providers:
        raise ValueError(f"Unknown provider: {provider_type}. Available: {list(providers.keys())}")

    return providers[provider_type](**kwargs)


def sync_dataset(
    provider: StorageProvider,
    remote_dataset: str,
    local_dataset: str,
    force: bool = False
) -> str:
    """
    Sync dataset from cloud storage to local disk.
    Returns the local path to the dataset.
    """
    local_path = Path(local_dataset)

    # Check if already exists
    if local_path.exists() and not force:
        logger.info(f"Dataset already exists at {local_path}, skipping download")
        return str(local_path)

    # Create parent directory
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    logger.info(f"Downloading dataset from {remote_dataset}...")
    success = provider.download(remote_dataset, str(local_path))

    if not success:
        raise RuntimeError(f"Failed to download dataset from {remote_dataset}")

    logger.info(f"Dataset ready at {local_path}")
    return str(local_path)


def sync_outputs(
    provider: StorageProvider,
    local_outputs: str,
    remote_outputs: str
) -> bool:
    """
    Sync training outputs from local disk to cloud storage.
    """
    local_path = Path(local_outputs)

    if not local_path.exists():
        logger.error(f"Local outputs not found: {local_path}")
        return False

    logger.info(f"Uploading outputs to {remote_outputs}...")
    return provider.upload(str(local_path), remote_outputs)


def main():
    """CLI for data sync operations."""
    parser = argparse.ArgumentParser(description='Sync data between cloud storage and local disk')

    parser.add_argument(
        'action',
        choices=['download', 'upload', 'sync'],
        help='Action to perform'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='gcs',
        choices=['gcs', 'azure'],
        help='Cloud storage provider'
    )
    parser.add_argument(
        '--remote',
        type=str,
        required=True,
        help='Remote path (e.g., gs://bucket/path or https://account.blob.core.windows.net/container/path)'
    )
    parser.add_argument(
        '--local',
        type=str,
        required=True,
        help='Local path'
    )
    parser.add_argument(
        '--key-file',
        type=str,
        default=None,
        help='Path to service account key file (GCS)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='GCP project ID'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if local path exists'
    )

    args = parser.parse_args()

    # Setup provider
    provider_kwargs = {}
    if args.provider == 'gcs':
        provider_kwargs = {
            'project_id': args.project,
            'key_file': args.key_file,
        }

    provider = get_provider(args.provider, **provider_kwargs)

    # Execute action
    if args.action == 'download':
        sync_dataset(provider, args.remote, args.local, args.force)
    elif args.action == 'upload':
        sync_outputs(provider, args.local, args.remote)
    elif args.action == 'sync':
        # Bidirectional sync (download first, then can upload after training)
        sync_dataset(provider, args.remote, args.local, args.force)

    logger.info("Sync completed successfully!")


if __name__ == '__main__':
    main()
