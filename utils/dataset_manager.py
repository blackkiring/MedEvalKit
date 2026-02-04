"""
Unified Dataset Manager for MedEvalKit
Provides centralized management of dataset storage, download, and migration.
"""

import os
import json
import shutil
import glob as glob_module
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class DatasetType(Enum):
    """Dataset source type"""
    HF_ONLY = "hf_only"           # Pure HuggingFace dataset (no local images)
    HF_WITH_IMAGES = "hf_images"  # HF metadata + separate image download
    URL_DOWNLOAD = "url_download" # Direct URL download
    MANUAL = "manual"             # Requires manual download (e.g., MIMIC_CXR)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str
    type: DatasetType
    hf_path: Optional[str] = None
    download_url: Optional[str] = None
    default_local_path: Optional[str] = None
    requires_extraction: bool = False
    archive_filename: Optional[str] = None
    notes: Optional[str] = None


# Registry of all supported datasets
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    # HuggingFace-only datasets (text or images embedded in HF)
    "MMMU-Medical-test": DatasetConfig(
        name="MMMU-Medical-test",
        type=DatasetType.HF_ONLY,
        hf_path="MMMU/MMMU"
    ),
    "MMMU-Medical-val": DatasetConfig(
        name="MMMU-Medical-val",
        type=DatasetType.HF_ONLY,
        hf_path="MMMU/MMMU"
    ),
    "VQA_RAD": DatasetConfig(
        name="VQA_RAD",
        type=DatasetType.HF_ONLY,
        hf_path="flaviagiammarino/vqa-rad"
    ),
    "PATH_VQA": DatasetConfig(
        name="PATH_VQA",
        type=DatasetType.HF_ONLY,
        hf_path="flaviagiammarino/path-vqa"
    ),
    "MMLU": DatasetConfig(
        name="MMLU",
        type=DatasetType.HF_ONLY,
        hf_path="cais/mmlu"
    ),
    "PubMedQA": DatasetConfig(
        name="PubMedQA",
        type=DatasetType.HF_ONLY,
        hf_path="openlifescienceai/pubmedqa"
    ),
    "MedMCQA": DatasetConfig(
        name="MedMCQA",
        type=DatasetType.HF_ONLY,
        hf_path="openlifescienceai/medmcqa"
    ),
    "MedQA_USMLE": DatasetConfig(
        name="MedQA_USMLE",
        type=DatasetType.HF_ONLY,
        hf_path="GBaker/MedQA-USMLE-4-options"
    ),
    "Medbullets_op4": DatasetConfig(
        name="Medbullets_op4",
        type=DatasetType.HF_ONLY,
        hf_path="tuenguyen/Medical-Eval-MedBullets_op4"
    ),
    "Medbullets_op5": DatasetConfig(
        name="Medbullets_op5",
        type=DatasetType.HF_ONLY,
        hf_path="LangAGI-Lab/medbullets_op5"
    ),
    "SuperGPQA": DatasetConfig(
        name="SuperGPQA",
        type=DatasetType.HF_ONLY,
        hf_path="m-a-p/SuperGPQA"
    ),

    # HuggingFace with separate image download
    "SLAKE": DatasetConfig(
        name="SLAKE",
        type=DatasetType.HF_WITH_IMAGES,
        hf_path="BoKelvin/SLAKE",
        default_local_path="./datas/SLAKE",
        requires_extraction=True,
        archive_filename="imgs.zip"
    ),
    "PMC_VQA": DatasetConfig(
        name="PMC_VQA",
        type=DatasetType.HF_WITH_IMAGES,
        hf_path="RadGenome/PMC-VQA",
        default_local_path="./datas/PMC-VQA",
        requires_extraction=True,
        archive_filename="images.zip"
    ),
    "MedXpertQA-MM": DatasetConfig(
        name="MedXpertQA-MM",
        type=DatasetType.HF_WITH_IMAGES,
        hf_path="TsinghuaC3I/MedXpertQA",
        default_local_path="./datas/MedXpertQA",
        download_url="https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/resolve/main/images.zip",
        requires_extraction=True,
        archive_filename="images.zip"
    ),
    "MedXpertQA-Text": DatasetConfig(
        name="MedXpertQA-Text",
        type=DatasetType.HF_ONLY,
        hf_path="TsinghuaC3I/MedXpertQA"
    ),

    # URL download datasets
    "OmniMedVQA": DatasetConfig(
        name="OmniMedVQA",
        type=DatasetType.URL_DOWNLOAD,
        download_url="https://huggingface.co/datasets/foreverbeliever/OmniMedVQA/resolve/main/OmniMedVQA.zip",
        default_local_path="./datas/OmniMedVQA",
        requires_extraction=True,
        archive_filename="OmniMedVQA.zip"
    ),
    "IU_XRAY": DatasetConfig(
        name="IU_XRAY",
        type=DatasetType.URL_DOWNLOAD,
        download_url="https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
        default_local_path="./datas/IU_XRAY",
        requires_extraction=True,
        archive_filename="NLMCXR_png.tgz"  # Fixed: was incorrectly "images.zip"
    ),

    # Manual download required
    "MIMIC_CXR": DatasetConfig(
        name="MIMIC_CXR",
        type=DatasetType.MANUAL,
        notes="Requires manual download from https://physionet.org/content/mimic-cxr/2.1.0/"
    ),
}


class DatasetManager:
    """
    Unified dataset manager for MedEvalKit.

    Features:
    - Centralized storage path configuration
    - Dataset status checking
    - Migration to/from AFS storage
    - HuggingFace cache management
    """

    def __init__(
        self,
        base_path: str = "./datas",
        hf_cache_dir: Optional[str] = None,
        afs_path: Optional[str] = None
    ):
        """
        Initialize dataset manager.

        Args:
            base_path: Base directory for local datasets (default: ./datas)
            hf_cache_dir: Custom HuggingFace cache directory (default: ~/.cache/huggingface)
            afs_path: Path to mounted AFS storage for migration
        """
        self.base_path = Path(base_path).resolve()
        self.hf_cache_dir = Path(hf_cache_dir) if hf_cache_dir else Path.home() / ".cache" / "huggingface"
        self.afs_path = Path(afs_path) if afs_path else None

        # Create base directory if not exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Load or create status file
        self.status_file = self.base_path / "dataset_status.json"
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        """Load dataset status from file"""
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                return json.load(f)
        return {}

    def _save_status(self):
        """Save dataset status to file"""
        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=2)

    def get_dataset_path(self, dataset_name: str) -> Path:
        """
        Get the local path for a dataset.

        For HF-only datasets, returns the HF cache path.
        For local datasets, returns path under base_path.
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = DATASET_REGISTRY[dataset_name]

        if config.type == DatasetType.HF_ONLY:
            # Return the hub-style cache path (newer HF versions)
            return self.hf_cache_dir / "hub" / f"datasets--{config.hf_path.replace('/', '--')}"
        else:
            return self.base_path / dataset_name

    def _check_hf_cache_exists(self, hf_path: str) -> Tuple[bool, Optional[Path], Optional[int]]:
        """
        Check if a HuggingFace dataset exists in the cache.

        HuggingFace uses two cache structures:
        1. Hub-style (newer): ~/.cache/huggingface/hub/datasets--{org}--{repo}/
        2. Legacy style: ~/.cache/huggingface/datasets/{org}___{repo}/

        Returns:
            Tuple of (exists, path, size_mb)
        """
        # Hub-style path (newer versions of HF)
        hub_path = self.hf_cache_dir / "hub" / f"datasets--{hf_path.replace('/', '--')}"

        # Legacy path (older versions)
        legacy_path = self.hf_cache_dir / "datasets" / hf_path.replace("/", "___")

        # Also check for the exact org/repo structure
        datasets_path = self.hf_cache_dir / "datasets"

        # Check hub-style first (most common in recent HF versions)
        if hub_path.exists():
            # Check if there are actual snapshot files
            snapshots = list(hub_path.glob("snapshots/*/"))
            if snapshots:
                size = self._get_dir_size(hub_path)
                return True, hub_path, size

        # Check legacy path
        if legacy_path.exists():
            # Check for arrow files or other data files
            has_data = any(legacy_path.rglob("*.arrow")) or any(legacy_path.rglob("*.parquet"))
            if has_data:
                size = self._get_dir_size(legacy_path)
                return True, legacy_path, size

        # Check datasets directory for partial matches
        if datasets_path.exists():
            # Look for directories that match the dataset name pattern
            org, repo = hf_path.split("/") if "/" in hf_path else ("", hf_path)
            pattern = f"{org}*{repo}*" if org else f"*{repo}*"

            for match in datasets_path.glob(pattern):
                if match.is_dir():
                    has_data = any(match.rglob("*.arrow")) or any(match.rglob("*.parquet"))
                    if has_data:
                        size = self._get_dir_size(match)
                        return True, match, size

        return False, None, None

    def _get_dir_size(self, path: Path) -> int:
        """Get directory size in MB"""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except (PermissionError, OSError):
            pass
        return total // (1024 * 1024)  # Convert to MB

    def check_dataset_status(self, dataset_name: str) -> Dict:
        """
        Check if a dataset is downloaded and ready.

        Returns:
            Dict with keys: exists, path, type, ready, notes, size_mb
        """
        if dataset_name not in DATASET_REGISTRY:
            return {"exists": False, "error": f"Unknown dataset: {dataset_name}"}

        config = DATASET_REGISTRY[dataset_name]
        path = self.get_dataset_path(dataset_name)

        result = {
            "name": dataset_name,
            "type": config.type.value,
            "path": str(path),
            "exists": False,
            "ready": False,
            "size_mb": None,
            "notes": config.notes
        }

        if config.type == DatasetType.HF_ONLY:
            # Actually check if the HF dataset is cached
            exists, actual_path, size_mb = self._check_hf_cache_exists(config.hf_path)
            result["exists"] = exists
            result["ready"] = exists
            result["size_mb"] = size_mb
            if actual_path:
                result["path"] = str(actual_path)
            result["notes"] = f"HuggingFace: {config.hf_path}"
            if not exists:
                result["notes"] += " (not cached, will download on first use)"
        elif config.type == DatasetType.MANUAL:
            result["exists"] = path.exists()
            result["ready"] = path.exists()
            if path.exists():
                result["size_mb"] = self._get_dir_size(path)
        else:
            # HF_WITH_IMAGES or URL_DOWNLOAD
            result["exists"] = path.exists()
            if path.exists():
                # Check for common data indicators
                has_images = (path / "images").exists() or (path / "imgs").exists() or (path / "figures").exists()
                has_json = any(path.glob("*.json"))
                has_csv = any(path.glob("*.csv"))
                result["ready"] = has_images or has_json or has_csv
                result["size_mb"] = self._get_dir_size(path)

        return result

    def list_all_datasets(self) -> List[Dict]:
        """List all datasets with their status"""
        results = []
        for name in DATASET_REGISTRY:
            results.append(self.check_dataset_status(name))
        return results

    def migrate_to_afs(self, dataset_name: str, force: bool = False) -> bool:
        """
        Migrate a dataset to AFS storage.

        Args:
            dataset_name: Name of the dataset to migrate
            force: Overwrite if exists on AFS

        Returns:
            True if successful
        """
        if not self.afs_path:
            raise ValueError("AFS path not configured")

        config = DATASET_REGISTRY.get(dataset_name)
        if not config:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if config.type == DatasetType.HF_ONLY:
            print(f"Skipping {dataset_name}: HuggingFace-only dataset")
            return False

        source = self.get_dataset_path(dataset_name)
        if not source.exists():
            raise FileNotFoundError(f"Dataset not found at {source}")

        dest = self.afs_path / dataset_name

        if dest.exists() and not force:
            print(f"Dataset already exists on AFS: {dest}")
            return False

        print(f"Migrating {dataset_name} to AFS...")
        print(f"  Source: {source}")
        print(f"  Destination: {dest}")

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(source, dest)

        # Update status
        self.status[dataset_name] = {
            "afs_path": str(dest),
            "migrated": True
        }
        self._save_status()

        print(f"Migration complete: {dataset_name}")
        return True

    def migrate_from_afs(self, dataset_name: str, force: bool = False) -> bool:
        """
        Copy a dataset from AFS to local storage.

        Args:
            dataset_name: Name of the dataset to copy
            force: Overwrite if exists locally

        Returns:
            True if successful
        """
        if not self.afs_path:
            raise ValueError("AFS path not configured")

        source = self.afs_path / dataset_name
        if not source.exists():
            raise FileNotFoundError(f"Dataset not found on AFS: {source}")

        dest = self.get_dataset_path(dataset_name)

        if dest.exists() and not force:
            print(f"Dataset already exists locally: {dest}")
            return False

        print(f"Copying {dataset_name} from AFS...")
        print(f"  Source: {source}")
        print(f"  Destination: {dest}")

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(source, dest)

        print(f"Copy complete: {dataset_name}")
        return True

    def create_symlink_from_afs(self, dataset_name: str) -> bool:
        """
        Create a symlink to AFS dataset instead of copying.
        More efficient for large datasets.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if successful
        """
        if not self.afs_path:
            raise ValueError("AFS path not configured")

        source = self.afs_path / dataset_name
        if not source.exists():
            raise FileNotFoundError(f"Dataset not found on AFS: {source}")

        dest = self.get_dataset_path(dataset_name)

        if dest.exists():
            if dest.is_symlink():
                dest.unlink()
            else:
                raise FileExistsError(f"Local path exists and is not a symlink: {dest}")

        dest.symlink_to(source)
        print(f"Created symlink: {dest} -> {source}")
        return True

    def set_hf_cache_dir(self, cache_dir: str):
        """
        Set custom HuggingFace cache directory.
        This also sets the HF_HOME environment variable.
        """
        self.hf_cache_dir = Path(cache_dir)
        os.environ["HF_HOME"] = str(self.hf_cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(self.hf_cache_dir / "datasets")
        print(f"HuggingFace cache set to: {self.hf_cache_dir}")

    def print_status_table(self):
        """Print a formatted status table of all datasets"""
        print("\n" + "=" * 100)
        print("Dataset Status")
        print(f"HF Cache: {self.hf_cache_dir}")
        print(f"Local Base: {self.base_path}")
        print("=" * 100)
        print(f"{'Dataset':<25} {'Type':<12} {'Ready':<6} {'Size':<10} {'Notes'}")
        print("-" * 100)

        total_size = 0
        ready_count = 0
        total_count = 0

        for status in self.list_all_datasets():
            ready_str = "✓" if status["ready"] else "✗"
            size_str = f"{status['size_mb']} MB" if status.get("size_mb") else "-"
            notes = status.get("notes", "") or ""
            # Truncate notes if too long
            if len(notes) > 45:
                notes = notes[:42] + "..."
            print(f"{status['name']:<25} {status['type']:<12} {ready_str:<6} {size_str:<10} {notes}")

            total_count += 1
            if status["ready"]:
                ready_count += 1
            if status.get("size_mb"):
                total_size += status["size_mb"]

        print("-" * 100)
        print(f"Summary: {ready_count}/{total_count} datasets ready, Total size: {total_size} MB ({total_size/1024:.1f} GB)")
        print("=" * 100)


# Convenience function for CLI usage
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MedEvalKit Dataset Manager")
    parser.add_argument("--base-path", default="./datas", help="Base path for local datasets")
    parser.add_argument("--afs-path", help="Path to AFS storage")
    parser.add_argument("--action", choices=["status", "migrate-to-afs", "migrate-from-afs", "symlink"],
                       default="status", help="Action to perform")
    parser.add_argument("--dataset", help="Dataset name (for migrate actions)")
    parser.add_argument("--force", action="store_true", help="Force overwrite")

    args = parser.parse_args()

    manager = DatasetManager(
        base_path=args.base_path,
        afs_path=args.afs_path
    )

    if args.action == "status":
        manager.print_status_table()
    elif args.action == "migrate-to-afs":
        if not args.dataset:
            print("Error: --dataset required for migrate action")
            return
        manager.migrate_to_afs(args.dataset, force=args.force)
    elif args.action == "migrate-from-afs":
        if not args.dataset:
            print("Error: --dataset required for migrate action")
            return
        manager.migrate_from_afs(args.dataset, force=args.force)
    elif args.action == "symlink":
        if not args.dataset:
            print("Error: --dataset required for symlink action")
            return
        manager.create_symlink_from_afs(args.dataset)


if __name__ == "__main__":
    main()
