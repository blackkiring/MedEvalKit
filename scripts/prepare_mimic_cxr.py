"""
MIMIC-CXR Dataset Preparation Script

This script downloads the MIMIC-CXR dataset from HuggingFace and converts it
to the format required by the evaluation code.

Usage:
    uv run python scripts/prepare_mimic_cxr.py --output_dir ./datas/MIMIC_CXR --test_size 500

Alternative HuggingFace datasets to try:
    - itsanmolgupta/mimic-cxr-dataset
    - StanfordAIMI/mimic-cxr (if available)
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def explore_hf_dataset(dataset_name: str):
    """Explore a HuggingFace dataset to understand its structure"""
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"Exploring dataset: {dataset_name}")
    print('='*60)

    try:
        # Try loading with different configurations
        ds = load_dataset(dataset_name, trust_remote_code=True)
        print(f"\nAvailable splits: {list(ds.keys())}")

        # Check first split
        first_split = list(ds.keys())[0]
        print(f"\nSplit '{first_split}' info:")
        print(f"  - Number of samples: {len(ds[first_split])}")
        print(f"  - Features: {ds[first_split].features}")

        # Show sample
        print(f"\nFirst sample:")
        sample = ds[first_split][0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  - {key}: {value[:100]}...")
            elif hasattr(value, 'size'):  # PIL Image
                print(f"  - {key}: Image {value.size}")
            else:
                print(f"  - {key}: {value}")

        return ds
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def prepare_from_hf(
    dataset_name: str,
    output_dir: str,
    test_size: int = 500,
    seed: int = 42
):
    """
    Download and prepare MIMIC-CXR dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Output directory
        test_size: Number of samples for test set
        seed: Random seed for sampling
    """
    from datasets import load_dataset
    import random

    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from HuggingFace: {dataset_name}")

    try:
        ds = load_dataset(dataset_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("\nTrying alternative loading methods...")

        # Try different configurations
        try:
            ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
            ds = {"train": ds}
        except:
            raise ValueError(f"Cannot load dataset {dataset_name}")

    # Determine which split to use
    available_splits = list(ds.keys())
    print(f"Available splits: {available_splits}")

    # Prefer test > validation > train
    if "test" in available_splits:
        split_name = "test"
    elif "validation" in available_splits:
        split_name = "validation"
    else:
        split_name = available_splits[0]

    data = ds[split_name]
    print(f"Using split: {split_name} with {len(data)} samples")

    # Get column names
    columns = list(data.features.keys())
    print(f"Columns: {columns}")

    # Try to identify the relevant columns
    image_col = None
    findings_col = None
    impression_col = None
    report_col = None

    for col in columns:
        col_lower = col.lower()
        if 'image' in col_lower or col_lower == 'img':
            image_col = col
        elif 'finding' in col_lower:
            findings_col = col
        elif 'impression' in col_lower:
            impression_col = col
        elif 'report' in col_lower or 'text' in col_lower:
            report_col = col

    print(f"\nDetected columns:")
    print(f"  - Image: {image_col}")
    print(f"  - Findings: {findings_col}")
    print(f"  - Impression: {impression_col}")
    print(f"  - Report (fallback): {report_col}")

    if not image_col:
        print("\nWARNING: No image column detected. Available columns:")
        for col in columns:
            print(f"  - {col}: {type(data[0][col])}")
        return

    # Sample data
    total_samples = len(data)
    if test_size > total_samples:
        print(f"Warning: test_size ({test_size}) > total samples ({total_samples})")
        test_size = total_samples

    random.seed(seed)
    indices = random.sample(range(total_samples), test_size)

    # Process samples
    test_data = []
    print(f"\nProcessing {test_size} samples...")

    for i, idx in enumerate(tqdm(indices)):
        sample = data[idx]

        # Handle image
        img = sample[image_col]
        if hasattr(img, 'save'):  # PIL Image
            img_filename = f"mimic_cxr_{i:05d}.jpg"
            img_path = images_dir / img_filename
            img.save(img_path)
        elif isinstance(img, str):
            img_filename = img
        elif isinstance(img, dict) and 'path' in img:
            img_filename = os.path.basename(img['path'])
        else:
            img_filename = f"mimic_cxr_{i:05d}.jpg"
            print(f"Warning: Unknown image format for sample {idx}")
            continue

        # Handle findings and impression
        if findings_col and impression_col:
            findings = sample.get(findings_col, "") or ""
            impression = sample.get(impression_col, "") or ""
        elif report_col:
            # Try to split report into findings and impression
            report = sample.get(report_col, "") or ""
            findings, impression = parse_report(report)
        else:
            findings = ""
            impression = ""

        # Skip empty samples
        if not findings.strip() and not impression.strip():
            continue

        test_data.append({
            "image": img_filename,
            "findings": findings,
            "impression": impression
        })

    # Save test.json
    test_json_path = output_path / "test.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Dataset prepared successfully!")
    print(f"  - Output directory: {output_path}")
    print(f"  - Test samples: {len(test_data)}")
    print(f"  - Images saved to: {images_dir}")
    print(f"  - Annotation file: {test_json_path}")
    print('='*60)


def parse_report(report: str) -> tuple:
    """
    Parse a radiology report into findings and impression sections.

    Returns:
        Tuple of (findings, impression)
    """
    report = report.strip()

    findings = ""
    impression = ""

    # Common section headers
    findings_headers = ["FINDINGS:", "FINDINGS", "Findings:"]
    impression_headers = ["IMPRESSION:", "IMPRESSION", "Impression:", "CONCLUSION:", "CONCLUSION"]

    report_lower = report.lower()

    # Find impression section
    impression_start = -1
    for header in impression_headers:
        pos = report_lower.find(header.lower())
        if pos != -1:
            impression_start = pos + len(header)
            break

    # Find findings section
    findings_start = -1
    for header in findings_headers:
        pos = report_lower.find(header.lower())
        if pos != -1:
            findings_start = pos + len(header)
            break

    if findings_start != -1 and impression_start != -1:
        if findings_start < impression_start:
            findings = report[findings_start:impression_start].strip()
            # Remove any header at the end
            for header in impression_headers:
                if findings.lower().endswith(header.lower()):
                    findings = findings[:-len(header)].strip()
            impression = report[impression_start:].strip()
        else:
            impression = report[impression_start:findings_start].strip()
            for header in findings_headers:
                if impression.lower().endswith(header.lower()):
                    impression = impression[:-len(header)].strip()
            findings = report[findings_start:].strip()
    elif findings_start != -1:
        findings = report[findings_start:].strip()
    elif impression_start != -1:
        impression = report[impression_start:].strip()
    else:
        # No clear sections, use whole report as findings
        findings = report

    return findings, impression


def create_sample_test_json(output_dir: str):
    """Create a sample test.json to show the expected format"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_data = [
        {
            "image": "sample_001.jpg",
            "findings": "The heart size is normal. The mediastinal contours are within normal limits. The lungs are clear without focal consolidation, pleural effusion, or pneumothorax. No acute osseous abnormality.",
            "impression": "No acute cardiopulmonary abnormality."
        },
        {
            "image": ["sample_002_frontal.jpg", "sample_002_lateral.jpg"],
            "findings": "PA and lateral views of the chest. The cardiac silhouette is mildly enlarged. There is mild pulmonary vascular congestion. Small bilateral pleural effusions are present.",
            "impression": "Mild cardiomegaly with pulmonary vascular congestion and small bilateral pleural effusions, compatible with mild congestive heart failure."
        }
    ]

    sample_path = output_path / "test_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"Sample test.json created at: {sample_path}")
    print("\nExpected format:")
    print(json.dumps(sample_data[0], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Prepare MIMIC-CXR dataset for evaluation")
    parser.add_argument("--dataset", type=str, default="itsanmolgupta/mimic-cxr-dataset",
                       help="HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str, default="./datas/MIMIC_CXR",
                       help="Output directory")
    parser.add_argument("--test_size", type=int, default=500,
                       help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--explore", action="store_true",
                       help="Only explore the dataset structure")
    parser.add_argument("--show_format", action="store_true",
                       help="Show expected test.json format")

    args = parser.parse_args()

    if args.show_format:
        create_sample_test_json(args.output_dir)
        return

    if args.explore:
        explore_hf_dataset(args.dataset)
        return

    prepare_from_hf(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
