"""
IU-XRAY Dataset Preparation Script

Downloads test annotations from HuggingFace and prepares test.json.
Images are downloaded separately from the official source.

Usage:
    export http_proxy="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
    export https_proxy="http://cmcproxy:WvUBhef4bQ@10.251.112.50:8128"
    uv run python scripts/prepare_iu_xray.py
"""

import os
import json
import re
from pathlib import Path
from tqdm import tqdm


def parse_report(report: str) -> tuple:
    """
    Parse a radiology report into findings and impression sections.

    IU-XRAY reports typically don't have explicit section headers,
    so we use heuristics to split them.
    """
    report = report.strip()

    # Common patterns for impression
    impression_patterns = [
        r'IMPRESSION[:\s]*',
        r'CONCLUSION[:\s]*',
        r'Impression[:\s]*',
        r'Conclusion[:\s]*',
    ]

    findings_patterns = [
        r'FINDINGS[:\s]*',
        r'Findings[:\s]*',
    ]

    findings = ""
    impression = ""

    # Try to find explicit sections
    for pattern in impression_patterns:
        match = re.search(pattern, report)
        if match:
            impression_start = match.end()
            # Everything before is findings (roughly)
            findings_text = report[:match.start()].strip()
            impression_text = report[impression_start:].strip()

            # Clean up findings
            for fp in findings_patterns:
                findings_text = re.sub(fp, '', findings_text).strip()

            findings = findings_text
            impression = impression_text
            return findings, impression

    # No explicit sections found - use the whole report as findings
    # and try to extract last sentence as impression
    sentences = report.split('.')
    if len(sentences) > 2:
        impression = sentences[-2].strip() + '.' if sentences[-2].strip() else ""
        findings = '.'.join(sentences[:-2]).strip()
        if findings and not findings.endswith('.'):
            findings += '.'
    else:
        findings = report
        impression = ""

    return findings, impression


def extract_image_id(image_path: str) -> str:
    """
    Extract image ID from HF dataset path.
    e.g., '/iu_xray/image/CXR2384_IM-0942/0.png' -> 'CXR2384_IM-0942_0'
    """
    # Extract the CXR ID and image number
    match = re.search(r'(CXR\d+_IM-\d+)/(\d+)\.png', image_path)
    if match:
        cxr_id = match.group(1)
        img_num = match.group(2)
        return f"{cxr_id}_{img_num}"
    return os.path.basename(image_path).replace('.png', '')


def prepare_iu_xray(output_dir: str = "./datas/IU_XRAY"):
    """
    Prepare IU-XRAY dataset from HuggingFace.
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading IU-XRAY dataset from HuggingFace...")
    ds = load_dataset("dz-osamu/IU-Xray")

    test_data = []

    # Process test split
    print(f"\nProcessing {len(ds['test'])} test samples...")

    for sample in tqdm(ds['test']):
        response = sample['response']
        image_paths = sample['images']

        # Parse report into findings and impression
        findings, impression = parse_report(response)

        # Skip empty samples
        if not findings.strip() and not impression.strip():
            continue

        # Extract image filenames
        # The HF paths are like '/iu_xray/image/CXR2384_IM-0942/0.png'
        # We need to map these to actual image files
        image_files = []
        for img_path in image_paths:
            # Extract CXR ID (e.g., CXR2384_IM-0942)
            match = re.search(r'(CXR\d+_IM-\d+)/(\d+)\.png', img_path)
            if match:
                cxr_id = match.group(1)
                img_num = match.group(2)
                # The actual files in NLMCXR_png are named like:
                # CXR2384_IM-0942-1001.png, CXR2384_IM-0942-2001.png
                # We'll need to find the actual filename
                image_files.append({
                    'cxr_id': cxr_id,
                    'img_num': img_num,
                    'original_path': img_path
                })

        if not image_files:
            continue

        test_data.append({
            'image_info': image_files,  # Temporary, will be resolved later
            'findings': findings,
            'impression': impression,
            'original_response': response
        })

    # Now we need to map to actual image files
    # First, check if images directory exists
    images_dir = output_path / "images"

    if images_dir.exists():
        print(f"\nMapping to actual image files in {images_dir}...")
        # Get list of actual image files
        actual_images = {f.stem: f.name for f in images_dir.glob("*.png")}

        final_test_data = []
        for sample in tqdm(test_data):
            image_files = []
            for img_info in sample['image_info']:
                cxr_id = img_info['cxr_id']
                # Find matching files
                matching = [f for stem, f in actual_images.items() if cxr_id in stem]
                if matching:
                    # Sort to get consistent ordering
                    matching.sort()
                    image_files.extend(matching)

            if image_files:
                final_test_data.append({
                    'image': list(set(image_files)),  # Remove duplicates
                    'findings': sample['findings'],
                    'impression': sample['impression']
                })

        test_data = final_test_data
        print(f"Successfully mapped {len(test_data)} samples")
    else:
        print(f"\nWARNING: Images directory not found at {images_dir}")
        print("Creating test.json with placeholder image paths.")
        print("Run the evaluation to trigger image download, then re-run this script.")

        # Create placeholder format
        final_test_data = []
        for sample in test_data:
            image_files = []
            for img_info in sample['image_info']:
                # Create expected filename pattern
                cxr_id = img_info['cxr_id']
                image_files.append(f"{cxr_id}.png")

            final_test_data.append({
                'image': image_files,
                'findings': sample['findings'],
                'impression': sample['impression']
            })
        test_data = final_test_data

    # Save test.json
    test_json_path = output_path / "test.json"
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"IU-XRAY dataset prepared!")
    print(f"  - Test samples: {len(test_data)}")
    print(f"  - Output: {test_json_path}")
    print(f"{'='*60}")

    # Show sample
    if test_data:
        print("\nSample entry:")
        print(json.dumps(test_data[0], indent=2, ensure_ascii=False)[:500])


def list_available_images(images_dir: str):
    """List available images to help with mapping"""
    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"Directory not found: {images_dir}")
        return

    images = list(images_path.glob("*.png"))
    print(f"Found {len(images)} images")
    print("Sample filenames:")
    for img in images[:10]:
        print(f"  {img.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare IU-XRAY dataset")
    parser.add_argument("--output_dir", default="./datas/IU_XRAY", help="Output directory")
    parser.add_argument("--list_images", action="store_true", help="List available images")

    args = parser.parse_args()

    if args.list_images:
        list_available_images(os.path.join(args.output_dir, "images"))
    else:
        prepare_iu_xray(args.output_dir)
