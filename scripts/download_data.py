"""
Download ASVspoof 2019 LA subset from HuggingFace Hub.
Usage:
    python scripts/download_data.py --output data/raw
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download ASVspoof 2019 from HuggingFace")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    parser.add_argument("--subset", default="LA", choices=["LA", "PA"], help="ASVspoof subset")
    args = parser.parse_args()

    from datasets import load_dataset

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ASVspoof2019 ({args.subset}) to {output} ...")
    dataset = load_dataset(
        "asvspoof/asvspoof2019",
        args.subset,
        trust_remote_code=True,
    )

    print("Dataset splits:", list(dataset.keys()))
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")

    print("\nDataset downloaded. Audio files are accessible via the 'audio' column.")
    print("To extract features, run: python scripts/extract_features.py")


if __name__ == "__main__":
    main()
