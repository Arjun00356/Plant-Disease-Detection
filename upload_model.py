"""
upload_model.py — One-time script to push the trained .pth checkpoint to HF Hub.

Usage:
    pip install huggingface_hub
    python upload_model.py --repo YOUR_HF_USERNAME/plant-disease-model
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        required=True,
        help="HF repo id, e.g. 'yourname/plant-disease-model'",
    )
    parser.add_argument(
        "--model",
        default="plant_disease__classification_model.pth",
        help="Path to the .pth file (default: plant_disease__classification_model.pth)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF write token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HF repo private (you will need HF_TOKEN in your Space secrets)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return

    api = HfApi(token=args.token)

    print(f"Creating repo '{args.repo}' (private={args.private})…")
    create_repo(
        repo_id=args.repo,
        repo_type="model",
        exist_ok=True,
        private=args.private,
        token=args.token,
    )

    print(f"Uploading '{model_path}' → '{args.repo}'…")
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=args.repo,
        repo_type="model",
    )

    size_mb = model_path.stat().st_size / 1e6
    print(f"[OK] Uploaded {size_mb:.1f} MB to https://huggingface.co/{args.repo}")
    print()
    print("Next steps:")
    print(f"  1. Set MODEL_REPO_ID={args.repo} as a Space secret")
    if args.private:
        print("  2. Set HF_TOKEN=<your read token> as a Space secret (private repo)")


if __name__ == "__main__":
    main()
