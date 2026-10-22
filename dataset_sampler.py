import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_POSITIVE_DIR = Path(r"C:\Users\darlis\Downloads\anya")
DEFAULT_NEGATIVE_DIR = Path(r"C:\Users\darlis\Pictures\Screenshots")
DEFAULT_OUTPUT = Path("dataset_manifest.json")


def list_images(directory, recursive=False):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: str(p).lower())


def verify_image(path):
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        if width <= 0 or height <= 0:
            return None
        return width, height
    except Exception as exc:
        print(f"[skip] Cannot open {path}: {exc}", file=sys.stderr)
        return None


def make_entry(image_id, path, label, is_art, width, height):
    return {
        "id": image_id,
        "path": str(Path(path).resolve()),
        "label": label,
        "is_art": is_art,
        "width": width,
        "height": height,
    }


def sample_valid(candidates, count, rng):
    if not candidates:
        return []
    if count >= len(candidates):
        selected = list(candidates)
        rng.shuffle(selected)
    else:
        selected = rng.sample(candidates, count)
    valid = []
    for path in selected:
        dimensions = verify_image(path)
        if dimensions is not None:
            valid.append((path, *dimensions))
    return valid


def build_manifest(positive_dir, negative_dir, negative_count, seed, recursive):
    rng = random.Random(seed)

    positives = list_images(positive_dir, recursive=recursive)
    negatives = list_images(negative_dir, recursive=recursive)

    if not positives:
        raise RuntimeError(f"No valid image files found in {positive_dir}")
    if not negatives:
        raise RuntimeError(f"No valid image files found in {negative_dir}")

    positive_samples = []
    for path in positives:
        dimensions = verify_image(path)
        if dimensions is not None:
            positive_samples.append((path, *dimensions))

    negative_samples = []
    remaining = list(negatives)
    while len(negative_samples) < min(negative_count, len(negatives)) and remaining:
        wanted = min(negative_count - len(negative_samples), len(remaining))
        batch = sample_valid(remaining, wanted, rng)
        for path, width, height in batch:
            negative_samples.append((path, width, height))
            remaining.remove(path)
        if not batch:
            break

    manifest = []
    for index, (path, width, height) in enumerate(positive_samples, start=1):
        manifest.append(
            make_entry(f"art_{index:03d}", path, "art", True, width, height)
        )
    for index, (path, width, height) in enumerate(negative_samples, start=1):
        manifest.append(
            make_entry(f"not_art_{index:03d}", path, "not_art", False, width, height)
        )
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a balanced image-classification evaluation manifest."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the generated JSON manifest (default: dataset_manifest.json)",
    )
    parser.add_argument(
        "--sample-negative-count",
        type=int,
        default=19,
        help="Number of negative (screenshot) samples to draw (default: 19)",
    )
    parser.add_argument(
        "--positive-dir",
        type=Path,
        default=DEFAULT_POSITIVE_DIR,
        help=f"Directory of positive (art) images (default: {DEFAULT_POSITIVE_DIR})",
    )
    parser.add_argument(
        "--negative-dir",
        type=Path,
        default=DEFAULT_NEGATIVE_DIR,
        help=f"Directory of negative (screenshot) images (default: {DEFAULT_NEGATIVE_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible negative sampling (default: 42)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search source directories recursively",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.sample_negative_count < 0:
        raise ValueError("--sample-negative-count must be non-negative")

    manifest = build_manifest(
        positive_dir=args.positive_dir,
        negative_dir=args.negative_dir,
        negative_count=args.sample_negative_count,
        seed=args.seed,
        recursive=args.recursive,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    art_count = sum(1 for entry in manifest if entry["is_art"])
    other_count = len(manifest) - art_count
    print(f"Wrote {len(manifest)} samples to {args.output.resolve()}")
    print(f"  art: {art_count}")
    print(f"  not_art: {other_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
