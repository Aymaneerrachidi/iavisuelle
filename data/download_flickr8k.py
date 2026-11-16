"""Download Flickr8k and create deterministic, grouped caption manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATASET_HANDLE = "dibyansudiptiman/flickr-8k"


def find_caption_file(root: Path) -> Path:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".txt"}
        and ("caption" in path.name.lower() or "token" in path.name.lower())
    ]
    if not candidates:
        raise FileNotFoundError("No caption CSV/TXT found in the downloaded dataset.")
    return max(candidates, key=lambda path: path.stat().st_size)


def read_captions(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    text = path.read_text("utf-8", errors="replace")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(text.splitlines(), dialect))
    if rows and any(cell.lower() in {"image", "caption", "image_name"} for cell in rows[0]):
        rows = rows[1:]
    for row in rows:
        if len(row) < 2:
            continue
        image_name = row[0].split("#", 1)[0].strip()
        caption = row[-1].strip()
        if image_name and caption:
            grouped[image_name].append(caption)
    return grouped


def write_manifest(
    path: Path, names: list[str], captions: dict[str, list[str]], images: dict[str, Path]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "references"])
        writer.writeheader()
        for name in names:
            writer.writerow(
                {
                    "image_id": name,
                    "image_path": str(images[name].resolve()),
                    "references": json.dumps(captions[name], ensure_ascii=False),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import kagglehub

    download_root = Path(kagglehub.dataset_download(DATASET_HANDLE))
    caption_file = find_caption_file(download_root)
    captions = read_captions(caption_file)
    image_paths = {
        path.name: path
        for path in download_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    names = sorted(set(captions) & set(image_paths))
    if len(names) < 100:
        raise RuntimeError(
            f"Only {len(names)} image-caption pairs found; dataset layout may have changed."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output / "flickr8k_all.csv", names, captions, image_paths)
    shuffled = names.copy()
    random.Random(args.seed).shuffle(shuffled)
    test_names = sorted(shuffled[: min(args.test_size, len(shuffled))])
    test_path = args.output / "flickr8k_test.csv"
    write_manifest(test_path, test_names, captions, image_paths)
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    metadata = {
        "dataset": "Flickr8k",
        "source": f"https://www.kaggle.com/datasets/{DATASET_HANDLE}",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "caption_file": str(caption_file),
        "images_detected": len(image_paths),
        "paired_images": len(names),
        "test_images": len(test_names),
        "seed": args.seed,
        "test_manifest_sha256": digest,
    }
    (args.output / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), "utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
