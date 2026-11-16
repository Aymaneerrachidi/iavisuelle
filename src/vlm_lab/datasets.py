from __future__ import annotations

import csv
import json
from pathlib import Path

from .domain import DatasetItem


class DatasetManifestError(ValueError):
    pass


def load_manifest(path: str, limit: int) -> list[DatasetItem]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise DatasetManifestError(f"Manifest not found: {path}")
    if manifest_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in manifest_path.read_text("utf-8").splitlines() if line]
    elif manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise DatasetManifestError("Manifest must be .csv or .jsonl.")

    grouped: dict[str, DatasetItem] = {}
    for index, row in enumerate(rows):
        image_value = row.get("image_path") or row.get("image")
        if not image_value:
            raise DatasetManifestError(f"Row {index + 1} has no image_path.")
        image_path = Path(str(image_value))
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        if not image_path.is_file():
            raise DatasetManifestError(f"Image not found at row {index + 1}: {image_path}")
        image_id = str(row.get("image_id") or image_path.name)
        raw_references = row.get("references") or row.get("caption") or ""
        if isinstance(raw_references, list):
            references = [str(item) for item in raw_references]
        else:
            value = str(raw_references)
            try:
                parsed = json.loads(value)
                references = parsed if isinstance(parsed, list) else [value]
            except json.JSONDecodeError:
                references = [value] if value else []
        if image_id in grouped:
            grouped[image_id].references.extend(references)
        else:
            grouped[image_id] = DatasetItem(str(image_path), references, image_id)
        if len(grouped) >= limit:
            break
    if not grouped:
        raise DatasetManifestError("Manifest contains no usable rows.")
    return list(grouped.values())
