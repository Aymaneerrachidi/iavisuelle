import csv
import json

from PIL import Image

from vlm_lab.datasets import load_manifest


def test_load_manifest_groups_multiple_references(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "references"])
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "sample",
                "image_path": "sample.jpg",
                "references": json.dumps(["a blue square", "a small blue image"]),
            }
        )

    items = load_manifest(str(manifest), limit=10)

    assert len(items) == 1
    assert items[0].references == ["a blue square", "a small blue image"]
    assert items[0].image_path == str(image_path.resolve())
