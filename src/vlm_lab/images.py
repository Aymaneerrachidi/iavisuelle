from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImage(ValueError):
    pass


def decode_image(payload: bytes, max_pixels: int) -> Image.Image:
    if not payload:
        raise InvalidImage("The uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(payload))
        image.verify()
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage("The file is not a valid JPEG, PNG, or WebP image.") from exc
    if image.format not in ALLOWED_FORMATS:
        raise InvalidImage("Only JPEG, PNG, and WebP images are supported.")
    if image.width * image.height > max_pixels:
        raise InvalidImage(f"Image exceeds the {max_pixels:,}-pixel safety limit.")
    return image.convert("RGB")


def load_local_image(path: str, max_pixels: int = 25_000_000) -> Image.Image:
    resolved = Path(path).resolve()
    return decode_image(resolved.read_bytes(), max_pixels)
