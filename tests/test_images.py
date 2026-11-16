import io

import pytest
from PIL import Image

from vlm_lab.images import InvalidImage, decode_image


def png_bytes(size=(10, 10)):
    stream = io.BytesIO()
    Image.new("RGB", size, "red").save(stream, format="PNG")
    return stream.getvalue()


def test_decode_valid_png():
    image = decode_image(png_bytes(), max_pixels=1000)
    assert image.size == (10, 10)
    assert image.mode == "RGB"


def test_rejects_non_image():
    with pytest.raises(InvalidImage, match="not a valid"):
        decode_image(b"not an image", max_pixels=1000)


def test_rejects_excessive_dimensions():
    with pytest.raises(InvalidImage, match="safety limit"):
        decode_image(png_bytes((20, 20)), max_pixels=100)
