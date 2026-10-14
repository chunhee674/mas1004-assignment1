"""Shared setup for the tests. You do not need to change this file."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CLASSES = ["alpha", "beta", "gamma"]
PER_CLASS = 8


@pytest.fixture
def image_folder(tmp_path):
    """A small folder of images that is messy in the same ways yours will be.

    Different sizes, different shapes, a PNG with transparency, an image that
    is already grayscale, an extension in capitals, a file that is not an image
    at all, and a stray text file. If load_folder survives this, it will
    survive your download folder.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / "images"

    for index, name in enumerate(CLASSES):
        folder = root / name
        folder.mkdir(parents=True)
        for number in range(PER_CLASS):
            width = int(rng.integers(40, 200))
            height = int(rng.integers(40, 200))
            # Each class gets its own colour, so a model can actually learn it.
            base = np.zeros((height, width, 3), dtype=np.uint8)
            base[:, :, index] = 200
            base += rng.integers(0, 40, base.shape, dtype=np.uint8)

            if number == 0:
                Image.fromarray(base).convert("RGBA").save(folder / "with_alpha.png")
            elif number == 1:
                Image.fromarray(base).convert("L").save(folder / "already_gray.png")
            elif number == 2 and index == 1:
                Image.fromarray(base).save(folder / "CAPITALS.JPG", "JPEG")
            else:
                Image.fromarray(base).save(folder / f"{number:03d}.jpg")

    (root / CLASSES[0] / "broken.jpg").write_bytes(b"this is not an image")
    (root / CLASSES[0] / "notes.txt").write_text("ignore me")
    return root


@pytest.fixture
def blobs():
    """A tiny problem that any working model should solve almost perfectly."""
    rng = np.random.default_rng(1)
    centres = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    X, y = [], []
    for label, centre in enumerate(centres):
        X.append(centre + rng.normal(0, 0.08, (40, 3)))
        y.extend([label] * 40)
    X = np.asarray(np.concatenate(X), dtype=np.float32)
    return X, np.asarray(y, dtype=np.int64)
