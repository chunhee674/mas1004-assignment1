"""Tests for Problem 2. Run: pytest tests/test_data.py"""

import numpy as np
import pytest
from PIL import Image

from conftest import CLASSES, PER_CLASS
from data import CROP, MEAN, STD, load_folder, prepare_image, split_train_test

TOTAL = len(CLASSES) * PER_CLASS


def reference(image):
    """How torchvision prepares a photo for ResNet18. Your answer must match."""
    from torchvision.models import ResNet18_Weights
    return ResNet18_Weights.IMAGENET1K_V1.transforms()(image.convert("RGB")).numpy()


@pytest.fixture
def photo():
    """A wide picture of smooth stripes, with a blue band on each side.

    The bands are outside the middle square, so a correct crop removes them.
    """
    rows, cols = np.mgrid[0:300, 0:500]
    pixels = np.stack([
        128 + 100 * np.sin(cols / 40),
        128 + 100 * np.cos(rows / 30),
        np.full((300, 500), 90.0),
    ], axis=-1).astype(np.uint8)
    pixels[:, :60] = (0, 0, 255)
    pixels[:, -60:] = (0, 0, 255)
    return Image.fromarray(pixels)


# ---------------------------------------------------------------------------
# prepare_image
# ---------------------------------------------------------------------------


def test_prepare_shape_and_type(photo):
    x = prepare_image(photo)
    assert isinstance(x, np.ndarray), (
        f"prepare_image should return a numpy array, got {type(x).__name__}. "
        "If you used torchvision, call .numpy() on the result."
    )
    assert x.dtype == np.float32, f"should be float32, got {x.dtype}"
    assert x.shape == (3, CROP, CROP), (
        f"should be (3, {CROP}, {CROP}), got {x.shape}. If it is "
        f"({CROP}, {CROP}, 3), the channels are last instead of first."
    )


def test_prepare_matches_how_resnet18_was_trained(photo):
    got = prepare_image(photo)
    wanted = reference(photo)
    gap = float(np.abs(got - wanted).max())
    assert gap < 0.05, (
        "Your numbers differ from the way ResNet18 was trained to see photos "
        f"(largest gap {gap:.3f}). Go through the six steps in the docstring "
        "one at a time."
    )


def test_prepare_normalises_each_channel():
    colour = tuple(round(m * 255) for m in MEAN)
    x = prepare_image(Image.new("RGB", (300, 300), colour))
    assert np.abs(x).max() < 0.02, (
        "A picture whose colour is exactly MEAN should come out as zeros. "
        "Subtract MEAN[c] and divide by STD[c] for each channel c, after "
        "dividing by 255."
    )
    x = prepare_image(Image.new("RGB", (300, 300), (255, 255, 255)))
    wanted = [(1 - m) / s for m, s in zip(MEAN, STD)]
    assert np.allclose(x[:, 0, 0], wanted, atol=0.02), (
        f"White should become {np.round(wanted, 3)}, got {np.round(x[:, 0, 0], 3)}."
    )


def test_prepare_keeps_red_green_blue_in_that_order():
    x = prepare_image(Image.new("RGB", (300, 300), (255, 0, 0)))
    assert x[0].mean() > x[1].mean() and x[0].mean() > x[2].mean(), (
        "A red picture should be brightest in channel 0. The channels are in "
        "the wrong order, or they are last instead of first."
    )


def test_prepare_cuts_out_the_middle(photo):
    x = prepare_image(photo)
    blue = (1 - MEAN[2]) / STD[2]
    left_column = x[2, :, 0]
    assert not np.allclose(left_column, blue, atol=0.1), (
        "The left edge of the result is the blue band from the edge of the "
        "photo. Resize the shorter side to 256 and cut out the middle 224 x 224. "
        "Do not squash the whole photo into a square."
    )


def test_prepare_accepts_grayscale_and_transparent_images(photo):
    for mode in ("L", "RGBA", "P"):
        x = prepare_image(photo.convert(mode))
        assert x.shape == (3, CROP, CROP), (
            f"A {mode} image gave shape {x.shape}. Convert to RGB first."
        )


# ---------------------------------------------------------------------------
# load_folder
# ---------------------------------------------------------------------------


def test_shapes_and_types(image_folder):
    X, y, class_names, paths = load_folder(image_folder)

    assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"
    assert y.dtype == np.int64, f"y should be int64, got {y.dtype}"
    assert X.shape == (TOTAL, 3, CROP, CROP), (
        f"X should be ({TOTAL}, 3, {CROP}, {CROP}), one prepared image per "
        f"row, got {X.shape}"
    )
    assert y.shape == (TOTAL,)
    assert len(paths) == TOTAL


def test_rows_are_prepared_images(image_folder):
    X, _, _, paths = load_folder(image_folder)
    wanted = reference(Image.open(paths[0]))
    assert np.abs(X[0] - wanted).max() < 0.05, (
        "Row 0 of X is not the prepared version of paths[0]. Use prepare_image "
        "on every file, and keep X and paths in the same order."
    )


def test_class_names_are_sorted(image_folder):
    _, _, class_names, _ = load_folder(image_folder)
    assert class_names == sorted(CLASSES)


def test_labels_match_the_folder_each_image_came_from(image_folder):
    _, y, class_names, paths = load_folder(image_folder)
    for label, path in zip(y, paths):
        assert class_names[label] == path.parent.name


def test_broken_and_non_image_files_are_skipped(image_folder):
    X, _, _, paths = load_folder(image_folder)
    assert len(X) == TOTAL, (
        "broken.jpg and notes.txt must not end up in X, and CAPITALS.JPG must. "
        f"Expected {TOTAL} rows, got {len(X)}."
    )
    names = [p.name for p in paths]
    assert "broken.jpg" not in names and "notes.txt" not in names
    assert "CAPITALS.JPG" in names, "A .JPG file is an image too."


def test_same_result_every_time(image_folder):
    first = load_folder(image_folder)
    second = load_folder(image_folder)
    assert [str(p) for p in first[3]] == [str(p) for p in second[3]], (
        "Two runs gave the files in a different order. Sort them."
    )
    assert np.array_equal(first[1], second[1])


# ---------------------------------------------------------------------------
# split_train_test
# ---------------------------------------------------------------------------


def test_split_keeps_everything(image_folder):
    X, y, _, paths = load_folder(image_folder)
    Xtr, ytr, ptr, Xte, yte, pte = split_train_test(X, y, paths, test_ratio=0.25, seed=0)

    assert len(Xtr) == len(ytr) == len(ptr)
    assert len(Xte) == len(yte) == len(pte)
    assert len(Xtr) + len(Xte) == TOTAL, "The split lost or duplicated rows."


def test_split_has_no_image_on_both_sides(image_folder):
    X, y, _, paths = load_folder(image_folder)
    _, _, ptr, _, _, pte = split_train_test(X, y, paths, test_ratio=0.25, seed=0)

    shared = set(map(str, ptr)) & set(map(str, pte))
    assert not shared, (
        "These images are in both the training set and the test set: "
        f"{sorted(shared)[:3]}. Then the test accuracy is a lie."
    )


def test_every_class_appears_on_both_sides(image_folder):
    X, y, _, paths = load_folder(image_folder)
    _, ytr, _, _, yte, _ = split_train_test(X, y, paths, test_ratio=0.25, seed=0)

    assert set(np.unique(ytr)) == set(range(len(CLASSES)))
    assert set(np.unique(yte)) == set(range(len(CLASSES))), (
        "A class has no test images, so its accuracy cannot be measured."
    )


def test_split_is_repeatable(image_folder):
    X, y, _, paths = load_folder(image_folder)
    first = split_train_test(X, y, paths, test_ratio=0.25, seed=7)
    second = split_train_test(X, y, paths, test_ratio=0.25, seed=7)
    assert [str(p) for p in first[2]] == [str(p) for p in second[2]], (
        "The same seed gave two different splits."
    )
