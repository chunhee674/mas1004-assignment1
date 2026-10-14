"""Checks that the export script writes a web page model that really is yours.

This one tests code that was given to you, so it should pass from the start.
It uses a tiny network instead of ResNet18 so that it runs in seconds, and a
copy of the preparation steps instead of your prepare_image, which you have
not written yet.

Run: pytest tests/test_export.py
"""

import base64
import io
import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from data import CROP, MEAN, RESIZE, STD
from export_web import export, selftest_picture


def prepare_for_test(image):
    """The six steps of prepare_image, written out so these tests can run."""
    image = image.convert("RGB")
    width, height = image.size
    scale = RESIZE / min(width, height)
    image = image.resize(
        (max(RESIZE, int(width * scale)), max(RESIZE, int(height * scale))),
        Image.BILINEAR,
    )
    left = int(round((image.width - CROP) / 2.0))
    top = int(round((image.height - CROP) / 2.0))
    image = image.crop((left, top, left + CROP, top + CROP))
    values = np.asarray(image, dtype=np.float32) / 255.0
    values = (values - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return values.transpose(2, 0, 1).astype(np.float32)


def tiny_network(outputs):
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 5, stride=4), nn.BatchNorm2d(8), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, outputs),
    ).eval()


@pytest.fixture(scope="module")
def pictures(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pictures")
    rng = np.random.default_rng(0)
    paths = []
    for number, size in enumerate([(400, 300), (300, 500), (256, 256)]):
        pixels = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        path = folder / f"{number}.jpg"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def exported(tmp_path_factory, pictures):
    out = tmp_path_factory.mktemp("docs")
    model = tiny_network(4)
    export(model, ["one", "two", "three", "four"], pictures,
           prepare=prepare_for_test, out_dir=out)
    return out, model


def test_every_file_is_written(exported):
    out, _ = exported
    for name in ("model.onnx", "model.json", "selftest.json"):
        assert (out / name).exists(), f"{name} was not written"


def test_model_json_says_how_to_prepare_a_photo(exported):
    out, _ = exported
    settings = json.loads((out / "model.json").read_text())
    assert settings["labels"] == ["one", "two", "three", "four"]
    assert settings["input"]["resize"] == RESIZE
    assert settings["input"]["crop"] == CROP
    assert settings["input"]["mean"] == list(MEAN)
    assert settings["input"]["std"] == list(STD)


def test_onnx_file_gives_the_answers_pytorch_gives(exported):
    import onnxruntime
    out, model = exported
    session = onnxruntime.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    x = np.random.default_rng(1).normal(size=(1, 3, CROP, CROP)).astype(np.float32)
    with torch.no_grad():
        wanted = model(torch.from_numpy(x)).numpy()
    got = session.run(None, {"image": x})[0]
    assert np.allclose(got, wanted, atol=1e-4), (
        "The exported file answers differently from PyTorch, so the browser "
        "would too."
    )


def test_selftest_holds_three_square_pictures_and_their_answers(exported):
    out, model = exported
    cases = json.loads((out / "selftest.json").read_text())["cases"]
    assert len(cases) == 3
    for case in cases:
        picture = Image.open(io.BytesIO(base64.b64decode(case["png"])))
        assert picture.size == (RESIZE, RESIZE)
        with torch.no_grad():
            wanted = model(torch.from_numpy(prepare_for_test(picture)[None])).numpy()[0]
        assert np.allclose(case["logits"], wanted, atol=1e-3)


def test_selftest_picture_is_already_resized(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (1000, 600), (10, 20, 30)).save(path)
    assert selftest_picture(path).size == (RESIZE, RESIZE)


def test_export_refuses_the_wrong_number_of_outputs(tmp_path, pictures):
    with pytest.raises(ValueError, match="outputs"):
        export(tiny_network(4), ["a", "b", "c"], pictures,
               prepare=prepare_for_test, out_dir=tmp_path)
