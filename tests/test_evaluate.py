"""Tests for Problems 3 and 5. Run: pytest tests/test_evaluate.py"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from evaluate import accuracy, confusion_matrix, predict_logits, worst_examples


@pytest.fixture
def known_model():
    """A model whose answers we already know, so the tests can check yours.

    It has no hidden layer and its weights are the identity, so the output for
    row x is simply x. Whichever of the three numbers is biggest is the class
    it picks.
    """
    model = nn.Sequential(nn.Linear(3, 3))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(3))
        model[0].bias.zero_()
    return model


@pytest.fixture
def known_data():
    X = np.array([
        [9.0, 0.0, 0.0],   # says 0
        [0.0, 9.0, 0.0],   # says 1
        [0.0, 0.0, 9.0],   # says 2
        [8.0, 0.0, 0.0],   # says 0, very sure
        [0.6, 0.5, 0.4],   # says 0, not sure at all
    ], dtype=np.float32)
    y = np.array([0, 1, 2, 2, 1], dtype=np.int64)  # last two are wrong
    paths = [Path(f"fake/{i}.jpg") for i in range(5)]
    return X, y, paths


def test_predict_logits_shape_and_values(known_model, known_data):
    X, _, _ = known_data
    logits = predict_logits(known_model, X)

    assert logits.shape == (5, 3), f"expected (5, 3), got {logits.shape}"
    assert np.allclose(logits, X, atol=1e-5), (
        "This model returns its input unchanged, so the logits should equal X. "
        "Did you apply a softmax? predict_logits must return raw outputs."
    )


def test_predict_logits_works_in_batches(known_model):
    X = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    assert predict_logits(known_model, X, batch_size=64).shape == (500, 3)


def test_predict_logits_takes_images():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(4, 2),
    )
    X = np.random.default_rng(0).random((7, 3, 16, 16)).astype(np.float32)
    logits = predict_logits(model, X, batch_size=3)
    assert isinstance(logits, np.ndarray) and logits.shape == (7, 2), (
        "Seven images should give a (7, 2) numpy array."
    )


def test_predict_logits_uses_eval_mode():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 64), nn.Dropout(0.5), nn.Linear(64, 2))
    model.train()
    X = np.random.default_rng(0).random((20, 3)).astype(np.float32)
    first = predict_logits(model, X)
    second = predict_logits(model, X)
    assert np.allclose(first, second), (
        "Two calls on the same rows gave different answers, so the model was "
        "still in training mode. Call model.eval() first. ResNet's BatchNorm "
        "layers give wrong answers in training mode."
    )


def test_accuracy(known_model, known_data):
    X, y, _ = known_data
    assert accuracy(known_model, X, y) == pytest.approx(3 / 5), (
        "Three of these five are right, so accuracy is 0.6. "
        "Return a share between 0 and 1, not a percentage."
    )


def test_confusion_matrix(known_model, known_data):
    X, y, _ = known_data
    matrix = confusion_matrix(known_model, X, y, num_classes=3)

    assert matrix.shape == (3, 3)
    assert matrix.sum() == len(y), "Every image belongs in exactly one square."
    assert matrix.trace() == 3, "Three were right, so the diagonal adds to 3."
    assert matrix[2][0] == 1, (
        "One image of class 2 was called class 0. Row is the true class and "
        "column is the prediction. You may have them the other way round."
    )
    assert matrix[1][0] == 1


def test_worst_examples_only_lists_mistakes(known_model, known_data):
    X, y, paths = known_data
    mistakes = worst_examples(known_model, X, y, paths, k=10)

    assert len(mistakes) == 2, f"There are exactly 2 mistakes, got {len(mistakes)}"
    for mistake in mistakes:
        assert mistake["true"] != mistake["predicted"]
        assert set(mistake) == {"path", "true", "predicted", "confidence"}
        assert 0.0 <= mistake["confidence"] <= 1.0, (
            "confidence is a softmax probability between 0 and 1."
        )


def test_worst_examples_puts_the_most_confident_mistake_first(known_model, known_data):
    X, y, paths = known_data
    mistakes = worst_examples(known_model, X, y, paths, k=10)

    assert mistakes[0]["path"].name == "3.jpg", (
        "Image 3 is the mistake the model was most sure about, so it goes first."
    )
    confidences = [m["confidence"] for m in mistakes]
    assert confidences == sorted(confidences, reverse=True)


def test_worst_examples_respects_k(known_model, known_data):
    X, y, paths = known_data
    assert len(worst_examples(known_model, X, y, paths, k=1)) == 1
