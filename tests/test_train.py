"""Tests for Problem 3. Run: pytest tests/test_train.py

The build_model tests download the ImageNet weights the first time, about
45 MB. The train tests use a tiny model instead of ResNet18, so they finish in
seconds even without a GPU.
"""

import copy

import pytest
import torch
import torch.nn as nn

from train import build_model, count_trainable, train


@pytest.fixture(scope="module")
def imagenet_weights():
    from torchvision.models import ResNet18_Weights
    return ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------


def test_last_layer_has_one_output_per_class():
    model = build_model(4)
    assert isinstance(model.fc, nn.Linear), (
        "model.fc should be an nn.Linear. Replace it, do not wrap it in "
        "anything, and do not add a softmax."
    )
    assert model.fc.in_features == 512 and model.fc.out_features == 4, (
        f"The last layer should be Linear(512, 4), got Linear("
        f"{model.fc.in_features}, {model.fc.out_features})"
    )


def test_output_shape():
    model = build_model(3).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 3), (
        f"Two images should give (2, 3) scores, got {tuple(out.shape)}"
    )


def test_there_is_no_softmax_at_the_end():
    model = build_model(3).eval()
    with torch.no_grad():
        rows = model(torch.randn(2, 3, 224, 224))
    assert not torch.allclose(rows.sum(dim=1), torch.ones(2), atol=1e-3), (
        "The outputs add up to 1, so there is a softmax in the model. Remove it."
    )


def test_pretrained_starts_from_the_imagenet_weights(imagenet_weights):
    model = build_model(3, pretrained=True)
    assert torch.equal(model.conv1.weight, imagenet_weights["conv1.weight"]), (
        "With pretrained=True the first layer should hold the ImageNet weights. "
        "Pass weights=ResNet18_Weights.IMAGENET1K_V1 to resnet18."
    )


def test_not_pretrained_starts_from_random_weights(imagenet_weights):
    model = build_model(3, pretrained=False)
    assert not torch.equal(model.conv1.weight, imagenet_weights["conv1.weight"]), (
        "With pretrained=False the model must not load the ImageNet weights."
    )


def test_everything_is_trained_by_default():
    model = build_model(3)
    assert all(p.requires_grad for p in model.parameters()), (
        "With freeze=False every parameter should have requires_grad True."
    )


def test_freeze_trains_only_the_last_layer():
    model = build_model(3, freeze=True)
    assert count_trainable(model) == 512 * 3 + 3, (
        "With freeze=True only the new last layer, 512 x 3 weights and 3 "
        f"biases, should be trainable. {count_trainable(model):,} are."
    )


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def tiny_model():
    return nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3))


def test_history_has_the_right_shape(blobs):
    X, y = blobs
    history = train(tiny_model(), X, y, X, y, epochs=4, lr=0.05, batch_size=16)

    for key in ("train_loss", "test_loss", "train_acc", "test_acc"):
        assert key in history, f"history is missing the key {key!r}"
        assert len(history[key]) == 4, (
            f"history[{key!r}] should have one number per epoch (4), "
            f"got {len(history[key])}"
        )
    for key in ("train_acc", "test_acc"):
        assert all(0.0 <= v <= 1.0 for v in history[key]), (
            f"history[{key!r}] must be between 0 and 1, not a percentage."
        )


def test_training_actually_changes_the_model(blobs):
    X, y = blobs
    model = tiny_model()
    before = copy.deepcopy(model.state_dict())
    train(model, X, y, X, y, epochs=3, lr=0.05, batch_size=16)

    changed = any(
        not torch.equal(before[name], value.cpu())
        for name, value in model.state_dict().items()
    )
    assert changed, (
        "The parameters are exactly what they were before training. "
        "Did you forget optimiser.step(), or train a copy of the model?"
    )


def test_frozen_parameters_stay_where_they_are(blobs):
    X, y = blobs
    model = tiny_model()
    for p in model[0].parameters():
        p.requires_grad = False
    before = model[0].weight.detach().clone()
    train(model, X, y, X, y, epochs=3, lr=0.05, batch_size=16)
    assert torch.equal(model[0].weight.detach().cpu(), before), (
        "A layer with requires_grad False was changed by training. Give the "
        "optimiser only the parameters that have requires_grad set."
    )


def test_training_learns_an_easy_problem(blobs):
    X, y = blobs
    history = train(tiny_model(), X, y, X, y, epochs=40, lr=0.05, batch_size=16)

    assert history["train_loss"][-1] < history["train_loss"][0], (
        "The loss did not go down at all over 40 epochs."
    )
    assert history["test_acc"][-1] > 0.9, (
        "These are three well separated blobs. A working training loop reaches "
        f"over 90% on them. Yours reached {history['test_acc'][-1]:.0%}."
    )
