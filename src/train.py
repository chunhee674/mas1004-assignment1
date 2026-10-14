"""Build a model and train it. YOU write the bodies of these functions.

The model is ResNet18, a convolutional network that was already trained on
ImageNet, 1.2 million photographs of 1,000 kinds of thing. You keep everything
it learned there and replace only its last layer, so that it answers with your
categories instead of those 1,000. Then you train it further on your images.

Run `pytest tests/test_train.py` after you fill them in. The first run
downloads the ImageNet weights, about 45 MB.
"""

import torch
import torch.nn as nn


def build_model(num_classes, pretrained=True, freeze=False):
    """Return a ResNet18 whose last layer has `num_classes` outputs.

    Arguments
        num_classes  int, how many categories you have
        pretrained   True: start from the ImageNet weights,
                     torchvision.models.ResNet18_Weights.IMAGENET1K_V1.
                     False: start from random numbers, as if ImageNet had
                     never happened. You need this for the comparison in
                     Problem 3.
        freeze       True: only the new last layer is trained. Every other
                     parameter keeps the value it came with, which you do by
                     setting requires_grad to False on it.
                     False: every parameter is trained.

    Returns the model from torchvision.models.resnet18, with its `fc`
    attribute replaced by a new nn.Linear that has 512 inputs and num_classes
    outputs.

    Do not put a softmax at the end. The loss function adds it for you, and the
    web page adds it for you.
    """
    import torchvision

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)

    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # The new layer is created after freezing, so it is always trainable.
    model.fc = nn.Linear(512, num_classes)
    return model


def train(model, X_train, y_train, X_test, y_test,
          epochs=10, lr=1e-4, batch_size=32):
    """Train the model and report what happened at every epoch.

    Arguments
        model        what build_model returned, or any other torch model
        X_train      np.float32 (N, ...)      y_train  np.int64 (N,)
        X_test       np.float32 (M, ...)      y_test   np.int64 (M,)
        epochs       int, how many times to go through the training set
        lr           float, the learning rate
        batch_size   int, how many rows per step

    Returns a dict named history with four keys. Each value is a list of
    `epochs` numbers, one per epoch:
        "train_loss"  the mean loss over the batches of that epoch
        "train_acc"   the share of training rows the model got right in those
                      batches, between 0.0 and 1.0
        "test_loss"   the mean loss over the test set, measured after the epoch
        "test_acc"    the share of the test set it gets right, measured after
                      the epoch, between 0.0 and 1.0

    What to do
        Use a GPU when there is one:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Move the model there once, and each batch there as you use it. Keep
        X_train and X_test themselves in ordinary memory: they may not fit on
        the GPU.
        Shuffle the training rows again at the start of every epoch.
        Use nn.CrossEntropyLoss, which expects raw outputs, and torch.optim.Adam
        over the parameters that have requires_grad set.
        Call model.train() before the training batches and model.eval() before
        measuring the test set. ResNet contains BatchNorm layers, which behave
        differently in the two modes.
        Do not compute gradients while measuring the test set, and never let
        the test rows change the parameters.
        Print one line per epoch, so that you can watch it while it runs.

    The model is trained in place. When this function returns, `model` is the
    trained one, and it is left on the device it was trained on.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    n_train = len(X_train_t)
    n_test = len(X_test_t)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_train)
        loss_sum = 0.0
        correct = 0
        batches = 0
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += (out.argmax(dim=1) == yb).sum().item()
            batches += 1

        train_loss = loss_sum / max(batches, 1)
        train_acc = correct / max(n_train, 1)

        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_batches = 0
        with torch.no_grad():
            for start in range(0, n_test, batch_size):
                xb = X_test_t[start:start + batch_size].to(device)
                yb = y_test_t[start:start + batch_size].to(device)
                out = model(xb)
                test_loss_sum += loss_fn(out, yb).item()
                test_correct += (out.argmax(dim=1) == yb).sum().item()
                test_batches += 1

        test_loss = test_loss_sum / max(test_batches, 1)
        test_acc = test_correct / max(n_test, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"epoch {epoch:2d}/{epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.3f}  "
            f"test loss {test_loss:.4f} acc {test_acc:.3f}"
        )

    return history


def count_trainable(model):
    """How many parameters training is allowed to change. Written for you."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(model, path):
    """Save a trained model. Written for you."""
    torch.save(model, path)


def load(path):
    """Load a model saved by `save`, onto the CPU. Written for you."""
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model
