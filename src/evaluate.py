"""Measure the model and look at what it got wrong.

YOU write the four functions at the top. The three plotting functions at the
bottom are already written, because fighting with matplotlib teaches you
nothing about machine learning.

Run `pytest tests/test_evaluate.py` after you fill them in.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, never to a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def predict_logits(model, X, batch_size=64):
    """Run the model over X and return its raw outputs.

    Arguments
        X  np.float32 (N, ...), for example (N, 3, 224, 224)

    Returns np.float32 (N, C), where C is the number of classes, as a numpy
    array in ordinary memory. These are raw outputs, not probabilities. Do not
    apply softmax here.

    Put the model in eval mode and do not compute gradients. The model may be
    on a GPU: send each batch to the device the model's parameters are on,
    next(model.parameters()).device, and bring the answers back with .cpu().
    Work in batches so that a large test set does not run you out of memory.
    """
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.asarray(X[start:start + batch_size]))
            outputs.append(model(batch.to(device)).cpu().numpy())
    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32)


def accuracy(model, X, y):
    """Return the share of rows the model gets right, as a float 0.0 to 1.0."""
    predicted = predict_logits(model, X).argmax(axis=1)
    return float((predicted == np.asarray(y)).mean())


def confusion_matrix(model, X, y, num_classes):
    """Return an np.int64 array of shape (num_classes, num_classes).

    Row is the true class, column is the predicted class. So element [i][j] is
    how many images of class i the model called class j. The diagonal is the
    ones it got right.
    """
    predicted = predict_logits(model, X).argmax(axis=1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(np.asarray(y), predicted):
        matrix[int(true)][int(pred)] += 1
    return matrix


def worst_examples(model, X, y, paths, k=10):
    """Return the k wrong answers the model was most confident about.

    Returns a list of k dicts, sorted so that the most confident mistake comes
    first. Each dict has these keys:
        "path"        the Path this image came from
        "true"        int, the true class index
        "predicted"   int, the class index the model chose
        "confidence"  float, the softmax probability it gave to "predicted"

    Only wrong answers belong in this list. If the model made fewer than k
    mistakes, return fewer than k items.

    These are the images to put in your report. A mistake the model was sure
    about tells you much more than a mistake it was unsure about.
    """
    logits = predict_logits(model, X)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    predicted = probs.argmax(axis=1)

    mistakes = []
    for i in range(len(predicted)):
        if int(predicted[i]) != int(y[i]):
            mistakes.append({
                "path": Path(paths[i]),
                "true": int(y[i]),
                "predicted": int(predicted[i]),
                "confidence": float(probs[i][predicted[i]]),
            })
    mistakes.sort(key=lambda m: m["confidence"], reverse=True)
    return mistakes[:k]


# ---------------------------------------------------------------------------
# Everything below is written for you.
# ---------------------------------------------------------------------------


def plot_history(history, out_path):
    """Draw the loss and accuracy curves from what train() returned."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    left.plot(epochs, history["train_loss"], label="train")
    left.plot(epochs, history["test_loss"], label="test")
    left.set_xlabel("epoch")
    left.set_ylabel("loss")
    left.set_title("Loss")
    left.legend()

    right.plot(epochs, history["train_acc"], label="train")
    right.plot(epochs, history["test_acc"], label="test")
    right.set_xlabel("epoch")
    right.set_ylabel("accuracy")
    right.set_ylim(0, 1)
    right.set_title("Accuracy")
    right.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_confusion(matrix, class_names, out_path):
    """Draw a confusion matrix with the counts written inside the squares."""
    size = len(class_names)
    fig, axes = plt.subplots(figsize=(1.4 * size + 2, 1.4 * size + 2))
    axes.imshow(matrix, cmap="Blues")

    axes.set_xticks(range(size), class_names, rotation=45, ha="right")
    axes.set_yticks(range(size), class_names)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")

    highest = matrix.max() if matrix.max() > 0 else 1
    for i in range(size):
        for j in range(size):
            axes.text(
                j, i, str(matrix[i][j]),
                ha="center", va="center",
                color="white" if matrix[i][j] > highest * 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_worst(mistakes, class_names, out_path, ncols=5):
    """Draw the images from worst_examples with what the model said."""
    if not mistakes:
        print("no mistakes to plot, which is suspicious. Check your split.")
        return

    nrows = (len(mistakes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for slot, mistake in enumerate(mistakes):
        axes[slot].imshow(Image.open(mistake["path"]).convert("RGB"))
        axes[slot].set_title(
            f'said {class_names[mistake["predicted"]]}'
            f' ({mistake["confidence"]:.0%})\n'
            f'really {class_names[mistake["true"]]}',
            fontsize=9,
        )
    for slot in range(len(mistakes), len(axes)):
        axes[slot].set_visible(False)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")
