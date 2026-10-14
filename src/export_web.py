"""Put a trained model on your web page.

This file is complete. You do not need to change it.

    python src/export_web.py --tag clean

reads results/clean_model.pt, the model that `python src/run.py --tag clean`
trained, and writes three files into docs/:

    model.onnx     the whole network, in the ONNX format the page can run.
                   About 45 MB for ResNet18.
    model.json     your class names and how a photo has to be prepared
    selftest.json  three of your test images plus the answers Python gave

The self test is the important one. When the page opens it runs those three
images through the network in the browser and compares the result with what
Python got. If the two disagree, the badge at the top of the page turns red.
That means the page prepares a photo differently from the way your
prepare_image did during training, and every answer on the page is wrong.

Export only the model you want to publish. Every time you commit a new
model.onnx, another 45 MB goes into the history of your repository.
"""

import argparse
import base64
import copy
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import CROP, MEAN, RESIZE, STD, prepare_image
from train import load

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "mas1004-onnx-v1"


def selftest_picture(path):
    """One of your images as a RESIZE x RESIZE square.

    It is already the size the page resizes to, so the page only has to cut
    out the middle and scale the numbers. What the self test checks is those
    steps, and the network itself.
    """
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = RESIZE / min(width, height)
    image = image.resize(
        (max(RESIZE, round(width * scale)), max(RESIZE, round(height * scale))),
        Image.BILINEAR,
    )
    left = (image.width - RESIZE) // 2
    top = (image.height - RESIZE) // 2
    return image.crop((left, top, left + RESIZE, top + RESIZE))


def write_onnx(model, path):
    example = torch.zeros(1, 3, CROP, CROP)
    torch.onnx.export(
        model, (example,), str(path),
        input_names=["image"], output_names=["logits"],
        dynamo=True, external_data=False,
    )


def run_onnx(path, x):
    import onnxruntime
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, {"image": np.asarray(x, dtype=np.float32)})[0]


def export(model, class_names, sample_paths, prepare=prepare_image, out_dir="docs"):
    """Write model.onnx, model.json and selftest.json.

    Arguments
        model         a trained model that takes (1, 3, 224, 224)
        class_names   the list load_folder gave you, in the same order
        sample_paths  image files, of which the first three go in the self test
        prepare       the function that prepared your training images
        out_dir       where to write. Leave this alone.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, CROP, CROP)).shape[-1]
    if outputs != len(class_names):
        raise ValueError(
            f"The model has {outputs} outputs but you gave {len(class_names)} "
            "class names."
        )

    pictures = [selftest_picture(p) for p in list(sample_paths)[:3]]
    if not pictures:
        raise ValueError("Give at least one image for the self test.")
    inputs = np.stack([prepare(picture) for picture in pictures]).astype(np.float32)
    if inputs.shape[1:] != (3, CROP, CROP):
        raise ValueError(
            f"prepare_image gave shape {inputs.shape[1:]}, not (3, {CROP}, {CROP})."
        )
    with torch.no_grad():
        expected = model(torch.from_numpy(inputs)).numpy()

    onnx_path = out_path / "model.onnx"
    write_onnx(model, onnx_path)

    # The ONNX file has to give the answers PyTorch gave, or nothing after
    # this point means anything.
    replayed = np.concatenate([run_onnx(onnx_path, row[None]) for row in inputs])
    gap = float(np.abs(replayed - expected).max())
    if gap > 1e-3:
        raise RuntimeError(
            f"model.onnx answers differently from PyTorch (gap {gap:.4f})."
        )

    (out_path / "model.json").write_text(json.dumps({
        "format": FORMAT,
        "labels": list(class_names),
        "input": {"name": "image", "resize": RESIZE, "crop": CROP,
                  "mean": list(MEAN), "std": list(STD)},
        "output": {"name": "logits"},
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }, indent=2))

    cases = []
    for picture, logits in zip(pictures, expected):
        buffer = io.BytesIO()
        picture.save(buffer, "PNG")
        cases.append({
            "png": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "logits": [round(float(v), 4) for v in logits],
        })
    (out_path / "selftest.json").write_text(json.dumps({"cases": cases}))

    megabytes = onnx_path.stat().st_size / 1e6
    print(f"wrote {onnx_path}   {megabytes:.1f} MB")
    print(f"wrote {out_path / 'model.json'}   {', '.join(class_names)}")
    print(f"wrote {out_path / 'selftest.json'}   {len(cases)} images, "
          f"ONNX agrees with PyTorch (largest gap {gap:.6f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True,
                        help="the --tag of the run.py run you want to publish")
    args = parser.parse_args()

    results = ROOT / "results"
    model_path = results / f"{args.tag}_model.pt"
    settings_path = results / f"{args.tag}_settings.json"
    if not model_path.exists() or not settings_path.exists():
        raise SystemExit(
            f"There is no {model_path.name} in results/. Run "
            f"`python src/run.py --tag {args.tag}` first."
        )

    settings = json.loads(settings_path.read_text())
    export(load(model_path), settings["class_names"], settings["test_paths"],
           out_dir=ROOT / "docs")
    print(f"\nThe page now shows the model from the run tagged {args.tag!r} "
          f"(test accuracy {settings['test_accuracy']:.1%}).")
    print("Look at it with: python -m http.server -d docs 8000")


if __name__ == "__main__":
    main()
