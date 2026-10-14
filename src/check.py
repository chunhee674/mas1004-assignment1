"""Check your own work. Run this before you submit.

    python src/check.py

It does not use any of the functions you wrote. It reads the files on disk and
the model you exported, and works everything out for itself. That is the point:
if it disagrees with the numbers your own code printed, one of the two is
wrong, and finding out which is part of the assignment.

Paste the whole output into your report.

    python src/check.py --compare run clean

also measures the models saved by those runs on your new images, which is the
fair way to see what cleaning did (Problem 4).
"""

import argparse
import base64
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEIC_SUFFIXES = {".heic", ".heif"}  # what iPhones save by default; unreadable here

problems = []
warnings = []


def title(text):
    print(f"\n{text}\n" + "-" * len(text))


def complain(text):
    problems.append(text)
    print(f"  PROBLEM  {text}")


def warn(text):
    warnings.append(text)
    print(f"  warning  {text}")


def good(text):
    print(f"  ok       {text}")


def image_files(folder):
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


# ---------------------------------------------------------------------------


def check_folder(name, folder, least_per_class, required):
    """required: fewer than least_per_class is a problem, not just a warning."""
    title(f"{name}  ({folder})")
    folder = Path(folder)
    if not folder.exists():
        complain(f"{folder} does not exist")
        return None

    classes = sorted(d for d in folder.iterdir() if d.is_dir())
    if not classes:
        complain(f"{folder} has no class sub-folders in it")
        return None

    counts = {}
    for class_dir in classes:
        files = image_files(class_dir)
        counts[class_dir.name] = len(files)
        print(f"  {class_dir.name:28s} {len(files):5d} images")

    heic = sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in HEIC_SUFFIXES)
    if heic:
        shown = ", ".join(str(p.relative_to(folder)) for p in heic[:3])
        complain(
            f"{len(heic)} HEIC photo(s) in {folder.name}/ cannot be read, so they "
            f"are not counted above: {shown}{' ...' if len(heic) > 3 else ''}. "
            "Convert them to JPEG (README, Problem 5)."
        )

    if len(counts) < 3:
        complain(f"only {len(counts)} classes. You need at least 3.")
    fewest = min(counts.values()) if counts else 0
    smallest = min(counts, key=counts.get)
    if fewest < least_per_class and required:
        complain(
            f'"{smallest}" has only {fewest} images, '
            f"fewer than the {least_per_class} this assignment asks for"
        )
    elif fewest < least_per_class:
        warn(
            f'"{smallest}" has only {fewest} images. With this few, the model '
            "has little to learn that class from, and its test accuracy rests "
            "on a handful of test images."
        )
    most = max(counts.values()) if counts else 0
    if fewest and most > fewest * 3:
        warn(
            f"the biggest class has {most} images and the smallest has {fewest}. "
            "A model can score well just by always guessing the big class."
        )
    return counts


def check_for_repeats(folders):
    title("The same image appearing twice")
    seen = defaultdict(list)
    for folder in folders:
        for path in image_files(folder):
            try:
                seen[hashlib.md5(path.read_bytes()).hexdigest()].append(path)
            except Exception:
                complain(f"could not read {path}")

    repeats = {k: v for k, v in seen.items() if len(v) > 1}
    if not repeats:
        good("no file appears twice byte for byte. Near copies (the same "
             "picture resized or re-saved) are only found by "
             "`python src/clean.py suspects` and `python src/clean.py overlap`.")
        return

    crossing = [
        group for group in repeats.values()
        if len({p.parent.parent.name for p in group}) > 1
        or len({p.parent.name for p in group}) > 1
    ]
    if crossing:
        complain(
            f"{len(crossing)} image(s) appear in more than one place, including "
            "across different classes or across your downloads and your new "
            "images. Test accuracy measured on an image the model trained on "
            "means nothing."
        )
        for group in crossing[:3]:
            print("           " + "  ==  ".join(str(p.relative_to(ROOT)) for p in group))
    else:
        warn(f"{len(repeats)} image(s) appear twice inside the same class")


# ---------------------------------------------------------------------------


def load_web_model():
    title("The model you exported for the web page")
    web = ROOT / "docs"
    missing = [
        name for name in ("model.onnx", "model.json", "selftest.json")
        if not (web / name).exists()
    ]
    if missing:
        complain(
            f"docs/ is missing {', '.join(missing)}. "
            "Run src/run.py and then src/export_web.py first."
        )
        return None

    try:
        import onnxruntime
    except ImportError:
        complain("onnxruntime is not installed. pip install -r requirements.txt")
        return None

    model = json.loads((web / "model.json").read_text())
    try:
        session = onnxruntime.InferenceSession(
            str(web / "model.onnx"), providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        complain(f"docs/model.onnx could not be opened: {error}")
        return None

    size = (web / "model.onnx").stat().st_size / 1e6
    good(f'{len(model["labels"])} classes: {", ".join(model["labels"])}')
    good(f'model.onnx is {size:.1f} MB, {model["n_parameters"]:,} parameters')
    shape = session.get_inputs()[0].shape
    print(f"           input {shape}, prepared as: shorter side "
          f'{model["input"]["resize"]}, centre {model["input"]["crop"]} x '
          f'{model["input"]["crop"]}, mean {model["input"]["mean"]}, '
          f'std {model["input"]["std"]}')
    check_committed(web / "model.onnx")
    return model, session


def check_committed(path):
    """Is the model file part of what you pushed? Pages can only serve that."""
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=ROOT, capture_output=True,
        ).returncode == 0
        if not inside:
            return  # not a git repository, so there is nothing to check
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=ROOT, capture_output=True,
        ).returncode == 0
        changed = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(path)],
            cwd=ROOT, capture_output=True,
        ).returncode != 0
    except (OSError, subprocess.SubprocessError):
        return  # git is not installed
    if not tracked:
        warn("docs/model.onnx is not committed. Your published page will not "
             "have a model until you commit and push it.")
    elif changed:
        warn("docs/model.onnx has changed since your last commit. The published "
             "page still shows the old model.")


def forward(session, x):
    """Run the exported network, the same file the browser runs."""
    return session.run(None, {session.get_inputs()[0].name: x[None]})[0][0]


def prepare(image, model):
    """The steps the web page does, written again from model.json."""
    settings = model["input"]
    image = image.convert("RGB")
    width, height = image.size
    short, long = min(width, height), max(width, height)
    new_long = int(settings["resize"] * long / short)
    new_size = (settings["resize"], new_long) if width <= height else (new_long, settings["resize"])
    image = image.resize(new_size, Image.BILINEAR)

    crop = settings["crop"]
    left = int(round((image.width - crop) / 2.0))
    top = int(round((image.height - crop) / 2.0))
    image = image.crop((left, top, left + crop, top + crop))

    values = np.asarray(image, dtype=np.float32) / 255.0
    values = (values - np.array(settings["mean"], dtype=np.float32)) / np.array(
        settings["std"], dtype=np.float32)
    return values.transpose(2, 0, 1).astype(np.float32)


def check_selftest(model, session):
    title("Does the exported model give the answers Python gave?")
    cases = json.loads((ROOT / "docs" / "selftest.json").read_text())["cases"]
    worst = 0.0
    for case in cases:
        image = Image.open(io.BytesIO(base64.b64decode(case["png"])))
        got = forward(session, prepare(image, model))
        worst = max(worst, float(np.abs(got - np.array(case["logits"])).max()))

    if worst < 0.02:
        good(f"yes, they agree (largest gap {worst:.4f})")
    else:
        complain(
            f"no, they differ by {worst:.3f}. The page prepares a photo the way "
            "model.json says, and your prepare_image did something else. The "
            "web page will be wrong too."
        )


def new_images_folder():
    """data/new_images, or data/my_photos, its name before 29 September."""
    new = ROOT / "data" / "new_images"
    old = ROOT / "data" / "my_photos"
    if not new.exists() and old.exists():
        warn("data/my_photos is the old name of this folder. Rename it to "
             "data/new_images.")
        return old
    return new


def check_new_images(model, session, folder):
    title(f"Your images from a new source  ({folder})")
    folder = Path(folder)
    if not folder.exists():
        complain(
            f"{folder} does not exist. Problem 5 asks for at least 5 images per "
            "class from a source you are sure is not in your downloads."
        )
        return

    labels = model["labels"]
    right = 0
    total = 0
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    confident_mistakes = []

    for class_dir in sorted(d for d in folder.iterdir() if d.is_dir()):
        if class_dir.name not in labels:
            complain(
                f'"{class_dir.name}" is not one of your classes. '
                f'Name the folders exactly: {", ".join(labels)}'
            )
            continue
        true_index = labels.index(class_dir.name)
        files = image_files(class_dir)
        if len(files) < 5:
            warn(f'"{class_dir.name}" has only {len(files)} new images')

        for path in files:
            try:
                logits = forward(session, prepare(Image.open(path), model))
            except Exception as error:
                complain(f"could not read {path}: {error}")
                continue
            shifted = logits - logits.max()
            probabilities = np.exp(shifted) / np.exp(shifted).sum()
            guess = int(np.argmax(logits))

            total += 1
            matrix[true_index][guess] += 1
            if guess == true_index:
                right += 1
            else:
                confident_mistakes.append(
                    (float(probabilities[guess]), path.name,
                     labels[guess], class_dir.name)
                )

    if total == 0:
        complain("no readable images found")
        return

    print(f"\n  accuracy on your new images: {right}/{total} = {right / total:.1%}")
    print("\n  rows are what it really is, columns are what the model said")
    width = max(len(name) for name in labels) + 2
    print(" " * (width + 4) + "".join(f"{name[:8]:>9s}" for name in labels))
    for i, name in enumerate(labels):
        print(f"    {name:<{width}s}" + "".join(f"{int(v):9d}" for v in matrix[i]))

    if confident_mistakes:
        confident_mistakes.sort(reverse=True)
        print("\n  the mistakes it was most sure about:")
        for confidence, name, said, really in confident_mistakes[:5]:
            print(f"    {name:28s} said {said} ({confidence:.0%}), really {really}")


# ---------------------------------------------------------------------------


IMAGENET_INPUT = {"resize": 256, "crop": 224,
                  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def compare_on_new_images(tags, folder):
    """Measure several saved runs on the same new images.

    Cleaning data/clean changes the test set as well as the training set, so
    the test accuracy before and after cleaning is measured on different
    images. Your new images stay the same, so they are a fair comparison.
    """
    import torch

    title(f"Your saved runs on your new images  ({folder})")
    folder = Path(folder)
    if not folder.exists():
        complain(f"{folder} does not exist")
        return
    settings_input = {"input": IMAGENET_INPUT}

    print(f"  {'run':16s} {'test accuracy':>14s} {'new images':>12s}")
    for tag in tags:
        model_path = ROOT / "results" / f"{tag}_model.pt"
        settings_path = ROOT / "results" / f"{tag}_settings.json"
        if not model_path.exists() or not settings_path.exists():
            complain(f"results/{tag}_model.pt is missing. Run src/run.py --tag {tag}")
            continue
        settings = json.loads(settings_path.read_text())
        labels = settings["class_names"]
        model = torch.load(model_path, map_location="cpu", weights_only=False).eval()

        right = total = 0
        for class_dir in sorted(d for d in folder.iterdir() if d.is_dir()):
            if class_dir.name not in labels:
                continue
            for path in image_files(class_dir):
                try:
                    x = prepare(Image.open(path), settings_input)
                except Exception:
                    continue
                with torch.no_grad():
                    guess = int(model(torch.from_numpy(x[None])).argmax())
                right += guess == labels.index(class_dir.name)
                total += 1
        if total:
            print(f"  {tag:16s} {settings['test_accuracy']:14.1%} "
                  f"{right / total:12.1%}   ({right}/{total})")


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare", nargs="+", metavar="TAG",
        help="also measure these saved runs on your new images, "
             "for example --compare run clean",
    )
    args = parser.parse_args()

    print("MAS1004 Assignment 1, checking your work")

    check_folder("Training images", ROOT / "data" / "clean", least_per_class=50,
                 required=False)
    new_images = new_images_folder()
    check_folder("Your images from a new source", new_images, least_per_class=5,
                 required=True)
    check_for_repeats([ROOT / "data" / "clean", new_images])

    loaded = load_web_model()
    if loaded:
        model, session = loaded
        check_selftest(model, session)
        check_new_images(model, session, new_images)
    if args.compare:
        compare_on_new_images(args.compare, new_images)

    title("Summary")
    if problems:
        print(f"  {len(problems)} thing(s) to fix:")
        for item in problems:
            print(f"    - {item}")
    else:
        print("  nothing is broken")
    if warnings:
        print(f"  {len(warnings)} thing(s) worth a sentence in your report:")
        for item in warnings:
            print(f"    - {item}")
    print()
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
