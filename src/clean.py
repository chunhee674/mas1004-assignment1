"""Clean your downloaded images. Problem 4 is done with this file.

This file is complete. You do not need to change it. It does not use any of
the functions you wrote, so it works before your model does.

Every image in data/clean got its label from a search engine: you typed
"wine glass", and whatever came back is now called wine_glass. Some of it is a
drawing, a logo, a bottle, a collage, or the same picture four times. The model
learns from all of it as if it were true, and your test set is full of it too.
Cleaning is deciding, image by image, what belongs, and it is the part of this
assignment that matters most.

    python src/clean.py look
        Draws every image in data/clean onto numbered sheets in
        results/cleaning/look/, with its file name under it. Open them and
        scroll through all of them. Do this first, before any model is
        involved.

    python src/clean.py suspects
        Asks the rest of your data about each image. It describes every image
        with a network trained on ImageNet, then for each image trains a small
        classifier on the other images only and asks it for this image's
        label. An image whose own label gets a low probability looks unlike
        the rest of its class. It draws, for each class, the images with the
        lowest probability first, and a sheet of pairs of images that are
        near copies of each other, into results/cleaning/.
        A suspect is not automatically junk. Look at it and apply your rule.

    python src/clean.py remove wine_glass/0012.jpg wine_glass/0101.jpg --reason "drawing"
        Moves those files out of data/clean into data/removed/, and writes each
        one into data/removed/log.csv with your reason. Nothing is deleted, so
        you can always put a file back by moving it yourself.

    python src/clean.py count
        How many images each class had, how many you removed, and for what
        reasons. This is the table your report asks for.

    python src/clean.py overlap
        Problem 5. Compares every image in data/new_images with every image
        you downloaded, in data/raw and data/clean, and draws the pairs that are
        near copies into results/cleaning/overlap.png. A new image that is a
        copy of a downloaded one is not new, so take it out of data/new_images.
"""

import argparse
import csv
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw"
NEW = ROOT / "data" / "new_images"
REMOVED = ROOT / "data" / "removed"
OUT = ROOT / "results" / "cleaning"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

THUMB = 180         # size of one picture on a sheet, in pixels
COLUMNS = 8
PER_SHEET = 48
SUSPECTS = 32       # how many suspects to draw per class
SAME = 0.95         # similarity above which two images count as near copies


def images_by_class(folder):
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {
        d.name: sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        for d in sorted(folder.iterdir()) if d.is_dir()
    }


# ---------------------------------------------------------------------------
# Drawing sheets
# ---------------------------------------------------------------------------


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow older than 10.1
        return ImageFont.load_default()


def thumbnail(path):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((THUMB, THUMB))
            return image
    except Exception:
        broken = Image.new("RGB", (THUMB, THUMB), (235, 235, 235))
        ImageDraw.Draw(broken).text((10, THUMB // 2), "cannot open", fill=(160, 0, 0),
                                    font=font(16))
        return broken


def draw_sheet(items, title, out_path):
    """items: a list of (path, caption lines). Draws them in a grid."""
    rows = (len(items) + COLUMNS - 1) // COLUMNS
    cell_w, cell_h = THUMB + 12, THUMB + 64
    sheet = Image.new("RGB", (COLUMNS * cell_w + 12, rows * cell_h + 56), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill="black", font=font(26))

    for slot, (path, lines) in enumerate(items):
        x = 12 + (slot % COLUMNS) * cell_w
        y = 56 + (slot // COLUMNS) * cell_h
        picture = thumbnail(path)
        sheet.paste(picture, (x + (THUMB - picture.width) // 2,
                              y + (THUMB - picture.height) // 2))
        for number, line in enumerate(lines):
            draw.text((x, y + THUMB + 4 + number * 18), line, fill="black", font=font(15))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def look(args):
    classes = images_by_class(CLEAN)
    if not classes:
        sys.exit(f"{CLEAN} has no class folders. Copy data/raw to data/clean first.")
    folder = OUT / "look"
    if folder.exists():
        shutil.rmtree(folder)
    for name, paths in classes.items():
        pages = (len(paths) + PER_SHEET - 1) // PER_SHEET
        for page in range(pages):
            chunk = paths[page * PER_SHEET:(page + 1) * PER_SHEET]
            items = [(p, [f"{name}/{p.name}"]) for p in chunk]
            out = draw_sheet(
                items, f"{name}   sheet {page + 1} of {pages}   ({len(paths)} images)",
                folder / f"{name}_{page + 1:02d}.png",
            )
            print(f"wrote {out.relative_to(ROOT)}")
    print("\nOpen every sheet and look at every picture. Write your rule down "
          "before you remove anything.")


# ---------------------------------------------------------------------------
# Asking the rest of the data
# ---------------------------------------------------------------------------


def describe(paths):
    """512 numbers per image from ResNet18 trained on ImageNet, one row each.

    Rows are cached in results/cleaning/features.npz, so a second run only
    reads the images that are new.
    """
    import torch
    from torchvision import models

    cache_path = OUT / "features.npz"
    cache = {}
    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(stored["keys"].tolist(), stored["rows"]))

    def key(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    keys = [key(p) for p in paths]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        network = models.resnet18(weights=weights)
        network.fc = torch.nn.Identity()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network.to(device).eval()
        steps = weights.transforms()
        print(f"describing {len(todo)} images with ResNet18 ...")
        with torch.no_grad():
            for start in range(0, len(todo), 32):
                batch, kept = [], []
                for i in todo[start:start + 32]:
                    try:
                        with Image.open(paths[i]) as image:
                            batch.append(steps(image.convert("RGB")))
                        kept.append(i)
                    except Exception:
                        cache[keys[i]] = np.full(512, np.nan, dtype=np.float32)
                if batch:
                    rows = network(torch.stack(batch).to(device)).cpu().numpy()
                    for i, row in zip(kept, rows):
                        cache[keys[i]] = row.astype(np.float32)
                print(f"  {min(start + 32, len(todo))}/{len(todo)}")
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.array(list(cache)), rows=np.stack(list(cache.values())))
    return np.stack([cache[k] for k in keys])


def out_of_fold_probabilities(features, labels, num_classes, folds=5, seed=0):
    """For every image, the probability of its own label, from a classifier
    that was trained on the other images and never saw this one."""
    import torch

    x = torch.from_numpy(features / np.linalg.norm(features, axis=1, keepdims=True))
    y = torch.from_numpy(labels)
    order = np.random.default_rng(seed).permutation(len(labels))
    probabilities = np.zeros((len(labels), num_classes), dtype=np.float32)

    for fold in range(folds):
        held = order[fold::folds]
        rest = np.setdiff1d(order, held)
        torch.manual_seed(seed)
        layer = torch.nn.Linear(x.shape[1], num_classes)
        optimiser = torch.optim.Adam(layer.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(300):
            optimiser.zero_grad()
            loss = torch.nn.functional.cross_entropy(layer(x[rest] * 10), y[rest])
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            probabilities[held] = torch.softmax(layer(x[held] * 10), dim=1).numpy()
    return probabilities


def suspects(args):
    classes = images_by_class(CLEAN)
    if not classes:
        sys.exit(f"{CLEAN} has no class folders. Copy data/raw to data/clean first.")
    names = list(classes)
    paths = [p for name in names for p in classes[name]]
    labels = np.array([names.index(p.parent.name) for p in paths], dtype=np.int64)

    features = describe(paths)
    readable = ~np.isnan(features).any(axis=1)
    for path in np.array(paths, dtype=object)[~readable]:
        print(f"  cannot open {path.relative_to(CLEAN)}: remove it")
    paths = [p for p, ok in zip(paths, readable) if ok]
    features, labels = features[readable], labels[readable]

    probabilities = out_of_fold_probabilities(features, labels, len(names))
    own = probabilities[np.arange(len(labels)), labels]
    guess = probabilities.argmax(1)

    if (OUT / "suspects").exists():
        shutil.rmtree(OUT / "suspects")
    rows = []
    for index, name in enumerate(names):
        members = np.flatnonzero(labels == index)
        ranked = members[np.argsort(own[members])]
        items = []
        for rank, i in enumerate(ranked[:SUSPECTS], start=1):
            lines = [f"{rank}. {paths[i].name}", f"{name}: {own[i]:.0%}"]
            if guess[i] != index:
                lines.append(f"looks like {names[guess[i]]}")
            items.append((paths[i], lines))
        out = draw_sheet(
            items, f"{name}: the {len(items)} images that look least like the rest "
                   f"of {name}", OUT / "suspects" / f"{name}.png")
        print(f"wrote {out.relative_to(ROOT)}")
        for rank, i in enumerate(ranked, start=1):
            rows.append([f"{name}/{paths[i].name}", name, rank, f"{own[i]:.4f}",
                         names[guess[i]]])

    # Near copies: the same picture resized, re-encoded, or with a watermark.
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, 0)
    first, second = np.nonzero(np.triu(similarity) > SAME)
    pairs = sorted(zip(first, second), key=lambda ij: -similarity[ij])
    items = []
    for a, b in pairs[:PER_SHEET // 2]:
        items.append((paths[a], [f"{paths[a].parent.name}/{paths[a].name}",
                                 f"same as next, {similarity[a, b]:.2f}"]))
        items.append((paths[b], [f"{paths[b].parent.name}/{paths[b].name}", ""]))
    if items:
        out = draw_sheet(items, f"{len(pairs)} pairs of near copies "
                                f"(similarity above {SAME})", OUT / "suspects" / "copies.png")
        print(f"wrote {out.relative_to(ROOT)}")
    else:
        print("no near copies found")

    with open(OUT / "suspects" / "suspects.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "label", "rank_in_class", "probability_of_label",
                         "looks_like"])
        writer.writerows(rows)
        for a, b in pairs:
            writer.writerow([f"{paths[a].parent.name}/{paths[a].name}", "copy of",
                             f"{paths[b].parent.name}/{paths[b].name}",
                             f"{similarity[a, b]:.4f}", ""])
    print(f"wrote {(OUT / 'suspects' / 'suspects.csv').relative_to(ROOT)}")

    print(f"\nAgreement with the label, over all {len(labels)} images: "
          f"{(guess == labels).mean():.1%} would be labelled the same by the rest "
          f"of your data.")
    print("Look at each suspect and decide by your rule. Some are fine pictures "
          "that are just unusual, and those are worth keeping.")


# ---------------------------------------------------------------------------
# Removing, and counting what you removed
# ---------------------------------------------------------------------------


def remove(args):
    log = REMOVED / "log.csv"
    new_log = not log.exists()
    REMOVED.mkdir(parents=True, exist_ok=True)
    moved = 0
    with open(log, "a", newline="") as handle:
        writer = csv.writer(handle)
        if new_log:
            writer.writerow(["file", "class", "reason", "when"])
        for name in args.files:
            source = CLEAN / name
            if not source.exists():
                print(f"  not found: data/clean/{name}. Write it as class/file, "
                      "for example wine_glass/0012.jpg")
                continue
            target = REMOVED / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            writer.writerow([name, source.parent.name, args.reason,
                             datetime.now().isoformat(timespec="seconds")])
            moved += 1
            print(f"  moved {name}")
    print(f"moved {moved} file(s) into data/removed/, reason: {args.reason}")


def count(args):
    downloaded = {k: len(v) for k, v in images_by_class(RAW).items()}
    kept = {k: len(v) for k, v in images_by_class(CLEAN).items()}
    removed = defaultdict(Counter)
    log = REMOVED / "log.csv"
    if log.exists():
        with open(log) as handle:
            for row in csv.DictReader(handle):
                if (REMOVED / row["file"]).exists():
                    removed[row["class"]][row["reason"]] += 1

    print(f"{'class':24s} {'downloaded':>10s} {'removed':>8s} {'kept':>6s} {'other':>6s}")
    for name in sorted(set(downloaded) | set(kept)):
        gone = sum(removed[name].values())
        other = downloaded.get(name, 0) - gone - kept.get(name, 0)
        print(f"{name:24s} {downloaded.get(name, 0):10d} {gone:8d} "
              f"{kept.get(name, 0):6d} {other:6d}")
    reasons = Counter()
    for tally in removed.values():
        reasons.update(tally)
    if reasons:
        print("\nreasons")
        for reason, number in reasons.most_common():
            print(f"  {number:4d}  {reason}")
    print("\n\"other\" is images that are in data/raw but neither in data/clean nor "
          "removed with `clean.py remove`, for example ones you deleted by hand. "
          "Their reasons are not recorded.")


def overlap(args):
    heic = sorted(p for p in NEW.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".heic", ".heif"})
    if heic:
        print(f"{len(heic)} HEIC photo(s) in data/new_images cannot be read and are "
              "left out. Convert them to JPEG (README, Problem 5).\n")
    new = [p for paths in images_by_class(NEW).values() for p in paths]
    if not new:
        sys.exit(f"{NEW} has no images in class folders yet.")
    seen, downloaded = set(), []
    for folder in (RAW, CLEAN, REMOVED):
        for paths in images_by_class(folder).values():
            for path in paths:
                digest = hashlib.md5(path.read_bytes()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    downloaded.append(path)

    features = describe(new + downloaded)
    readable = ~np.isnan(features).any(axis=1)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    similarity = unit[:len(new)] @ unit[len(new):].T
    similarity[~readable[:len(new)]] = 0
    similarity[:, ~readable[len(new):]] = 0

    closest = similarity.argmax(1)
    scores = similarity[np.arange(len(new)), closest]
    copies = [i for i in np.argsort(-scores) if scores[i] > SAME]

    items = []
    for i in copies[:PER_SHEET // 2]:
        j = closest[i]
        items.append((new[i], [f"{new[i].parent.name}/{new[i].name}", "new"]))
        items.append((downloaded[j], [f"{downloaded[j].parent.name}/{downloaded[j].name}",
                                      f"downloaded, {scores[i]:.2f}"]))
    if (OUT / "overlap.png").exists():
        (OUT / "overlap.png").unlink()
    if items:
        verb = "is a near copy" if len(copies) == 1 else "are near copies"
        out = draw_sheet(items, f"{len(copies)} of your new images {verb} of "
                                f"downloaded ones", OUT / "overlap.png")
        print(f"wrote {out.relative_to(ROOT)}")
        for i in copies:
            print(f"  data/new_images/{new[i].parent.name}/{new[i].name}  looks like  "
                  f"{downloaded[closest[i]].relative_to(ROOT)}  ({scores[i]:.2f})")
        print("\nThese are not new. Take them out of data/new_images, and think "
              "about whether the rest of that source is really separate.")
    else:
        print(f"none of your {len(new)} new images is a near copy of any of the "
              f"{len(downloaded)} images you downloaded")
    print(f"\nthe closest any new image comes to a downloaded one: "
          f"{scores.max():.2f} (near copies are above {SAME})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("look", help="draw every image onto sheets")
    commands.add_parser("suspects", help="find images that look unlike their class, and near copies")
    removing = commands.add_parser("remove", help="move images out of data/clean")
    removing.add_argument("files", nargs="+", help="class/file, e.g. wine_glass/0012.jpg")
    removing.add_argument("--reason", required=True,
                          help="which part of your rule it breaks")
    commands.add_parser("count", help="how many you removed, by class and by reason")
    commands.add_parser("overlap",
                        help="check that data/new_images holds no copy of a downloaded image")
    args = parser.parse_args()
    {"look": look, "suspects": suspects, "remove": remove, "count": count,
     "overlap": overlap}[args.command](args)


if __name__ == "__main__":
    main()
