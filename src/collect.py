"""Download images for each class from DuckDuckGo image search.

This file is complete. You do not need to change it.

Usage:
    uv run python src/collect.py --classes "espresso cup,mug,wine glass" --n 120

It writes into data/raw/<class name>/0001.jpg and so on. Images that fail to
download, fail to open, or are too small are skipped and counted, so the number
you get back is usually smaller than the number you asked for.
"""

import argparse
import hashlib
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# The same versions as in pyproject.toml and requirements.txt. Other versions
# of these two have crashed the whole program on macOS or failed to search,
# with errors that do not say why.
PINNED = {"ddgs": "9.16.0", "primp": "1.3.1"}


def check_versions():
    """Stop before anything else if the search library is not what we tested."""
    wrong = []
    for name, wanted in PINNED.items():
        try:
            have = version(name)
        except PackageNotFoundError:
            have = "not installed"
        if have != wanted:
            wrong.append(f"  {name}: {have}, needs {wanted}")
    if not wrong:
        return
    python = ".".join(str(n) for n in sys.version_info[:3])
    sys.exit(
        f"The Python running this ({python}, {sys.executable}) has the wrong\n"
        "versions of the image search library:\n"
        + "\n".join(wrong)
        + "\n\nOn your own computer, run it through uv, which installs exactly"
        " the right ones:\n"
        '  uv run python src/collect.py --classes "..." --n 150\n'
        "On Colab, run the install cell of the notebook first:\n"
        "  !pip install -q -r requirements.txt\n"
        "Do not install other versions by hand."
    )


check_versions()

import requests  # noqa: E402  (after the check, which may stop us first)
from PIL import Image  # noqa: E402

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
MIN_SIDE = 64
TIMEOUT = 8
WORKERS = 16        # how many downloads to run at once
OVERSAMPLE = 3      # most search results are dead links, so gather far more
MAX_PAGES = 12      # one search page only ever returns about 35 results
BACKENDS = ["duckduckgo", "bing"]  # if one search engine fails, use the next


def safe_dirname(name):
    """Turn a search phrase into a folder name we can rely on."""
    keep = [c if (c.isalnum() or c in " -_") else " " for c in name]
    return "_".join("".join(keep).split()).lower()


def search_urls(query, how_many):
    """Collect links, a page at a time, until we have plenty.

    One search page only gives about 35 results however many you ask for, and
    a good share of those links are dead by the time we get to them. So we walk
    through pages and gather far more links than we actually need. If one search
    engine fails or runs out of results, we continue with the next one.
    """
    from ddgs import DDGS

    wanted = how_many * OVERSAMPLE
    urls = []
    seen = set()

    with DDGS() as ddgs:
        for backend in BACKENDS:
            for page in range(1, MAX_PAGES + 1):
                try:
                    hits = ddgs.images(
                        query, max_results=100, page=page, backend=backend
                    )
                except Exception as error:
                    print(f"  {backend} stopped at page {page}: {error}")
                    break

                before = len(urls)
                for hit in hits:
                    url = hit.get("image")
                    if url and url not in seen:
                        seen.add(url)
                        urls.append(url)

                if len(urls) == before:
                    break  # this page told us nothing new, so there is no more
                if len(urls) >= wanted:
                    break

            if len(urls) >= wanted:
                break  # enough links, no need to ask the next search engine

    return urls


def collect_one_class(query, how_many, root):
    """Download up to `how_many` usable images for one search phrase.

    The downloads run side by side, because most of the waiting is the network
    rather than us. A lock keeps the counter and the duplicate list honest.
    """
    out_dir = root / safe_dirname(query)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = search_urls(query, how_many)
    print(f'"{query}": search returned {len(urls)} links, downloading...')

    lock = threading.Lock()
    taken_hashes = set()
    kept = [0]

    def fetch(url):
        if kept[0] >= how_many:
            return
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            raw = response.content

            image = Image.open(io.BytesIO(raw))
            image.load()
            if min(image.size) < MIN_SIDE:
                return
            image = image.convert("RGB")
        except Exception:
            return  # a dead link is completely normal, just move on

        digest = hashlib.md5(raw).hexdigest()
        with lock:
            if kept[0] >= how_many or digest in taken_hashes:
                return
            taken_hashes.add(digest)
            kept[0] += 1
            number = kept[0]
        image.save(out_dir / f"{number:04d}.jpg", "JPEG", quality=90)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(fetch, urls))

    print(f'"{query}": kept {kept[0]} images in {out_dir}')
    return kept[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes",
        required=True,
        help='comma separated search phrases, e.g. "mug,wine glass,paper cup"',
    )
    parser.add_argument("--n", type=int, default=120, help="images per class")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    queries = [q.strip() for q in args.classes.split(",") if q.strip()]
    if len(queries) < 3:
        sys.exit("You need at least 3 classes.")

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    counts = {}
    for query in queries:
        counts[query] = collect_one_class(query, args.n, root)

    print("\nSummary")
    for query, kept in counts.items():
        print(f"  {query:30s} {kept}")
    smallest = min(counts.values())
    if smallest < 50:
        print(
            f"\nWarning: the smallest class has only {smallest} images. "
            "Try a different search phrase for that class."
        )


if __name__ == "__main__":
    main()
