"""Move p1..p5 evaluation images from data/clean/<class>/ to data/new_images/<class>/."""
import shutil
from pathlib import Path

data = Path(__file__).parent / "data"
for cls_dir in sorted((data / "clean").iterdir()):
    if not cls_dir.is_dir():
        continue
    dest = data / "new_images" / cls_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for i in range(1, 6):
        src = cls_dir / f"p{i}.jpg"
        if src.exists():
            shutil.move(str(src), str(dest / src.name))
            print("moved", src.relative_to(data), "->", (dest / src.name).relative_to(data))
for cls_dir in sorted((data / "new_images").iterdir()):
    print(cls_dir.name, len(list(cls_dir.glob("*"))))
