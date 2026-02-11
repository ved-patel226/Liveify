import os
import re
import sys


def sanitize(filename: str) -> str:
    base, ext = os.path.splitext(filename)

    if " - Live" in base:
        parts = base.split(" - ")
        for i, part in enumerate(parts):
            if part == "Live" and i > 0:
                title = parts[i - 1]
                title = re.sub(r"^\d+\s+", "", title).strip()
                return title + ext

    if " - " in base:
        base = base.split(" - ", 1)[1]
    base = re.sub(r"\s*\([^)]*\)", "", base).strip()
    if " - " in base:
        base = base.split(" - ", 1)[0].strip()
    base = re.sub(r"\s+", " ", base)
    return base + ext


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base} ({i}){ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def main(root=None, dry=False):
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    for sub in ("studio", "live"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            src = os.path.join(d, name)
            if not os.path.isfile(src):
                continue
            newname = sanitize(name)
            dst = os.path.join(d, newname)
            dst = unique_path(dst)
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            if dry:
                print(f"Would rename: {src} -> {dst}")
            else:
                print(f"Renaming: {src} -> {dst}")
                os.rename(src, dst)


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    main(dry=dry)
