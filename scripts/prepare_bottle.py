"""Audit MVTec bottle, create a fixed normal-only validation split and previews."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def prepare(root, seed=42, val_fraction=0.2):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    # Accept the common extra bottle/bottle layer without moving original files.
    if not (root / "train").is_dir() and (root / "bottle" / "train").is_dir():
        root = root / "bottle"
    root = root.resolve()
    for name in ("train", "test", "ground_truth"):
        if not (root / name).is_dir():
            raise ValueError(f"Missing directory: {root / name}")
    rows, masks_used, hashes = [], set(), {}
    sizes = Counter()
    for split in ("train", "test"):
        images = sorted(p for p in (root / split).rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f"No images in {split}")
        for path in images:
            if len(path.relative_to(root / split).parts) != 2:
                raise ValueError(f"Unexpected image nesting: {path}")
            defect = path.parent.name
            if split == "train" and defect != "good":
                raise ValueError(f"Training data must be normal-only: {path}")
            with Image.open(path) as image:
                image.load()
                size = image.size
            sizes[f"{size[0]}x{size[1]}"] += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # Exact duplicate images are unsafe for evaluation and splitting.
            if digest in hashes:
                raise ValueError(f"Duplicate image content: {hashes[digest]} and {path}")
            hashes[digest] = str(path)
            mask_path = None
            if split == "test" and defect != "good":
                mask_path = root / "ground_truth" / defect / f"{path.stem}_mask.png"
                if not mask_path.is_file():
                    raise ValueError(f"Missing mask: {mask_path}")
                with Image.open(mask_path) as mask:
                    mask.load()
                    if mask.size != size:
                        raise ValueError(f"Mask size mismatch: {mask_path}")
                    values = np.asarray(mask.convert("L"))
                if not set(np.unique(values)).issubset({0, 255}) or not values.any():
                    raise ValueError(f"Expected nonempty binary 0/255 mask: {mask_path}")
                masks_used.add(mask_path.resolve())
            rows.append({"image": path.relative_to(ROOT).as_posix(),
                         "mask": mask_path.relative_to(ROOT).as_posix() if mask_path else None,
                         "label": int(defect != "good"), "defect_type": defect,
                         "official_split": split, "split": split,
                         "width": size[0], "height": size[1], "sha256": digest})
    actual_masks = {p.resolve() for p in (root / "ground_truth").rglob("*")
                    if p.suffix.lower() in EXTENSIONS}
    if actual_masks != masks_used:
        raise ValueError(f"Orphan mask files: {sorted(actual_masks - masks_used)}")
    normal = [r for r in rows if r["official_split"] == "train"]
    if len(normal) < 2:
        raise ValueError("At least two normal training images are required")
    random.Random(seed).shuffle(normal)
    count = max(1, min(len(normal) - 1, round(len(normal) * val_fraction)))
    for row in normal[:count]:
        row["split"] = "val"
    for split in ("train", "val", "test"):
        if not any(r["split"] == split for r in rows):
            raise ValueError(f"Empty split: {split}")
    return root, rows, sizes


def create_preview(rows, output):
    groups = [("train", "good"), ("test", "good")]
    groups += [("test", name) for name in sorted({r["defect_type"] for r in rows if r["label"]})]
    cell, header = 240, 32
    canvas = Image.new("RGB", (cell * 3, (cell + header) * len(groups)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (split, defect) in enumerate(groups):
        row = next(r for r in rows if r["official_split"] == split and r["defect_type"] == defect)
        with Image.open(ROOT / row["image"]) as img:
            image = img.convert("RGB")
        if row["mask"]:
            with Image.open(ROOT / row["mask"]) as mask_img:
                mask = mask_img.convert("L")
        else:
            mask = Image.new("L", image.size, 0)
        array = np.asarray(image).copy()
        selected = np.asarray(mask) > 0
        array[selected] = (0.55 * array[selected] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        panels = (image, mask.convert("RGB"), Image.fromarray(array))
        for j, panel in enumerate(panels):
            y = i * (cell + header)
            draw.text((j * cell + 5, y + 5), f"{split}/{defect} | {['image','mask','overlay'][j]}", fill="black")
            panel.thumbnail((cell, cell))
            canvas.paste(panel, (j * cell + (cell-panel.width)//2, y+header+(cell-panel.height)//2))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/mvtec_ad/bottle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    root, rows, sizes = prepare(args.root, args.seed, args.val_fraction)
    configs = ROOT / "configs"
    reports = ROOT / "reports"
    configs.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    manifest = {"category": "bottle", "root": root.relative_to(ROOT).as_posix(),
                "seed": args.seed, "val_fraction": args.val_fraction,
                "split_policy": "Validation from official train/good only; official test untouched",
                "samples": rows}
    (configs / "bottle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = {"data_root": str(root), "total_images": len(rows),
               "image_sizes_wh": dict(sizes),
               "official_counts": dict(Counter(f"{r['official_split']}/{r['defect_type']}" for r in rows)),
               "split_counts": dict(Counter(r["split"] for r in rows)),
               "mask_count": sum(r["mask"] is not None for r in rows),
               "checks": {"images_decodable": True, "unique_image_sha256": True,
                          "anomaly_masks_matched": True, "binary_nonempty_masks": True,
                          "image_mask_sizes_match": True, "no_orphan_masks": True,
                          "normal_only_train_and_val": True, "official_test_unchanged": True},
               "seed": args.seed}
    (reports / "bottle_data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_preview(rows, reports / "bottle_mask_preview.png")
    print(json.dumps(summary, indent=2))
    print("Saved configs/bottle_manifest.json and reports/bottle_mask_preview.png")


if __name__ == "__main__":
    main()
