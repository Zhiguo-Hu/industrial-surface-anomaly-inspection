"""Extract the 512-dimensional pooled ResNet18 embedding of one image."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "assets" / "synthetic_inspection_sample.png")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights" / "resnet18-f37072fd.pth")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "smoke_test")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}. See README for the official download command.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; choose the py312 interpreter or pass --device cpu")

    # Verify the official filename's SHA256 prefix before loading downloaded weights.
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise RuntimeError(f"Unexpected weight checksum: {digest}")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    # Remove the 1000-class classifier, retaining global average pooling.
    model.fc = nn.Identity()
    model = model.to(args.device).eval()
    with Image.open(args.image) as image:
        original_size = list(image.size)
        batch = weights.transforms()(image.convert("RGB")).unsqueeze(0).to(args.device)

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        features = model(batch)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if tuple(features.shape) != (1, 512) or not torch.isfinite(features).all().item():
        raise RuntimeError("Invalid embedding shape or non-finite features")
    cpu_features = features.cpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_features, args.output_dir / "features.pt")
    report = {
        "image": str(args.image.resolve()),
        "original_image_size_wh": original_size,
        "weights": "ResNet18 IMAGENET1K_V1 (official pretrained weights)",
        "weights_sha256": digest,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "preprocessing": "Official weights.transforms(): resize 256, center crop 224, ImageNet normalization",
        "input_shape": list(batch.shape),
        "feature_shape": list(cpu_features.shape),
        "features_finite": True,
        "feature_mean": cpu_features.mean().item(),
        "feature_std": cpu_features.std().item(),
        "single_cold_forward_seconds": elapsed,
        "timing_note": "Single cold forward, not a warmed-up performance benchmark.",
        "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2)
            if args.device == "cuda" else None,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved features and report: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
