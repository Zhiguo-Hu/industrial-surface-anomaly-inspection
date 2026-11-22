"""Record the active Python environment and verify a real CUDA operation."""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": {name: version(name) for name in
                     ["torch", "torchvision", "numpy", "Pillow", "scikit-learn"]},
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select the py312 interpreter and check the GPU driver.")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    torch.manual_seed(42)
    x = torch.randn(256, 256, device=device)
    result = x @ x.T
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise RuntimeError("CUDA matrix operation produced non-finite values")
    report.update({
        "gpu": props.name,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_mib": round(props.total_memory / 2**20, 1),
        "cuda_matrix_test": "passed",
        "cuda_matrix_shape": list(result.shape),
    })
    try:
        report["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
             "--format=csv"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report["nvidia_smi_error"] = str(exc)
    folder = ROOT / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "environment.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (folder / "environment-freeze.txt").write_text(frozen, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
