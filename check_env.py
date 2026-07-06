"""Environment check for gazetrack.

Verifies that every runtime dependency is importable and, critically, that
PyTorch is a CUDA build with a working GPU. A CPU-only PyTorch install is
the most common setup failure on Windows and would silently make inference
10-50x slower, so it is treated as a hard error.

Usage:
    python check_env.py [--camera]
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "models" / "eth-xgaze_resnet18.pth"

REINSTALL_HINT = (
    "    .venv\\Scripts\\python.exe -m pip install --force-reinstall "
    "torch==2.5.1+cu121 torchvision==0.20.1+cu121 "
    "--extra-index-url https://download.pytorch.org/whl/cu121"
)

_failures = 0


def _report(ok: bool, label: str, detail: str = "", fatal: bool = True) -> bool:
    global _failures
    tag = "[ OK ]" if ok else ("[FAIL]" if fatal else "[WARN]")
    line = f"{tag} {label}"
    if detail:
        line += f": {detail}"
    print(line)
    if not ok and fatal:
        _failures += 1
    return ok


def check_python() -> None:
    version = sys.version.split()[0]
    ok = sys.version_info[:2] == (3, 10)
    _report(ok, "Python", f"{version} (expected 3.10.x)", fatal=False)


def check_import(module: str, label: str | None = None):
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "unknown")
        _report(True, label or module, f"version {version}")
        return mod
    except ImportError as exc:
        _report(False, label or module, f"import failed ({exc}). "
                "Run: pip install -r requirements.txt")
        return None


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        _report(False, "PyTorch", f"import failed ({exc})")
        return

    _report(True, "PyTorch", f"version {torch.__version__}")

    if torch.version.cuda is None:
        _report(False, "PyTorch build",
                "this is a CPU-only build of PyTorch. Reinstall the CUDA "
                "build:\n" + REINSTALL_HINT)
        return
    _report(True, "PyTorch build", f"CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        _report(False, "torch.cuda.is_available()",
                "False. PyTorch has CUDA support but cannot reach the GPU. "
                "Check the NVIDIA driver (nvidia-smi) and that no other "
                "process holds the device.")
        return
    device_name = torch.cuda.get_device_name(0)
    _report(True, "torch.cuda.is_available()", f"True ({device_name})")

    # Exercise the GPU end to end; driver/toolkit mismatches often only
    # surface on the first real kernel launch.
    try:
        a = torch.randn(256, 256, device="cuda")
        b = (a @ a).sum().item()
        _report(abs(b) >= 0, "CUDA kernel launch", "matmul on GPU succeeded")
    except Exception as exc:
        _report(False, "CUDA kernel launch", str(exc))


def check_weights() -> None:
    if WEIGHTS_PATH.exists():
        size_mb = WEIGHTS_PATH.stat().st_size / 1e6
        _report(True, "ETH-XGaze weights",
                f"{WEIGHTS_PATH.name} ({size_mb:.1f} MB)")
    else:
        _report(False, "ETH-XGaze weights",
                f"not found at {WEIGHTS_PATH}. They are downloaded "
                "automatically on first model load, or run: "
                "python -m gaze.eth_xgaze --download-only", fatal=False)


def check_camera() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from capture.camera import Camera
    print("--- camera check ---")
    try:
        with Camera() as cam:
            ok = cam.verify()
        _report(ok, "Camera", "C920 delivering requested mode" if ok
                else "camera opened but not at 1920x1080")
    except RuntimeError as exc:
        _report(False, "Camera", str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="gazetrack environment check")
    parser.add_argument("--camera", action="store_true",
                        help="also open the camera and verify its mode")
    args = parser.parse_args()

    check_python()
    check_torch_cuda()
    check_import("torchvision")
    check_import("mediapipe", "MediaPipe")
    check_import("cv2", "OpenCV")
    check_import("sklearn", "scikit-learn")
    check_import("yaml", "PyYAML")
    check_weights()
    if args.camera:
        check_camera()

    print()
    if _failures:
        print(f"RESULT: {_failures} check(s) failed. Fix the items marked "
              "[FAIL] before running gazetrack.")
        return 1
    print("RESULT: environment OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
