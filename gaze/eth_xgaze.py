"""ETH-XGaze gaze direction estimator (layer 1).

Pipeline per frame:
1. Take the FaceObservation from gaze.mediapipe_head (head pose + center).
2. Warp the face into the ETH-XGaze normalized space: a virtual camera at a
   fixed 600 mm distance and 960 px focal length looking straight at the
   face center, rolled so the head x-axis is horizontal. This is what makes
   the model robust to head pose.
3. Run the CNN (ResNet-18 trained on ETH-XGaze) on the 224x224 patch. The
   network outputs (pitch, yaw) in radians in the normalized space.
4. Rotate the gaze vector back to the real camera frame and report yaw and
   pitch in degrees.

This module owns everything model-specific (weights, preprocessing,
normalization). It knows nothing about screens or calibration - that is
layer 2 (mapping/).

Weights: ResNet-18 checkpoint trained on ETH-XGaze, published as a release
asset of https://github.com/hysts/pytorch_mpiigaze_demo (v0.2.2). It is
downloaded automatically on first use and verified by SHA-256.

Angle conventions (camera space): yaw positive = subject looking to their
right (toward image left); pitch positive = looking up.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import resnet18
from torchvision.transforms.functional import normalize, to_tensor

from gaze.mediapipe_head import FaceObservation, make_camera_matrix

WEIGHTS_URL = ("https://github.com/hysts/pytorch_mpiigaze_demo/releases/"
               "download/v0.2.2/eth-xgaze_resnet18.pth")
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models" \
    / "eth-xgaze_resnet18.pth"
WEIGHTS_SIZE = 44790835
WEIGHTS_SHA256 = ("9cad8d47ae99e769732c8eeda54f58a0d644c983"
                  "b5269aa74b45f08d013ba63f")

# ETH-XGaze data-normalization constants (Zhang et al.).
NORM_DISTANCE_MM = 600.0
NORM_FOCAL_PX = 960.0
INPUT_SIZE = 224

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_NORM_CAMERA_MATRIX = np.array([
    [NORM_FOCAL_PX, 0.0, INPUT_SIZE / 2.0],
    [0.0, NORM_FOCAL_PX, INPUT_SIZE / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


@dataclass
class GazeResult:
    """Gaze direction for one frame."""

    yaw_deg: float                 # camera space, positive = subject's right
    pitch_deg: float               # camera space, positive = up
    gaze_vector: np.ndarray        # (3,) unit vector, camera frame
    yaw_norm_deg: float            # normalized space (model raw output)
    pitch_norm_deg: float
    patch: np.ndarray | None = None  # normalized 224x224 face (debug)


def download_weights(path: Path = WEIGHTS_PATH) -> Path:
    """Download and verify the model checkpoint if not already present."""
    if path.exists() and _weights_valid(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ETH-XGaze weights ({WEIGHTS_SIZE / 1e6:.1f} MB) "
          f"to {path} ...")
    tmp = path.with_suffix(".tmp")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    if not _weights_valid(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded weights failed verification (size or SHA-256 "
            f"mismatch). Retry, or fetch manually from {WEIGHTS_URL}")
    tmp.replace(path)
    print("Download complete and verified.")
    return path


def _weights_valid(path: Path) -> bool:
    if path.stat().st_size != WEIGHTS_SIZE:
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == WEIGHTS_SHA256


def _vector_from_angles(pitch: float, yaw: float) -> np.ndarray:
    """ETH-XGaze convention: (pitch, yaw) radians -> unit gaze vector."""
    return -np.array([
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
        np.cos(pitch) * np.cos(yaw),
    ])


def _angles_from_vector(vec: np.ndarray) -> tuple[float, float]:
    """Inverse of _vector_from_angles: unit vector -> (pitch, yaw) radians."""
    pitch = np.arcsin(np.clip(-vec[1], -1.0, 1.0))
    yaw = np.arctan2(-vec[0], -vec[2])
    return float(pitch), float(yaw)


class EthXGazeEstimator:
    """CUDA wrapper around the ETH-XGaze ResNet-18 model."""

    def __init__(self, device: str = "cuda",
                 weights_path: Path | None = None):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run 'python check_env.py' for a "
                "diagnosis (most likely a CPU-only PyTorch build or a "
                "driver problem). Pass device='cpu' only for debugging.")
        self.device = torch.device(device)

        path = download_weights(weights_path or WEIGHTS_PATH)
        checkpoint = torch.load(path, map_location="cpu",
                                weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = {k.removeprefix("module."): v
                      for k, v in state_dict.items()}

        self._model = resnet18(num_classes=2)
        self._model.load_state_dict(state_dict)
        self._model.eval().to(self.device)

        # Warm up so the first real frame does not pay CUDA init cost.
        with torch.inference_mode():
            self._model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE,
                                    device=self.device))

    def normalize_face(self, frame_bgr: np.ndarray, obs: FaceObservation,
                       camera_matrix: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Warp the face into the ETH-XGaze normalized camera.

        Returns (patch_bgr 224x224, rot_norm) where rot_norm is the rotation
        from the real camera frame to the normalized frame.
        """
        center = obs.face_center.astype(np.float64)
        distance = float(np.linalg.norm(center))
        forward = center / distance
        head_x = obs.rotation[:, 0]
        down = np.cross(forward, head_x)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        rot_norm = np.vstack([right, down, forward])

        scale = np.diag([1.0, 1.0, NORM_DISTANCE_MM / distance])
        warp = _NORM_CAMERA_MATRIX @ scale @ rot_norm \
            @ np.linalg.inv(camera_matrix)
        patch = cv2.warpPerspective(frame_bgr, warp,
                                    (INPUT_SIZE, INPUT_SIZE))
        return patch, rot_norm

    @torch.inference_mode()
    def predict_normalized(self, patch_bgr: np.ndarray
                           ) -> tuple[float, float]:
        """Run the CNN on a normalized patch. Returns (pitch, yaw) radians."""
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        tensor = normalize(to_tensor(rgb), _IMAGENET_MEAN, _IMAGENET_STD)
        output = self._model(tensor.unsqueeze(0).to(self.device))
        pitch, yaw = output[0].tolist()
        return pitch, yaw

    def estimate(self, frame_bgr: np.ndarray, obs: FaceObservation,
                 camera_matrix: np.ndarray | None = None,
                 keep_patch: bool = False) -> GazeResult:
        """Full layer-1 step: observation -> gaze angles in camera space."""
        if camera_matrix is None:
            h, w = frame_bgr.shape[:2]
            camera_matrix = make_camera_matrix(w, h)

        patch, rot_norm = self.normalize_face(frame_bgr, obs, camera_matrix)
        pitch_n, yaw_n = self.predict_normalized(patch)

        gaze_norm = _vector_from_angles(pitch_n, yaw_n)
        gaze_cam = rot_norm.T @ gaze_norm
        gaze_cam /= np.linalg.norm(gaze_cam)
        pitch_c, yaw_c = _angles_from_vector(gaze_cam)

        return GazeResult(
            yaw_deg=float(np.degrees(yaw_c)),
            pitch_deg=float(np.degrees(pitch_c)),
            gaze_vector=gaze_cam,
            yaw_norm_deg=float(np.degrees(yaw_n)),
            pitch_norm_deg=float(np.degrees(pitch_n)),
            patch=patch if keep_patch else None,
        )


# Live-check HUD: px of ray length per degree of gaze angle. Generous on
# purpose so a 5 deg yaw is already clearly visible (~45 px).
_RAY_GAIN_PX_PER_DEG = 9.0
_BAR_RANGE_DEG = 25.0


def _draw_bar_h(frame: np.ndarray, x: int, y: int, w: int, h: int,
                value_px: float, label: str) -> None:
    """Horizontal bar filled from the center; value_px is signed fill."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (110, 110, 110), 1)
    cx = x + w // 2
    cv2.line(frame, (cx, y), (cx, y + h), (110, 110, 110), 1)
    fill = int(np.clip(value_px, -w / 2, w / 2))
    cv2.rectangle(frame, (cx, y + 2), (cx + fill, y + h - 2),
                  (235, 235, 235), -1)
    cv2.putText(frame, label, (x + w + 8, y + h - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1,
                cv2.LINE_AA)


def _draw_bar_v(frame: np.ndarray, x: int, y: int, w: int, h: int,
                value_px: float, label: str) -> None:
    """Vertical bar filled from the center; positive fills upward."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (110, 110, 110), 1)
    cy = y + h // 2
    cv2.line(frame, (x, cy), (x + w, cy), (110, 110, 110), 1)
    fill = int(np.clip(value_px, -h / 2, h / 2))
    cv2.rectangle(frame, (x + 2, cy), (x + w - 2, cy - fill),
                  (235, 235, 235), -1)
    cv2.putText(frame, label, (x, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1,
                cv2.LINE_AA)


def _draw_overlay(frame: np.ndarray, obs: FaceObservation, gaze: GazeResult,
                  camera_matrix: np.ndarray, fps: float) -> None:
    """Status panel, gaze ray and yaw/pitch bars for the live check.

    The ray is drawn in 2D with a fixed px-per-degree gain instead of
    projecting the 3D gaze vector: a gaze aimed roughly at the camera
    projects to nearly a point, which made the previous ray invisible.
    Screen mapping: yaw positive (subject's right) points toward image
    left; pitch positive (up) points toward image top.
    """
    origin, _ = cv2.projectPoints(obs.face_center.reshape(1, 3),
                                  np.zeros(3), np.zeros(3), camera_matrix,
                                  None)
    p0 = origin.reshape(2).astype(int)
    dx = -_RAY_GAIN_PX_PER_DEG * gaze.yaw_deg
    dy = -_RAY_GAIN_PX_PER_DEG * gaze.pitch_deg
    p1 = (int(p0[0] + dx), int(p0[1] + dy))
    cv2.line(frame, tuple(p0), p1, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.circle(frame, p1, 6, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.circle(frame, tuple(p0), 3, (110, 110, 110), -1, cv2.LINE_AA)

    lines = [
        f"gaze  yaw {gaze.yaw_deg:+6.1f} deg   pitch {gaze.pitch_deg:+6.1f} deg",
        f"head  yaw {obs.head_yaw:+6.1f} deg   pitch {obs.head_pitch:+6.1f} deg",
        f"{fps:5.1f} fps   q to quit",
    ]
    cv2.rectangle(frame, (0, 0), (430, 22 * len(lines) + 12), (32, 32, 32),
                  -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
                    cv2.LINE_AA)

    # Bars share the ray's screen convention so all three visualizations
    # agree: yaw fills left for positive yaw, pitch fills up for looking up.
    bar_scale = (300 / 2) / _BAR_RANGE_DEG  # half-bar px per degree
    _draw_bar_h(frame, 10, 92, 300, 14, -gaze.yaw_deg * bar_scale,
                f"yaw +/-{_BAR_RANGE_DEG:.0f} deg")
    bar_scale_v = (200 / 2) / _BAR_RANGE_DEG
    _draw_bar_v(frame, 10, 120, 14, 200, gaze.pitch_deg * bar_scale_v,
                "pitch")


def main() -> int:
    """Live check: camera -> face mesh -> ETH-XGaze -> yaw/pitch on screen.

    This is the verification target for the end of implementation step (d):
    the printed yaw/pitch and the drawn ray must follow the eyes, including
    under moderate head movement.
    """
    from capture.camera import Camera
    from gaze.mediapipe_head import PLAUSIBLE_DISTANCE_MM, HeadTracker

    parser = argparse.ArgumentParser(
        description="ETH-XGaze live inference check")
    parser.add_argument("--index", type=int, default=None,
                        help="camera index (default: data/camera.yaml, "
                             "else 0)")
    parser.add_argument("--device", default="cuda",
                        help="inference device (default: cuda)")
    parser.add_argument("--show-patch", action="store_true",
                        help="also show the normalized 224x224 face patch")
    parser.add_argument("--download-only", action="store_true",
                        help="download and verify the weights, then exit")
    args = parser.parse_args()

    if args.download_only:
        download_weights()
        return 0

    estimator = EthXGazeEstimator(device=args.device)
    window = "gazetrack - live gaze check"

    with Camera(index=args.index) as camera, HeadTracker() as tracker:
        # Startup 'fps (measured)' is pure capture throughput; if it reads
        # ~30 here but the HUD fps is low, the bottleneck is processing,
        # not the camera.
        camera.verify()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        frame_times: list[float] = []
        last_print = 0.0
        sanity_done = False
        while True:
            frame = camera.read()
            if frame is None:
                print("ERROR: frame grab failed.")
                return 1
            h, w = frame.shape[:2]
            camera_matrix = tracker.camera_matrix(w, h)

            now = time.perf_counter()
            frame_times = [t for t in frame_times if now - t < 2.0] + [now]
            fps = len(frame_times) / 2.0

            obs = tracker.process(frame)
            if obs is not None:
                if not sanity_done:
                    # Startup sanity check: the estimated eye-to-camera
                    # distance is the canary for a wrong 3D face model
                    # scale, which silently crops the eyes out of the
                    # normalized patch and kills the gaze signal.
                    dist = float(np.linalg.norm(obs.face_center))
                    lo, hi = PLAUSIBLE_DISTANCE_MM
                    in_range = lo <= dist <= hi
                    print(f"PnP sanity: eye-to-camera distance ~{dist:.0f} mm"
                          f"  [{'ok' if in_range else 'OUT OF RANGE'}]")
                    if not in_range:
                        print(f"WARNING: distance is outside the plausible "
                              f"desktop band ({lo:.0f}-{hi:.0f} mm). If the "
                              "gaze ignores your eyes, the 3D face model "
                              "scale in gaze/mediapipe_head.py is likely "
                              "wrong (the normalized patch would crop the "
                              "eyes). Use --show-patch to confirm both eyes "
                              "are visible and centered.")
                    sanity_done = True
                gaze = estimator.estimate(frame, obs, camera_matrix,
                                          keep_patch=args.show_patch)
                _draw_overlay(frame, obs, gaze, camera_matrix, fps)
                if args.show_patch and gaze.patch is not None:
                    cv2.imshow("normalized face", gaze.patch)
                if now - last_print >= 1.0:
                    print(f"yaw {gaze.yaw_deg:+6.1f} deg   "
                          f"pitch {gaze.pitch_deg:+6.1f} deg   "
                          f"{fps:5.1f} fps")
                    last_print = now
            else:
                cv2.putText(frame, "no face detected", (10, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 220),
                            1, cv2.LINE_AA)
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
